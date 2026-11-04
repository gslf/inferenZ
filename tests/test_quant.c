// Accuracy and speed check for the LUT quant kernels against a float reference.
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../src/quant.h"
#include "../src/util.h"

static float randn(uint64_t *s) {
    double u1, u2;
    *s = *s * 6364136223846793005ull + 1442695040888963407ull;
    u1 = ((*s >> 11) + 1.0) / 9007199254740993.0;
    *s = *s * 6364136223846793005ull + 1442695040888963407ull;
    u2 = (*s >> 11) / 9007199254740992.0;
    return (float)(sqrt(-2 * log(u1)) * cos(2 * M_PI * u2));
}

int main(int argc, char **argv) {
    int rows = argc > 1 ? atoi(argv[1]) : 1024, cols = argc > 2 ? atoi(argv[2]) : 2048;
    uint64_t seed = 42;
    float *W = malloc(sizeof(float) * rows * cols), *x = malloc(sizeof(float) * cols);
    // heavy-ish tails + a few outlier columns, like real LLM weights
    for (int i = 0; i < rows * cols; i++) {
        float v = randn(&seed) * 0.02f;
        if ((i % cols) % 97 == 0) v *= 6;
        W[i] = v;
    }
    for (int j = 0; j < cols; j++) x[j] = randn(&seed) * (j % 61 == 0 ? 8.0f : 1.0f);
    float *yref = malloc(sizeof(float) * rows), *y = malloc(sizeof(float) * rows);
    for (int r = 0; r < rows; r++) {
        double s = 0;
        for (int j = 0; j < cols; j++) s += (double)W[(size_t)r * cols + j] * x[j];
        yref[r] = (float)s;
    }
    int types[] = {INZ_T_Q2L, INZ_T_Q3L, INZ_T_Q4L, INZ_T_Q8G, INZ_T_Q3P, INZ_T_Q3P, INZ_T_Q4N};
    printf("kernel ISA: %s, matrix %dx%d\n", inz_kernel_isa(), rows, cols);
    for (int rot = 0; rot < 2; rot++)
        for (size_t ti = 0; ti < 7; ti++) {
            int t = types[ti];
            if (t == INZ_T_Q4N && rot) continue;  // NVFP4 keeps the checkpoint's basis
            size_t nb = inz_qmat_bytes(t, rows, cols);
            void *buf = inz_aligned(64, nb);
            inz_quant_params p = {.type = t, .rot_block = rot ? 0 : -1};
            double t0 = inz_now();
            double qerr = inz_quantize(W, rows, cols, &p, buf);
            double tq = inz_now() - t0;
            inz_qmat m;
            inz_qmat_view(&m, buf);
            if (ti == 5) m.coarse = true;  // coarse-only decode of a planar Q3P
            // reference check: dequantized row vs original
            float *row = malloc(sizeof(float) * cols);
            double e2 = 0, n2 = 0;
            for (int r = 0; r < rows; r += 37) {
                inz_dequantize_row_orig(&m, r, row);
                for (int j = 0; j < cols; j++) {
                    double d = row[j] - W[(size_t)r * cols + j];
                    e2 += d * d;
                    n2 += (double)W[(size_t)r * cols + j] * W[(size_t)r * cols + j];
                }
            }
            inz_act a;
            inz_act_alloc(&a, cols);
            inz_act_quantize(&a, x, cols, m.rot_block, m.rot_seed);
            inz_matvec_rows(&m, &a, y, 0, rows);
            double ye = 0, yn = 0;
            for (int r = 0; r < rows; r++) { ye += (y[r] - yref[r]) * (y[r] - yref[r]); yn += yref[r] * yref[r]; }
            // multi-token and prefill-GEMM paths must agree with the matvec
            double me = 0, ge = 0;
            {
                const int nt = 3;
                inz_act acts[3] = {a, a, a};
                float *Y = malloc(sizeof(float) * rows * nt);
                inz_matmul_rows(&m, acts, nt, Y, rows, 0, rows);
                for (int r = 0; r < rows; r++) me = fmax(me, fabs(Y[(size_t)2 * rows + r] - y[r]));
                inz_xq xq;
                inz_xq_alloc(&xq, 16, cols);
                float *X = malloc(sizeof(float) * cols * 16);
                for (int t2 = 0; t2 < 16; t2++) memcpy(X + (size_t)t2 * cols, x, sizeof(float) * cols);
                inz_xq_quantize(&xq, X, cols, 16, cols, m.rot_block, m.rot_seed);
                float *G = malloc(sizeof(float) * rows * 16);
                uint8_t *tb = inz_aligned(64, inz_gemm_tilebuf_bytes(cols));
                inz_gemm_rows(&m, &xq, G, rows, 0, rows, tb);
                double gn = 0;
                for (int r = 0; r < rows; r++) { ge += (G[(size_t)5 * rows + r] - yref[r]) * (G[(size_t)5 * rows + r] - yref[r]); gn += yref[r] * yref[r]; }
                ge = sqrt(ge / gn);
                free(Y); free(X); free(G); free(tb); inz_xq_free(&xq);
            }
            printf("     matmul max|diff| %.2e   gemm y-relerr %.5f\n", me, ge);
            // speed: repeat matvec on a larger working set to stay out of cache
            int reps = 200;
            t0 = inz_now();
            for (int k = 0; k < reps; k++) inz_matvec_rows(&m, &a, y, 0, rows);
            double el = inz_now() - t0;
            double bytes = (double)nb * reps;
            printf("%-4s%s rot=%-3d bpw=%.3f  w-relerr=%.5f  y-relerr=%.5f (lut-fit %.5f)  1-thread %.2f GB/s (%.1f Gw/s)  quant %.2fs\n",
                   inz_qtype_name(t), m.coarse ? "-coarse" : "", m.rot_block, 8.0 * nb / ((double)rows * cols), sqrt(e2 / n2), sqrt(ye / yn), sqrt(qerr),
                   bytes / el / 1e9, (double)rows * cols * reps / el / 1e9, tq);
            free(row);
            inz_act_free(&a);
            free(buf);
        }
    int fails = 0;
    {
        // F8T: e4m3 codes (subnormals and negatives included) with a per-tensor scale must be exact, and the matmul
        // rows bit-identical to the matvec's
        uint8_t *w8 = malloc((size_t)rows * cols);
        for (size_t i = 0; i < (size_t)rows * cols; i++) {
            uint8_t v = (uint8_t)(seed = seed * 6364136223846793005ull + 1442695040888963407ull) >> 0;
            v = (uint8_t)((seed >> 33) & 0xFF);
            if ((v & 0x7F) == 0x7F) v ^= 1;  // no NaN
            w8[i] = v;
        }
        const float scale = 0.0123f;
        void *buf = inz_aligned(64, inz_qmat_bytes(INZ_T_F8T, rows, cols));
        inz_pack_f8t(w8, scale, rows, cols, buf);
        inz_qmat m;
        inz_qmat_view(&m, buf);
        float *row = malloc(sizeof(float) * cols);
        double me = 0, worst = 0;
        for (int r = 0; r < rows; r++) {
            inz_dequantize_row(&m, r, row);
            double s = 0, sa = 0;
            for (int j = 0; j < cols; j++) {
                const float ref = inz_e4m3_to_f32(w8[(size_t)r * cols + j]) * scale;
                if (row[j] != ref) me = 1;
                s += (double)ref * x[j];
                sa += fabs((double)ref * x[j]);
            }
            yref[r] = (float)s;
            row[0] = (float)sa;
            y[r] = row[0];  // |terms| for the error bound
        }
        float *yv = malloc(sizeof(float) * rows), *Y = malloc(sizeof(float) * rows * 5);
        inz_matvec_f8t_rows(&m, x, yv, 0, rows);
        for (int r = 0; r < rows; r++) worst = fmax(worst, fabs(yv[r] - yref[r]) / (y[r] + 1e-30));
        float *X = malloc(sizeof(float) * cols * 5);
        for (int t = 0; t < 5; t++) memcpy(X + (size_t)t * cols, x, sizeof(float) * cols);
        int ident = 1;
        for (int n = 2; n <= 5; n++) {
            inz_matmul_f8t_rows(&m, X, cols, n, Y, rows, 0, rows);
            for (int t = 0; t < n; t++)
                for (int r = 0; r < rows; r++) ident &= Y[(size_t)t * rows + r] == yv[r];
        }
        const int reps = 200;
        double t0 = inz_now();
        for (int k = 0; k < reps; k++) inz_matvec_f8t_rows(&m, x, yv, 0, rows);
        const double el = inz_now() - t0;
        printf("F8T  dequant exact %s  matvec max rel err %.2e  matmul == matvec %s  1-thread %.2f GB/s\n", me ? "NO" : "yes", worst,
               ident ? "yes" : "NO", (double)rows * cols * reps / el / 1e9);
        fails += me != 0 || worst > 1e-5 || !ident;
        free(w8); free(buf); free(row); free(yv); free(Y); free(X);
    }
    {
        // Q4N with zero and subnormal block scales (they occur in real checkpoints): the kernels must match the
        // dequantized weights
        uint8_t *pk = malloc((size_t)rows * cols / 2), *sc = malloc((size_t)rows * cols / 16);
        for (size_t i = 0; i < (size_t)rows * cols / 2; i++) pk[i] = (uint8_t)((seed = seed * 6364136223846793005ull + 1442695040888963407ull) >> 40);
        for (size_t i = 0; i < (size_t)rows * cols / 16; i++) {
            const uint8_t v = (uint8_t)((seed = seed * 6364136223846793005ull + 1442695040888963407ull) >> 40) & 0x7F;
            sc[i] = i % 7 == 0 ? 0 : i % 5 == 0 ? (v & 7) : v == 0x7F ? 0x7E : v;  // zeros, subnormals, normals
        }
        void *buf = inz_aligned(64, inz_qmat_bytes(INZ_T_Q4N, rows, cols));
        const int ok = inz_pack_q4n_mul(pk, sc, 0.37f, rows, cols, buf);
        inz_qmat m;
        inz_qmat_view(&m, buf);
        inz_act a;
        inz_act_alloc(&a, cols);
        inz_act_quantize(&a, x, cols, 0, 0);
        float *row = malloc(sizeof(float) * cols);
        double worst = 0;
        inz_matvec_rows(&m, &a, y, 0, rows);
        for (int r = 0; r < rows; r++) {
            inz_dequantize_row(&m, r, row);
            double s = 0, sa = 0;
            for (int g = 0; g < cols / 64; g++)
                for (int j = 0; j < 64; j++) {
                    s += (double)row[g * 64 + j] * a.q[g * 64 + j] * a.s[g];  // same int8 activation as the kernel
                    sa += fabs((double)row[g * 64 + j] * a.q[g * 64 + j] * a.s[g]);
                }
            worst = fmax(worst, fabs(y[r] - s) / (sa + 1e-30));
        }
        printf("Q4N  zero/subnormal block scales: pack %s, matvec max rel err %.2e\n", ok ? "ok" : "REFUSED", worst);
        fails += !ok || worst > 1e-5;
        inz_act_free(&a);
        free(pk); free(sc); free(buf); free(row);
    }
    if (fails) printf("FAILED: %d check(s)\n", fails);
    return fails ? 1 : 0;
}

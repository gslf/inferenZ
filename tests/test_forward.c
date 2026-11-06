#include <assert.h>
#include "../src/model_fwd.c"

int main(void) {
    enum { R = 64, C = 256, N = 17 };
    inz_engine *e = calloc(1, sizeof(*e));
    e->tp = tp_create(2, NULL);
    q_scratch_alloc(e);
    float W[R * C], X[N * C], a[N * R], b[N * R];
    for (int i = 0; i < R * C; i++) W[i] = sinf(i * 0.17f) * 0.03f;
    for (int i = 0; i < N * C; i++) X[i] = cosf(i * 0.13f);
    int types[] = {INZ_T_Q4L, INZ_T_Q8G};
    int counts[] = {1, 2, 3, 4, 7, 8, 9, 12, N};
    for (int rotation = 0; rotation < 2; rotation++) {
        void *data[2]; lin m[2];
        for (int t = 0; t < 2; t++) {
            data[t] = inz_aligned(64, inz_qmat_bytes(types[t], R, C));
            inz_quant_params p = {.type = types[t], .rot_block = rotation ? 128 : -1, .rot_seed = 42, .threads = 1};
            inz_quantize(W, R, C, &p, data[t]);
            assert(q_bind_lin(&m[t], types[t], R, C, data[t]));
        }
        for (int t = 0; t < 2; t++) for (size_t k = 0; k < sizeof(counts) / sizeof(*counts); k++) {
            int n = counts[k];
            lin_mm(e, &m[t], X, C, n, a, R);
            lin_mm_after(e, &m[1 - t], &m[t], X, C, n, b, R);
            lin_mm(e, &m[1 - t], X, C, n, a, R);
            assert(!memcmp(a, b, sizeof(float) * n * R));
            m[t].q.rot_seed++;
            lin_mm(e, &m[t], X, C, n, b, R);
            lin_mm_after(e, &m[1 - t], &m[t], X, C, n, b, R);
            assert(!memcmp(a, b, sizeof(float) * n * R));
            m[t].q.rot_seed--;
        }
        free(data[0]); free(data[1]);
    }
    // matrices sharing an input in one dispatch, the input quantized by it in slices == each kernel on its own
    // input quantized whole (K = 320: slices of 64)
    static float W2[R * 320], X2[N * 320], Yg[3][N * R], Ys[N * R];
    for (int i = 0; i < R * 320; i++) W2[i] = sinf(i * 0.11f) * 0.03f;
    for (int i = 0; i < N * 320; i++) X2[i] = cosf(i * 0.07f) * (i % 97 == 0 ? 9.0f : 1.0f);
    inz_act acts[N];
    for (int t = 0; t < N; t++) inz_act_alloc(&acts[t], 320);
    for (int rotation = 0; rotation < 2; rotation++)
        for (int K = 256; K <= 320; K += 64) {
            const int rows[3] = {R, 40, 6};
            void *data[2]; lin m[3];
            for (int t = 0; t < 2; t++) {
                data[t] = inz_aligned(64, inz_qmat_bytes(types[t], rows[t], K));
                inz_quant_params p = {.type = types[t], .rot_block = rotation ? 64 : -1, .rot_seed = 7, .threads = 1};
                inz_quantize(W2, rows[t], K, &p, data[t]);
                assert(q_bind_lin(&m[t], types[t], rows[t], K, data[t]));
            }
            assert(q_bind_lin(&m[2], INZ_T_F32, 6, K, (const uint8_t *)W2));
            for (size_t k = 0; k < sizeof(counts) / sizeof(*counts); k++) {
                const int n = counts[k];
                if (n >= GEMM_MIN_T) continue;
                const lin *Wg[3] = {&m[0], &m[1], &m[2]};
                float *Y[3] = {Yg[0], Yg[1], Yg[2]};
                lin_mm_group(e, Wg, Y, rows, 3, X2, 320, n);
                for (int i = 0; i < 3; i++) {
                    if (i == 2)
                        for (int t = 0; t < n; t++) inz_matvec_f32_rows(W2, K, X2 + t * 320, Ys + t * rows[i], 0, rows[i]);
                    else {
                        for (int t = 0; t < n; t++) inz_act_quantize(&acts[t], X2 + t * 320, K, m[i].q.rot_block, m[i].q.rot_seed);
                        if (n == 1) inz_matvec_rows(&m[i].q, acts, Ys, 0, rows[i]);
                        else inz_matmul_rows(&m[i].q, acts, n, Ys, rows[i], 0, rows[i]);
                    }
                    assert(!memcmp(Yg[i], Ys, sizeof(float) * n * rows[i]));
                }
            }
            free(data[0]); free(data[1]);
        }
    for (int t = 0; t < N; t++) inz_act_free(&acts[t]);
    q_scratch_free(e);
    tp_destroy(e->tp);
    free(e);
    puts("Projection reuse and grouped dispatch: identical outputs, including rotation changes");
}

#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "../src/quant.h"
#include "../src/util.h"

int main(void) {
    enum { R = 9, C = 320, N = 17 };
    float W[R * C], x[C], ref[N * R], out[N * R];
    int types[] = {INZ_T_Q4L, INZ_T_Q8G};
    int counts[] = {1, 2, 3, 4, 7, 8, 9, N};
    for (int rotation = 0; rotation < 2; rotation++) for (int t = 0; t < 2; t++) {
        for (int i = 0; i < R * C; i++) W[i] = sinf(i * .17f) * .03f;
        size_t bytes = inz_qmat_bytes(types[t], R, C);
        void *data = inz_aligned(64, bytes);
        inz_quant_params p = {.type = types[t], .rot_block = rotation ? 64 : -1, .rot_seed = 42, .threads = 1};
        inz_quantize(W, R, C, &p, data);
        inz_qmat m; assert(inz_qmat_view(&m, data));
        inz_act acts[N];
        for (int n = 0; n < N; n++) {
            for (int j = 0; j < C; j++) x[j] = cosf(j * .13f + n * .23f);
            inz_act_alloc(&acts[n], C);
            inz_act_quantize(&acts[n], x, C, m.rot_block, m.rot_seed);
            inz_matvec_rows(&m, &acts[n], ref + n * R, 0, R);
        }
        for (size_t k = 0; k < sizeof(counts) / sizeof(*counts); k++) {
            int n = counts[k];
            for (int i = 0; i < N * R; i++) out[i] = 12345;
            inz_matmul_rows(&m, acts, n, out, R, 1, R - 1);
            for (int i = 0; i < N; i++) for (int r = 0; r < R; r++)
                assert(out[i * R + r] == (i < n && r > 0 && r < R - 1 ? ref[i * R + r] : 12345));
        }
        for (int n = 0; n < N; n++) inz_act_free(&acts[n]);
        free(data);
    }
    uint16_t half[R * C];
    float inputs[N * (C + 1)];
    for (int i = 0; i < R * C; i++) half[i] = inz_f32_to_f16(W[i]);
    for (int i = 0; i < N * (C + 1); i++) inputs[i] = cosf(i * .13f);
    for (int n = 2; n <= 3; n++) for (int cols = C - 1; cols <= C; cols++) {
        for (int t = 0; t < n; t++) inz_matvec_f16_rows(half, cols, inputs + t * (C + 1), ref + t * R, 0, R);
        inz_matmul_f16_rows(half, cols, inputs, C + 1, n, out, R, 0, R);
        assert(!memcmp(ref, out, sizeof(float) * n * R));
    }
    // Four large int8 products must be added in 32 bits, including on AVX2.
    for (int i = 0; i < R * C; i++) W[i] = i / C % 2 ? -1 : 1;
    for (int j = 0; j < C; j++) x[j] = 1;
    void *data = inz_aligned(64, inz_qmat_bytes(INZ_T_Q8G, R, C));
    inz_quant_params p = {.type = INZ_T_Q8G, .rot_block = -1, .threads = 1};
    inz_quantize(W, R, C, &p, data);
    inz_qmat m; assert(inz_qmat_view(&m, data));
    inz_act a; inz_act_alloc(&a, C); inz_act_quantize(&a, x, C, 0, 0);
    inz_matvec_rows(&m, &a, out, 0, R);
    for (int r = 0; r < R; r++) assert(fabsf(out[r] - (r % 2 ? -C : C)) < .1f);
    inz_act_free(&a); free(data);
    printf("%s: distinct batch inputs match, Q8 accumulation does not overflow\n", inz_kernel_isa());
}

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
    q_scratch_free(e);
    tp_destroy(e->tp);
    free(e);
    puts("Projection reuse: identical outputs, including rotation changes");
}

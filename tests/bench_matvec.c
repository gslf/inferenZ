// Multi-threaded matvec bandwidth: how close each format gets to DRAM bandwidth.
//   bench_matvec [threads] [all|TYPE] [rows cols]   (threads 0: the engine's placement, one worker per P and E core)
#include <math.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>

#include "../src/quant.h"
#include "../src/threadpool.h"
#include "../src/util.h"

typedef struct { const inz_qmat *m; const uint16_t *h; int cols; const inz_act *a; const float *x; float *y; } job;

static void run(void *ctx, int64_t b, int64_t e, int tid) {
    (void)tid;
    job *j = ctx;
    if (j->h) inz_matvec_f16_rows(j->h, j->cols, j->x, j->y, (int)b, (int)e);
    else inz_matvec_rows(j->m, j->a, j->y, (int)b, (int)e);
}

int main(int argc, char **argv) {
    int nthreads = argc > 1 ? atoi(argv[1]) : 0;
    const char *only = argc > 2 && strcasecmp(argv[2], "all") ? argv[2] : NULL;
    int rows = argc > 4 ? atoi(argv[3]) : 16384, cols = argc > 4 ? atoi(argv[4]) : 8192;
    tp_topology topo;
    tp_detect_topology(&topo);
    int cpus[256];
    tp_pool *tp = nthreads > 0 ? tp_create(nthreads, NULL) : tp_create(tp_default_cpus(&topo, cpus, 256), cpus);
    printf("threads=%d isa=%s cpus:", tp_size(tp), inz_kernel_isa());
    for (int i = 0; i < tp_size(tp); i++) printf(" %d", tp_cpus(tp)[i]);
    printf("\n%d x %d\n", rows, cols);
    float *W = malloc(sizeof(float) * (size_t)rows * cols);
    uint64_t s = 1;
    for (size_t i = 0; i < (size_t)rows * cols; i++) { s = s * 6364136223846793005ull + 1; W[i] = ((int64_t)(s >> 33) % 2001 - 1000) * 1e-5f; }
    float *x = malloc(sizeof(float) * cols), *y = malloc(sizeof(float) * rows);
    for (int j = 0; j < cols; j++) x[j] = sinf(j * 0.37f);
    // Q3P appears twice: full (coarse + fine planes) and coarse-only (2-bit plane, as after a partial read)
    int types[] = {INZ_T_Q2L, INZ_T_Q3L, INZ_T_Q3P, INZ_T_Q3P, INZ_T_Q4L, INZ_T_Q8G, INZ_T_Q4N, INZ_T_F16};
    for (int ti = 0; ti < 8; ti++) {
        int t = types[ti];
        const bool coarse = ti == 3;
        if (only && (strcasecmp(only, inz_qtype_name(t)) || coarse)) continue;
        size_t nb = inz_mat_bytes(t, rows, cols);
        // 4 copies so the working set is far beyond L3
        void *buf[4];
        inz_qmat m[4];
        inz_quant_params p = {.type = t, .rot_block = -1, .lut_iters = 1};
        buf[0] = inz_aligned(64, nb);
        if (t == INZ_T_F16) for (size_t i = 0; i < (size_t)rows * cols; i++) ((uint16_t *)buf[0])[i] = inz_f32_to_f16(W[i]);
        else inz_quantize(W, rows, cols, &p, buf[0]);
        for (int k = 1; k < 4; k++) { buf[k] = inz_aligned(64, nb); memcpy(buf[k], buf[0], nb); }
        if (t != INZ_T_F16) for (int k = 0; k < 4; k++) { inz_qmat_view(&m[k], buf[k]); m[k].coarse = coarse; }
        const double rb = coarse ? (double)inz_q3p_coarse_bytes(rows, cols) : (double)nb;  // bytes actually read
        inz_act a;
        inz_act_alloc(&a, cols);
        inz_act_quantize(&a, x, cols, 0, 0);
        int reps = 40;
        for (int grain = 16; grain <= 64; grain *= 2) {
            double t0 = inz_now();
            for (int r = 0; r < reps; r++) {
                job j = {&m[r & 3], t == INZ_T_F16 ? buf[r & 3] : NULL, cols, &a, x, y};
                tp_for(tp, rows, grain, run, &j);
            }
            double el = inz_now() - t0;
            printf("%-4s%s grain=%-3d %6.1f GB/s  %6.1f Gweights/s  %.2f ms/matvec (%.1f MB)\n", inz_qtype_name(t),
                   coarse ? "c" : " ", grain, rb * reps / el / 1e9, (double)rows * cols * reps / el / 1e9, el / reps * 1e3, rb / 1e6);
        }
        inz_act_free(&a);
        for (int k = 0; k < 4; k++) free(buf[k]);
    }
    const tp_stats *st = tp_get_stats(tp);
    printf("per-thread busy share:");
    double tot = 0;
    for (int i = 0; i < tp_size(tp); i++) tot += st->chunks[i];
    for (int i = 0; i < tp_size(tp); i++) printf(" %d:%.1f%%", tp_cpus(tp)[i], 100.0 * st->chunks[i] / tot);
    printf("\n");
    tp_destroy(tp);
    return 0;
}

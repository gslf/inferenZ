// bench_q38: speed mock of one Qwen3.8-Flash-Next decode / verify step on this machine, before any download.
//
// Every weight matrix has the model's exact shape and the chosen inferenZ format (random values: speed only).
// Per step and per layer it runs what the real forward pass reads:
//   - 36 Gated DeltaNet layers: in_proj_qkv 10240x2560, in_proj_z 6144x2560, in_proj_a/b 2x48x2560,
//     recurrent state update (48 heads x 128x128 f32, read + write), out_proj 2560x6144
//   - 12 QSA layers: q 12288x2560 (gated), k/v 1024x2560, indexer 640x2560, sparse attention over 2048 positions
//     (2 KV heads x 256, f16), o 2560x6144
//   - every layer: hyper-connection mixers (2 x 2 x 320x2560), router 512x2560, shared expert, 10 routed experts
//     (gate|up 1280x2560, down 2560x640) drawn from a large resident pool, union over the n tokens of a step
//   - LM head screen (Q3L, 248320x2560); the exact rows of a few candidates are ignored
//
//   cc -O3 -march=native -std=gnu23 -fopenmp -o tools/bench_q38 tools/bench_q38.c src/quant.c src/threadpool.c src/util.c -lm
//   tools/bench_q38 NONEXPERT_TYPE EXPERT_TYPE [seconds]      e.g.  tools/bench_q38 q8g q3l 60
#include <immintrin.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../src/quant.h"
#include "../src/threadpool.h"
#include "../src/util.h"

#define H 2560
#define NL 48
#define NE 512
#define TOPK 10
#define EFF 640
#define POOL 96            // distinct experts allocated per layer (working set >> caches)
#define MAXT 8

typedef struct { inz_qmat m; void *buf; } qm;

typedef struct { int type, rows, cols; void *buf; } tmpl;
static tmpl tmpls[64];
static int ntmpl;

// quantize one random template per (type, shape) and copy it: allocation is fast, contents irrelevant
static qm make(int type, int rows, int cols, int slot) {
    qm r;
    size_t nb = inz_qmat_bytes(type, rows, cols);
    int ti = -1;
    for (int i = 0; i < ntmpl; i++) if (tmpls[i].type == type && tmpls[i].rows == rows && tmpls[i].cols == cols) ti = i;
    if (ti < 0) {
        ti = ntmpl++;
        tmpls[ti] = (tmpl){type, rows, cols, NULL};
        float *W = malloc(sizeof(float) * (size_t)rows * cols);
        uint64_t s = 0x1234 + slot;
        for (size_t i = 0; i < (size_t)rows * cols; i++) { s = s * 6364136223846793005ull + 1; W[i] = ((int)(s >> 41) % 2001 - 1000) * 1e-4f; }
        tmpls[ti].buf = inz_aligned(64, nb);
        inz_quant_params p = {.type = type, .rot_block = type == INZ_T_Q8G || type == INZ_T_Q4N ? -1 : 0, .lut_iters = 1};
        inz_quantize(W, rows, cols, &p, tmpls[ti].buf);
        free(W);
    }
    r.buf = inz_aligned(64, nb);
    memcpy(r.buf, tmpls[ti].buf, nb);
    inz_qmat_view(&r.m, r.buf);
    return r;
}

typedef struct { const inz_qmat *m; const inz_act *a; int n; float *y; int ld; } mv_job;
static void mv_task(void *ctx, int64_t b, int64_t e, int tid) {
    (void)tid;
    mv_job *j = ctx;
    if (j->n == 1) inz_matvec_rows(j->m, j->a, j->y, (int)b, (int)e);
    else inz_matmul_rows(j->m, j->a, j->n, j->y, j->ld, (int)b, (int)e);
}

// several matrices with the same input in one dispatch: tasks = (matrix, row chunk)
typedef struct { const inz_qmat *m[64]; const inz_act *a[64]; int nt[64]; float *y[64]; int n; int64_t start[65]; } multi_job;
static void multi_task(void *ctx, int64_t b, int64_t e, int tid) {
    (void)tid;
    multi_job *j = ctx;
    for (int64_t t = b; t < e; t++) {
        int i = 0;
        while (j->start[i + 1] <= t) i++;
        int r0 = (int)(t - j->start[i]) * 32, r1 = r0 + 32;
        if (r1 > j->m[i]->rows) r1 = j->m[i]->rows;
        if (j->nt[i] == 1) inz_matvec_rows(j->m[i], j->a[i], j->y[i], r0, r1);
        else inz_matmul_rows(j->m[i], j->a[i], j->nt[i], j->y[i], j->m[i]->rows, r0, r1);
    }
}

static tp_pool *TP;
static float *Y;

static void run_multi(multi_job *j) {
    j->start[0] = 0;
    for (int i = 0; i < j->n; i++) j->start[i + 1] = j->start[i] + (j->m[i]->rows + 31) / 32;
    tp_for(TP, j->start[j->n], 1, multi_task, j);
}

typedef struct { float *S; int n; } st_job;
static void state_task(void *ctx, int64_t b, int64_t e, int tid) {
    (void)tid;
    st_job *j = ctx;
    for (int64_t h = b; h < e; h++) {
        float *s = j->S + h * 128 * 128;
        for (int t = 0; t < j->n; t++)
            for (int i = 0; i < 128 * 128; i += 8) {
                __m256 v = _mm256_loadu_ps(s + i);
                _mm256_storeu_ps(s + i, _mm256_fmadd_ps(v, _mm256_set1_ps(0.999f), _mm256_set1_ps(1e-6f)));
            }
    }
}

typedef struct { const uint16_t *kv; int n; float *out; } at_job;
static void att_task(void *ctx, int64_t b, int64_t e, int tid) {
    (void)tid;
    at_job *j = ctx;
    for (int64_t c = b; c < e; c++) {  // (token, kv head, 256-position chunk)
        const uint16_t *k = j->kv + (size_t)(c % 16) * 256 * 512;
        __m256 acc = _mm256_setzero_ps();
        for (int p = 0; p < 256; p++)
            for (int d = 0; d < 512; d += 8) acc = _mm256_add_ps(acc, _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)(k + p * 512 + d))));
        _mm256_storeu_ps(j->out + c * 8, acc);
    }
}

int main(int argc, char **argv) {
    if (argc < 3) { fprintf(stderr, "usage: %s NONEXPERT_TYPE EXPERT_TYPE [seconds]\n", argv[0]); return 1; }
    const int tne = inz_qtype_from_name(argv[1]), tex = inz_qtype_from_name(argv[2]);
    const double secs = argc > 3 ? atof(argv[3]) : 30;
    tp_topology topo;
    tp_detect_topology(&topo);
    int cpus[64];
    int nt = tp_default_cpus(&topo, cpus, 64);
    // weights
    double t0 = inz_now();
    qm *qkv = calloc(NL, sizeof(qm)), *z = calloc(NL, sizeof(qm)), *ab = calloc(NL, sizeof(qm)), *out = calloc(NL, sizeof(qm));
    qm *hc = calloc(NL, sizeof(qm)), *router = calloc(NL, sizeof(qm)), *sgu = calloc(NL, sizeof(qm)), *sd = calloc(NL, sizeof(qm));
    qm (*egu)[POOL] = calloc(NL, sizeof(*egu)), (*ed)[POOL] = calloc(NL, sizeof(*ed));
    size_t dense_b = 0, exp_b = 0;
    for (int l = 0; l < NL; l++) {
        const bool qsa = l % 4 == 3;
        qkv[l] = qsa ? make(tne, 12288 + 1024 + 640, H, 0) : make(tne, 10240, H, 1);
        z[l] = qsa ? make(INZ_T_Q8G, 64, H, 2) : make(tne, 6144, H, 2);
        ab[l] = make(INZ_T_Q8G, 96, H, 3);
        out[l] = make(tne, H, 6144, 0);
        hc[l] = make(INZ_T_Q8G, 4 * 320, H, 1);
        router[l] = make(INZ_T_Q8G, NE, H, 2);
        sgu[l] = make(tne, 2 * EFF, H, 3);
        sd[l] = make(tne, H, EFF, 0);
        for (int e = 0; e < POOL; e++) { egu[l][e] = make(tex, 2 * EFF, H, 1); ed[l][e] = make(tex, H, EFF, 2); }
        dense_b += qkv[l].m.rows * 0 + inz_qmat_bytes(qkv[l].m.type, qkv[l].m.rows, H) + inz_qmat_bytes(z[l].m.type, z[l].m.rows, H) +
                   inz_qmat_bytes(out[l].m.type, H, 6144) + inz_qmat_bytes(INZ_T_Q8G, 96 + 1280 + NE, H) +
                   inz_qmat_bytes(tne, 2 * EFF, H) + inz_qmat_bytes(tne, H, EFF);
    }
    exp_b = inz_qmat_bytes(tex, 2 * EFF, H) + inz_qmat_bytes(tex, H, EFF);
    qm head = make(INZ_T_Q3L, 248320, H, 3);
    uint16_t *kv = inz_aligned(64, (size_t)16 * 256 * 512 * 2);
    memset(kv, 0x3c, (size_t)16 * 256 * 512 * 2);
    float *S = inz_aligned(64, sizeof(float) * 36 * 48 * 128 * 128);
    memset(S, 0, sizeof(float) * 36 * 48 * 128 * 128);
    TP = tp_create(nt, cpus);
    Y = inz_aligned(64, sizeof(float) * MAXT * 300000);
    float *att_out = inz_aligned(64, sizeof(float) * MAXT * 2 * 16 * 8 * 64);
    inz_act acts[MAXT], acts_e[MAXT], acts_6144[MAXT], acts_640[MAXT * (TOPK + 1)];
    float *x = malloc(sizeof(float) * 16384);
    for (int i = 0; i < 16384; i++) x[i] = sinf(i * 0.1f);
    for (int t = 0; t < MAXT; t++) {
        inz_act_alloc(&acts[t], H); inz_act_alloc(&acts_e[t], H); inz_act_alloc(&acts_6144[t], 6144);
        inz_act_quantize(&acts[t], x, H, qkv[0].m.rot_block, qkv[0].m.rot_seed);
        inz_act_quantize(&acts_e[t], x, H, egu[0][0].m.rot_block, egu[0][0].m.rot_seed);
        inz_act_quantize(&acts_6144[t], x, 6144, out[0].m.rot_block, out[0].m.rot_seed);
    }
    for (int i = 0; i < MAXT * (TOPK + 1); i++) { inz_act_alloc(&acts_640[i], EFF); inz_act_quantize(&acts_640[i], x, EFF, ed[0][0].m.rot_block, ed[0][0].m.rot_seed); }
    printf("non-expert %s, experts %s: dense %.2f GB (all layers), expert %.2f MiB each, setup %.0f s, %d threads\n",
           argv[1], argv[2], dense_b / 1e9, exp_b / 1048576.0, inz_now() - t0, nt);
    printf("model in RAM at this recipe: dense %.1f GB + experts %.1f GB + head screen %.2f GB\n", dense_b / 1e9,
           (double)exp_b * NE * NL / 1e9, inz_qmat_bytes(INZ_T_Q3L, 248320, H) / 1e9);

    uint64_t rng = 99;
    const int nlist[4] = {1, 1, 2, 4};
    for (int ni = 0; ni < 4; ni++) {
        const int n = nlist[ni];
        double tstart = inz_now();
        int steps = 0;
        double busy = 0;
        while (inz_now() - tstart < secs / 3) {
            double ts = inz_now();
            for (int l = 0; l < NL; l++) {
                const bool qsa = l % 4 == 3;
                // hyper-connection mixers + attention input projections in one dispatch
                multi_job a = {0};
                a.m[a.n] = &hc[l].m; a.a[a.n] = acts; a.nt[a.n] = n; a.y[a.n++] = Y;
                a.m[a.n] = &qkv[l].m; a.a[a.n] = acts; a.nt[a.n] = n; a.y[a.n++] = Y + 20000;
                a.m[a.n] = &z[l].m; a.a[a.n] = acts; a.nt[a.n] = n; a.y[a.n++] = Y + 80000;
                a.m[a.n] = &ab[l].m; a.a[a.n] = acts; a.nt[a.n] = n; a.y[a.n++] = Y + 120000;
                run_multi(&a);
                if (qsa) {
                    at_job aj = {kv, n, att_out};
                    tp_for(TP, (int64_t)n * 2 * 8, 1, att_task, &aj);
                } else {
                    st_job sj = {S + (size_t)(l - l / 4) * 48 * 128 * 128, n};
                    tp_for(TP, 48, 1, state_task, &sj);
                }
                mv_job oj = {&out[l].m, acts_6144, n, Y, H};
                tp_for(TP, H, 32, mv_task, &oj);
                // router, then shared + routed experts (union over the step's tokens)
                mv_job rj = {&router[l].m, acts, n, Y, NE};
                tp_for(TP, NE, 32, mv_task, &rj);
                int sel[MAXT * TOPK], ns = 0;
                for (int t = 0; t < n * TOPK; t++) {
                    rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17;
                    int e = (int)(rng % POOL), dup = 0;
                    for (int q = 0; q < ns; q++) if (sel[q] == e) dup = 1;
                    if (!dup) sel[ns++] = e;
                }
                const int per = (n * TOPK + ns - 1) / ns;  // tokens per expert (approximation)
                multi_job g = {0};
                g.m[g.n] = &sgu[l].m; g.a[g.n] = acts; g.nt[g.n] = n; g.y[g.n++] = Y;
                for (int q = 0; q < ns; q++) { g.m[g.n] = &egu[l][sel[q]].m; g.a[g.n] = acts_e; g.nt[g.n] = per > n ? n : per; g.y[g.n++] = Y + 5000 + q * 2600 * MAXT; }
                run_multi(&g);
                multi_job d = {0};
                d.m[d.n] = &sd[l].m; d.a[d.n] = acts_640; d.nt[d.n] = n; d.y[d.n++] = Y;
                for (int q = 0; q < ns; q++) { d.m[d.n] = &ed[l][sel[q]].m; d.a[d.n] = acts_640 + (q % TOPK); d.nt[d.n] = per > n ? n : per; d.y[d.n++] = Y + 5000 + q * 2600 * MAXT; }
                run_multi(&d);
            }
            mv_job hj = {&head.m, acts, n, Y, 248320};
            tp_for(TP, 248320, 256, mv_task, &hj);
            busy += inz_now() - ts;
            steps++;
        }
        const double ms = 1e3 * busy / steps;
        printf("%sstep with %d token(s): %.1f ms  -> %.1f steps/s  (%d steps)\n", ni == 0 ? "(warm-up) " : "", n, ms, 1e3 / ms, steps);
        fflush(stdout);
    }
    return 0;
}

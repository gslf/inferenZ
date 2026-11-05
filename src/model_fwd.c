// Forward pass of Laguna S 2.1 (docs/MODEL.md gives the math; every block below follows it line by line).
//
// One code path serves decode (n = 1) and prefill chunks (n <= Q_MAX_T). Weights are `lin` matrices in any
// storage type; activations stay f32 between operators. Projections, attention and experts are batched over the
// chunk.
#define _GNU_SOURCE
#include <immintrin.h>
#include <stdatomic.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "diag.h"
#include "engine_internal.h"
#include "util.h"
#include "vmath.h"

/* ------------------------------------------------------------------ */
/* Small math                                                          */
/* ------------------------------------------------------------------ */

static inline float dot_f32(const float *a, const float *b, int n) {
    __m256 s0 = _mm256_setzero_ps(), s1 = _mm256_setzero_ps();
    int i = 0;
    for (; i + 16 <= n; i += 16) {
        s0 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i), _mm256_loadu_ps(b + i), s0);
        s1 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i + 8), _mm256_loadu_ps(b + i + 8), s1);
    }
    float s = hsum256_ps(_mm256_add_ps(s0, s1));
    for (; i < n; i++) s += a[i] * b[i];
    return s;
}

// out = x * rsqrt(mean(x^2) + eps) * w (NULL: no weight)
static void rmsnorm(float *out, const float *x, const float *w, int n) {
    const float r = 1.0f / sqrtf(dot_f32(x, x, n) / n + Q_RMS_EPS);
    for (int i = 0; i < n; i++) out[i] = x[i] * r * (w ? w[i] : 1.0f);
}

static inline float siluf(float x) { return x / (1.0f + expf(-x)); }
static inline float sigmoidf(float x) { return 1.0f / (1.0f + expf(-x)); }
static inline float softplusf(float x) { return x > 20.0f ? x : log1pf(expf(x)); }  // torch threshold 20

static void swiglu(float *gu, int ff) {  // gu = [gate | up] -> silu(gate) * up in the first ff values
    for (int c = 0; c < ff; c++) gu[c] = siluf(gu[c]) * gu[ff + c];
}

static void f32_to_f16(uint16_t *dst, const float *src, int n) {
    int i = 0;
    for (; i + 8 <= n; i += 8) _mm_storeu_si128((__m128i *)(dst + i), _mm256_cvtps_ph(_mm256_loadu_ps(src + i), 0));
    for (; i < n; i++) dst[i] = inz_f32_to_f16(src[i]);
}

static inline float dot_f16(const uint16_t *a, const float *b, int n) {  // n multiple of 16
    __m256 s0 = _mm256_setzero_ps(), s1 = _mm256_setzero_ps();
    for (int i = 0; i < n; i += 16) {
        s0 = _mm256_fmadd_ps(_mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)(a + i))), _mm256_loadu_ps(b + i), s0);
        s1 = _mm256_fmadd_ps(_mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)(a + i + 8))), _mm256_loadu_ps(b + i + 8), s1);
    }
    return hsum256_ps(_mm256_add_ps(s0, s1));
}

static inline void axpy_f16(float *y, float w, const uint16_t *x, int n) {  // y += w * x, n multiple of 8
    const __m256 vw = _mm256_set1_ps(w);
    for (int i = 0; i < n; i += 8)
        _mm256_storeu_ps(y + i, _mm256_fmadd_ps(vw, _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)(x + i))), _mm256_loadu_ps(y + i)));
}

/* ------------------------------------------------------------------ */
/* Matrices                                                            */
/* ------------------------------------------------------------------ */

// types whose kernels take f32 activations (the others quantize the activation to int8 first)
static inline bool float_act(int type) { return type == INZ_T_F32 || type == INZ_T_F16 || type == INZ_T_F8T; }

bool q_bind_lin(lin *m, int type, int rows, int cols, const uint8_t *data) {
    memset(m, 0, sizeof(*m));
    m->type = type;
    m->rows = rows;
    m->cols = cols;
    if (type == INZ_T_F16) { m->h = (const uint16_t *)data; return true; }
    if (type == INZ_T_F32) { m->f = (const float *)data; return true; }
    if (!inz_qmat_view(&m->q, data) || m->q.type != type) return false;
    return m->q.rows == rows && m->q.cols == cols;
}

uint64_t q_lin_bytes(const lin *W) { return inz_mat_bytes(W->type, W->rows, W->cols); }

typedef struct { const lin *W; const float *X; int ldx, n; float *Y; int ldy; const inz_act *acts; } mm_job;

static void mm_task(void *ctx, int64_t b, int64_t e_, int tid) {
    (void)tid;
    mm_job *j = ctx;
    const lin *W = j->W;
    const int r0 = (int)b, r1 = (int)e_;
    if (W->type == INZ_T_F8T) {
        if (j->n == 1) inz_matvec_f8t_rows(&W->q, j->X, j->Y, r0, r1);
        else inz_matmul_f8t_rows(&W->q, j->X, j->ldx, j->n, j->Y, j->ldy, r0, r1);
        return;
    }
    if (W->type == INZ_T_F16 && j->n > 1) {
        inz_matmul_f16_rows(W->h, W->cols, j->X, j->ldx, j->n, j->Y, j->ldy, r0, r1);
        return;
    }
    if (W->type == INZ_T_F32 || W->type == INZ_T_F16) {
        for (int t = 0; t < j->n; t++) {
            const float *x = j->X + (size_t)t * j->ldx;
            float *y = j->Y + (size_t)t * j->ldy;
            if (W->type == INZ_T_F32) inz_matvec_f32_rows(W->f, W->cols, x, y, r0, r1);
            else inz_matvec_f16_rows(W->h, W->cols, x, y, r0, r1);
        }
        return;
    }
    if (j->n == 1) inz_matvec_rows(&W->q, j->acts, j->Y, r0, r1);
    else inz_matmul_rows(&W->q, j->acts, j->n, j->Y, j->ldy, r0, r1);
}

#define GEMM_MIN_T 12

typedef struct { inz_xq *x; const float *X; int ldx; float **tmp; } xq_job;
static void xq_task(void *ctx, int64_t b, int64_t e_, int tid) {
    xq_job *j = ctx;
    inz_xq_rows(j->x, j->X, j->ldx, (int)b, (int)e_, j->tmp[tid]);
}

typedef struct { const inz_qmat *m; const inz_xq *x; float *Y; int ldy; uint8_t **tb; } gemm_job;
static void gemm_task(void *ctx, int64_t b, int64_t e_, int tid) {
    gemm_job *j = ctx;
    int r1 = (int)e_ * 16;
    if (r1 > j->m->rows) r1 = j->m->rows;
    inz_gemm_rows(j->m, j->x, j->Y, j->ldy, (int)b * 16, r1, j->tb[tid]);
}

typedef struct { inz_act *acts; const float *X; int ldx, n; int rot; uint32_t seed; } aq_job;
static void aq_task(void *ctx, int64_t b, int64_t e_, int tid) {
    (void)tid;
    aq_job *j = ctx;
    for (int64_t t = b; t < e_; t++) inz_act_quantize(&j->acts[t], j->X + (size_t)t * j->ldx, j->n, j->rot, j->seed);
}

/* ------------------------------------------------------------------ */
/* Importance matrix                                                   */
/* ------------------------------------------------------------------ */

int q_imx_register(inz_engine *e, const char *name, int cols) {
    if (e->n_imx == e->cap_imx) {
        e->cap_imx = e->cap_imx ? 2 * e->cap_imx : 512;
        e->imx = realloc(e->imx, sizeof(imx_entry) * (size_t)e->cap_imx);
    }
    imx_entry *x = &e->imx[e->n_imx];
    snprintf(x->name, sizeof(x->name), "%s", name);
    x->cols = cols;
    x->n = 0;
    x->acc = inz_calloc((size_t)cols, sizeof(double));
    return ++e->n_imx;
}

static void imx_add(double *acc, const float *X, int ldx, int n, int cols) {
    for (int t = 0; t < n; t++) {
        const float *x = X + (size_t)t * ldx;
        for (int c = 0; c < cols; c++) acc[c] += (double)x[c] * x[c];
    }
}

// file: "INZIMX01", u32 count, then per entry: u16 name length, name, u32 cols, u64 samples, f32[cols] mean x^2
void q_imx_write(inz_engine *e) {
    FILE *f = fopen(e->imx_path, "wb");
    if (!f) { LOGI("imatrix: cannot write %s", e->imx_path); return; }
    const int nl = e->n_moe;
    uint32_t count = 0;
    for (int i = 0; i < e->n_imx; i++) count += e->imx[i].n > 0;
    for (int k = 0; k < nl * Q_N_EXPERT; k++) count += 2 * (e->imx_exp_n[k] > 0);
    fwrite("INZIMX01", 1, 8, f);
    fwrite(&count, 4, 1, f);
    float *buf = inz_malloc(sizeof(float) * Q_DENSE_FF);
    for (int i = 0; i < e->n_imx; i++) {
        const imx_entry *x = &e->imx[i];
        if (!x->n) continue;
        const uint16_t len = (uint16_t)strlen(x->name);
        const uint32_t cols = (uint32_t)x->cols;
        for (int c = 0; c < x->cols; c++) buf[c] = (float)(x->acc[c] / (double)x->n);
        fwrite(&len, 2, 1, f); fwrite(x->name, 1, len, f); fwrite(&cols, 4, 1, f); fwrite(&x->n, 8, 1, f);
        fwrite(buf, 4, cols, f);
    }
    for (int k = 0; k < nl * Q_N_EXPERT; k++) {
        if (!e->imx_exp_n[k]) continue;
        const int l = k / Q_N_EXPERT, xid = k % Q_N_EXPERT;
        const double *acc = e->imx_exp + (size_t)k * (Q_H + Q_FF);
        for (int part = 0; part < 2; part++) {
            char name[48];
            snprintf(name, sizeof(name), "blk.%d.experts.%d.%s", l + 1, xid, part ? "down" : "gate_up");
            const uint16_t len = (uint16_t)strlen(name);
            const uint32_t cols = part ? Q_FF : Q_H;
            for (uint32_t c = 0; c < cols; c++) buf[c] = (float)(acc[(part ? Q_H : 0) + c] / (double)e->imx_exp_n[k]);
            fwrite(&len, 2, 1, f); fwrite(name, 1, len, f); fwrite(&cols, 4, 1, f); fwrite(&e->imx_exp_n[k], 8, 1, f);
            fwrite(buf, 4, cols, f);
        }
    }
    free(buf);
    fclose(f);
    LOGI("imatrix: %u entries written to %s", count, e->imx_path);
}

static void mm_group_run(inz_engine *e, const lin *const *W, float *const *Y, const int *ldy, int nw, const float *X, int ldx,
                         int n, const inz_qmat *qm, bool quantize);

// Y[t*ldy + r] = W[r] . X[t*ldx]  for t < n, r < W->rows. `prev`: the matrix run just before on the same input,
// whose quantized activation can be reused.
static void lin_mm_after(inz_engine *e, const lin *W, const lin *prev, const float *X, int ldx, int n, float *Y, int ldy) {
    if (W->imx) {
        imx_entry *x = &e->imx[W->imx - 1];
        imx_add(x->acc, X, ldx, n, W->cols);
        x->n += (uint64_t)n;
    }
    if (n == 1 && !W->imx) {
        const inz_qmat *qm = float_act(W->type) ? NULL : &W->q;
        const bool reuse = qm && prev && !float_act(prev->type) && prev->cols == W->cols && prev->q.rot_block == qm->rot_block &&
                           prev->q.rot_seed == qm->rot_seed;
        mm_group_run(e, &W, &Y, &ldy, 1, X, ldx, n, qm, !reuse);
        return;
    }
    if (!float_act(W->type)) {
        const inz_qmat *m = &W->q;
        const bool reuse = prev && !float_act(prev->type) && prev->cols == W->cols && prev->q.rot_block == m->rot_block &&
                           prev->q.rot_seed == m->rot_seed;
        if (n >= GEMM_MIN_T) {
            if (!reuse) {
                inz_xq_begin(&e->gemm_x, n, m->cols, m->rot_block, m->rot_seed);
                xq_job xj = {&e->gemm_x, X, ldx, e->gemm_tmp};
                tp_for(e->tp, n, 4, xq_task, &xj);
            }
            gemm_job gj = {m, &e->gemm_x, Y, ldy, e->gemm_tb};
            tp_for(e->tp, (m->rows + 15) / 16, 1, gemm_task, &gj);
            return;
        }
        if (!reuse) {
            aq_job aj = {e->acts, X, ldx, W->cols, m->rot_block, m->rot_seed};
            tp_for(e->tp, n, 1, aq_task, &aj);
        }
    }
    mm_job j = {W, X, ldx, n, Y, ldy, e->acts};
    const bool big_k = W->cols >= 6144;
    int grain = W->type != INZ_T_F32 && n > 1 && float_act(W->type) ? (big_k ? 12 : 24) : big_k ? 16 : 32;
    if (n > 32 && (W->type == INZ_T_F16 || W->type == INZ_T_F8T)) {
        // many tokens: a task streams every token block once, so it should own many rows (fewer X passes), while
        // leaving about 8 tasks per thread for the balance between P- and E-cores
        const int g8 = W->rows / (8 * tp_size(e->tp));
        grain = g8 > 128 ? 128 : g8 > grain ? g8 : grain;
    }
    tp_for(e->tp, W->rows, grain, mm_task, &j);
}

static void lin_mm(inz_engine *e, const lin *W, const float *X, int ldx, int n, float *Y, int ldy) {
    lin_mm_after(e, W, NULL, X, ldx, n, Y, ldy);
}

// Several matrices on the same input in one dispatch, for a few tokens: one wake-up of the pool instead of one per
// matrix, and the small matrices stream at the big one's bandwidth. For one token the input is quantized by the
// same dispatch: its first tasks quantize slices of it in parallel (the previous kernels' threads hold its cache
// lines, which a single thread would fetch one by one), the others wait for them and then run rows.
#define MMG_MAX 4
#define MMG_SLICE 256
typedef struct {
    mm_job j[MMG_MAX];
    int start[MMG_MAX + 1], rows[MMG_MAX];   // first row block and rows per block of each matrix (after the slices)
    int nq;                                  // quantization slices
    inz_act *a;
    const float *x;
    atomic_int qleft;
} mmg_job;

static void mmg_task(void *ctx, int64_t b, int64_t e_, int tid) {
    mmg_job *g = ctx;
    for (int64_t k = b; k < e_; k++) {
        if (k < g->nq) {
            inz_act_quantize_part(g->a, g->x, (int)k * MMG_SLICE, (int)(k + 1) * MMG_SLICE);
            atomic_fetch_sub_explicit(&g->qleft, 1, memory_order_release);
            continue;
        }
        // slices are handed out first, so every one still pending is held by a running thread
        while (atomic_load_explicit(&g->qleft, memory_order_acquire) > 0) _mm_pause();
        const int64_t kb = k - g->nq;
        int w = 0;
        while (kb >= g->start[w + 1]) w++;
        const int r0 = (int)(kb - g->start[w]) * g->rows[w], rows = g->j[w].W->rows;
        mm_task(&g->j[w], r0, r0 + g->rows[w] < rows ? r0 + g->rows[w] : rows, tid);
    }
}

// whether the group can run as one dispatch: no importance matrix, same input width, one rotation for the quantized
static bool mm_groupable(const lin *const *W, int nw, int n, const inz_qmat **qm) {
    *qm = NULL;
    if (n >= GEMM_MIN_T || nw > MMG_MAX) return false;
    for (int i = 0; i < nw; i++) {
        if (W[i]->imx || W[i]->cols != W[0]->cols) return false;
        if (float_act(W[i]->type)) continue;
        if (!*qm) *qm = &W[i]->q;
        else if (W[i]->q.rot_block != (*qm)->rot_block || W[i]->q.rot_seed != (*qm)->rot_seed) return false;
    }
    return true;
}

// quantize: whether the input still has to be quantized (false when the previous matrix left it in e->acts)
static void mm_group_run(inz_engine *e, const lin *const *W, float *const *Y, const int *ldy, int nw, const float *X, int ldx,
                         int n, const inz_qmat *qm, bool quantize) {
    mmg_job g;
    g.nq = 0;
    if (qm && quantize) {
        const int K = W[0]->cols;
        if (n == 1 && K % MMG_SLICE == 0 && qm->rot_block <= MMG_SLICE) {
            inz_act_begin(&e->acts[0], K, qm->rot_block, qm->rot_seed);
            g.nq = K / MMG_SLICE;
        } else {
            aq_job aj = {e->acts, X, ldx, K, qm->rot_block, qm->rot_seed};
            tp_for(e->tp, n, 1, aq_task, &aj);
        }
    }
    g.a = &e->acts[0];
    g.x = X;
    atomic_init(&g.qleft, g.nq);
    g.start[0] = 0;
    for (int i = 0; i < nw; i++) {
        g.j[i] = (mm_job){W[i], X, ldx, n, Y[i], ldy[i], e->acts};
        g.rows[i] = W[i]->cols >= 6144 ? 16 : 32;
        g.start[i + 1] = g.start[i] + (W[i]->rows + g.rows[i] - 1) / g.rows[i];
    }
    tp_for(e->tp, g.nq + g.start[nw], 1, mmg_task, &g);
}

static void lin_mm_group(inz_engine *e, const lin *const *W, float *const *Y, const int *ldy, int nw, const float *X, int ldx,
                         int n) {
    const inz_qmat *qm;
    if (!mm_groupable(W, nw, n, &qm)) {
        for (int i = 0; i < nw; i++) lin_mm_after(e, W[i], i ? W[i - 1] : NULL, X, ldx, n, Y[i], ldy[i]);
        return;
    }
    mm_group_run(e, W, Y, ldy, nw, X, ldx, n, qm, true);
}

/* ------------------------------------------------------------------ */
/* Embedding, norms, residual                                          */
/* ------------------------------------------------------------------ */

// one row of the table, read from the file (the 1 GB table never needs to be resident)
static void embed_row(inz_engine *e, int tok, float *out) {
    if (tok < 0 || tok >= Q_VOCAB) inz_die("token id %d out of range", tok);
    if (e->embed_type == INZ_T_F16) {
        uint16_t h[Q_H];
        if (pread(e->embed_fd, h, sizeof(h), (off_t)(e->embed_offset + (uint64_t)tok * sizeof(h))) != (ssize_t)sizeof(h))
            inz_die("embedding read failed (token %d)", tok);
        for (int i = 0; i < Q_H; i++) out[i] = inz_f16_to_f32(h[i]);
        return;
    }
    const inz_qmat *m = &e->embed;
    const int ng = m->ngroups;
    const size_t qb = (size_t)ng * m->qbytes, db = (size_t)ng * 2;
    uint8_t q[Q_H];
    uint16_t d[Q_H / INZ_GROUP];
    const uint64_t qoff = e->embed_offset + (uint64_t)((const uint8_t *)m->q - e->embed_hdr) + (uint64_t)tok * qb;
    const uint64_t doff = e->embed_offset + (uint64_t)((const uint8_t *)m->d - e->embed_hdr) + (uint64_t)tok * db;
    if (pread(e->embed_fd, q, qb, (off_t)qoff) != (ssize_t)qb || pread(e->embed_fd, d, db, (off_t)doff) != (ssize_t)db)
        inz_die("embedding read failed (token %d)", tok);
    inz_qmat row = *m;
    row.rows = 1;
    row.q = q;
    row.d = d;
    inz_dequantize_row(&row, 0, out);  // embeddings are never rotated
}

typedef struct { const float *X; int ldx; float *Y; int ldy; const float *w; } norm_job;
static void norm_task(void *ctx, int64_t b, int64_t e_, int tid) {
    (void)tid;
    norm_job *j = ctx;
    for (int64_t t = b; t < e_; t++) rmsnorm(j->Y + (size_t)t * j->ldy, j->X + (size_t)t * j->ldx, j->w, Q_H);
}

static void norm_rows(inz_engine *e, const float *X, int ldx, int n, float *Y, int ldy, const float *w) {
    norm_job j = {X, ldx, Y, ldy, w};
    tp_for(e->tp, n, 1, norm_task, &j);
}

static void add_rows(float *X, const float *y, int n) {
    for (size_t i = 0; i < (size_t)n * Q_H; i++) X[i] += y[i];
}

/* ------------------------------------------------------------------ */
/* RoPE                                                                */
/* ------------------------------------------------------------------ */

// Inverse frequencies as transformers computes them (float32): plain RoPE for the sliding layers, YaRN for the full
// ones (factor 128 over 8192 positions, beta_fast 32 / beta_slow 1, truncated correction range).
void q_rope_init(inz_engine *e) {
    for (int i = 0; i < Q_HD / 2; i++) e->rope_swa[i] = 1.0f / powf((float)Q_ROPE_SWA_THETA, (float)(2 * i) / (float)Q_HD);
    const int dim = Q_ROPE_FULL_DIM;
    const double lb = log(Q_ROPE_FULL_THETA);
#define CORR_DIM(rot) ((dim * log(Q_ROPE_FULL_ORIG / ((rot) * 2 * M_PI))) / (2 * lb))
    double low = floor(CORR_DIM(32.0)), high = ceil(CORR_DIM(1.0));
#undef CORR_DIM
    if (low < 0) low = 0;
    if (high > dim - 1) high = dim - 1;
    if (low == high) high += 0.001;
    for (int i = 0; i < dim / 2; i++) {
        const float pos = powf((float)Q_ROPE_FULL_THETA, (float)(2 * i) / (float)dim);
        const float extra = 1.0f / pos, inter = 1.0f / ((float)Q_ROPE_FULL_FACTOR * pos);
        float ramp = (float)(((double)i - low) / (high - low));
        ramp = ramp < 0 ? 0 : ramp > 1 ? 1 : ramp;
        const float ef = 1.0f - ramp;
        e->rope_full[i] = inter * (1.0f - ef) + extra * ef;
    }
}

// NeoX halves on the first `rd` dims of a head (pairs i, i + rd/2), cos and sin scaled by `scale`
static void rope(float *x, int pos, const float *inv, int rd, float scale) {
    const int half = rd / 2;
    for (int i = 0; i < half; i++) {
        const float a = (float)pos * inv[i];
        const float c = cosf(a) * scale, s = sinf(a) * scale;
        const float x1 = x[i], x2 = x[i + half];
        x[i] = x1 * c - x2 * s;
        x[i + half] = x2 * c + x1 * s;
    }
}

/* ------------------------------------------------------------------ */
/* Attention                                                           */
/* ------------------------------------------------------------------ */

typedef struct { inz_engine *e; int l, n, p0, nh; const float *q, *g; float *out; } att_job;

// K/V rows of one layer: full layers by position, sliding layers in a ring
static inline const uint16_t *kv_row(const inz_engine *e, const uint16_t *base, int l, int pos) {
    return base + (size_t)(q_is_full(l) ? pos : pos % Q_RING) * Q_KV;
}

static void kv_bases(const inz_engine *e, int l, uint16_t **K, uint16_t **V) {
    if (q_is_full(l)) {
        const size_t off = (size_t)(l / 4) * e->ctx * Q_KV;
        *K = e->st.kf + off;
        *V = e->st.vf + off;
    } else {
        const size_t off = (size_t)(l - l / 4 - 1) * Q_RING * Q_KV;
        *K = e->st.ks + off;
        *V = e->st.vs + off;
    }
}

// one (token, head) pair: softmax(q k^T / sqrt(128)) v over the visible positions, times softplus(g)
static void att_head_task(void *ctx, int64_t b, int64_t e_, int tid) {
    att_job *j = ctx;
    inz_engine *e = j->e;
    uint16_t *K, *V;
    kv_bases(e, j->l, &K, &V);
    float *sc = e->att_scr[tid];
    const float scale = 1.0f / sqrtf((float)Q_HD);
    const int rep = j->nh / Q_NKV;
    for (int64_t th = b; th < e_; th++) {
        const int t = (int)(th / j->nh), h = (int)(th % j->nh), kh = h / rep, p = j->p0 + t;
        const int first = q_is_full(j->l) || p < Q_WINDOW ? 0 : p - Q_WINDOW + 1, np = p - first + 1;
        const float *qh = j->q + ((size_t)t * j->nh + h) * Q_HD;
        for (int i = 0; i < np; i++) sc[i] = dot_f16(kv_row(e, K, j->l, first + i) + kh * Q_HD, qh, Q_HD) * scale;
        softmax_inplace(sc, np);
        float o[Q_HD] = {0};
        for (int i = 0; i < np; i++) axpy_f16(o, sc[i], kv_row(e, V, j->l, first + i) + kh * Q_HD, Q_HD);
        const float gt = softplusf(j->g[(size_t)t * j->nh + h]);
        float *dst = j->out + ((size_t)t * j->nh + h) * Q_HD;
        for (int d = 0; d < Q_HD; d++) dst[d] = o[d] * gt;
    }
}

static void attn_block(inz_engine *e, const q_layer *L, int l, int n, int p0, const float *xin, float *y) {
    const int nh = L->nh, qd = nh * Q_HD;
    float *q = e->big, *att = e->big + (size_t)Q_MAX_T * Q_QD_MAX;  // [n][nh*128] each
    float *k = e->tmp, *v = e->tmp + (size_t)Q_MAX_T * Q_KV, *g = e->tmp2;  // [n][1024], [n][1024], [n][nh]
    const lin *W[4] = {&L->at_q, &L->at_k, &L->at_v, &L->at_g};
    float *Y[4] = {q, k, v, g};
    const int ldy[4] = {qd, Q_KV, Q_KV, nh};
    lin_mm_group(e, W, Y, ldy, 4, xin, Q_H, n);
    const bool full = q_is_full(l);
    const float *inv = full ? e->rope_full : e->rope_swa;
    const int rd = full ? Q_ROPE_FULL_DIM : Q_HD;
    const float rs = full ? Q_ROPE_FULL_ATTN : 1.0f;
    uint16_t *K, *V;
    kv_bases(e, l, &K, &V);
    for (int t = 0; t < n; t++) {
        const int p = p0 + t;
        for (int h = 0; h < nh; h++) {
            float *qh = q + ((size_t)t * nh + h) * Q_HD;
            rmsnorm(qh, qh, L->q_norm, Q_HD);
            rope(qh, p, inv, rd, rs);
        }
        float kk[Q_KV];
        for (int kh = 0; kh < Q_NKV; kh++) {
            rmsnorm(kk + kh * Q_HD, k + (size_t)t * Q_KV + kh * Q_HD, L->k_norm, Q_HD);
            rope(kk + kh * Q_HD, p, inv, rd, rs);
        }
        f32_to_f16((uint16_t *)kv_row(e, K, l, p), kk, Q_KV);
        f32_to_f16((uint16_t *)kv_row(e, V, l, p), v + (size_t)t * Q_KV, Q_KV);
    }
    att_job j = {e, l, n, p0, nh, q, g, att};
    tp_for(e->tp, (int64_t)n * nh, 1, att_head_task, &j);
    lin_mm(e, &L->at_o, att, qd, n, y, Q_H);
}

/* ------------------------------------------------------------------ */
/* Dense MLP (layer 0)                                                 */
/* ------------------------------------------------------------------ */

typedef struct { float *gu; int ff, ld; } swiglu_job;
static void swiglu_task(void *ctx, int64_t b, int64_t e_, int tid) {
    (void)tid;
    swiglu_job *j = ctx;
    for (int64_t t = b; t < e_; t++) swiglu(j->gu + (size_t)t * j->ld, j->ff);
}

static void dense_block(inz_engine *e, const q_layer *L, int n, const float *xin, float *y) {
    float *GU = e->big;                         // [n][2 * 12288] (big holds T x 2*9216 floats)
    lin_mm(e, &L->d_gu, xin, Q_H, n, GU, 2 * Q_DENSE_FF);
    swiglu_job j = {GU, Q_DENSE_FF, 2 * Q_DENSE_FF};
    tp_for(e->tp, n, 4, swiglu_task, &j);
    lin_mm(e, &L->d_down, GU, 2 * Q_DENSE_FF, n, y, Q_H);
}

/* ------------------------------------------------------------------ */
/* MoE                                                                 */
/* ------------------------------------------------------------------ */

// tokens routed to one expert; slot = t * Q_TOPK + rank, where the expert's weighted output for that token goes
typedef struct { int expert, ntok; int tok[Q_MAX_T]; int slot[Q_MAX_T]; float w[Q_MAX_T]; } unit;

// out[slot] = w * down(silu(gate x) * up x) for the tokens routed to one expert. The slots are summed in rank order
// afterwards, so the result does not depend on which experts were resident or in which order reads completed.
static void expert_compute(inz_engine *e, const q_xclass *xc, const uint8_t *blob, const unit *u, const float *xin, float *out) {
    lin gu, dn;
    if (!q_bind_lin(&gu, xc->type, 2 * Q_FF, Q_H, blob) || !q_bind_lin(&dn, xc->type, Q_H, Q_FF, blob + xc->gu_bytes))
        inz_die("expert blob is corrupt (expert %d)", u->expert);
    float *X = e->ex_x, *GU = e->ex_gu, *D = e->ex_d;
    for (int i = 0; i < u->ntok; i++) memcpy(X + (size_t)i * Q_H, xin + (size_t)u->tok[i] * Q_H, sizeof(float) * Q_H);
    lin_mm(e, &gu, X, Q_H, u->ntok, GU, 2 * Q_FF);
    for (int i = 0; i < u->ntok; i++) swiglu(GU + (size_t)i * 2 * Q_FF, Q_FF);
    if (e->imx_exp) {                           // importance matrix: this expert's gate|up and down inputs
        const size_t k = (size_t)e->imx_layer * Q_N_EXPERT + (size_t)u->expert;
        double *acc = e->imx_exp + k * (Q_H + Q_FF);
        imx_add(acc, X, Q_H, u->ntok, Q_H);
        imx_add(acc + Q_H, GU, 2 * Q_FF, u->ntok, Q_FF);
        e->imx_exp_n[k] += (uint64_t)u->ntok;
    }
    lin_mm(e, &dn, GU, 2 * Q_FF, u->ntok, D, Q_H);
    for (int i = 0; i < u->ntok; i++) {
        float *o = out + (size_t)u->slot[i] * Q_H;
        const float w = u->w[i], *dt = D + (size_t)i * Q_H;
        for (int c = 0; c < Q_H; c++) o[c] = w * dt[c];
    }
}

/* Decode / verify (n <= 8 tokens): the resident experts of a layer are computed together, in three dispatches over
 * (expert, 16-row chunk) items, instead of two small dispatches per expert. Per row the arithmetic is the same as
 * expert_compute's, so the results are identical. */
#define EB_TOK 8
#define EB_MAX (Q_TOPK * EB_TOK)                 // experts x tokens of one batch
typedef struct {
    const unit *u;
    lin gu, dn;
    inz_act ga[EB_TOK], da[EB_TOK];              // per token: the quantized input of gate|up, of down
    float *GU;                                   // [ntok][2 FF]
    float *D;                                    // [ntok][H]
} ebatch_item;

typedef struct { inz_engine *e; ebatch_item *it; int ni; } ebatch_job;

#define EB_CHUNK 64  // rows per call: the kernels prefetch the next rows within a call
static void eb_gu_task(void *ctx, int64_t b, int64_t e_, int tid) {
    (void)tid;
    ebatch_job *j = ctx;
    const int per = 2 * Q_FF / EB_CHUNK;
    for (int64_t k = b; k < e_; k++) {
        ebatch_item *it = &j->it[k / per];
        const int r0 = (int)(k % per) * EB_CHUNK, r1 = r0 + EB_CHUNK;
        if (it->u->ntok == 1) inz_matvec_rows(&it->gu.q, it->ga, it->GU, r0, r1);
        else inz_matmul_rows(&it->gu.q, it->ga, it->u->ntok, it->GU, 2 * Q_FF, r0, r1);
    }
}

static void eb_act_task(void *ctx, int64_t b, int64_t e_, int tid) {
    (void)tid;
    ebatch_job *j = ctx;
    for (int64_t k = b; k < e_; k++) {
        ebatch_item *it = &j->it[k];
        for (int i = 0; i < it->u->ntok; i++) {
            float *r = it->GU + (size_t)i * 2 * Q_FF;
            swiglu(r, Q_FF);
            inz_act_quantize(&it->da[i], r, Q_FF, it->dn.q.rot_block, it->dn.q.rot_seed);
        }
    }
}

static void eb_down_task(void *ctx, int64_t b, int64_t e_, int tid) {
    (void)tid;
    ebatch_job *j = ctx;
    const int per = Q_H / EB_CHUNK;
    for (int64_t k = b; k < e_; k++) {
        ebatch_item *it = &j->it[k / per];
        const int r0 = (int)(k % per) * EB_CHUNK, r1 = r0 + EB_CHUNK;
        const unit *u = it->u;
        if (u->ntok == 1) inz_matvec_rows(&it->dn.q, it->da, it->D, r0, r1);
        else inz_matmul_rows(&it->dn.q, it->da, u->ntok, it->D, Q_H, r0, r1);
        for (int i = 0; i < u->ntok; i++) {
            float *o = j->e->moe_out + (size_t)u->slot[i] * Q_H;
            const float w = u->w[i], *dt = it->D + (size_t)i * Q_H;
            for (int r = r0; r < r1; r++) o[r] = w * dt[r];
        }
    }
}

// compute the given units (blobs pinned) as one batch; quantized expert types only. The tokens' inputs are quantized
// once for the gate|up rotation, which all experts of a layer share; an expert with another rotation (not produced by
// the converter) is computed on its own.
static void experts_batch(inz_engine *e, const q_xclass *xc, const unit *const *us, const uint8_t *const *blobs, int nb, int n,
                          const float *xin) {
    ebatch_item *it = e->eb_items;
    int ni = 0, rows = 0, qrot = -1;
    uint32_t qseed = 0;
    for (int i = 0; i < nb; i++) {
        ebatch_item *x = &it[ni];
        if (!q_bind_lin(&x->gu, xc->type, 2 * Q_FF, Q_H, blobs[i]) || !q_bind_lin(&x->dn, xc->type, Q_H, Q_FF, blobs[i] + xc->gu_bytes))
            inz_die("expert blob is corrupt (expert %d)", us[i]->expert);
        if (qrot < 0) {
            qrot = x->gu.q.rot_block;
            qseed = x->gu.q.rot_seed;
            for (int t = 0; t < n; t++) inz_act_quantize(&e->eb_xa[t], xin + (size_t)t * Q_H, Q_H, qrot, qseed);
        } else if (x->gu.q.rot_block != qrot || x->gu.q.rot_seed != qseed) {
            expert_compute(e, xc, blobs[i], us[i], xin, e->moe_out);
            continue;
        }
        x->u = us[i];
        x->GU = e->eb_gu + (size_t)rows * 2 * Q_FF;
        x->D = e->eb_d + (size_t)rows * Q_H;
        for (int t = 0; t < us[i]->ntok; t++) {
            x->ga[t] = e->eb_xa[us[i]->tok[t]];
            x->da[t] = e->eb_acts[rows + t];
        }
        rows += us[i]->ntok;
        ni++;
    }
    if (!ni) return;
    ebatch_job j = {e, it, ni};
    tp_for(e->tp, (int64_t)ni * (2 * Q_FF / EB_CHUNK), 1, eb_gu_task, &j);
    tp_for(e->tp, ni, 1, eb_act_task, &j);
    tp_for(e->tp, (int64_t)ni * (Q_H / EB_CHUNK), 2, eb_down_task, &j);
}

// top-10 by sigmoid score + selection bias; weights are the chosen scores, normalized and scaled by 2.5
static void route(const float *lg, const float *bias, int32_t *sel, float *w) {
    float sc[Q_N_EXPERT], ch[Q_N_EXPERT];
    for (int x = 0; x < Q_N_EXPERT; x++) { sc[x] = sigmoidf(lg[x]); ch[x] = sc[x] + bias[x]; }
    topk_f32(ch, Q_N_EXPERT, Q_TOPK, sel);
    float sum = 0;
    for (int k = 0; k < Q_TOPK; k++) { w[k] = sc[sel[k]]; sum += w[k]; }
    for (int k = 0; k < Q_TOPK; k++) w[k] = w[k] / sum * Q_ROUTED_SCALE;
}

static void moe_block(inz_engine *e, const q_layer *L, int l, int n, const float *xin, float *y) {
    const q_xclass *xc = L->xc;
    ec_cache *ec = xc->ec;
    const int xl = L->xl, mi = q_moe_index(l);
    const uint64_t io_start = ec_read_bytes(ec);
    e->imx_layer = mi;
    float *logits = e->tmp;                     // [n][256]
    uint64_t t0 = diag_tsc();
    lin_mm(e, &L->router, xin, Q_H, n, logits, Q_N_EXPERT);
    unit *units = e->units;
    int uid[Q_N_EXPERT];
    for (int x = 0; x < Q_N_EXPERT; x++) uid[x] = -1;
    int nu = 0;
    for (int t = 0; t < n; t++) {
        int32_t sel[Q_TOPK];
        float w[Q_TOPK];
        route(logits + (size_t)t * Q_N_EXPERT, L->router_bias, sel, w);
        diag_routing(mi, sel, w, Q_TOPK, e->n_past + t);
        if (n == 1 && e->pred_valid[l]) {         // how well the previous layer's lookahead predicted this routing
            int covered = 0;
            for (int k = 0; k < Q_TOPK; k++)
                for (int q = 0; q < Q_TOPK; q++) if (e->pred[l][q] == sel[k]) { covered++; break; }
            g_diag.pred_needed += Q_TOPK;
            g_diag.pred_covered += (uint64_t)covered;
            g_diag.pred_hits += (uint64_t)covered;
            e->pred_valid[l] = false;
        }
        for (int k = 0; k < Q_TOPK; k++) {
            const int x = sel[k];
            if (uid[x] < 0) { uid[x] = nu; units[nu].expert = x; units[nu].ntok = 0; nu++; }
            unit *u = &units[uid[x]];
            u->tok[u->ntok] = t;
            u->slot[u->ntok] = t * Q_TOPK + k;
            u->w[u->ntok] = w[k];
            u->ntok++;
        }
    }
    diag_add(DG_ROUTER, l, t0, q_lin_bytes(&L->router), 2ull * n * Q_N_EXPERT * Q_H);
    // routed experts: read requests for every missing one first, then the shared expert and the resident experts
    // while the SSD works, then the rest as they arrive
    bool done[Q_N_EXPERT] = {0}, asked[Q_N_EXPERT] = {0};
    int left = nu, lvl;
    if (n <= EB_TOK)
        for (int i = 0; i < nu; i++)
            if (!ec_is_resident(ec, xl, units[i].expert)) { ec_request(ec, xl, units[i].expert, 1.0f, EC_FULL); asked[i] = true; }
    t0 = diag_tsc();
    float *GU = e->big, *D = e->tmp2;           // [n][2048], [n][H]
    lin_mm(e, &L->sh_gu, xin, Q_H, n, GU, 2 * Q_SHARED_FF);
    for (int t = 0; t < n; t++) swiglu(GU + (size_t)t * 2 * Q_SHARED_FF, Q_SHARED_FF);
    lin_mm(e, &L->sh_down, GU, 2 * Q_SHARED_FF, n, D, Q_H);
    diag_add(DG_SHARED, l, t0, q_lin_bytes(&L->sh_gu) + q_lin_bytes(&L->sh_down), 0);
    t0 = diag_tsc();
    const bool batch = n <= EB_TOK && !e->imx_exp && !float_act(xc->type);
    const unit *bu[EB_MAX];
    const uint8_t *bb[EB_MAX];
    int nb = 0;
    for (int i = 0; i < nu; i++) {
        const uint8_t *b = ec_try_acquire(ec, xl, units[i].expert, 1.0f, EC_FULL, &lvl);
        if (!b) { if (!asked[i]) ec_request(ec, xl, units[i].expert, 1.0f, EC_FULL); continue; }
        if (batch) { bu[nb] = &units[i]; bb[nb++] = b; }
        else {
            expert_compute(e, xc, b, &units[i], xin, e->moe_out);
            ec_release(ec, xl, units[i].expert);
        }
        done[i] = true;
        left--;
    }
    if (nb) {
        experts_batch(e, xc, bu, bb, nb, n, xin);
        for (int i = 0; i < nb; i++) ec_release(ec, xl, bu[i]->expert);
    }
    diag_add(DG_EXPERT_HIT, l, t0, (uint64_t)(nu - left) * xc->blob_bytes, 0);
    t0 = diag_tsc();
    while (left > 0) {
        bool progress = false;
        nb = 0;
        for (int i = 0; i < nu; i++) {
            if (done[i]) continue;
            const uint8_t *b = ec_try_acquire(ec, xl, units[i].expert, 0, EC_FULL, &lvl);
            if (!b) continue;
            if (batch) { bu[nb] = &units[i]; bb[nb++] = b; }
            else {
                expert_compute(e, xc, b, &units[i], xin, e->moe_out);
                ec_release(ec, xl, units[i].expert);
            }
            done[i] = true;
            left--;
            progress = true;
        }
        if (nb) {
            experts_batch(e, xc, bu, bb, nb, n, xin);
            for (int i = 0; i < nb; i++) ec_release(ec, xl, bu[i]->expert);
        }
        if (progress || left == 0) continue;
        diag_add(DG_EXPERT_MISS, l, t0, 0, 0);
        t0 = diag_tsc();
        if (!ec_wait_any(ec)) {                 // nothing in flight: fall back to a blocking read
            for (int i = 0; i < nu; i++)
                if (!done[i]) {
                    const uint8_t *b = ec_acquire(ec, xl, units[i].expert, 0, EC_FULL, &lvl);
                    diag_add(DG_EXPERT_WAIT, l, t0, 0, 0);
                    t0 = diag_tsc();
                    expert_compute(e, xc, b, &units[i], xin, e->moe_out);
                    ec_release(ec, xl, units[i].expert);
                    done[i] = true;
                    left--;
                    diag_add(DG_EXPERT_MISS, l, t0, xc->blob_bytes, 0);
                    t0 = diag_tsc();
                    break;
                }
        }
        diag_add(DG_EXPERT_WAIT, l, t0, 0, 0);
        t0 = diag_tsc();
    }
    diag_add(DG_EXPERT_MISS, l, t0, 0, 0);
    // routed outputs in rank order (deterministic), plus the shared expert
    for (int t = 0; t < n; t++) {
        float *yt = y + (size_t)t * Q_H;
        memcpy(yt, e->moe_out + (size_t)t * Q_TOPK * Q_H, sizeof(float) * Q_H);
        for (int k = 1; k < Q_TOPK; k++) {
            const float *o = e->moe_out + ((size_t)t * Q_TOPK + k) * Q_H;
            for (int c = 0; c < Q_H; c++) yt[c] += o[c];
        }
    }
    add_rows(y, D, n);
    diag_add_ssd(DG_PREFETCH, l, ec_read_bytes(ec) - io_start);
}

// Router lookahead (--lookahead 1, the default): once layer l has updated the residual, layer l + 1's MLP norm and
// router on it predict that layer's experts, which are prefetched while its attention runs. The early variant
// (--lookahead 2, from the MoE input) exists in the Nemotron branch; here the next router is only one attention
// block away, so the late prediction is both close and early enough.
static void moe_lookahead(inz_engine *e, int l, int n, const float *X) {
    if (l + 1 >= e->n_layer || n > EB_TOK || !e->opt.lookahead) return;
    const q_layer *Ln = &e->L[l + 1];
    if (!Ln->xc || !Ln->xc->ec || Ln->xc->all_resident) return;
    const uint64_t t0 = diag_tsc();
    norm_rows(e, X, Q_H, n, e->Xn, Q_H, Ln->mlp_norm);
    float *lg = e->tmp;
    lin_mm(e, &Ln->router, e->Xn, Q_H, n, lg, Q_N_EXPERT);
    for (int t = 0; t < n; t++) {
        int32_t ps[Q_TOPK];
        float w[Q_TOPK];
        route(lg + (size_t)t * Q_N_EXPERT, Ln->router_bias, ps, w);
        for (int k = 0; k < Q_TOPK; k++) ec_prefetch(Ln->xc->ec, Ln->xl, ps[k], 0.75f, EC_FULL);
        if (n == 1) { memcpy(e->pred[l + 1], ps, sizeof(ps)); e->pred_valid[l + 1] = true; g_diag.pred_total += Q_TOPK; }
    }
    diag_add(DG_PREFETCH, l, t0, q_lin_bytes(&Ln->router), 0);
}

/* ------------------------------------------------------------------ */
/* Scratch, state, forward                                             */
/* ------------------------------------------------------------------ */

#define ACT_MAX Q_DENSE_FF                      // widest input of a matrix with int8 activations (layer 0 down)

void q_scratch_alloc(inz_engine *e) {
    const size_t T = Q_MAX_T;
    e->X = inz_aligned(64, sizeof(float) * T * Q_H);
    e->X2 = inz_aligned(64, sizeof(float) * 64 * Q_H);
    e->Xn = inz_aligned(64, sizeof(float) * T * Q_H);
    e->y = inz_aligned(64, sizeof(float) * T * Q_H);
    e->tmp = inz_aligned(64, sizeof(float) * T * 2 * Q_KV);          // k | v; router logits; head rows
    e->tmp2 = inz_aligned(64, sizeof(float) * T * Q_H);              // gates; shared expert output
    e->big = inz_aligned(64, sizeof(float) * T * 2 * Q_DENSE_FF);    // q | attention output; dense gate|up
    e->moe_out = inz_aligned(64, sizeof(float) * T * Q_TOPK * Q_H);
    e->ex_x = inz_aligned(64, sizeof(float) * T * Q_H);
    e->ex_gu = inz_aligned(64, sizeof(float) * T * 2 * Q_FF);
    e->ex_d = inz_aligned(64, sizeof(float) * T * Q_H);
    e->eb_items = inz_calloc(EB_MAX, sizeof(ebatch_item));
    e->eb_gu = inz_aligned(64, sizeof(float) * EB_MAX * 2 * Q_FF);
    e->eb_d = inz_aligned(64, sizeof(float) * EB_MAX * Q_H);
    e->eb_acts = inz_calloc(EB_MAX, sizeof(inz_act));
    for (int i = 0; i < EB_MAX; i++) inz_act_alloc(&e->eb_acts[i], Q_FF);
    e->eb_xa = inz_calloc(EB_TOK, sizeof(inz_act));
    for (int i = 0; i < EB_TOK; i++) inz_act_alloc(&e->eb_xa[i], Q_H);
    e->units = inz_calloc(Q_N_EXPERT, sizeof(unit));
    e->acts = inz_calloc(T, sizeof(inz_act));
    for (size_t i = 0; i < T; i++) inz_act_alloc(&e->acts[i], ACT_MAX);
    const int nthr = tp_size(e->tp);
    e->att_scr = inz_calloc((size_t)nthr, sizeof(float *));
    for (int i = 0; i < nthr; i++) e->att_scr[i] = inz_aligned(64, sizeof(float) * ((size_t)e->ctx + 64));
    e->gemm_tb = inz_calloc((size_t)nthr, sizeof(uint8_t *));
    e->gemm_tmp = inz_calloc((size_t)nthr, sizeof(float *));
    for (int i = 0; i < nthr; i++) {
        e->gemm_tb[i] = inz_aligned(64, inz_gemm_tilebuf_bytes(ACT_MAX));
        e->gemm_tmp[i] = inz_aligned(64, sizeof(float) * ACT_MAX + 64);
    }
    inz_xq_alloc(&e->gemm_x, Q_MAX_T, ACT_MAX);
}

void q_scratch_free(inz_engine *e) {
    free(e->X); free(e->X2); free(e->Xn); free(e->y); free(e->tmp); free(e->tmp2); free(e->big);
    free(e->moe_out); free(e->ex_x); free(e->ex_gu); free(e->ex_d); free(e->units);
    for (int i = 0; i < EB_MAX; i++) inz_act_free(&e->eb_acts[i]);
    for (int i = 0; i < EB_TOK; i++) inz_act_free(&e->eb_xa[i]);
    free(e->eb_items); free(e->eb_gu); free(e->eb_d); free(e->eb_acts); free(e->eb_xa);
    for (int i = 0; i < Q_MAX_T; i++) inz_act_free(&e->acts[i]);
    free(e->acts);
    const int nthr = tp_size(e->tp);
    for (int i = 0; i < nthr; i++) { free(e->att_scr[i]); free(e->gemm_tb[i]); free(e->gemm_tmp[i]); }
    free(e->att_scr); free(e->gemm_tb); free(e->gemm_tmp);
    inz_xq_free(&e->gemm_x);
}

void q_state_reset(inz_engine *e) {
    e->n_past = 0;  // the caches are positional: nothing to clear
    e->mtp_valid = 0;
}

// one decoder layer on rows X [n][H]: X += attn(norm(X)); X += mlp(norm(X))
static void run_layer(inz_engine *e, int l, int n, int p0, float *X) {
    const q_layer *L = &e->L[l];
    uint64_t t0 = diag_tsc();
    norm_rows(e, X, Q_H, n, e->Xn, Q_H, L->attn_norm);
    diag_add(DG_ATTN_NORM, l, t0, 0, 0);
    t0 = diag_tsc();
    attn_block(e, L, l, n, p0, e->Xn, e->y);
    diag_add(DG_ATTN_QKV, l, t0, q_lin_bytes(&L->at_q) + q_lin_bytes(&L->at_k) + q_lin_bytes(&L->at_v) + q_lin_bytes(&L->at_o) +
                                     q_lin_bytes(&L->at_g), 0);
    add_rows(X, e->y, n);
    t0 = diag_tsc();
    norm_rows(e, X, Q_H, n, e->Xn, Q_H, L->mlp_norm);
    diag_add(DG_FFN_NORM, l, t0, 0, 0);
    if (L->xc) moe_block(e, L, l, n, e->Xn, e->y);
    else {
        t0 = diag_tsc();
        dense_block(e, L, n, e->Xn, e->y);
        diag_add(DG_DENSE_FFN, l, t0, q_lin_bytes(&L->d_gu) + q_lin_bytes(&L->d_down), 0);
    }
    add_rows(X, e->y, n);
    moe_lookahead(e, l, n, X);
}

void q_forward(inz_engine *e, const int32_t *tokens, int n, const int *logits_rows, int n_logits, float *out_logits) {
    if (n > Q_MAX_T) inz_die("q_forward: %d tokens > %d", n, Q_MAX_T);
    if (e->n_past + n > e->ctx) inz_die("q_forward: context full (%d + %d > %d)", e->n_past, n, e->ctx);
    const int p0 = e->n_past;
    memcpy(e->hist + p0, tokens, sizeof(int32_t) * n);
    uint64_t t0 = diag_tsc();
    for (int t = 0; t < n; t++) embed_row(e, tokens[t], e->X + (size_t)t * Q_H);
    diag_add(DG_EMBED, -1, t0, (uint64_t)n * Q_H * 2, 0);
    static FILE *dump;  // validation: INZ_DUMP_HIDDEN=file appends the residual after every layer
    static int dump_init;
    if (!dump_init) {
        dump_init = 1;
        const char *pth = getenv("INZ_DUMP_HIDDEN");
        if (pth && *pth) dump = fopen(pth, "wb");
    }
    for (int l = 0; l < e->n_layer; l++) {
        run_layer(e, l, n, p0, e->X);
        if (dump) { fwrite(e->X, sizeof(float), (size_t)n * Q_H, dump); fflush(dump); }
    }
    e->n_past += n;
    memcpy(e->x_last, e->X + (size_t)(n - 1) * Q_H, sizeof(float) * Q_H);
    for (int c = 0; c < e->n_xc; c++) if (e->xc[c].ec) ec_tick(e->xc[c].ec, n);
    if (n_logits <= 0) return;
    float *rows = e->X2;                        // final norm of the requested rows, 64 at a time
    int seq[64];
    for (int i = 0; i < 64; i++) seq[i] = i;
    for (int g = 0; g < n_logits; g += 64) {
        const int k = n_logits - g < 64 ? n_logits - g : 64;
        for (int i = 0; i < k; i++) rmsnorm(rows + (size_t)i * Q_H, e->X + (size_t)logits_rows[g + i] * Q_H, e->norm_f, Q_H);
        q_head(e, rows, k, seq, k, out_logits + (size_t)g * Q_VOCAB);
    }
}

void q_head(inz_engine *e, const float *Hn, int n, const int *logits_rows, int n_logits, float *out_logits) {
    (void)n;
    uint64_t t0 = diag_tsc();
    float *rows = e->tmp;                       // [n_logits][H]
    for (int i = 0; i < n_logits; i++) memcpy(rows + (size_t)i * Q_H, Hn + (size_t)logits_rows[i] * Q_H, sizeof(float) * Q_H);
    lin_mm(e, &e->head, rows, Q_H, n_logits, out_logits, Q_VOCAB);
    diag_add(DG_LM_HEAD, -1, t0, q_lin_bytes(&e->head), 2ull * n_logits * Q_VOCAB * Q_H);
}

void q_mtp_forward(inz_engine *e, const float *H, int ldh, bool residual, const int32_t *next, int n, int p0,
                   const int *logits_rows, int n_logits, float *out_logits) {
    (void)e; (void)H; (void)ldh; (void)residual; (void)next; (void)n; (void)p0; (void)logits_rows; (void)n_logits; (void)out_logits;
    inz_die("Laguna has no MTP head");
}

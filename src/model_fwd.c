// Forward pass of Qwen3.8-Flash-Next (docs/MODEL.md gives the math; every block below follows it line by line).
//
// One code path serves decode (n = 1) and prefill chunks (n <= Q_MAX_T). Weights are `lin` matrices in any
// storage type; activations stay f32 between operators. Recurrent parts (Gated DeltaNet, its short convolution,
// the PLE convolution) advance token by token; projections, attention and experts are batched over the chunk.
#define _GNU_SOURCE
#include <immintrin.h>
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

// out = x * rsqrt(mean(x^2) + eps) * w; w already holds the +1 of zero-centered norms (NULL: no weight)
static void rmsnorm(float *out, const float *x, const float *w, int n) {
    const float r = 1.0f / sqrtf(dot_f32(x, x, n) / n + Q_RMS_EPS);
    for (int i = 0; i < n; i++) out[i] = x[i] * r * (w ? w[i] : 1.0f);
}

static inline float sigmoidf(float x) { return 1.0f / (1.0f + expf(-x)); }
static inline float siluf(float x) { return x / (1.0f + expf(-x)); }
static inline float softplusf(float x) { return x > 20.0f ? x : log1pf(expf(x)); }  // torch threshold 20

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

// RoPE on the first Q_ROT dims of a head: NeoX halves, pairs (i, i + 32), inv_freq_i = theta^(-2i/64).
// cos/sin are evaluated in f32 like the reference (positions * inv_freq in f32).
static float rope_inv[Q_ROT / 2];
static void rope(float *x, int pos) {
    const int half = Q_ROT / 2;
    if (!rope_inv[0])
        for (int i = 0; i < half; i++) rope_inv[i] = (float)(1.0 / pow(Q_ROPE_THETA, (double)(2 * i) / Q_ROT));
    for (int i = 0; i < half; i++) {
        const float a = (float)pos * rope_inv[i];
        const float c = cosf(a), s = sinf(a);
        const float x1 = x[i], x2 = x[i + half];
        x[i] = x1 * c - x2 * s;
        x[i + half] = x2 * c + x1 * s;
    }
}

/* ------------------------------------------------------------------ */
/* Matrices                                                            */
/* ------------------------------------------------------------------ */

bool q_bind_lin(lin *m, int type, int rows, int cols, const uint8_t *data) {
    memset(m, 0, sizeof(*m));
    m->type = type;
    m->rows = rows;
    m->cols = cols;
    if (type == INZ_T_F16) { m->h = (const uint16_t *)data; return true; }
    if (type == INZ_T_F32) { m->f = (const float *)data; return true; }
    if (!inz_qmat_view(&m->q, data)) return false;
    return m->q.rows == rows && m->q.cols == cols;
}

uint64_t q_lin_bytes(const lin *W) { return inz_mat_bytes(W->type, W->rows, W->cols); }

typedef struct { const lin *W; const float *X; int ldx, n; float *Y; int ldy; const inz_act *acts; } mm_job;

static void mm_task(void *ctx, int64_t b, int64_t e_, int tid) {
    (void)tid;
    mm_job *j = ctx;
    const lin *W = j->W;
    const int r0 = (int)b, r1 = (int)e_;
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
    const int nl = e->n_layer + (e->mtp.present ? 1 : 0);
    uint32_t count = 0;
    for (int i = 0; i < e->n_imx; i++) count += e->imx[i].n > 0;
    for (int k = 0; k < nl * Q_N_EXPERT; k++) count += 2 * (e->imx_exp_n[k] > 0);
    fwrite("INZIMX01", 1, 8, f);
    fwrite(&count, 4, 1, f);
    float *buf = inz_malloc(sizeof(float) * Q_HCW);
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
            if (l < e->n_layer) snprintf(name, sizeof(name), "blk.%d.experts.%d.%s", l, xid, part ? "down" : "gate_up");
            else snprintf(name, sizeof(name), "mtp.experts.%d.%s", xid, part ? "down" : "gate_up");
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

// Y[t*ldy + r] = W[r] . X[t*ldx]  for t < n, r < W->rows
static void lin_mm(inz_engine *e, const lin *W, const float *X, int ldx, int n, float *Y, int ldy) {
    if (W->imx) {
        imx_entry *x = &e->imx[W->imx - 1];
        imx_add(x->acc, X, ldx, n, W->cols);
        x->n += (uint64_t)n;
    }
    if (W->type != INZ_T_F32 && W->type != INZ_T_F16) {
        const inz_qmat *m = &W->q;
        if (n >= GEMM_MIN_T) {
            inz_xq_begin(&e->gemm_x, n, m->cols, m->rot_block, m->rot_seed);
            xq_job xj = {&e->gemm_x, X, ldx, e->gemm_tmp};
            tp_for(e->tp, n, 4, xq_task, &xj);
            gemm_job gj = {m, &e->gemm_x, Y, ldy, e->gemm_tb};
            tp_for(e->tp, (m->rows + 15) / 16, 1, gemm_task, &gj);
            return;
        }
        aq_job aj = {e->acts, X, ldx, W->cols, m->rot_block, m->rot_seed};
        tp_for(e->tp, n, 1, aq_task, &aj);
    }
    mm_job j = {W, X, ldx, n, Y, ldy, e->acts};
    const int grain = W->type == INZ_T_F16 && n > 1 ? (W->cols >= 6144 ? 12 : 24) : W->cols >= 6144 ? 16 : 32;
    tp_for(e->tp, W->rows, grain, mm_task, &j);
}

/* ------------------------------------------------------------------ */
/* Embedding                                                           */
/* ------------------------------------------------------------------ */

// one row of the quantized table, read from the file (the 1.3 GB table never needs to be resident)
static void embed_row(inz_engine *e, int tok, float *out) {
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

/* ------------------------------------------------------------------ */
/* Hyper-connections                                                   */
/* ------------------------------------------------------------------ */

static inline __m256 sigmoid256(__m256 x) {
    const __m256 one = _mm256_set1_ps(1.0f);
    return _mm256_div_ps(one, _mm256_add_ps(one, exp256_ps(_mm256_sub_ps(_mm256_setzero_ps(), x))));
}

typedef struct { const hcw *w; const float *X; float *Xn; const float *m; float *xin, *inj; } hc_job;

static void hc_norm_task(void *ctx, int64_t b, int64_t e_, int tid) {
    (void)tid;
    hc_job *j = ctx;
    for (int64_t t = b; t < e_; t++)
        for (int s = 0; s < Q_HC; s++)
            rmsnorm(j->Xn + (size_t)t * Q_HCW + s * Q_H, j->X + (size_t)t * Q_HCW + s * Q_H, j->w->norm + s * Q_H, Q_H);
}

static void hc_mix_task(void *ctx, int64_t b, int64_t e_, int tid) {
    (void)tid;
    hc_job *j = ctx;
    const __m256 quarter = _mm256_set1_ps(1.0f / Q_HC);
    for (int64_t t = b; t < e_; t++) {
        const float *mt = j->m + (size_t)t * Q_HCW, *xt = j->Xn + (size_t)t * Q_HCW;
        float *o = j->xin + (size_t)t * Q_H;
        for (int i = 0; i < Q_H; i += 8) {
            __m256 acc = _mm256_setzero_ps();
            for (int s = 0; s < Q_HC; s++)
                acc = _mm256_fmadd_ps(sigmoid256(_mm256_loadu_ps(mt + s * Q_H + i)), _mm256_loadu_ps(xt + s * Q_H + i), acc);
            _mm256_storeu_ps(o + i, _mm256_mul_ps(acc, quarter));
        }
        if (j->inj)
            for (int s = 0; s < Q_HC; s++)
                j->inj[t * Q_HC + s] = 2.0f * sigmoidf(dot_f32(j->w->inject + (size_t)s * Q_HCW, xt, Q_HCW) / Q_HC);
    }
}

// Xn = group_rmsnorm(X) * norm; xin = mean_s(sigmoid(up . silu(down . Xn / 4))_s * Xn_s); inj = 2 sigmoid(inject . Xn / 4)
static void hc_in(inz_engine *e, const hcw *w, int n, const float *X, float *xin, float *inj) {
    float *d = e->tmp, *m = e->big;             // [n][320], [n][4H]
    hc_job j = {w, X, e->Xn, m, xin, inj};
    tp_for(e->tp, n, 1, hc_norm_task, &j);
    lin_mm(e, &w->down, e->Xn, Q_HCW, n, d, Q_HC_RANK);
    for (int i = 0; i < n * Q_HC_RANK; i++) d[i] = siluf(d[i] / Q_HC);
    lin_mm(e, &w->up, d, Q_HC_RANK, n, m, Q_HCW);
    tp_for(e->tp, n, 1, hc_mix_task, &j);
}

// X_s += inj_s * y
static void hc_out(int n, float *X, const float *inj, const float *y) {
    for (int t = 0; t < n; t++)
        for (int s = 0; s < Q_HC; s++) {
            float *x = X + (size_t)t * Q_HCW + s * Q_H;
            const float *yt = y + (size_t)t * Q_H;
            const float w = inj[t * Q_HC + s];
            for (int i = 0; i < Q_H; i++) x[i] += w * yt[i];
        }
}

/* ------------------------------------------------------------------ */
/* Per-layer n-gram embedding (before layer 1)                         */
/* ------------------------------------------------------------------ */

// token at position p - s, or EOS when the shift crosses an end-of-text (or the start of the sequence)
static int64_t shifted(const int32_t *hist, int p, int s) {
    for (int k = 1; k <= s; k++)
        if (p - k < 0 || hist[p - k] == Q_EOS) return Q_EOS;
    return hist[p - s];
}

typedef struct { inz_engine *e; int n, p0; float *X, *E, *K, *V, *G; } ple_job;

// gather the 16 rows of every token (random 160-byte reads of the mapped table: in parallel, so the page faults
// of cold rows overlap on the SSD)
static void ple_gather_task(void *ctx, int64_t b, int64_t e_, int tid) {
    (void)tid;
    ple_job *j = ctx;
    const q_ple *P = &j->e->ple;
    for (int64_t i = b; i < e_; i++) {
        const int t = (int)(i / Q_NG_HEADS), h = (int)(i % Q_NG_HEADS), p = j->p0 + t;
        const int32_t *hist = j->e->hist;
        const int64_t t0 = hist[p], t1 = shifted(hist, p, 1), t2 = shifted(hist, p, 2);
        // products stay below 2^63 (multipliers < 2^63 / vocab), so int64 is exact; ids are non-negative
        int64_t mixed = (t0 * P->mult[0]) ^ (t1 * P->mult[1]);
        if (h >= Q_NG_HEADS / 2) mixed ^= t2 * P->mult[2];
        const uint8_t *row = P->table + (uint64_t)(mixed % P->prime[h] + P->offset[h]) * Q_NG_DIM;
        float *dst = j->E + (size_t)t * Q_H + h * Q_NG_DIM;
        for (int c = 0; c < Q_NG_DIM; c++) dst[c] = j->e->e4m3[row[c]] * P->scale;
    }
}

// G_s = sigmoid(g_s) V with g_s = signed sqrt of <norm_key(K)_s, norm_query(X)_s> / sqrt(H); then K := norm_conv(G)
static void ple_gate_task(void *ctx, int64_t b, int64_t e_, int tid) {
    (void)tid;
    ple_job *j = ctx;
    const q_ple *P = &j->e->ple;
    float kn[Q_H], qn[Q_H];
    for (int64_t t = b; t < e_; t++) {
        float *gt = j->G + (size_t)t * Q_HCW, *kt = j->K + (size_t)t * Q_HCW;
        const float *v = j->V + (size_t)t * Q_H;
        for (int s = 0; s < Q_HC; s++) {
            rmsnorm(kn, kt + s * Q_H, P->norm_key + s * Q_H, Q_H);
            rmsnorm(qn, j->X + (size_t)t * Q_HCW + s * Q_H, P->norm_query + s * Q_H, Q_H);
            float g = dot_f32(kn, qn, Q_H) / sqrtf((float)Q_H);
            g = sqrtf(fabsf(g) > 1e-6f ? fabsf(g) : 1e-6f) * (g > 0 ? 1.0f : g < 0 ? -1.0f : 0.0f);
            const float sg = sigmoidf(g);
            for (int i = 0; i < Q_H; i++) gt[s * Q_H + i] = sg * v[i];
        }
        for (int s = 0; s < Q_HC; s++) rmsnorm(kt + s * Q_H, gt + s * Q_H, P->norm_conv + s * Q_H, Q_H);
    }
}

// X += G + silu(depthwise conv of the normalized G), kernel 4, dilation 3: taps at p - 9, p - 6, p - 3, p
static void ple_conv_task(void *ctx, int64_t b, int64_t e_, int tid) {
    (void)tid;
    ple_job *j = ctx;
    const q_ple *P = &j->e->ple;
    const float *hist = j->e->st.ple_conv;      // normalized G of the 9 positions before the chunk, oldest first
    for (int64_t t = b; t < e_; t++) {
        const float *tap[4];
        for (int k = 0; k < 3; k++) {
            const int d = 9 - 3 * k;
            tap[k] = t >= d ? j->K + (size_t)(t - d) * Q_HCW : hist + (size_t)(9 - d + t) * Q_HCW;
        }
        tap[3] = j->K + (size_t)t * Q_HCW;
        float *x = j->X + (size_t)t * Q_HCW;
        const float *gt = j->G + (size_t)t * Q_HCW;
        for (int c = 0; c < Q_HCW; c++) {
            const float *w = P->conv + (size_t)c * Q_PLE_CONV_K;
            x[c] += gt[c] + siluf(w[0] * tap[0][c] + w[1] * tap[1][c] + w[2] * tap[2][c] + w[3] * tap[3][c]);
        }
    }
}

static void ple_forward(inz_engine *e, int n, int p0, float *X) {
    q_ple *P = &e->ple;
    ple_job j = {e, n, p0, X, e->tmp2, e->big, e->tmp, e->Xn};  // E [n][H], K [n][4H], V [n][H], G [n][4H]
    tp_for(e->tp, (int64_t)n * Q_NG_HEADS, 4, ple_gather_task, &j);
    lin_mm(e, &P->key, j.E, Q_H, n, j.K, Q_HCW);
    lin_mm(e, &P->value, j.E, Q_H, n, j.V, Q_H);
    tp_for(e->tp, n, 1, ple_gate_task, &j);    // K now holds norm_conv(G)
    tp_for(e->tp, n, 1, ple_conv_task, &j);
    // the conv history becomes the last 9 normalized rows (part of the old history when the chunk is short)
    float *hist = e->st.ple_conv;
    for (int s = 0; s < e->snap_n && s < n; s++)   // and, for speculative verify, the history after token s
        for (int i = 0; i < 9; i++) {
            const int src = s + 1 - 9 + i;
            memcpy(e->snap[s].ple_conv + (size_t)i * Q_HCW, src >= 0 ? j.K + (size_t)src * Q_HCW : hist + (size_t)(i + s + 1) * Q_HCW,
                   sizeof(float) * Q_HCW);
        }
    for (int i = 0; i < 9; i++) {
        const int src = n - 9 + i;
        if (src >= 0) memcpy(hist + (size_t)i * Q_HCW, j.K + (size_t)src * Q_HCW, sizeof(float) * Q_HCW);
        else memmove(hist + (size_t)i * Q_HCW, hist + (size_t)(i + n) * Q_HCW, sizeof(float) * Q_HCW);
    }
}

/* ------------------------------------------------------------------ */
/* Gated DeltaNet                                                      */
/* ------------------------------------------------------------------ */

typedef struct {
    inz_engine *e;
    const q_layer *L;
    int li, n;
    const float *qkv;    // [n][10240] projections (conv input)
    float *cv;           // [n][10240] conv + silu, q/k normalized
    const float *ab;     // [n][96] a | b projections
    float *g, *beta;     // [n][48]
    float *o;            // [n][6144] recurrence output, then gated-normalized in place
    const float *z;      // [n][6144]
} dn_job;

// causal depthwise conv (kernel 4) over q | k | v, silu, l2-normalized q (scaled by 1/sqrt(dk)) and k; decay and
// update strength. Tokens are independent here: the conv reads raw inputs, from the chunk or the saved state.
static void dn_prep_task(void *ctx, int64_t b, int64_t e_, int tid) {
    (void)tid;
    dn_job *j = ctx;
    const q_layer *L = j->L;
    const float *cs = j->e->st.conv + (size_t)j->li * (Q_DN_KS - 1) * Q_DN_CONV;  // last 3 inputs, oldest first
    for (int64_t t = b; t < e_; t++) {
        const float *tap[4];
        for (int k = 0; k < 3; k++) {
            const int d = 3 - k;                // input at t - d
            tap[k] = t >= d ? j->qkv + (size_t)(t - d) * Q_DN_CONV : cs + (size_t)(3 - d + t) * Q_DN_CONV;
        }
        tap[3] = j->qkv + (size_t)t * Q_DN_CONV;
        float *cv = j->cv + (size_t)t * Q_DN_CONV;
        for (int c = 0; c < Q_DN_CONV; c++) {
            const float *w = L->dn_conv + (size_t)c * Q_DN_KS;
            cv[c] = siluf(w[0] * tap[0][c] + w[1] * tap[1][c] + w[2] * tap[2][c] + w[3] * tap[3][c]);
        }
        float *q = cv, *k = cv + Q_DN_KEY;
        for (int h = 0; h < Q_DN_NK; h++) {     // l2norm (eps inside the root), then q /= sqrt(dk)
            float *qh = q + h * Q_DN_DK, *kh = k + h * Q_DN_DK;
            const float sq = 1.0f / sqrtf(dot_f32(qh, qh, Q_DN_DK) + 1e-6f) / sqrtf((float)Q_DN_DK);
            const float sk = 1.0f / sqrtf(dot_f32(kh, kh, Q_DN_DK) + 1e-6f);
            for (int i = 0; i < Q_DN_DK; i++) { qh[i] *= sq; kh[i] *= sk; }
        }
        const float *ab = j->ab + (size_t)t * 2 * Q_DN_NV;
        for (int h = 0; h < Q_DN_NV; h++) {
            j->g[t * Q_DN_NV + h] = L->dn_negA[h] * softplusf(ab[h] + L->dn_dt[h]);
            j->beta[t * Q_DN_NV + h] = sigmoidf(ab[Q_DN_NV + h]);
        }
    }
}

// one V head (QK head h / 3) through all n tokens: S *= exp(g); d = beta (v - S^T k); S += k d^T; o = S^T q.
// The head's 64 KB state stays in L2 for the whole chunk.
static void dn_scan_task(void *ctx, int64_t b, int64_t e_, int tid) {
    (void)tid;
    dn_job *j = ctx;
    inz_engine *e = j->e;
    const size_t hs = (size_t)Q_DN_DK * Q_DN_DV;
    for (int64_t h = b; h < e_; h++) {
        float *S = e->st.S + ((size_t)j->li * Q_DN_NV + h) * hs;
        const int kh = (int)h / (Q_DN_NV / Q_DN_NK);
        for (int t = 0; t < j->n; t++) {
            const float *cv = j->cv + (size_t)t * Q_DN_CONV;
            const float *q = cv + kh * Q_DN_DK, *k = cv + Q_DN_KEY + kh * Q_DN_DK, *v = cv + 2 * Q_DN_KEY + h * Q_DN_DV;
            const __m256 dv = _mm256_set1_ps(expf(j->g[t * Q_DN_NV + h]));
            const float beta = j->beta[t * Q_DN_NV + h];
            float kv[Q_DN_DV] = {0}, d[Q_DN_DV], o[Q_DN_DV] = {0};
            for (int r = 0; r < Q_DN_DK; r++) {
                float *Sr = S + (size_t)r * Q_DN_DV;
                const __m256 kr = _mm256_set1_ps(k[r]);
                for (int c = 0; c < Q_DN_DV; c += 8) {
                    const __m256 sv = _mm256_mul_ps(_mm256_loadu_ps(Sr + c), dv);
                    _mm256_storeu_ps(Sr + c, sv);
                    _mm256_storeu_ps(kv + c, _mm256_fmadd_ps(sv, kr, _mm256_loadu_ps(kv + c)));
                }
            }
            for (int c = 0; c < Q_DN_DV; c++) d[c] = (v[c] - kv[c]) * beta;
            for (int r = 0; r < Q_DN_DK; r++) {
                float *Sr = S + (size_t)r * Q_DN_DV;
                const __m256 kr = _mm256_set1_ps(k[r]), qr = _mm256_set1_ps(q[r]);
                for (int c = 0; c < Q_DN_DV; c += 8) {
                    const __m256 sv = _mm256_fmadd_ps(kr, _mm256_loadu_ps(d + c), _mm256_loadu_ps(Sr + c));
                    _mm256_storeu_ps(Sr + c, sv);
                    _mm256_storeu_ps(o + c, _mm256_fmadd_ps(sv, qr, _mm256_loadu_ps(o + c)));
                }
            }
            memcpy(j->o + (size_t)t * Q_DN_VAL + h * Q_DN_DV, o, sizeof(o));
            if (t < e->snap_n)                  // speculative verify: this head's state after draft position t
                memcpy(e->snap[t].S + ((size_t)j->li * Q_DN_NV + h) * hs, S, sizeof(float) * hs);
        }
    }
}

// RMSNormGated per head: rmsnorm(o) * w * sigmoid(z), in place
static void dn_out_task(void *ctx, int64_t b, int64_t e_, int tid) {
    (void)tid;
    dn_job *j = ctx;
    for (int64_t t = b; t < e_; t++) {
        float *ot = j->o + (size_t)t * Q_DN_VAL;
        const float *zt = j->z + (size_t)t * Q_DN_VAL;
        for (int h = 0; h < Q_DN_NV; h++) {
            rmsnorm(ot + h * Q_DN_DV, ot + h * Q_DN_DV, j->L->dn_norm, Q_DN_DV);
            for (int i = 0; i < Q_DN_DV; i++) ot[h * Q_DN_DV + i] *= sigmoidf(zt[h * Q_DN_DV + i]);
        }
    }
}

static void dn_block(inz_engine *e, int l, int n, const float *xin, float *y) {
    const q_layer *L = &e->L[l];
    const int li = q_dn_index(l);
    float *qkv = e->big, *z = e->tmp2, *o = e->tmp, *cv = e->Xn;  // [n][10240], [n][6144], [n][6144], [n][10240]
    lin_mm(e, &L->dn_qkv, xin, Q_H, n, qkv, Q_DN_CONV);
    lin_mm(e, &L->dn_z, xin, Q_H, n, z, Q_DN_VAL);
    lin abl;
    q_bind_lin(&abl, INZ_T_F32, 2 * Q_DN_NV, Q_H, (const uint8_t *)L->dn_ab);
    float *abm = e->dn_ab, *g = e->dn_g, *beta = e->dn_beta;
    lin_mm(e, &abl, xin, Q_H, n, abm, 2 * Q_DN_NV);
    dn_job j = {e, L, li, n, qkv, cv, abm, g, beta, o, z};
    tp_for(e->tp, n, 1, dn_prep_task, &j);
    tp_for(e->tp, Q_DN_NV, 1, dn_scan_task, &j);
    tp_for(e->tp, n, 1, dn_out_task, &j);
    // conv state: the last 3 raw inputs (part of the old state when the chunk is short); speculative snapshots
    // take the state after each draft position
    float *cs = e->st.conv + (size_t)li * (Q_DN_KS - 1) * Q_DN_CONV;
    const size_t cz = (size_t)(Q_DN_KS - 1) * Q_DN_CONV;
    for (int s = 0; s < e->snap_n && s < n; s++)
        for (int i = 0; i < 3; i++) {
            const int src = s + 1 - 3 + i;
            memcpy(e->snap[s].conv + (size_t)li * cz + (size_t)i * Q_DN_CONV,
                   src >= 0 ? qkv + (size_t)src * Q_DN_CONV : cs + (size_t)(i + s + 1) * Q_DN_CONV, sizeof(float) * Q_DN_CONV);
        }
    for (int i = 0; i < 3; i++) {
        const int src = n - 3 + i;
        if (src >= 0) memcpy(cs + (size_t)i * Q_DN_CONV, qkv + (size_t)src * Q_DN_CONV, sizeof(float) * Q_DN_CONV);
        else memmove(cs + (size_t)i * Q_DN_CONV, cs + (size_t)(i + n) * Q_DN_CONV, sizeof(float) * Q_DN_CONV);
    }
    lin_mm(e, &L->dn_out, o, Q_DN_VAL, n, y, Q_H);
}

/* ------------------------------------------------------------------ */
/* QSA attention                                                       */
/* ------------------------------------------------------------------ */

#define Q_SEL_MAX (Q_IDX_TOPK * Q_IDX_BLOCK + Q_IDX_BLOCK)  // selected blocks + the incomplete tail

typedef struct {
    inz_engine *e;
    int ai, n, p0;
    const float *q, *iq, *gate;  // [n][24][256], [n][4][128], [n][24][256]
    float *out;                  // [n][24 * 256]
} qsa_job;

// k-th largest of a[0..n) (k >= 1), reordering a: Wirth's selection, linear on average
static float kth_largest(float *a, int n, int k) {
    int lo = 0, hi = n - 1;
    const int target = k - 1;
    while (lo < hi) {
        const float pivot = a[(lo + hi) >> 1];
        int i = lo, j = hi;
        do {
            while (a[i] > pivot) i++;
            while (a[j] < pivot) j--;
            if (i <= j) { const float t = a[i]; a[i] = a[j]; a[j] = t; i++; j--; }
        } while (i <= j);
        if (j < target) lo = i;
        if (target < i) hi = j;
    }
    return a[target];
}

// token selection: the top-512 complete blocks by sum_h relu(iq_h . kblock) / sqrt(128), then the incomplete tail
static void qsa_select_task(void *ctx, int64_t b, int64_t e_, int tid) {
    qsa_job *j = ctx;
    inz_engine *e = j->e;
    const float *BK = e->st.bk + (size_t)j->ai * (e->ctx / Q_IDX_BLOCK) * Q_IDX_HD;
    float *pairs = e->att_scr[tid];             // [nblk][2] (score, block)
    for (int64_t t = b; t < e_; t++) {
        const int p = j->p0 + (int)t, nblk = (p + 1) / Q_IDX_BLOCK;
        int32_t *sel = e->att_sel + (size_t)t * Q_SEL_MAX;
        int ns = 0;
        if (nblk > Q_IDX_TOPK) {
            const float *iq = j->iq + (size_t)t * Q_IDX_NH * Q_IDX_HD;
            float *sc = pairs, *tmp = pairs + nblk;   // scores, and a copy the selection reorders
            for (int bi = 0; bi < nblk; bi++) {
                const float *k = BK + (size_t)bi * Q_IDX_HD;
                float s = 0;
                for (int h = 0; h < Q_IDX_NH; h++) {
                    const float d = dot_f32(iq + h * Q_IDX_HD, k, Q_IDX_HD);
                    s += d > 0 ? d : 0;
                }
                sc[bi] = tmp[bi] = s / sqrtf((float)Q_IDX_HD);
            }
            // the 512 best blocks (a set: order does not matter); ties at the threshold go to the earliest blocks
            const float thr = kth_largest(tmp, nblk, Q_IDX_TOPK);
            int above = 0;
            for (int bi = 0; bi < nblk; bi++) above += sc[bi] > thr;
            int ties = Q_IDX_TOPK - above;
            for (int bi = 0; bi < nblk; bi++)
                if (sc[bi] > thr || (sc[bi] == thr && ties-- > 0))
                    for (int k = 0; k < Q_IDX_BLOCK; k++) sel[ns++] = bi * Q_IDX_BLOCK + k;
        } else {
            // every complete block is selected (the order of the selection does not matter)
            for (int x = 0; x < nblk * Q_IDX_BLOCK; x++) sel[ns++] = x;
        }
        for (int x = nblk * Q_IDX_BLOCK; x <= p; x++) sel[ns++] = x;
        e->att_nsel[t] = ns;
    }
}

// one (token, head) pair: softmax(q k^T / 16) v over the selected tokens, times sigmoid(gate)
static void qsa_head_task(void *ctx, int64_t b, int64_t e_, int tid) {
    qsa_job *j = ctx;
    inz_engine *e = j->e;
    const uint16_t *K = e->st.k + (size_t)j->ai * e->ctx * Q_KV, *V = e->st.v + (size_t)j->ai * e->ctx * Q_KV;
    float *sc = e->att_scr[tid];
    for (int64_t th = b; th < e_; th++) {
        const int t = (int)(th / Q_NH), h = (int)(th % Q_NH), g = h / (Q_NH / Q_NKV);
        const int32_t *sel = e->att_sel + (size_t)t * Q_SEL_MAX;
        const int ns = e->att_nsel[t];
        const float *qh = j->q + ((size_t)t * Q_NH + h) * Q_HD;
        for (int i = 0; i < ns; i++) sc[i] = dot_f16(K + (size_t)sel[i] * Q_KV + g * Q_HD, qh, Q_HD) / 16.0f;
        softmax_inplace(sc, ns);
        float o[Q_HD] = {0};
        for (int i = 0; i < ns; i++) axpy_f16(o, sc[i], V + (size_t)sel[i] * Q_KV + g * Q_HD, Q_HD);
        const float *gt = j->gate + ((size_t)t * Q_NH + h) * Q_HD;
        float *dst = j->out + (size_t)t * Q_NH * Q_HD + h * Q_HD;
        for (int d = 0; d < Q_HD; d++) dst[d] = o[d] * sigmoidf(gt[d]);
    }
}

// ai: attention cache index (0..11 for the model's QSA layers, Q_N_ATTN for the MTP layer)
static void qsa_block(inz_engine *e, const q_layer *L, int ai, int n, int p0, const float *xin, float *y) {
    float *qg = e->big;                         // [n][12288]: per head q (256) | gate (256)
    float *kv = e->tmp;                         // [n][1024]: k | v
    float *idx = e->tmp2;                       // [n][640]: 4 index query heads | index key
    lin_mm(e, &L->at_q, xin, Q_H, n, qg, Q_QPROJ);
    lin_mm(e, &L->at_kv, xin, Q_H, n, kv, 2 * Q_KV);
    lin_mm(e, &L->at_idx, xin, Q_H, n, idx, Q_IDX_PROJ);
    float *q = e->Xn;                           // [n][24][256] (Xn is free inside a block)
    float *gate = e->Xn + (size_t)Q_MAX_T * Q_NH * Q_HD;
    float *iq = e->y;                           // [n][4][128] (y is written only by o_proj, at the end)
    q_state *st = &e->st;
    uint16_t *K = st->k + (size_t)ai * e->ctx * Q_KV, *V = st->v + (size_t)ai * e->ctx * Q_KV;
    float *IK = st->ik + (size_t)ai * e->ctx * Q_IDX_HD;
    float *BK = st->bk + (size_t)ai * (e->ctx / Q_IDX_BLOCK) * Q_IDX_HD;
    for (int t = 0; t < n; t++) {
        const int p = p0 + t;
        for (int h = 0; h < Q_NH; h++) {
            const float *src = qg + (size_t)t * Q_QPROJ + h * 2 * Q_HD;
            float *qh = q + ((size_t)t * Q_NH + h) * Q_HD;
            rmsnorm(qh, src, L->q_norm, Q_HD);
            rope(qh, p);
            memcpy(gate + ((size_t)t * Q_NH + h) * Q_HD, src + Q_HD, sizeof(float) * Q_HD);
        }
        float kk[Q_KV];
        for (int g = 0; g < Q_NKV; g++) {
            rmsnorm(kk + g * Q_HD, kv + (size_t)t * 2 * Q_KV + g * Q_HD, L->k_norm, Q_HD);
            rope(kk + g * Q_HD, p);
        }
        f32_to_f16(K + (size_t)p * Q_KV, kk, Q_KV);
        f32_to_f16(V + (size_t)p * Q_KV, kv + (size_t)t * 2 * Q_KV + Q_KV, Q_KV);
        const float *it = idx + (size_t)t * Q_IDX_PROJ;
        for (int h = 0; h < Q_IDX_NH; h++) {
            float *dst = iq + ((size_t)t * Q_IDX_NH + h) * Q_IDX_HD;
            rmsnorm(dst, it + h * Q_IDX_HD, L->iq_norm, Q_IDX_HD);
            rope(dst, p);
        }
        memcpy(IK + (size_t)p * Q_IDX_HD, it + Q_IDX_NH * Q_IDX_HD, sizeof(float) * Q_IDX_HD);
        if ((p + 1) % Q_IDX_BLOCK == 0) {       // a block completes: mean of raw keys, normalize, RoPE at its start
            const int bi = p / Q_IDX_BLOCK;
            float pool[Q_IDX_HD];
            for (int d = 0; d < Q_IDX_HD; d++) {
                float s = 0;
                for (int k = 0; k < Q_IDX_BLOCK; k++) s += IK[(size_t)(bi * Q_IDX_BLOCK + k) * Q_IDX_HD + d];
                pool[d] = s / Q_IDX_BLOCK;
            }
            rmsnorm(BK + (size_t)bi * Q_IDX_HD, pool, L->ik_norm, Q_IDX_HD);
            rope(BK + (size_t)bi * Q_IDX_HD, bi * Q_IDX_BLOCK);
        }
    }
    float *att = e->big;                        // [n][6144] (q | gate were copied out of big)
    qsa_job j = {e, ai, n, p0, q, iq, gate, att};
    tp_for(e->tp, n, 1, qsa_select_task, &j);
    tp_for(e->tp, (int64_t)n * Q_NH, 1, qsa_head_task, &j);
    lin_mm(e, &L->at_o, att, Q_NH * Q_HD, n, y, Q_H);
}

/* ------------------------------------------------------------------ */
/* MoE                                                                 */
/* ------------------------------------------------------------------ */

// tokens routed to one expert; slot = t * Q_TOPK + rank, where the expert's weighted output for that token goes
typedef struct { int expert, ntok; int tok[Q_MAX_T]; int slot[Q_MAX_T]; float w[Q_MAX_T]; } unit;

// out[slot] = w * down(silu(gate x) * up x) for the tokens routed to one expert. The slots are summed in rank order
// afterwards, so the result does not depend on which experts were resident or in which order reads completed.
static void expert_compute(inz_engine *e, const uint8_t *blob, const unit *u, const float *xin, float *out) {
    lin gu, dn;
    if (!q_bind_lin(&gu, e->expert_type, 2 * Q_FF, Q_H, blob) || !q_bind_lin(&dn, e->expert_type, Q_H, Q_FF, blob + e->gu_bytes))
        inz_die("expert blob is corrupt (expert %d)", u->expert);
    float *X = e->ex_x, *GU = e->ex_gu, *D = e->ex_d;
    for (int i = 0; i < u->ntok; i++) memcpy(X + (size_t)i * Q_H, xin + (size_t)u->tok[i] * Q_H, sizeof(float) * Q_H);
    lin_mm(e, &gu, X, Q_H, u->ntok, GU, 2 * Q_FF);
    for (int i = 0; i < u->ntok; i++) {
        float *r = GU + (size_t)i * 2 * Q_FF;
        for (int c = 0; c < Q_FF; c++) r[c] = siluf(r[c]) * r[Q_FF + c];
    }
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
#define EB_MAX 80                                // experts x tokens of one batch (10 experts x 8 tokens)
typedef struct {
    const unit *u;
    lin gu, dn;
    inz_act ga[8], da[8];                        // per token: the quantized input of gate|up, of down
    float *GU;                                   // [ntok][2 FF]
    float *D;                                    // [ntok][H]
} ebatch_item;

typedef struct { inz_engine *e; ebatch_item *it; int ni; const float *xin; } ebatch_job;

#define EB_CHUNK 16
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
            for (int c = 0; c < Q_FF; c++) r[c] = siluf(r[c]) * r[Q_FF + c];
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

// compute the given units (blobs pinned) as one batch; quantized expert types only. The tokens' MoE inputs are
// quantized once for the gate|up rotation, which all experts of a layer share; an expert with another rotation
// (not produced by the converter) is computed on its own.
static void experts_batch(inz_engine *e, const unit *const *us, const uint8_t *const *blobs, int nb, int n, const float *xin) {
    ebatch_item *it = e->eb_items;
    int ni = 0, rows = 0, qrot = -1;
    uint32_t qseed = 0;
    for (int i = 0; i < nb; i++) {
        ebatch_item *x = &it[ni];
        if (!q_bind_lin(&x->gu, e->expert_type, 2 * Q_FF, Q_H, blobs[i]) ||
            !q_bind_lin(&x->dn, e->expert_type, Q_H, Q_FF, blobs[i] + e->gu_bytes))
            inz_die("expert blob is corrupt (expert %d)", us[i]->expert);
        if (qrot < 0) {
            qrot = x->gu.q.rot_block;
            qseed = x->gu.q.rot_seed;
            for (int t = 0; t < n; t++) inz_act_quantize(&e->eb_xa[t], xin + (size_t)t * Q_H, Q_H, qrot, qseed);
        } else if (x->gu.q.rot_block != qrot || x->gu.q.rot_seed != qseed) {
            expert_compute(e, blobs[i], us[i], xin, e->moe_out);
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
    ebatch_job j = {e, it, ni, xin};
    tp_for(e->tp, (int64_t)ni * (2 * Q_FF / EB_CHUNK), 4, eb_gu_task, &j);
    tp_for(e->tp, ni, 1, eb_act_task, &j);
    tp_for(e->tp, (int64_t)ni * (Q_H / EB_CHUNK), 8, eb_down_task, &j);
}

// l: expert cache layer (0..47, n_layer for the MTP layer)
static void moe_block(inz_engine *e, const q_layer *L, int l, int n, const float *xin, float *y) {
    e->imx_layer = l;
    float *logits = e->tmp;                     // [n][512]
    uint64_t t0 = diag_tsc();
    lin_mm(e, &L->router, xin, Q_H, n, logits, Q_N_EXPERT);
    unit *units = e->units;
    int uid[Q_N_EXPERT];
    for (int x = 0; x < Q_N_EXPERT; x++) uid[x] = -1;
    int nu = 0;
    for (int t = 0; t < n; t++) {
        float *pr = logits + (size_t)t * Q_N_EXPERT;
        softmax_inplace(pr, Q_N_EXPERT);
        int32_t sel[Q_TOPK];
        float w[Q_TOPK], sum = 0;
        bool used[Q_N_EXPERT] = {0};
        for (int k = 0; k < Q_TOPK; k++) {      // top-k by probability, lowest index first on ties
            int best = -1;
            for (int x = 0; x < Q_N_EXPERT; x++)
                if (!used[x] && (best < 0 || pr[x] > pr[best])) best = x;
            used[best] = true;
            sel[k] = best;
            w[k] = pr[best];
            sum += w[k];
        }
        diag_routing(l, sel, w, Q_TOPK, e->n_past + t);
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
            u->w[u->ntok] = w[k] / sum;
            u->ntok++;
        }
    }
    diag_add(DG_ROUTER, l, t0, q_lin_bytes(&L->router), 2ull * n * Q_N_EXPERT * Q_H);
    memset(y, 0, sizeof(float) * (size_t)n * Q_H);
    // routed experts: read requests for every missing one first, then the next layer's predicted experts
    // (decode), then compute the resident ones while the SSD works, then the rest as they arrive
    t0 = diag_tsc();
    bool done[Q_N_EXPERT] = {0}, asked[Q_N_EXPERT] = {0};
    int left = nu, lvl;
    if (n <= 8)
        for (int i = 0; i < nu; i++)
            if (!ec_is_resident(e->ec, l, units[i].expert)) { ec_request(e->ec, l, units[i].expert, 1.0f, EC_FULL); asked[i] = true; }
    if (n <= 8 && e->opt.lookahead > 0 && l + 1 < e->n_layer && !e->ec_all_resident) {
        // the residual drifts slowly: this layer's MoE input through the next layer's router predicts its experts
        const uint64_t tl = diag_tsc();
        float *lg2 = e->tmp + (size_t)n * Q_N_EXPERT;
        lin_mm(e, &e->L[l + 1].router, xin, Q_H, n, lg2, Q_N_EXPERT);
        for (int t = 0; t < n; t++) {
            const float *lg = lg2 + (size_t)t * Q_N_EXPERT;
            int32_t ps[Q_TOPK];
            bool used[Q_N_EXPERT] = {0};
            for (int k = 0; k < Q_TOPK; k++) {
                int best = -1;
                for (int x = 0; x < Q_N_EXPERT; x++) if (!used[x] && (best < 0 || lg[x] > lg[best])) best = x;
                used[best] = true;
                ps[k] = best;
                ec_prefetch(e->ec, l + 1, best, 0.5f, EC_FULL);
            }
            if (n == 1) { memcpy(e->pred[l + 1], ps, sizeof(ps)); e->pred_valid[l + 1] = true; g_diag.pred_total += Q_TOPK; }
        }
        diag_add(DG_PREFETCH, l, tl, q_lin_bytes(&e->L[l + 1].router), 0);
    }
    const bool batch = n <= 8 && nu <= EB_MAX && !e->imx_exp && e->expert_type != INZ_T_F16 && e->expert_type != INZ_T_F32;
    const unit *bu[EB_MAX];
    const uint8_t *bb[EB_MAX];
    int nb = 0;
    for (int i = 0; i < nu; i++) {
        const uint8_t *b = ec_try_acquire(e->ec, l, units[i].expert, 1.0f, EC_FULL, &lvl);
        if (!b) { if (!asked[i]) ec_request(e->ec, l, units[i].expert, 1.0f, EC_FULL); continue; }
        if (batch) { bu[nb] = &units[i]; bb[nb++] = b; }
        else {
            expert_compute(e, b, &units[i], xin, e->moe_out);
            ec_release(e->ec, l, units[i].expert);
        }
        done[i] = true;
        left--;
    }
    if (nb) {
        experts_batch(e, bu, bb, nb, n, xin);
        for (int i = 0; i < nb; i++) ec_release(e->ec, l, bu[i]->expert);
    }
    diag_add(DG_EXPERT_HIT, l, t0, (uint64_t)(nu - left) * e->blob_bytes, 0);
    t0 = diag_tsc();
    while (left > 0) {
        bool progress = false;
        for (int i = 0; i < nu; i++) {
            if (done[i]) continue;
            const uint8_t *b = ec_try_acquire(e->ec, l, units[i].expert, 0, EC_FULL, &lvl);
            if (!b) continue;
            expert_compute(e, b, &units[i], xin, e->moe_out);
            ec_release(e->ec, l, units[i].expert);
            done[i] = true;
            left--;
            progress = true;
        }
        if (progress || left == 0) continue;
        if (!ec_wait_any(e->ec)) {              // nothing in flight: fall back to a blocking read
            for (int i = 0; i < nu; i++)
                if (!done[i]) {
                    const uint8_t *b = ec_acquire(e->ec, l, units[i].expert, 0, EC_FULL, &lvl);
                    expert_compute(e, b, &units[i], xin, e->moe_out);
                    ec_release(e->ec, l, units[i].expert);
                    done[i] = true;
                    left--;
                    break;
                }
        }
    }
    diag_add(DG_EXPERT_MISS, l, t0, 0, 0);
    for (int t = 0; t < n; t++) {                // routed outputs in rank order: deterministic
        float *yt = y + (size_t)t * Q_H;
        for (int k = 0; k < Q_TOPK; k++) {
            const float *o = e->moe_out + ((size_t)t * Q_TOPK + k) * Q_H;
            for (int c = 0; c < Q_H; c++) yt[c] += o[c];
        }
    }
    // shared expert: y += sigmoid(sh_gate . x) * down(silu(gate x) * up x)
    t0 = diag_tsc();
    float *GU = e->big, *D = e->tmp2;
    lin_mm(e, &L->sh_gu, xin, Q_H, n, GU, 2 * Q_SHARED_FF);
    for (int t = 0; t < n; t++) {
        float *r = GU + (size_t)t * 2 * Q_SHARED_FF;
        for (int c = 0; c < Q_SHARED_FF; c++) r[c] = siluf(r[c]) * r[Q_SHARED_FF + c];
    }
    lin_mm(e, &L->sh_down, GU, 2 * Q_SHARED_FF, n, D, Q_H);
    for (int t = 0; t < n; t++) {
        const float sg = sigmoidf(dot_f32(L->sh_gate, xin + (size_t)t * Q_H, Q_H));
        float *yt = y + (size_t)t * Q_H;
        const float *dt = D + (size_t)t * Q_H;
        for (int c = 0; c < Q_H; c++) yt[c] += sg * dt[c];
    }
    diag_add(DG_SHARED, l, t0, q_lin_bytes(&L->sh_gu) + q_lin_bytes(&L->sh_down), 0);
}

/* ------------------------------------------------------------------ */
/* Scratch, state, forward                                             */
/* ------------------------------------------------------------------ */

void q_scratch_alloc(inz_engine *e) {
    const size_t T = Q_MAX_T;
    e->X = inz_aligned(64, sizeof(float) * T * Q_HCW);
    e->X2 = inz_aligned(64, sizeof(float) * T * Q_HCW);
    e->Xn = inz_aligned(64, sizeof(float) * T * Q_QPROJ); // 4H normed residual; attention q | gate (2 x 6144)
    e->xin = inz_aligned(64, sizeof(float) * T * Q_H);
    e->y = inz_aligned(64, sizeof(float) * T * Q_H);
    e->inj = inz_aligned(64, sizeof(float) * T * Q_HC);
    e->tmp = inz_aligned(64, sizeof(float) * T * Q_DN_VAL);
    e->tmp2 = inz_aligned(64, sizeof(float) * T * Q_DN_VAL);
    e->big = inz_aligned(64, sizeof(float) * T * Q_QPROJ);
    e->dn_ab = inz_aligned(64, sizeof(float) * T * 2 * Q_DN_NV);
    e->dn_g = inz_aligned(64, sizeof(float) * T * Q_DN_NV);
    e->dn_beta = inz_aligned(64, sizeof(float) * T * Q_DN_NV);
    e->ex_x = inz_aligned(64, sizeof(float) * T * Q_H);
    e->moe_out = inz_aligned(64, sizeof(float) * T * Q_TOPK * Q_H);
    e->eb_items = inz_calloc(EB_MAX, sizeof(ebatch_item));
    e->eb_gu = inz_aligned(64, sizeof(float) * EB_MAX * 2 * Q_FF);
    e->eb_d = inz_aligned(64, sizeof(float) * EB_MAX * Q_H);
    e->eb_acts = inz_calloc(EB_MAX, sizeof(inz_act));
    for (int i = 0; i < EB_MAX; i++) inz_act_alloc(&e->eb_acts[i], Q_FF);
    e->eb_xa = inz_calloc(8, sizeof(inz_act));
    for (int i = 0; i < 8; i++) inz_act_alloc(&e->eb_xa[i], Q_H);
    e->ex_gu = inz_aligned(64, sizeof(float) * T * 2 * Q_FF);
    e->ex_d = inz_aligned(64, sizeof(float) * T * Q_H);
    e->units = inz_calloc(Q_N_EXPERT, sizeof(unit));
    e->acts = inz_calloc(T, sizeof(inz_act));
    for (size_t i = 0; i < T; i++) inz_act_alloc(&e->acts[i], Q_HCW);
    const int nthr = tp_size(e->tp);
    e->att_scr = inz_calloc((size_t)nthr, sizeof(float *));
    const size_t scr = 2 * ((size_t)e->ctx / Q_IDX_BLOCK + 2) + Q_SEL_MAX + 64;
    for (int i = 0; i < nthr; i++) e->att_scr[i] = inz_aligned(64, sizeof(float) * scr);
    e->att_sel = inz_aligned(64, sizeof(int32_t) * T * Q_SEL_MAX);
    e->att_nsel = inz_calloc(T, sizeof(int));
    e->gemm_tb = inz_calloc((size_t)nthr, sizeof(uint8_t *));
    e->gemm_tmp = inz_calloc((size_t)nthr, sizeof(float *));
    for (int i = 0; i < nthr; i++) {
        e->gemm_tb[i] = inz_aligned(64, inz_gemm_tilebuf_bytes(Q_HCW));
        e->gemm_tmp[i] = inz_aligned(64, sizeof(float) * Q_HCW + 64);
    }
    inz_xq_alloc(&e->gemm_x, Q_MAX_T, Q_HCW);
    for (int i = 0; i < 256; i++) e->e4m3[i] = inz_e4m3_to_f32((uint8_t)i);
}

void q_scratch_free(inz_engine *e) {
    free(e->X); free(e->X2); free(e->Xn); free(e->xin); free(e->y); free(e->inj); free(e->tmp); free(e->tmp2); free(e->big);
    free(e->ex_x); free(e->ex_gu); free(e->ex_d); free(e->units); free(e->moe_out);
    for (int i = 0; i < EB_MAX; i++) inz_act_free(&e->eb_acts[i]);
    for (int i = 0; i < 8; i++) inz_act_free(&e->eb_xa[i]);
    free(e->eb_items); free(e->eb_gu); free(e->eb_d); free(e->eb_acts); free(e->eb_xa);
    free(e->dn_ab); free(e->dn_g); free(e->dn_beta);
    for (int i = 0; i < Q_MAX_T; i++) inz_act_free(&e->acts[i]);
    free(e->acts);
    const int nthr = tp_size(e->tp);
    for (int i = 0; i < nthr; i++) { free(e->att_scr[i]); free(e->gemm_tb[i]); free(e->gemm_tmp[i]); }
    free(e->att_scr); free(e->gemm_tb); free(e->gemm_tmp);
    free(e->att_sel); free(e->att_nsel);
    inz_xq_free(&e->gemm_x);
}

void q_state_reset(inz_engine *e) {
    memset(e->st.S, 0, sizeof(float) * (size_t)Q_N_DN * Q_DN_NV * Q_DN_DK * Q_DN_DV);
    memset(e->st.conv, 0, sizeof(float) * (size_t)Q_N_DN * (Q_DN_KS - 1) * Q_DN_CONV);
    memset(e->st.ple_conv, 0, sizeof(float) * 9 * Q_HCW);
    e->n_past = 0;
    e->mtp_valid = 0;
}

void q_forward(inz_engine *e, const int32_t *tokens, int n, const int *logits_rows, int n_logits, float *out_logits) {
    if (n > Q_MAX_T) inz_die("q_forward: %d tokens > %d", n, Q_MAX_T);
    if (e->n_past + n > e->ctx) inz_die("q_forward: context full (%d + %d > %d)", e->n_past, n, e->ctx);
    const int p0 = e->n_past;
    memcpy(e->hist + p0, tokens, sizeof(int32_t) * n);
    uint64_t t0 = diag_tsc();
    for (int t = 0; t < n; t++) {
        float x[Q_H];
        embed_row(e, tokens[t], x);
        for (int s = 0; s < Q_HC; s++) memcpy(e->X + (size_t)t * Q_HCW + s * Q_H, x, sizeof(x));
    }
    diag_add(DG_EMBED, -1, t0, (uint64_t)n * Q_H, 0);
    static FILE *dump;  // validation: INZ_DUMP_HIDDEN=file appends the 4H residual after every layer
    static int dump_init;
    if (!dump_init) {
        dump_init = 1;
        const char *pth = getenv("INZ_DUMP_HIDDEN");
        if (pth && *pth) dump = fopen(pth, "wb");
    }
    for (int l = 0; l < e->n_layer; l++) {
        const q_layer *L = &e->L[l];
        if (l == Q_PLE_LAYER && e->ple.present) {
            t0 = diag_tsc();
            ple_forward(e, n, p0, e->X);
            diag_add(DG_OTHER, l, t0, q_lin_bytes(&e->ple.key) + q_lin_bytes(&e->ple.value), 0);
        }
        t0 = diag_tsc();
        hc_in(e, &L->ahc, n, e->X, e->xin, e->inj);
        diag_add(DG_ATTN_NORM, l, t0, q_lin_bytes(&L->ahc.down) + q_lin_bytes(&L->ahc.up), 0);
        t0 = diag_tsc();
        if (q_is_attn(l)) {
            qsa_block(e, L, q_attn_index(l), n, p0, e->xin, e->y);
            diag_add(DG_ATTN_CORE, l, t0, q_lin_bytes(&L->at_q) + q_lin_bytes(&L->at_kv) + q_lin_bytes(&L->at_idx) + q_lin_bytes(&L->at_o), 0);
        } else {
            dn_block(e, l, n, e->xin, e->y);
            diag_add(DG_ATTN_CORE, l, t0, q_lin_bytes(&L->dn_qkv) + q_lin_bytes(&L->dn_z) + q_lin_bytes(&L->dn_out), 0);
        }
        hc_out(n, e->X, e->inj, e->y);
        t0 = diag_tsc();
        hc_in(e, &L->mhc, n, e->X, e->xin, e->inj);
        diag_add(DG_FFN_NORM, l, t0, q_lin_bytes(&L->mhc.down) + q_lin_bytes(&L->mhc.up), 0);
        moe_block(e, L, l, n, e->xin, e->y);
        hc_out(n, e->X, e->inj, e->y);
        if (dump) { fwrite(e->X, sizeof(float), (size_t)n * Q_HCW, dump); fflush(dump); }
    }
    e->n_past += n;
    memcpy(e->x_last, e->X + (size_t)(n - 1) * Q_HCW, sizeof(float) * Q_HCW);
    if (e->ec) ec_tick(e->ec);
    if (n_logits <= 0) return;
    q_head(e, &e->mixer, e->X, n, logits_rows, n_logits, out_logits);
}

#define HEAD_MIN_CAND 32
#define HEAD_MAX_CAND 2048

typedef struct { inz_engine *e; const int32_t *cand; uint8_t *qbuf; uint16_t *dbuf; const float *x; float *ex; } head_job;

// exact rows of the candidates, from the file (hot rows stay in the page cache), and their logits (F16 rows)
static void head_rows_task(void *ctx, int64_t b, int64_t e_, int tid) {
    (void)tid;
    head_job *j = ctx;
    inz_engine *e = j->e;
    for (int64_t k = b; k < e_; k++) {
        const int r = j->cand[k];
        if (e->head.type == INZ_T_F16) {
            uint16_t row[Q_H];
            if (pread(e->head_fd, row, sizeof(row), (off_t)(e->head_off + (uint64_t)r * sizeof(row))) != (ssize_t)sizeof(row))
                inz_die("LM head row read failed");
            j->ex[k] = dot_f16(row, j->x, Q_H);
            continue;
        }
        const inz_qmat *m = &e->head.q;
        const size_t qrow = (size_t)m->ngroups * m->qbytes, drow = (size_t)m->ngroups * 2;
        const uint64_t qbase = e->head_off + (uint64_t)((const uint8_t *)m->q - (const uint8_t *)&e->head_hdr);
        const uint64_t dbase = e->head_off + (uint64_t)((const uint8_t *)m->d - (const uint8_t *)&e->head_hdr);
        if (pread(e->head_fd, j->qbuf + (size_t)k * qrow, qrow, (off_t)(qbase + (uint64_t)r * qrow)) != (ssize_t)qrow ||
            pread(e->head_fd, j->dbuf + (size_t)k * m->ngroups, drow, (off_t)(dbase + (uint64_t)r * drow)) != (ssize_t)drow)
            inz_die("LM head row read failed");
    }
}

// Two-stage head for one row: the screen scored every token; rescore exactly the tokens within a fixed margin of
// the best (6 sigma + 2 nats for the Q4L screen's measured error, sigma ~ 0.23), then remove the noise bias of the
// unscored tail. The margin is fixed, not adaptive, so a row's logits depend on that row only: a token gets the
// same logits whether it is decoded alone or inside a speculative verify pass.
#define HEAD_MARGIN 3.5f
static void head_rescore(inz_engine *e, const float *x, float *lg) {
    float mx = -INFINITY;
    for (int v = 0; v < Q_VOCAB; v++) mx = lg[v] > mx ? lg[v] : mx;
    int32_t *cand = inz_malloc(sizeof(int32_t) * HEAD_MAX_CAND);
    float margin = HEAD_MARGIN;
    int nc;
    for (;;) {
        nc = 0;
        const float th = mx - margin;
        for (int v = 0; v < Q_VOCAB && nc < HEAD_MAX_CAND; v++) if (lg[v] >= th) cand[nc++] = v;
        if (nc >= HEAD_MAX_CAND && margin > 1.0f) { margin *= 0.7f; continue; }
        if (nc < HEAD_MIN_CAND && margin < 64) { margin *= 1.5f; continue; }
        break;
    }
    float *ex = inz_malloc(sizeof(float) * nc);
    head_job j = {e, cand, NULL, NULL, x, ex};
    if (e->head.type != INZ_T_F16) {
        j.qbuf = inz_malloc((size_t)e->head.q.ngroups * e->head.q.qbytes * nc);
        j.dbuf = inz_malloc(sizeof(uint16_t) * (size_t)e->head.q.ngroups * nc);
    }
    tp_for(e->tp, nc, 8, head_rows_task, &j);
    if (e->head.type != INZ_T_F16) {
        inz_qmat sub = e->head.q;
        sub.rows = nc;
        sub.q = j.qbuf;
        sub.d = j.dbuf;
        inz_act_quantize(&e->acts[0], x, Q_H, 0, 0);
        inz_matvec_rows(&sub, &e->acts[0], ex, 0, nc);
        free(j.qbuf);
        free(j.dbuf);
    }
    double se = 0;
    for (int k = 0; k < nc; k++) {
        const double d = ex[k] - lg[cand[k]];
        se += d * d;
    }
    const double sigma = sqrt(se / (nc ? nc : 1));
    e->head_sigma = 0.9 * e->head_sigma + 0.1 * sigma;  // diagnostics only
    // zero-mean logit noise of variance s^2 inflates exp() by e^(s^2/2) on average: remove that from the tail
    const float corr = (float)(0.5 * sigma * sigma);
    for (int v = 0; v < Q_VOCAB; v++) lg[v] -= corr;
    for (int k = 0; k < nc; k++) lg[cand[k]] = ex[k];
    e->head_rows++;
    e->head_cands += (uint64_t)nc;
    free(ex);
    free(cand);
}

// final mixer (no other norm) of residual rows X, then the LM head on the requested rows
void q_head(inz_engine *e, const hcw *mixer, const float *X, int n, const int *logits_rows, int n_logits, float *out_logits) {
    uint64_t t0 = diag_tsc();
    hc_in(e, mixer, n, X, e->xin, NULL);
    float *rows = e->tmp;                       // [n_logits][H]
    for (int i = 0; i < n_logits; i++) memcpy(rows + (size_t)i * Q_H, e->xin + (size_t)logits_rows[i] * Q_H, sizeof(float) * Q_H);
    if (!e->two_stage) {
        lin_mm(e, &e->head, rows, Q_H, n_logits, out_logits, Q_VOCAB);
        diag_add(DG_LM_HEAD, -1, t0, q_lin_bytes(&e->head), 2ull * n_logits * Q_VOCAB * Q_H);
        return;
    }
    lin_mm(e, &e->head_screen, rows, Q_H, n_logits, out_logits, Q_VOCAB);
    const uint64_t c0 = e->head_cands;
    for (int i = 0; i < n_logits; i++) head_rescore(e, rows + (size_t)i * Q_H, out_logits + (size_t)i * Q_VOCAB);
    diag_add(DG_LM_HEAD, -1, t0, q_lin_bytes(&e->head_screen) + (e->head_cands - c0) * (uint64_t)inz_mat_bytes(e->head.type, 1, Q_H),
             2ull * n_logits * Q_VOCAB * Q_H);
}

/* ------------------------------------------------------------------ */
/* MTP: one extra QSA + MoE layer that drafts token p+2 from (X_p, p+1) */
/* ------------------------------------------------------------------ */

typedef struct { const float *X; int ldx; float *Xn; const float *w; } mtp_norm_job;
static void mtp_norm_task(void *ctx, int64_t b, int64_t e_, int tid) {
    (void)tid;
    mtp_norm_job *j = ctx;
    for (int64_t r = b; r < e_; r++) {          // row r = token r / 4, stream r % 4
        const int t = (int)(r / Q_HC), s = (int)(r % Q_HC);
        rmsnorm(j->Xn + (size_t)r * Q_H, j->X + (size_t)t * j->ldx + s * Q_H, j->w + s * Q_H, Q_H);
    }
}

// Slots p0 .. p0+n-1 of the MTP layer: the main residual X[t] (after layer 47, before the mixer) of position p0+t
// with the token at position p0+t+1. Writes the MTP residual to e->X2 (input of a chained draft step) and logits
// for the requested rows.
void q_mtp_forward(inz_engine *e, const float *X, int ldx, const int32_t *next, int n, int p0, const int *logits_rows,
                   int n_logits, float *out_logits) {
    if (n > Q_MAX_T) inz_die("q_mtp_forward: %d tokens > %d", n, Q_MAX_T);
    const q_mtp *M = &e->mtp;
    uint64_t t0 = diag_tsc();
    // X0 = [fc_hidden . (rmsnorm(X_s) * norm_hid_s)]_s : 4n rows of H through one H x H matrix
    mtp_norm_job nj = {X, ldx, e->Xn, M->norm_hid};
    tp_for(e->tp, (int64_t)n * Q_HC, 4, mtp_norm_task, &nj);
    for (int r = 0; r < n * Q_HC; r += Q_MAX_T)  // activation scratch holds Q_MAX_T rows
        lin_mm(e, &M->fc_hid, e->Xn + (size_t)r * Q_H, Q_H, n * Q_HC - r < Q_MAX_T ? n * Q_HC - r : Q_MAX_T,
               e->X2 + (size_t)r * Q_H, Q_H);
    // + fc_embedding . (rmsnorm(embed(next)) * norm_emb) on every stream (unit-weight combine)
    float *em = e->tmp2, *ef = e->y;
    for (int t = 0; t < n; t++) {
        float x[Q_H];
        embed_row(e, next[t], x);
        rmsnorm(em + (size_t)t * Q_H, x, M->norm_emb, Q_H);
    }
    lin_mm(e, &M->fc_emb, em, Q_H, n, ef, Q_H);
    for (int t = 0; t < n; t++)
        for (int s = 0; s < Q_HC; s++) {
            float *xs = e->X2 + (size_t)t * Q_HCW + s * Q_H;
            for (int i = 0; i < Q_H; i++) xs[i] += ef[(size_t)t * Q_H + i];
        }
    diag_add(DG_DRAFT, -1, t0, q_lin_bytes(&M->fc_hid) + q_lin_bytes(&M->fc_emb), 0);
    const q_layer *L = &M->L;
    const int l = e->n_layer;                   // the MTP layer's index in the expert cache and in diagnostics
    hc_in(e, &L->ahc, n, e->X2, e->xin, e->inj);
    qsa_block(e, L, Q_N_ATTN, n, p0, e->xin, e->y);
    hc_out(n, e->X2, e->inj, e->y);
    hc_in(e, &L->mhc, n, e->X2, e->xin, e->inj);
    moe_block(e, L, l, n, e->xin, e->y);
    hc_out(n, e->X2, e->inj, e->y);
    if (n_logits > 0) q_head(e, &M->mixer, e->X2, n, logits_rows, n_logits, out_logits);
}

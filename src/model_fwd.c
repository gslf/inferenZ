// Forward pass of Nemotron 3.5 Super (docs/MODEL.md gives the math; every block below follows it line by line).
//
// One code path serves decode (n = 1) and prefill chunks (n <= Q_MAX_T). Weights are `lin` matrices in any
// storage type; activations stay f32 between operators. The Mamba-2 recurrence and its short convolution advance
// token by token; projections, attention and experts are batched over the chunk.
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

// out = x * rsqrt(mean(x^2) + eps) * w (NULL: no weight)
static void rmsnorm(float *out, const float *x, const float *w, int n) {
    const float r = 1.0f / sqrtf(dot_f32(x, x, n) / n + Q_RMS_EPS);
    for (int i = 0; i < n; i++) out[i] = x[i] * r * (w ? w[i] : 1.0f);
}

static inline float siluf(float x) { return x / (1.0f + expf(-x)); }
static inline float sigmoidf(float x) { return 1.0f / (1.0f + expf(-x)); }
static inline float softplusf(float x) { return x > 20.0f ? x : log1pf(expf(x)); }  // torch threshold 20

static void relu2(float *x, int n) {  // relu(x)^2 in place
    const __m256 z = _mm256_setzero_ps();
    int i = 0;
    for (; i + 8 <= n; i += 8) {
        const __m256 v = _mm256_max_ps(_mm256_loadu_ps(x + i), z);
        _mm256_storeu_ps(x + i, _mm256_mul_ps(v, v));
    }
    for (; i < n; i++) x[i] = x[i] > 0 ? x[i] * x[i] : 0;
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
    const int nl = e->n_moe + (e->mtp.present ? 1 : 0);
    uint32_t count = 0;
    for (int i = 0; i < e->n_imx; i++) count += e->imx[i].n > 0;
    for (int k = 0; k < nl * Q_N_EXPERT; k++) count += 2 * (e->imx_exp_n[k] > 0);
    fwrite("INZIMX01", 1, 8, f);
    fwrite(&count, 4, 1, f);
    float *buf = inz_malloc(sizeof(float) * 2 * Q_H);
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
        const double *acc = e->imx_exp + (size_t)k * (Q_LAT + Q_FF);
        for (int part = 0; part < 2; part++) {
            char name[48];
            if (l < e->n_moe) snprintf(name, sizeof(name), "moe.%d.experts.%d.%s", l, xid, part ? "down" : "up");
            else snprintf(name, sizeof(name), "mtp.experts.%d.%s", xid, part ? "down" : "up");
            const uint16_t len = (uint16_t)strlen(name);
            const uint32_t cols = part ? Q_FF : Q_LAT;
            for (uint32_t c = 0; c < cols; c++) buf[c] = (float)(acc[(part ? Q_LAT : 0) + c] / (double)e->imx_exp_n[k]);
            fwrite(&len, 2, 1, f); fwrite(name, 1, len, f); fwrite(&cols, 4, 1, f); fwrite(&e->imx_exp_n[k], 8, 1, f);
            fwrite(buf, 4, cols, f);
        }
    }
    free(buf);
    fclose(f);
    LOGI("imatrix: %u entries written to %s", count, e->imx_path);
}

// Y[t*ldy + r] = W[r] . X[t*ldx]  for t < n, r < W->rows. `prev`: the matrix run just before on the same input,
// whose quantized activation can be reused.
static void lin_mm_after(inz_engine *e, const lin *W, const lin *prev, const float *X, int ldx, int n, float *Y, int ldy) {
    if (W->imx) {
        imx_entry *x = &e->imx[W->imx - 1];
        imx_add(x->acc, X, ldx, n, W->cols);
        x->n += (uint64_t)n;
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
    const int grain = W->type != INZ_T_F32 && n > 1 && float_act(W->type) ? (big_k ? 12 : 24) : big_k ? 16 : 32;
    tp_for(e->tp, W->rows, grain, mm_task, &j);
}

static void lin_mm(inz_engine *e, const lin *W, const float *X, int ldx, int n, float *Y, int ldy) {
    lin_mm_after(e, W, NULL, X, ldx, n, Y, ldy);
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
/* Mamba-2                                                             */
/* ------------------------------------------------------------------ */

#define MB_CCHUNK 512
typedef struct {
    inz_engine *e;
    const q_layer *L;
    int n;
    const float *P;      // [n][Q_MB_PROJ] in_proj output: z | x B C | dt
    float *cv;           // [n][Q_MB_CONV] conv + silu: x | B | C
    float *dt;           // [n][NH] step sizes
    float *yo;           // [n][INNER] scan output, then gated-normalized in place
} mb_job;

// causal depthwise conv (kernel 4, bias) over x | B | C and silu; step sizes dt = softplus(dt_raw + dt_bias). Tokens
// are independent here: the conv reads raw inputs, from the chunk or from the saved state.
static void mb_conv_task(void *ctx, int64_t b, int64_t e_, int tid) {
    (void)tid;
    mb_job *j = ctx;
    const q_layer *L = j->L;
    const float *cs = j->e->st.conv + (size_t)L->idx * (Q_MB_KS - 1) * Q_MB_CONV;  // last 3 inputs, oldest first
    const int nch = Q_MB_CONV / MB_CCHUNK;
    for (int64_t item = b; item < e_; item++) {
        const int t = (int)(item / nch), c0 = (int)(item % nch) * MB_CCHUNK;
        const float *tap[4];
        for (int k = 0; k < 3; k++) {
            const int d = 3 - k;
            tap[k] = t >= d ? j->P + (size_t)(t - d) * Q_MB_PROJ + Q_MB_INNER : cs + (size_t)(3 - d + t) * Q_MB_CONV;
        }
        tap[3] = j->P + (size_t)t * Q_MB_PROJ + Q_MB_INNER;
        float *cv = j->cv + (size_t)t * Q_MB_CONV;
        for (int c = c0; c < c0 + MB_CCHUNK; c++) {
            const float *w = L->mb_conv + (size_t)c * Q_MB_KS;
            cv[c] = siluf(L->mb_conv_b[c] + w[0] * tap[0][c] + w[1] * tap[1][c] + w[2] * tap[2][c] + w[3] * tap[3][c]);
        }
        if (c0 == 0) {
            const float *dr = j->P + (size_t)t * Q_MB_PROJ + Q_MB_INNER + Q_MB_CONV;
            for (int h = 0; h < Q_MB_NH; h++) j->dt[t * Q_MB_NH + h] = softplusf(dr[h] + L->mb_dt[h]);
        }
    }
}

// one head through all n tokens: S = S * exp(dt A) + dt x B^T; y = S C + D x. B and C are shared by the 16 heads of
// a group. The head's 32 KB state stays in L1/L2 for the whole chunk.
static void mb_scan_task(void *ctx, int64_t b, int64_t e_, int tid) {
    (void)tid;
    mb_job *j = ctx;
    inz_engine *e = j->e;
    const q_layer *L = j->L;
    const size_t hs = (size_t)Q_MB_HD * Q_MB_DS;
    for (int64_t h = b; h < e_; h++) {
        float *S = e->st.S + ((size_t)L->idx * Q_MB_NH + h) * hs;
        const int g = (int)h / (Q_MB_NH / Q_MB_NG);
        for (int t = 0; t < j->n; t++) {
            const float *cv = j->cv + (size_t)t * Q_MB_CONV;
            const float *x = cv + h * Q_MB_HD, *B = cv + Q_MB_INNER + g * Q_MB_DS, *C = cv + Q_MB_INNER + Q_MB_NG * Q_MB_DS + g * Q_MB_DS;
            const float dt = j->dt[t * Q_MB_NH + h];
            const __m256 dA = _mm256_set1_ps(expf(dt * L->mb_negA[h]));
            float *y = j->yo + (size_t)t * Q_MB_INNER + h * Q_MB_HD;
            for (int p = 0; p < Q_MB_HD; p++) {
                float *Sp = S + (size_t)p * Q_MB_DS;
                const __m256 dx = _mm256_set1_ps(dt * x[p]);
                __m256 acc = _mm256_setzero_ps();
                for (int s = 0; s < Q_MB_DS; s += 8) {
                    const __m256 sv = _mm256_fmadd_ps(dx, _mm256_loadu_ps(B + s), _mm256_mul_ps(_mm256_loadu_ps(Sp + s), dA));
                    _mm256_storeu_ps(Sp + s, sv);
                    acc = _mm256_fmadd_ps(sv, _mm256_loadu_ps(C + s), acc);
                }
                y[p] = hsum256_ps(acc) + L->mb_D[h] * x[p];
            }
            if (t < e->snap_n)                  // speculative verify: this head's state after draft position t
                memcpy(e->snap[t].S + ((size_t)L->idx * Q_MB_NH + h) * hs, S, sizeof(float) * hs);
        }
    }
}

// gated RMSNorm per group of 1024: y = rmsnorm(y * silu(z)) * w, in place
static void mb_norm_task(void *ctx, int64_t b, int64_t e_, int tid) {
    (void)tid;
    mb_job *j = ctx;
    for (int64_t item = b; item < e_; item++) {
        const int t = (int)(item / Q_MB_NG), g = (int)(item % Q_MB_NG);
        float *y = j->yo + (size_t)t * Q_MB_INNER + g * Q_MB_NORM_GROUP;
        const float *z = j->P + (size_t)t * Q_MB_PROJ + g * Q_MB_NORM_GROUP;
        for (int i = 0; i < Q_MB_NORM_GROUP; i++) y[i] *= siluf(z[i]);
        rmsnorm(y, y, j->L->mb_norm + g * Q_MB_NORM_GROUP, Q_MB_NORM_GROUP);
    }
}

static void mamba_block(inz_engine *e, const q_layer *L, int n, const float *xin, float *y) {
    float *P = e->big, *cv = e->tmp, *yo = e->tmp2;  // [n][18560], [n][10240], [n][8192]
    lin_mm(e, &L->mb_in, xin, Q_H, n, P, Q_MB_PROJ);
    mb_job j = {e, L, n, P, cv, e->mb_dt, yo};
    tp_for(e->tp, (int64_t)n * (Q_MB_CONV / MB_CCHUNK), 1, mb_conv_task, &j);
    tp_for(e->tp, Q_MB_NH, 1, mb_scan_task, &j);
    tp_for(e->tp, (int64_t)n * Q_MB_NG, 2, mb_norm_task, &j);
    // conv state: the last 3 raw inputs (part of the old state when the chunk is short); speculative snapshots
    // take the state after each draft position
    float *cs = e->st.conv + (size_t)L->idx * (Q_MB_KS - 1) * Q_MB_CONV;
    const size_t cz = (size_t)(Q_MB_KS - 1) * Q_MB_CONV;
    for (int s = 0; s < e->snap_n && s < n; s++)
        for (int i = 0; i < 3; i++) {
            const int src = s + 1 - 3 + i;
            memcpy(e->snap[s].conv + (size_t)L->idx * cz + (size_t)i * Q_MB_CONV,
                   src >= 0 ? P + (size_t)src * Q_MB_PROJ + Q_MB_INNER : cs + (size_t)(i + s + 1) * Q_MB_CONV, sizeof(float) * Q_MB_CONV);
        }
    for (int i = 0; i < 3; i++) {
        const int src = n - 3 + i;
        if (src >= 0) memcpy(cs + (size_t)i * Q_MB_CONV, P + (size_t)src * Q_MB_PROJ + Q_MB_INNER, sizeof(float) * Q_MB_CONV);
        else memmove(cs + (size_t)i * Q_MB_CONV, cs + (size_t)(i + n) * Q_MB_CONV, sizeof(float) * Q_MB_CONV);
    }
    lin_mm(e, &L->mb_out, yo, Q_MB_INNER, n, y, Q_H);
}

/* ------------------------------------------------------------------ */
/* Attention                                                           */
/* ------------------------------------------------------------------ */

typedef struct { inz_engine *e; int ai, n, p0; const float *q; float *out; } att_job;

// one (token, head) pair: softmax(q k^T / sqrt(128)) v over positions 0..p (causal, no positional encoding)
static void att_head_task(void *ctx, int64_t b, int64_t e_, int tid) {
    att_job *j = ctx;
    inz_engine *e = j->e;
    const uint16_t *K = e->st.k + (size_t)j->ai * e->ctx * Q_KV, *V = e->st.v + (size_t)j->ai * e->ctx * Q_KV;
    float *sc = e->att_scr[tid];
    const float scale = 1.0f / sqrtf((float)Q_HD);
    for (int64_t th = b; th < e_; th++) {
        const int t = (int)(th / Q_NH), h = (int)(th % Q_NH), g = h / (Q_NH / Q_NKV), np = j->p0 + t + 1;
        const float *qh = j->q + (size_t)t * Q_QD + h * Q_HD;
        for (int i = 0; i < np; i++) sc[i] = dot_f16(K + (size_t)i * Q_KV + g * Q_HD, qh, Q_HD) * scale;
        softmax_inplace(sc, np);
        float o[Q_HD] = {0};
        for (int i = 0; i < np; i++) axpy_f16(o, sc[i], V + (size_t)i * Q_KV + g * Q_HD, Q_HD);
        memcpy(j->out + (size_t)t * Q_QD + h * Q_HD, o, sizeof(o));
    }
}

// ai: attention cache index (0..7 for the model's attention blocks, Q_N_ATTN for the MTP layer)
static void attn_block(inz_engine *e, const q_layer *L, int ai, int n, int p0, const float *xin, float *y) {
    float *q = e->big, *att = e->big + (size_t)Q_MAX_T * Q_QD;  // [n][4096] each
    float *k = e->tmp, *v = e->tmp2;                             // [n][256]
    lin_mm(e, &L->at_q, xin, Q_H, n, q, Q_QD);
    lin_mm(e, &L->at_k, xin, Q_H, n, k, Q_KV);
    lin_mm(e, &L->at_v, xin, Q_H, n, v, Q_KV);
    uint16_t *K = e->st.k + (size_t)ai * e->ctx * Q_KV, *V = e->st.v + (size_t)ai * e->ctx * Q_KV;
    for (int t = 0; t < n; t++) {
        f32_to_f16(K + (size_t)(p0 + t) * Q_KV, k + (size_t)t * Q_KV, Q_KV);
        f32_to_f16(V + (size_t)(p0 + t) * Q_KV, v + (size_t)t * Q_KV, Q_KV);
    }
    att_job j = {e, ai, n, p0, q, att};
    tp_for(e->tp, (int64_t)n * Q_NH, 1, att_head_task, &j);
    lin_mm(e, &L->at_o, att, Q_QD, n, y, Q_H);
}

/* ------------------------------------------------------------------ */
/* LatentMoE                                                           */
/* ------------------------------------------------------------------ */

// tokens routed to one expert; slot = t * Q_TOPK + rank, where the expert's weighted latent output for that token goes
typedef struct { int expert, ntok; int tok[Q_MAX_T]; int slot[Q_MAX_T]; float w[Q_MAX_T]; } unit;

// out[slot] = w * down(relu(up x)^2) for the tokens routed to one expert, in the latent space. The slots are summed
// in rank order afterwards, so the result does not depend on which experts were resident or in which order reads
// completed.
static void expert_compute(inz_engine *e, const uint8_t *blob, const unit *u, const float *xlat, float *out) {
    lin up, dn;
    if (!q_bind_lin(&up, e->expert_type, Q_FF, Q_LAT, blob) || !q_bind_lin(&dn, e->expert_type, Q_LAT, Q_FF, blob + e->up_bytes))
        inz_die("expert blob is corrupt (expert %d)", u->expert);
    float *X = e->ex_x, *U = e->ex_up, *D = e->ex_d;
    for (int i = 0; i < u->ntok; i++) memcpy(X + (size_t)i * Q_LAT, xlat + (size_t)u->tok[i] * Q_LAT, sizeof(float) * Q_LAT);
    lin_mm(e, &up, X, Q_LAT, u->ntok, U, Q_FF);
    relu2(U, u->ntok * Q_FF);
    if (e->imx_exp) {                           // importance matrix: this expert's up and down inputs
        const size_t k = (size_t)e->imx_layer * Q_N_EXPERT + (size_t)u->expert;
        double *acc = e->imx_exp + k * (Q_LAT + Q_FF);
        imx_add(acc, X, Q_LAT, u->ntok, Q_LAT);
        imx_add(acc + Q_LAT, U, Q_FF, u->ntok, Q_FF);
        e->imx_exp_n[k] += (uint64_t)u->ntok;
    }
    lin_mm(e, &dn, U, Q_FF, u->ntok, D, Q_LAT);
    for (int i = 0; i < u->ntok; i++) {
        float *o = out + (size_t)u->slot[i] * Q_LAT;
        const float w = u->w[i], *dt = D + (size_t)i * Q_LAT;
        for (int c = 0; c < Q_LAT; c++) o[c] = w * dt[c];
    }
}

/* Decode / verify (n <= 8 tokens): the resident experts of a block are computed together, in three dispatches over
 * (expert, 16-row chunk) items, instead of two small dispatches per expert. Per row the arithmetic is the same as
 * expert_compute's, so the results are identical. */
#define EB_TOK 8
#define EB_MAX (Q_TOPK * EB_TOK)                 // experts x tokens of one batch
typedef struct {
    const unit *u;
    lin up, dn;
    inz_act ua[EB_TOK], da[EB_TOK];              // per token: the quantized input of up, of down
    float *U;                                    // [ntok][FF]
    float *D;                                    // [ntok][LAT]
} ebatch_item;

typedef struct { inz_engine *e; ebatch_item *it; int ni; } ebatch_job;

#define EB_CHUNK 16
static void eb_up_task(void *ctx, int64_t b, int64_t e_, int tid) {
    (void)tid;
    ebatch_job *j = ctx;
    const int per = Q_FF / EB_CHUNK;
    for (int64_t k = b; k < e_; k++) {
        ebatch_item *it = &j->it[k / per];
        const int r0 = (int)(k % per) * EB_CHUNK, r1 = r0 + EB_CHUNK;
        if (it->u->ntok == 1) inz_matvec_rows(&it->up.q, it->ua, it->U, r0, r1);
        else inz_matmul_rows(&it->up.q, it->ua, it->u->ntok, it->U, Q_FF, r0, r1);
    }
}

static void eb_act_task(void *ctx, int64_t b, int64_t e_, int tid) {
    (void)tid;
    ebatch_job *j = ctx;
    for (int64_t k = b; k < e_; k++) {
        ebatch_item *it = &j->it[k];
        for (int i = 0; i < it->u->ntok; i++) {
            float *r = it->U + (size_t)i * Q_FF;
            relu2(r, Q_FF);
            inz_act_quantize(&it->da[i], r, Q_FF, it->dn.q.rot_block, it->dn.q.rot_seed);
        }
    }
}

static void eb_down_task(void *ctx, int64_t b, int64_t e_, int tid) {
    (void)tid;
    ebatch_job *j = ctx;
    const int per = Q_LAT / EB_CHUNK;
    for (int64_t k = b; k < e_; k++) {
        ebatch_item *it = &j->it[k / per];
        const int r0 = (int)(k % per) * EB_CHUNK, r1 = r0 + EB_CHUNK;
        const unit *u = it->u;
        if (u->ntok == 1) inz_matvec_rows(&it->dn.q, it->da, it->D, r0, r1);
        else inz_matmul_rows(&it->dn.q, it->da, u->ntok, it->D, Q_LAT, r0, r1);
        for (int i = 0; i < u->ntok; i++) {
            float *o = j->e->moe_lat + (size_t)u->slot[i] * Q_LAT;
            const float w = u->w[i], *dt = it->D + (size_t)i * Q_LAT;
            for (int r = r0; r < r1; r++) o[r] = w * dt[r];
        }
    }
}

// compute the given units (blobs pinned) as one batch; quantized expert types only. The tokens' latent inputs are
// quantized once for the up rotation, which all experts of a block share; an expert with another rotation (not
// produced by the converter) is computed on its own.
static void experts_batch(inz_engine *e, const unit *const *us, const uint8_t *const *blobs, int nb, int n, const float *xlat) {
    ebatch_item *it = e->eb_items;
    int ni = 0, rows = 0, qrot = -1;
    uint32_t qseed = 0;
    for (int i = 0; i < nb; i++) {
        ebatch_item *x = &it[ni];
        if (!q_bind_lin(&x->up, e->expert_type, Q_FF, Q_LAT, blobs[i]) ||
            !q_bind_lin(&x->dn, e->expert_type, Q_LAT, Q_FF, blobs[i] + e->up_bytes))
            inz_die("expert blob is corrupt (expert %d)", us[i]->expert);
        if (qrot < 0) {
            qrot = x->up.q.rot_block;
            qseed = x->up.q.rot_seed;
            for (int t = 0; t < n; t++) inz_act_quantize(&e->eb_xa[t], xlat + (size_t)t * Q_LAT, Q_LAT, qrot, qseed);
        } else if (x->up.q.rot_block != qrot || x->up.q.rot_seed != qseed) {
            expert_compute(e, blobs[i], us[i], xlat, e->moe_lat);
            continue;
        }
        x->u = us[i];
        x->U = e->eb_up + (size_t)rows * Q_FF;
        x->D = e->eb_d + (size_t)rows * Q_LAT;
        for (int t = 0; t < us[i]->ntok; t++) {
            x->ua[t] = e->eb_xa[us[i]->tok[t]];
            x->da[t] = e->eb_acts[rows + t];
        }
        rows += us[i]->ntok;
        ni++;
    }
    if (!ni) return;
    ebatch_job j = {e, it, ni};
    tp_for(e->tp, (int64_t)ni * (Q_FF / EB_CHUNK), 4, eb_up_task, &j);
    tp_for(e->tp, ni, 1, eb_act_task, &j);
    tp_for(e->tp, (int64_t)ni * (Q_LAT / EB_CHUNK), 8, eb_down_task, &j);
}

// the next MoE block after block l (lookahead), or NULL
static const q_layer *next_moe(const inz_engine *e, int l) {
    for (int k = l + 1; k < e->n_layer; k++)
        if (e->L[k].kind == QB_MOE) return &e->L[k];
    return NULL;
}

// L->idx: expert cache layer (0..39, n_moe for the MTP layer); dl: block index for diagnostics
static void moe_block(inz_engine *e, const q_layer *L, int dl, int n, const float *xin, float *y) {
    const int l = L->idx;
    const uint64_t io_start = ec_read_bytes(e->ec);
    e->imx_layer = l;
    float *logits = e->tmp;                     // [n][512]
    uint64_t t0 = diag_tsc();
    lin_mm(e, &L->router, xin, Q_H, n, logits, Q_N_EXPERT);
    unit *units = e->units;
    int uid[Q_N_EXPERT];
    for (int x = 0; x < Q_N_EXPERT; x++) uid[x] = -1;
    int nu = 0;
    for (int t = 0; t < n; t++) {
        // sigmoid scores; the correction bias only shifts the choice; weights are the chosen scores, normalized and
        // scaled by 5
        const float *lg = logits + (size_t)t * Q_N_EXPERT;
        float sc[Q_N_EXPERT], ch[Q_N_EXPERT];
        for (int x = 0; x < Q_N_EXPERT; x++) { sc[x] = sigmoidf(lg[x]); ch[x] = sc[x] + L->router_bias[x]; }
        int32_t sel[Q_TOPK];
        float w[Q_TOPK], sum = 0;
        topk_f32(ch, Q_N_EXPERT, Q_TOPK, sel);
        for (int k = 0; k < Q_TOPK; k++) {
            w[k] = sc[sel[k]];
            sum += w[k];
        }
        const float norm = Q_ROUTED_SCALE / (sum + 1e-20f);
        for (int k = 0; k < Q_TOPK; k++) w[k] *= norm;
        diag_routing(l, sel, w, Q_TOPK, e->n_past + t);
        if (n == 1 && e->pred_valid[l]) {         // how well the previous block's lookahead predicted this routing
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
    diag_add(DG_ROUTER, dl, t0, q_lin_bytes(&L->router), 2ull * n * Q_N_EXPERT * Q_H);
    // routed experts: read requests for every missing one first, then the next block's predicted experts
    // (decode), then the latent projection and the resident experts while the SSD works, then the rest as they
    // arrive
    bool done[Q_N_EXPERT] = {0}, asked[Q_N_EXPERT] = {0};
    int left = nu, lvl;
    if (n <= EB_TOK)
        for (int i = 0; i < nu; i++)
            if (!ec_is_resident(e->ec, l, units[i].expert)) { ec_request(e->ec, l, units[i].expert, 1.0f, EC_FULL); asked[i] = true; }
    const q_layer *Ln = next_moe(e, dl);
    if (n <= EB_TOK && e->opt.lookahead == 2 && Ln && !e->ec_all_resident) {
        // early lookahead (--lookahead 2): this block's MoE input through the next MoE block's router predicts its
        // experts; earlier but less precise than the late one (run_block)
        const uint64_t tl = diag_tsc();
        float *lg2 = e->tmp + (size_t)n * Q_N_EXPERT;
        lin_mm(e, &Ln->router, xin, Q_H, n, lg2, Q_N_EXPERT);
        for (int t = 0; t < n; t++) {
            float ch[Q_N_EXPERT];
            const float *lg = lg2 + (size_t)t * Q_N_EXPERT;
            for (int x = 0; x < Q_N_EXPERT; x++) ch[x] = sigmoidf(lg[x]) + Ln->router_bias[x];
            int32_t ps[Q_TOPK];
            topk_f32(ch, Q_N_EXPERT, Q_TOPK, ps);
            for (int k = 0; k < Q_TOPK; k++) ec_prefetch(e->ec, Ln->idx, ps[k], 0.5f, EC_FULL);
            if (n == 1) { memcpy(e->pred[Ln->idx], ps, sizeof(ps)); e->pred_valid[Ln->idx] = true; g_diag.pred_total += Q_TOPK; }
        }
        diag_add(DG_PREFETCH, dl, tl, q_lin_bytes(&Ln->router), 0);
    }
    // latent input, and the shared expert, while the requested experts are arriving
    t0 = diag_tsc();
    float *xlat = e->xlat;
    lin_mm(e, &L->lat_in, xin, Q_H, n, xlat, Q_LAT);
    float *U = e->big, *D = e->tmp2;            // [n][5376], [n][H]
    lin_mm(e, &L->sh_up, xin, Q_H, n, U, Q_SHARED_FF);
    relu2(U, n * Q_SHARED_FF);
    lin_mm(e, &L->sh_down, U, Q_SHARED_FF, n, D, Q_H);
    diag_add(DG_SHARED, dl, t0, q_lin_bytes(&L->lat_in) + q_lin_bytes(&L->sh_up) + q_lin_bytes(&L->sh_down), 0);
    t0 = diag_tsc();
    const bool batch = n <= EB_TOK && !e->imx_exp && !float_act(e->expert_type);
    const unit *bu[EB_MAX];
    const uint8_t *bb[EB_MAX];
    int nb = 0;
    for (int i = 0; i < nu; i++) {
        const uint8_t *b = ec_try_acquire(e->ec, l, units[i].expert, 1.0f, EC_FULL, &lvl);
        if (!b) { if (!asked[i]) ec_request(e->ec, l, units[i].expert, 1.0f, EC_FULL); continue; }
        if (batch) { bu[nb] = &units[i]; bb[nb++] = b; }
        else {
            expert_compute(e, b, &units[i], xlat, e->moe_lat);
            ec_release(e->ec, l, units[i].expert);
        }
        done[i] = true;
        left--;
    }
    if (nb) {
        experts_batch(e, bu, bb, nb, n, xlat);
        for (int i = 0; i < nb; i++) ec_release(e->ec, l, bu[i]->expert);
    }
    diag_add(DG_EXPERT_HIT, dl, t0, (uint64_t)(nu - left) * e->blob_bytes, 0);
    t0 = diag_tsc();
    while (left > 0) {
        bool progress = false;
        nb = 0;
        for (int i = 0; i < nu; i++) {
            if (done[i]) continue;
            const uint8_t *b = ec_try_acquire(e->ec, l, units[i].expert, 0, EC_FULL, &lvl);
            if (!b) continue;
            if (batch) { bu[nb] = &units[i]; bb[nb++] = b; }
            else {
                expert_compute(e, b, &units[i], xlat, e->moe_lat);
                ec_release(e->ec, l, units[i].expert);
            }
            done[i] = true;
            left--;
            progress = true;
        }
        if (nb) {
            experts_batch(e, bu, bb, nb, n, xlat);
            for (int i = 0; i < nb; i++) ec_release(e->ec, l, bu[i]->expert);
        }
        if (progress || left == 0) continue;
        diag_add(DG_EXPERT_MISS, dl, t0, 0, 0);
        t0 = diag_tsc();
        if (!ec_wait_any(e->ec)) {              // nothing in flight: fall back to a blocking read
            for (int i = 0; i < nu; i++)
                if (!done[i]) {
                    const uint8_t *b = ec_acquire(e->ec, l, units[i].expert, 0, EC_FULL, &lvl);
                    diag_add(DG_EXPERT_WAIT, dl, t0, 0, 0);
                    t0 = diag_tsc();
                    expert_compute(e, b, &units[i], xlat, e->moe_lat);
                    ec_release(e->ec, l, units[i].expert);
                    done[i] = true;
                    left--;
                    diag_add(DG_EXPERT_MISS, dl, t0, e->blob_bytes, 0);
                    t0 = diag_tsc();
                    break;
                }
        }
        diag_add(DG_EXPERT_WAIT, dl, t0, 0, 0);
        t0 = diag_tsc();
    }
    diag_add(DG_EXPERT_MISS, dl, t0, 0, 0);
    // routed outputs in rank order (deterministic), back to the hidden size, plus the shared expert
    t0 = diag_tsc();
    float *ls = e->tmp;                          // [n][LAT] (the router logits are no longer needed)
    for (int t = 0; t < n; t++) {
        float *st = ls + (size_t)t * Q_LAT;
        memcpy(st, e->moe_lat + (size_t)t * Q_TOPK * Q_LAT, sizeof(float) * Q_LAT);
        for (int k = 1; k < Q_TOPK; k++) {
            const float *o = e->moe_lat + ((size_t)t * Q_TOPK + k) * Q_LAT;
            for (int c = 0; c < Q_LAT; c++) st[c] += o[c];
        }
    }
    lin_mm(e, &L->lat_out, ls, Q_LAT, n, y, Q_H);
    add_rows(y, D, n);
    diag_add(DG_SHARED, dl, t0, q_lin_bytes(&L->lat_out), 0);
    diag_add_ssd(DG_PREFETCH, dl, ec_read_bytes(e->ec) - io_start);
}

/* ------------------------------------------------------------------ */
/* Scratch, state, forward                                             */
/* ------------------------------------------------------------------ */

#define ACT_MAX Q_MB_INNER                      // widest input of a matrix with int8 activations

void q_scratch_alloc(inz_engine *e) {
    const size_t T = Q_MAX_T;
    e->X = inz_aligned(64, sizeof(float) * T * Q_H);
    e->X2 = inz_aligned(64, sizeof(float) * T * Q_H);
    e->Xn = inz_aligned(64, sizeof(float) * T * Q_H);
    e->y = inz_aligned(64, sizeof(float) * T * Q_H);
    e->tmp = inz_aligned(64, sizeof(float) * T * Q_MB_CONV);
    e->tmp2 = inz_aligned(64, sizeof(float) * T * Q_MB_INNER);
    e->big = inz_aligned(64, sizeof(float) * T * Q_MB_PROJ);
    e->mb_dt = inz_aligned(64, sizeof(float) * T * Q_MB_NH);
    e->xlat = inz_aligned(64, sizeof(float) * T * Q_LAT);
    e->moe_lat = inz_aligned(64, sizeof(float) * T * Q_TOPK * Q_LAT);
    e->ex_x = inz_aligned(64, sizeof(float) * T * Q_LAT);
    e->ex_up = inz_aligned(64, sizeof(float) * T * Q_FF);
    e->ex_d = inz_aligned(64, sizeof(float) * T * Q_LAT);
    e->eb_items = inz_calloc(EB_MAX, sizeof(ebatch_item));
    e->eb_up = inz_aligned(64, sizeof(float) * EB_MAX * Q_FF);
    e->eb_d = inz_aligned(64, sizeof(float) * EB_MAX * Q_LAT);
    e->eb_acts = inz_calloc(EB_MAX, sizeof(inz_act));
    for (int i = 0; i < EB_MAX; i++) inz_act_alloc(&e->eb_acts[i], Q_FF);
    e->eb_xa = inz_calloc(EB_TOK, sizeof(inz_act));
    for (int i = 0; i < EB_TOK; i++) inz_act_alloc(&e->eb_xa[i], Q_LAT);
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
    free(e->X); free(e->X2); free(e->Xn); free(e->y); free(e->tmp); free(e->tmp2); free(e->big); free(e->mb_dt);
    free(e->xlat); free(e->moe_lat); free(e->ex_x); free(e->ex_up); free(e->ex_d); free(e->units);
    for (int i = 0; i < EB_MAX; i++) inz_act_free(&e->eb_acts[i]);
    for (int i = 0; i < EB_TOK; i++) inz_act_free(&e->eb_xa[i]);
    free(e->eb_items); free(e->eb_up); free(e->eb_d); free(e->eb_acts); free(e->eb_xa);
    for (int i = 0; i < Q_MAX_T; i++) inz_act_free(&e->acts[i]);
    free(e->acts);
    const int nthr = tp_size(e->tp);
    for (int i = 0; i < nthr; i++) { free(e->att_scr[i]); free(e->gemm_tb[i]); free(e->gemm_tmp[i]); }
    free(e->att_scr); free(e->gemm_tb); free(e->gemm_tmp);
    inz_xq_free(&e->gemm_x);
}

void q_state_reset(inz_engine *e) {
    memset(e->st.S, 0, sizeof(float) * (size_t)Q_N_MAMBA * Q_MB_NH * Q_MB_HD * Q_MB_DS);
    memset(e->st.conv, 0, sizeof(float) * (size_t)Q_N_MAMBA * (Q_MB_KS - 1) * Q_MB_CONV);
    e->n_past = 0;
    e->mtp_valid = 0;
}

// Router lookahead (--lookahead 1, the default): once a MoE block has updated the residual, the next MoE block's own
// norm and router on it predict that block's experts (only a Mamba or attention block lies in between), and they are
// prefetched while that block runs. On the first blocks this predicts 69% of the experts against 53% for the early
// lookahead from the MoE input, and the shorter lead time costs less than the extra precision gains.
static void moe_lookahead_late(inz_engine *e, int dl, int n, const float *X) {
    const q_layer *Ln = next_moe(e, dl);
    if (!Ln || n > EB_TOK || !e->ec || e->ec_all_resident) return;
    const uint64_t t0 = diag_tsc();
    norm_rows(e, X, Q_H, n, e->Xn, Q_H, Ln->norm);
    float *lg = e->tmp;
    lin_mm(e, &Ln->router, e->Xn, Q_H, n, lg, Q_N_EXPERT);
    for (int t = 0; t < n; t++) {
        float ch[Q_N_EXPERT];
        for (int x = 0; x < Q_N_EXPERT; x++) ch[x] = sigmoidf(lg[(size_t)t * Q_N_EXPERT + x]) + Ln->router_bias[x];
        int32_t ps[Q_TOPK];
        topk_f32(ch, Q_N_EXPERT, Q_TOPK, ps);
        for (int k = 0; k < Q_TOPK; k++) ec_prefetch(e->ec, Ln->idx, ps[k], 0.75f, EC_FULL);
        if (n == 1) { memcpy(e->pred[Ln->idx], ps, sizeof(ps)); e->pred_valid[Ln->idx] = true; g_diag.pred_total += Q_TOPK; }
    }
    diag_add(DG_PREFETCH, dl, t0, q_lin_bytes(&Ln->router), 0);
}

// one block on rows X [n][H]: X += mixer(rmsnorm(X) * norm); dl: block index for diagnostics
static void run_block(inz_engine *e, const q_layer *L, int dl, int ai, int n, int p0, float *X) {
    uint64_t t0 = diag_tsc();
    norm_rows(e, X, Q_H, n, e->Xn, Q_H, L->norm);
    diag_add(L->kind == QB_MOE ? DG_FFN_NORM : DG_ATTN_NORM, dl, t0, 0, 0);
    t0 = diag_tsc();
    switch (L->kind) {
    case QB_MAMBA:
        mamba_block(e, L, n, e->Xn, e->y);
        diag_add(DG_ATTN_CORE, dl, t0, q_lin_bytes(&L->mb_in) + q_lin_bytes(&L->mb_out), 0);
        break;
    case QB_ATTN:
        attn_block(e, L, ai, n, p0, e->Xn, e->y);
        diag_add(DG_ATTN_QKV, dl, t0, q_lin_bytes(&L->at_q) + q_lin_bytes(&L->at_k) + q_lin_bytes(&L->at_v) + q_lin_bytes(&L->at_o), 0);
        break;
    default:
        moe_block(e, L, dl, n, e->Xn, e->y);
        break;
    }
    add_rows(X, e->y, n);
    if (L->kind == QB_MOE && e->opt.lookahead == 1) moe_lookahead_late(e, dl, n, X);
}

void q_forward(inz_engine *e, const int32_t *tokens, int n, const int *logits_rows, int n_logits, float *out_logits) {
    if (n > Q_MAX_T) inz_die("q_forward: %d tokens > %d", n, Q_MAX_T);
    if (e->n_past + n > e->ctx) inz_die("q_forward: context full (%d + %d > %d)", e->n_past, n, e->ctx);
    const int p0 = e->n_past;
    memcpy(e->hist + p0, tokens, sizeof(int32_t) * n);
    uint64_t t0 = diag_tsc();
    for (int t = 0; t < n; t++) embed_row(e, tokens[t], e->X + (size_t)t * Q_H);
    diag_add(DG_EMBED, -1, t0, (uint64_t)n * Q_H * 2, 0);
    static FILE *dump;  // validation: INZ_DUMP_HIDDEN=file appends the residual after every block
    static int dump_init;
    if (!dump_init) {
        dump_init = 1;
        const char *pth = getenv("INZ_DUMP_HIDDEN");
        if (pth && *pth) dump = fopen(pth, "wb");
    }
    for (int l = 0; l < e->n_layer; l++) {
        const q_layer *L = &e->L[l];
        run_block(e, L, l, L->idx, n, p0, e->X);
        if (dump) { fwrite(e->X, sizeof(float), (size_t)n * Q_H, dump); fflush(dump); }
    }
    e->n_past += n;
    memcpy(e->x_last, e->X + (size_t)(n - 1) * Q_H, sizeof(float) * Q_H);
    if (e->ec) ec_tick(e->ec, n);
    if (n_logits <= 0) return;
    float *rows = e->X2;                        // final norm of the requested rows (X2 is the MTP's, free here)
    for (int i = 0; i < n_logits; i++) rmsnorm(rows + (size_t)i * Q_H, e->X + (size_t)logits_rows[i] * Q_H, e->norm_f, Q_H);
    if (n_logits > 64) inz_die("q_forward: %d logits rows (max 64)", n_logits);
    int seq[64];
    for (int i = 0; i < n_logits; i++) seq[i] = i;
    q_head(e, rows, n_logits, seq, n_logits, out_logits);
}

void q_head(inz_engine *e, const float *Hn, int n, const int *logits_rows, int n_logits, float *out_logits) {
    (void)n;
    uint64_t t0 = diag_tsc();
    float *rows = e->tmp;                       // [n_logits][H]
    for (int i = 0; i < n_logits; i++) memcpy(rows + (size_t)i * Q_H, Hn + (size_t)logits_rows[i] * Q_H, sizeof(float) * Q_H);
    lin_mm(e, &e->head, rows, Q_H, n_logits, out_logits, Q_VOCAB);
    diag_add(DG_LM_HEAD, -1, t0, q_lin_bytes(&e->head), 2ull * n_logits * Q_VOCAB * Q_H);
}

/* ------------------------------------------------------------------ */
/* MTP: one attention + MoE step that drafts token p+2 from (h_p, p+1) */
/* ------------------------------------------------------------------ */

void q_mtp_forward(inz_engine *e, const float *H, int ldh, bool residual, const int32_t *next, int n, int p0,
                   const int *logits_rows, int n_logits, float *out_logits) {
    if (n > Q_MAX_T) inz_die("q_mtp_forward: %d tokens > %d", n, Q_MAX_T);
    const diag_phase phase = g_diag.phase;
    diag_set_phase(PH_DRAFT);
    g_diag.phase_tokens[PH_DRAFT] += (uint64_t)n;
    const q_mtp *M = &e->mtp;
    const int dl = e->n_layer;                  // the MTP blocks in diagnostics: n_layer, n_layer + 1
    uint64_t t0 = diag_tsc();
    // F = [enorm(embed(next)) | hnorm(h)], with h = norm_f(residual) for rows of the main model; read before X2 is
    // written (a chained step passes X2 itself)
    float *F = e->big;                          // [n][2H]
    for (int t = 0; t < n; t++) {
        float x[Q_H], h[Q_H];
        embed_row(e, next[t], x);
        rmsnorm(F + (size_t)t * 2 * Q_H, x, M->enorm, Q_H);
        const float *src = H + (size_t)t * ldh;
        if (residual) { rmsnorm(h, src, e->norm_f, Q_H); src = h; }
        rmsnorm(F + (size_t)t * 2 * Q_H + Q_H, src, M->hnorm, Q_H);
    }
    float *X = e->X2;
    lin_mm(e, &M->eh, F, 2 * Q_H, n, X, Q_H);
    diag_add(DG_DRAFT, -1, t0, q_lin_bytes(&M->eh), 0);
    run_block(e, &M->at, dl, Q_N_ATTN, n, p0, X);
    run_block(e, &M->moe, dl + 1, 0, n, p0, X);
    norm_rows(e, X, Q_H, n, X, Q_H, M->final_norm);
    if (n_logits > 0) q_head(e, X, n, logits_rows, n_logits, out_logits);
    diag_set_phase(phase);
}

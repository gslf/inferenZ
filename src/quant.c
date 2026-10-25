#include "quant.h"

#include <float.h>
#include <immintrin.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "util.h"
#include <omp.h>

#define QMAGIC 0x515A4E49u  // "INZQ"

int inz_qtype_bits(int type) {
    switch (type) {
    case INZ_T_Q2L: return 2;
    case INZ_T_Q3L: return 3;
    case INZ_T_Q3P: return 3;
    case INZ_T_Q4L: return 4;
    case INZ_T_Q4N: return 4;
    case INZ_T_Q8G: return 8;
    default: return 0;
    }
}

size_t inz_qtype_group_bytes(int type) { return type == INZ_T_Q3P ? 16 : (size_t)inz_qtype_bits(type) * INZ_GROUP / 8; }

static int type_bias(int type) { return type == INZ_T_Q8G ? 128 : INZ_LUT_BIAS; }

size_t inz_qmat_bytes(int type, int rows, int cols) {
    size_t ng = (size_t)cols / INZ_GROUP;
    size_t n = sizeof(inz_qhdr) + (size_t)rows * ng * inz_qtype_group_bytes(type) + (size_t)rows * ng * 2;
    if (type == INZ_T_Q3P) n += (size_t)rows * ng * 8;
    if (type == INZ_T_Q4N) n += (size_t)rows * ng * 2;  // 4 e4m3 scales per group instead of one fp16
    return INZ_ALIGN_UP(n, 64);
}

size_t inz_mat_bytes(int type, int rows, int cols) {
    if (type == INZ_T_F32) return (size_t)rows * cols * 4;
    if (type == INZ_T_F16 || type == INZ_T_BF16) return (size_t)rows * cols * 2;
    if (type == INZ_T_FP8) return (size_t)rows * cols;
    return inz_qmat_bytes(type, rows, cols);
}

const char *inz_qtype_name(int type) {
    switch (type) {
    case INZ_T_F32: return "F32";
    case INZ_T_F16: return "F16";
    case INZ_T_BF16: return "BF16";
    case INZ_T_Q2L: return "Q2L";
    case INZ_T_Q3L: return "Q3L";
    case INZ_T_Q4L: return "Q4L";
    case INZ_T_Q8G: return "Q8G";
    case INZ_T_Q3P: return "Q3P";
    case INZ_T_Q4N: return "Q4N";
    case INZ_T_FP8: return "FP8";
    case INZ_T_EXPERTS: return "EXPERTS";
    default: return "?";
    }
}

int inz_qtype_from_name(const char *s) {
    static const int all[] = {INZ_T_F32, INZ_T_F16, INZ_T_BF16, INZ_T_Q2L, INZ_T_Q3L, INZ_T_Q4L, INZ_T_Q8G, INZ_T_Q3P, INZ_T_Q4N};
    for (size_t i = 0; i < sizeof(all) / sizeof(all[0]); i++)
        if (!strcasecmp(s, inz_qtype_name(all[i]))) return all[i];
    return -1;
}

bool inz_qmat_view(inz_qmat *m, const void *payload) {
    const inz_qhdr *h = payload;
    if (h->magic != QMAGIC) return false;
    memset(m, 0, sizeof(*m));
    m->type = h->type;
    m->bits = h->bits;
    m->rows = (int)h->rows;
    m->cols = (int)h->cols;
    m->ngroups = m->cols / INZ_GROUP;
    m->rot_block = h->rot_block;
    m->rot_seed = h->rot_seed;
    m->lut = h->lut;
    m->qbytes = inz_qtype_group_bytes(m->type);
    m->q = (const uint8_t *)payload + sizeof(inz_qhdr);
    m->d = (const uint16_t *)(m->q + (size_t)m->rows * m->ngroups * m->qbytes);
    if (m->type == INZ_T_Q3P) m->f = (const uint8_t *)(m->d + (size_t)m->rows * m->ngroups);
    if (m->type == INZ_T_Q4N) {
        m->s8 = (const uint8_t *)m->d;
        m->d = NULL;
        memcpy(&m->gscale, h->pad2, sizeof(float));
    }
    return true;
}

size_t inz_q3p_coarse_bytes(int rows, int cols) { return (size_t)rows * (cols / INZ_GROUP) * 18; }
size_t inz_q3p_fine_bytes(int rows, int cols) { return (size_t)rows * (cols / INZ_GROUP) * 8; }

void inz_qmat_view_q3p(inz_qmat *m, const void *hdr, const uint8_t *coarse, const uint8_t *fine) {
    const inz_qhdr *h = hdr;
    memset(m, 0, sizeof(*m));
    m->type = INZ_T_Q3P;
    m->bits = 3;
    m->rows = (int)h->rows;
    m->cols = (int)h->cols;
    m->ngroups = m->cols / INZ_GROUP;
    m->rot_block = h->rot_block;
    m->rot_seed = h->rot_seed;
    m->lut = h->lut;
    m->qbytes = 16;
    m->q = coarse;
    m->d = (const uint16_t *)(coarse + (size_t)m->rows * m->ngroups * 16);
    m->f = fine;
    m->coarse = fine == NULL;
}

/* ------------------------------------------------------------------ */
/* Randomized Hadamard rotation                                        */
/* ------------------------------------------------------------------ */

static inline float rot_sign(uint32_t seed, uint32_t i) {
    uint32_t h = (i + 1) * 0x9E3779B1u ^ seed;
    h ^= h >> 15; h *= 0x2C1B3C6Du; h ^= h >> 12; h *= 0x297A2D39u; h ^= h >> 15;
    return (h & 1) ? -1.0f : 1.0f;
}

static void fwht(float *x, int n) {
    // in-place unnormalized Walsh-Hadamard transform, n power of 2
    for (int len = 1; len < n; len <<= 1) {
        if (len >= 8) {
            for (int i = 0; i < n; i += len << 1) {
                for (int j = i; j < i + len; j += 8) {
                    __m256 a = _mm256_loadu_ps(x + j), b = _mm256_loadu_ps(x + j + len);
                    _mm256_storeu_ps(x + j, _mm256_add_ps(a, b));
                    _mm256_storeu_ps(x + j + len, _mm256_sub_ps(a, b));
                }
            }
        } else {
            for (int i = 0; i < n; i += len << 1)
                for (int j = i; j < i + len; j++) {
                    float a = x[j], b = x[j + len];
                    x[j] = a + b;
                    x[j + len] = a - b;
                }
        }
    }
}

void inz_rotate(float *x, int n, int block, uint32_t seed) {
    if (block <= 1) return;
    const float norm = 1.0f / sqrtf((float)block);
    for (int i = 0; i < n; i++) x[i] *= rot_sign(seed, (uint32_t)i);
    for (int b = 0; b < n; b += block) {
        fwht(x + b, block);
        for (int i = 0; i < block; i++) x[b + i] *= norm;
    }
}

void inz_unrotate(float *x, int n, int block, uint32_t seed) {
    if (block <= 1) return;
    const float norm = 1.0f / sqrtf((float)block);
    for (int b = 0; b < n; b += block) {
        fwht(x + b, block);
        for (int i = 0; i < block; i++) x[b + i] *= norm;
    }
    for (int i = 0; i < n; i++) x[i] *= rot_sign(seed, (uint32_t)i);
}

int inz_auto_rot_block(int cols) {
    for (int b = 256; b >= INZ_GROUP; b >>= 1)
        if (cols % b == 0) return b;
    return 0;
}

/* ------------------------------------------------------------------ */
/* Offline quantizer                                                   */
/* ------------------------------------------------------------------ */

// Lloyd-Max optimal levels for a unit Gaussian (normalized so max = 1)
static void gaussian_levels(int bits, float *L) {
    static const float g2[] = {-1.5104f, -0.4528f, 0.4528f, 1.5104f};
    static const float g3[] = {-2.1520f, -1.3439f, -0.7560f, -0.2451f, 0.2451f, 0.7560f, 1.3439f, 2.1520f};
    static const float g4[] = {-2.7326f, -2.0690f, -1.6181f, -1.2562f, -0.9423f, -0.6568f, -0.3880f, -0.1284f,
                               0.1284f,  0.3880f,  0.6568f,  0.9423f,  1.2562f,  1.6181f,  2.0690f,  2.7326f};
    const float *src = bits == 2 ? g2 : bits == 3 ? g3 : g4;
    int n = 1 << bits;
    for (int i = 0; i < n; i++) L[i] = src[i] / src[n - 1];
}


// Vectorized nearest-level assignment for <=16 sorted levels: index = number of midpoints below v.
typedef struct {
    int nl;
    float lmax;
    __m256 T[15];          // midpoints
    __m256 lo, hi;         // levels 0..7 and 8..15 for permutevar gathers
} lvlset;

static void lvlset_init(lvlset *ls, const float *L, int nl) {
    ls->nl = nl;
    ls->lmax = 0;
    float lo[8] = {0}, hi[8] = {0};
    for (int k = 0; k < nl; k++) {
        ls->lmax = fmaxf(ls->lmax, fabsf(L[k]));
        if (k < 8) lo[k] = L[k]; else hi[k - 8] = L[k];
    }
    for (int k = 0; k + 1 < nl; k++) ls->T[k] = _mm256_set1_ps(0.5f * (L[k] + L[k + 1]));
    ls->lo = _mm256_loadu_ps(lo);
    ls->hi = _mm256_loadu_ps(hi);
}

static inline __m256i lvl_assign(const lvlset *ls, __m256 v) {
    __m256i c = _mm256_setzero_si256();
    for (int k = 0; k + 1 < ls->nl; k++) c = _mm256_sub_epi32(c, _mm256_castps_si256(_mm256_cmp_ps(v, ls->T[k], _CMP_GT_OQ)));
    return c;
}

static inline __m256 lvl_value(const lvlset *ls, __m256i idx) {
    __m256 a = _mm256_permutevar8x32_ps(ls->lo, idx);
    if (ls->nl <= 8) return a;
    __m256 b = _mm256_permutevar8x32_ps(ls->hi, _mm256_sub_epi32(idx, _mm256_set1_epi32(8)));
    __m256 sel = _mm256_castsi256_ps(_mm256_cmpgt_epi32(idx, _mm256_set1_epi32(7)));
    return _mm256_blendv_ps(a, b, sel);
}

static inline float hsum8f(__m256 v) {
    __m128 lo = _mm256_castps256_ps128(v), hi = _mm256_extractf128_ps(v, 1);
    lo = _mm_add_ps(lo, hi);
    lo = _mm_add_ps(lo, _mm_movehl_ps(lo, lo));
    lo = _mm_add_ss(lo, _mm_movehdup_ps(lo));
    return _mm_cvtss_f32(lo);
}

// Fit scale and indices of one 64-weight group against levels (weighted MSE). Returns weighted error.
static float fit_group(const float *x, const float *w, const lvlset *ls, float *d_out, uint8_t *idx) {
    const __m256 absmask = _mm256_castsi256_ps(_mm256_set1_epi32(0x7FFFFFFF));
    __m256 X[8], Wt[8];
    __m256 mx = _mm256_setzero_ps();
    for (int j = 0; j < 8; j++) {
        X[j] = _mm256_loadu_ps(x + 8 * j);
        Wt[j] = _mm256_loadu_ps(w + 8 * j);
        mx = _mm256_max_ps(mx, _mm256_and_ps(X[j], absmask));
    }
    float m8[8];
    _mm256_storeu_ps(m8, mx);
    float amax = 0;
    for (int k = 0; k < 8; k++) amax = fmaxf(amax, m8[k]);
    if (amax == 0 || ls->lmax == 0) {
        *d_out = 0;
        __m256i z = lvl_assign(ls, _mm256_setzero_ps());
        int32_t t[8];
        _mm256_storeu_si256((__m256i *)t, z);
        memset(idx, t[0], INZ_GROUP);
        return 0;
    }
    static const float cand[] = {0.82f, 0.88f, 0.94f, 1.0f, 1.06f};
    float best_err = FLT_MAX, best_d = amax / ls->lmax;
    __m256i best_i[8];
    for (size_t c = 0; c < sizeof(cand) / sizeof(cand[0]); c++) {
        float d = amax / ls->lmax * cand[c];
        __m256 inv = _mm256_set1_ps(1.0f / d);
        __m256 num = _mm256_setzero_ps(), den = _mm256_setzero_ps();
        for (int j = 0; j < 8; j++) {
            __m256 lv = lvl_value(ls, lvl_assign(ls, _mm256_mul_ps(X[j], inv)));
            __m256 wl = _mm256_mul_ps(Wt[j], lv);
            num = _mm256_fmadd_ps(wl, X[j], num);
            den = _mm256_fmadd_ps(wl, lv, den);
        }
        float dn = hsum8f(den);
        if (dn > 0) d = hsum8f(num) / dn;
        if (!(d > 0)) continue;
        inv = _mm256_set1_ps(1.0f / d);
        __m256 vd = _mm256_set1_ps(d), err = _mm256_setzero_ps();
        __m256i I[8];
        for (int j = 0; j < 8; j++) {
            I[j] = lvl_assign(ls, _mm256_mul_ps(X[j], inv));
            __m256 e = _mm256_fnmadd_ps(vd, lvl_value(ls, I[j]), X[j]);
            err = _mm256_fmadd_ps(_mm256_mul_ps(e, e), Wt[j], err);
        }
        float er = hsum8f(err);
        if (er < best_err) {
            best_err = er;
            best_d = d;
            for (int j = 0; j < 8; j++) best_i[j] = I[j];
        }
    }
    if (best_err == FLT_MAX) {  // degenerate group: fall back to the amax scale
        __m256 inv = _mm256_set1_ps(1.0f / best_d);
        best_err = 0;
        for (int j = 0; j < 8; j++) best_i[j] = lvl_assign(ls, _mm256_mul_ps(X[j], inv));
    }
    for (int j = 0; j < 8; j++) {
        int32_t t[8];
        _mm256_storeu_si256((__m256i *)t, best_i[j]);
        for (int k = 0; k < 8; k++) idx[8 * j + k] = (uint8_t)t[k];
    }
    *d_out = best_d;
    return best_err;
}

static void pack_group(int type, const uint8_t *idx, uint8_t *dst) {
    switch (type) {
    case INZ_T_Q4L:
        for (int i = 0; i < 32; i++) dst[i] = (uint8_t)(idx[i] | (idx[i + 32] << 4));
        break;
    case INZ_T_Q2L:
        for (int i = 0; i < 16; i++)
            dst[i] = (uint8_t)(idx[i] | (idx[i + 16] << 2) | (idx[i + 32] << 4) | (idx[i + 48] << 6));
        break;
    case INZ_T_Q3L: {
        for (int i = 0; i < 16; i++)
            dst[i] = (uint8_t)((idx[i] & 3) | ((idx[i + 16] & 3) << 2) | ((idx[i + 32] & 3) << 4) | ((idx[i + 48] & 3) << 6));
        // high bit plane: byte (i % 8), bit (i / 8) -> decodes with one SRLV per 32 weights
        uint8_t hb[8] = {0};
        for (int i = 0; i < 64; i++) hb[i % 8] |= (uint8_t)(((idx[i] >> 2) & 1) << (i / 8));
        memcpy(dst + 16, hb, 8);
        break;
    }
    case INZ_T_Q8G:
        for (int i = 0; i < 64; i++) dst[i] = idx[i];
        break;
    case INZ_T_Q3P: {
        // coarse plane = idx >> 1 in the Q2 layout; fine bit = idx & 1 in the high-bit layout (written by caller)
        for (int i = 0; i < 16; i++)
            dst[i] = (uint8_t)((idx[i] >> 1) | ((idx[i + 16] >> 1) << 2) | ((idx[i + 32] >> 1) << 4) | ((idx[i + 48] >> 1) << 6));
        break;
    }
    }
}

static void pack_fine(const uint8_t *idx, uint8_t *dst) {
    uint8_t hb[8] = {0};
    for (int i = 0; i < 64; i++) hb[i % 8] |= (uint8_t)((idx[i] & 1) << (i / 8));
    memcpy(dst, hb, 8);
}

static void unpack_group_q3p(const uint8_t *q, const uint8_t *f, uint8_t *idx) {
    for (int i = 0; i < 16; i++) {
        idx[i] = (uint8_t)((q[i] & 3) << 1); idx[i + 16] = (uint8_t)(((q[i] >> 2) & 3) << 1);
        idx[i + 32] = (uint8_t)(((q[i] >> 4) & 3) << 1); idx[i + 48] = (uint8_t)(((q[i] >> 6) & 3) << 1);
    }
    if (f) for (int i = 0; i < 64; i++) idx[i] |= (uint8_t)((f[i % 8] >> (i / 8)) & 1);
}

static void unpack_group(int type, const uint8_t *src, uint8_t *idx) {
    switch (type) {
    case INZ_T_Q4L:
        for (int i = 0; i < 32; i++) { idx[i] = src[i] & 15; idx[i + 32] = src[i] >> 4; }
        break;
    case INZ_T_Q2L:
        for (int i = 0; i < 16; i++) {
            idx[i] = src[i] & 3; idx[i + 16] = (src[i] >> 2) & 3;
            idx[i + 32] = (src[i] >> 4) & 3; idx[i + 48] = (src[i] >> 6) & 3;
        }
        break;
    case INZ_T_Q3L: {
        for (int i = 0; i < 16; i++) {
            idx[i] = src[i] & 3; idx[i + 16] = (src[i] >> 2) & 3;
            idx[i + 32] = (src[i] >> 4) & 3; idx[i + 48] = (src[i] >> 6) & 3;
        }
        for (int i = 0; i < 64; i++) idx[i] |= (uint8_t)(((src[16 + i % 8] >> (i / 8)) & 1) << 2);
        break;
    }
    case INZ_T_Q8G:
        for (int i = 0; i < 64; i++) idx[i] = src[i];
        break;
    }
}

// Weighted Lloyd-Max on (already rotated) rows, using a deterministic subsample of groups.
// Output: levels in integer units (-63..63), sorted ascending.
void inz_learn_lut_rotated(const float *Wr, int rows, int cols, const float *imp, int bits, int iters, int sample_groups,
                           float *Lout) {
    const int nl = 1 << bits, ng = cols / INZ_GROUP;
    const int64_t total = (int64_t)rows * ng;
    int64_t ns = sample_groups > 0 ? sample_groups : 8192;
    if (ns > total) ns = total;
    if (iters <= 0) iters = 5;
    float L[16];
    gaussian_levels(bits, L);
    float *dsc = inz_malloc(sizeof(float) * (size_t)ns);
    uint8_t *ix = inz_malloc((size_t)ns * INZ_GROUP);
    const int64_t stride = total / ns;
    for (int it = 0; it < iters; it++) {
        lvlset ls;
        lvlset_init(&ls, L, nl);
        double num[16] = {0}, den[16] = {0};
        for (int64_t s = 0; s < ns; s++) {
            int64_t gi = s * stride;
            int r = (int)(gi / ng), g = (int)(gi % ng);
            const float *x = Wr + (size_t)r * cols + (size_t)g * INZ_GROUP;
            const float *w = imp + g * INZ_GROUP;
            fit_group(x, w, &ls, &dsc[s], ix + s * INZ_GROUP);
            float d = dsc[s];
            if (d == 0) continue;
            for (int j = 0; j < INZ_GROUP; j++) {
                int k = ix[s * INZ_GROUP + j];
                num[k] += (double)w[j] * d * x[j];
                den[k] += (double)w[j] * d * d;
            }
        }
        for (int k = 0; k < nl; k++) if (den[k] > 0) L[k] = (float)(num[k] / den[k]);
        for (int a = 1; a < nl; a++) {
            float v = L[a];
            int b = a - 1;
            while (b >= 0 && L[b] > v) { L[b + 1] = L[b]; b--; }
            L[b + 1] = v;
        }
    }
    float lmax = 0;
    for (int k = 0; k < nl; k++) lmax = fmaxf(lmax, fabsf(L[k]));
    for (int k = 0; k < nl; k++) {
        int v = (int)lrintf(L[k] / lmax * 63.0f);
        v = v < -63 ? -63 : v > 63 ? 63 : v;
        Lout[k] = (float)v;
    }
    // rounding may create duplicates; nudge them apart to keep levels distinct and sorted
    for (int k = 1; k < nl; k++) if (Lout[k] <= Lout[k - 1]) Lout[k] = Lout[k - 1] + 1;
    free(dsc);
    free(ix);
}

static double quantize_q4n(const float *W, int rows, int cols, void *out);

double inz_quantize(const float *W, int rows, int cols, const inz_quant_params *p, void *out) {
    const int type = p->type;
    if (type == INZ_T_Q4N) return quantize_q4n(W, rows, cols, out);
    const int bits = inz_qtype_bits(type);
    if (!bits || cols % INZ_GROUP) inz_die("inz_quantize: unsupported type %d or cols %d", type, cols);
    const int ng = cols / INZ_GROUP;
    int rot = p->rot_block == 0 ? inz_auto_rot_block(cols) : p->rot_block < 0 ? 0 : p->rot_block;
    if (rot && cols % rot) inz_die("rotation block %d does not divide %d", rot, cols);
    // int8 needs no incoherence processing for its own error, but rotating spreads activation outliers before
    // their per-group int8 quantization; Q8G is rotated only when the caller asks for a block explicitly
    if (type == INZ_T_Q8G && p->rot_block <= 0) rot = 0;
    const uint32_t seed = p->rot_seed ? p->rot_seed : 0x5EEDu;

    memset(out, 0, inz_qmat_bytes(type, rows, cols));
    inz_qhdr *h = out;
    h->magic = QMAGIC;
    h->type = (uint16_t)type;
    h->rot_block = (uint16_t)rot;
    h->rows = (uint32_t)rows;
    h->cols = (uint32_t)cols;
    h->rot_seed = seed;
    h->bits = (uint8_t)bits;
    uint8_t *qout = (uint8_t *)out + sizeof(inz_qhdr);
    uint16_t *dout = (uint16_t *)(qout + (size_t)rows * ng * inz_qtype_group_bytes(type));

    // importance in the rotated basis: diagonal approximation = block mean
    float *imp = inz_malloc(sizeof(float) * cols);
    for (int j = 0; j < cols; j++) imp[j] = p->imatrix ? fmaxf(p->imatrix[j], 1e-8f) : 1.0f;
    if (rot && p->imatrix) {
        for (int b = 0; b < cols; b += rot) {
            double s = 0;
            for (int j = 0; j < rot; j++) s += imp[b + j];
            for (int j = 0; j < rot; j++) imp[b + j] = (float)(s / rot);
        }
    }

    // rotated copy of the weights
    float *Wr = inz_malloc(sizeof(float) * (size_t)rows * cols);
    const int nthr = p->threads > 0 ? p->threads : omp_get_max_threads();
    #pragma omp parallel for schedule(static) if(nthr != 1) num_threads(nthr)
    for (int r = 0; r < rows; r++) {
        memcpy(Wr + (size_t)r * cols, W + (size_t)r * cols, sizeof(float) * cols);
        if (rot) inz_rotate(Wr + (size_t)r * cols, cols, rot, seed);
    }

    const int nl = 1 << bits;
    double total_err = 0, total_energy = 0;
    uint8_t *idx_all = inz_malloc((size_t)rows * cols);
    float *d_all = inz_malloc(sizeof(float) * (size_t)rows * ng);

    if (type == INZ_T_Q8G) {
        #pragma omp parallel for schedule(static) reduction(+:total_err,total_energy) if(nthr != 1) num_threads(nthr)
        for (int r = 0; r < rows; r++) {
            for (int g = 0; g < ng; g++) {
                const float *x = Wr + (size_t)r * cols + (size_t)g * INZ_GROUP;
                const float *w = imp + g * INZ_GROUP;
                float amax = 0;
                for (int j = 0; j < INZ_GROUP; j++) amax = fmaxf(amax, fabsf(x[j]));
                float d = amax / 127.0f;
                float inv = d > 0 ? 1.0f / d : 0;
                uint8_t *ix = idx_all + (size_t)r * cols + (size_t)g * INZ_GROUP;
                for (int j = 0; j < INZ_GROUP; j++) {
                    int q = (int)lrintf(x[j] * inv);
                    q = q < -127 ? -127 : q > 127 ? 127 : q;
                    ix[j] = (uint8_t)(q + 128);
                    total_energy += w[j] * x[j] * x[j];
                }
                float stored = inz_f16_to_f32(inz_f32_to_f16(d));
                double best = 0;
                for (int j = 0; j < INZ_GROUP; j++) {
                    float decoded = stored * ((int)ix[j] - 128);
                    double diff = (double)x[j] - decoded;
                    best += w[j] * diff * diff;
                }
                total_err += best;
                d_all[(size_t)r * ng + g] = d;
            }
        }
    } else {
        // Levels: either fixed by the caller (shared per layer/projection) or learned here with
        // weighted Lloyd-Max on a subsample of groups, then frozen to their 7-bit biased form.
        float Lq[16];
        if (p->fixed_lut) {
            for (int k = 0; k < nl; k++) Lq[k] = p->fixed_lut[k];
        } else {
            inz_learn_lut_rotated(Wr, rows, cols, imp, bits, p->lut_iters, p->lut_sample_groups, Lq);
        }
        for (int k = 0; k < nl; k++) h->lut[k] = (uint8_t)((int)Lq[k] + INZ_LUT_BIAS);
        lvlset ls;
        lvlset_init(&ls, Lq, nl);
        total_err = 0;
        total_energy = 0;
        #pragma omp parallel for schedule(static) reduction(+:total_err,total_energy) if(nthr != 1) num_threads(nthr)
        for (int r = 0; r < rows; r++) {
            for (int g = 0; g < ng; g++) {
                const float *x = Wr + (size_t)r * cols + (size_t)g * INZ_GROUP;
                const float *w = imp + g * INZ_GROUP;
                uint8_t *ix = idx_all + (size_t)r * cols + (size_t)g * INZ_GROUP;
                float d;
                total_err += fit_group(x, w, &ls, &d, ix);
                d_all[(size_t)r * ng + g] = d;
                for (int jj = 0; jj < INZ_GROUP; jj++) total_energy += w[jj] * x[jj] * x[jj];
            }
        }
    }
    // pack
    const size_t gb = inz_qtype_group_bytes(type);
    uint8_t *fout = type == INZ_T_Q3P ? (uint8_t *)(dout + (size_t)rows * ng) : NULL;
    #pragma omp parallel for schedule(static) if(nthr != 1) num_threads(nthr)
    for (int r = 0; r < rows; r++)
        for (int g = 0; g < ng; g++) {
            const uint8_t *ix = idx_all + (size_t)r * cols + (size_t)g * INZ_GROUP;
            pack_group(type, ix, qout + ((size_t)r * ng + g) * gb);
            if (fout) pack_fine(ix, fout + ((size_t)r * ng + g) * 8);
            dout[(size_t)r * ng + g] = inz_f32_to_f16(d_all[(size_t)r * ng + g]);
        }
    if (type == INZ_T_Q3P) {
        // coarse levels: occupancy-weighted mean of each pair of adjacent levels (the conditional mean of
        // a weight given only its coarse code), so a coarse-only read still decodes sensibly
        uint64_t cnt[8] = {0};
        for (size_t i = 0; i < (size_t)rows * cols; i++) cnt[idx_all[i] & 7]++;
        for (int c = 0; c < 4; c++) {
            double a = (double)h->lut[2 * c] - INZ_LUT_BIAS, b = (double)h->lut[2 * c + 1] - INZ_LUT_BIAS;
            double na_ = (double)cnt[2 * c], nb_ = (double)cnt[2 * c + 1];
            double v = na_ + nb_ > 0 ? (a * na_ + b * nb_) / (na_ + nb_) : 0.5 * (a + b);
            int iv = (int)lrint(v);
            h->lut[8 + c] = (uint8_t)((iv < -63 ? -63 : iv > 63 ? 63 : iv) + INZ_LUT_BIAS);
        }
    }
    free(idx_all);
    free(d_all);
    free(Wr);
    free(imp);
    return total_energy > 0 ? total_err / total_energy : 0;
}

void inz_dequantize_row(const inz_qmat *m, int row, float *out) {
    uint8_t idx[INZ_GROUP];
    const int bias = type_bias(m->type);
    for (int g = 0; g < m->ngroups; g++) {
        const size_t gi = (size_t)row * m->ngroups + g;
        if (m->type == INZ_T_Q4N) {
            unpack_group(INZ_T_Q4L, m->q + gi * 32, idx);
            for (int j = 0; j < INZ_GROUP; j++)
                out[g * INZ_GROUP + j] = (float)((int)m->lut[idx[j]] - INZ_LUT_BIAS) * inz_e4m3_to_f32(m->s8[gi * 4 + j / 16]) * m->gscale;
            continue;
        }
        if (m->type == INZ_T_Q3P) {
            unpack_group_q3p(m->q + gi * 16, m->coarse ? NULL : m->f + gi * 8, idx);
            const float d = inz_f16_to_f32(m->d[gi]);
            for (int j = 0; j < INZ_GROUP; j++)
                out[g * INZ_GROUP + j] = d * (float)((m->coarse ? m->lut[8 + (idx[j] >> 1)] : m->lut[idx[j]]) - bias);
            continue;
        }
        unpack_group(m->type, m->q + gi * m->qbytes, idx);
        float d = inz_f16_to_f32(m->d[gi]);
        for (int j = 0; j < INZ_GROUP; j++) {
            int v = m->type == INZ_T_Q8G ? idx[j] : m->lut[idx[j]];
            out[g * INZ_GROUP + j] = d * (float)(v - bias);
        }
    }
}

void inz_dequantize_row_orig(const inz_qmat *m, int row, float *out) {
    inz_dequantize_row(m, row, out);
    if (m->rot_block) inz_unrotate(out, m->cols, m->rot_block, m->rot_seed);
}

/* ------------------------------------------------------------------ */
/* Activations                                                         */
/* ------------------------------------------------------------------ */

void inz_act_alloc(inz_act *a, int n) {
    memset(a, 0, sizeof(*a));
    a->n = n;
    a->ng = n / INZ_GROUP;
    a->q = inz_aligned(64, (size_t)n + 64);
    a->s = inz_aligned(64, sizeof(float) * (a->ng + 8));
    a->bsum = inz_aligned(64, sizeof(float) * (a->ng + 8));
    a->xrot = inz_aligned(64, sizeof(float) * (size_t)n + 64);
    a->nb4 = inz_aligned(64, sizeof(int32_t) * ((size_t)n / 4 + 16));
}

void inz_act_free(inz_act *a) {
    free(a->q); free(a->s); free(a->bsum); free(a->xrot); free(a->nb4);
    memset(a, 0, sizeof(*a));
}

void inz_act_quantize(inz_act *a, const float *x, int n, int rot_block, uint32_t rot_seed) {
    a->n = n;
    a->ng = n / INZ_GROUP;
    a->rot_block = rot_block;
    a->rot_seed = rot_seed;
    const float *src = x;
    if (rot_block > 1) {
        memcpy(a->xrot, x, sizeof(float) * n);
        inz_rotate(a->xrot, n, rot_block, rot_seed);
        src = a->xrot;
    }
    for (int g = 0; g < a->ng; g++) {
        const float *xg = src + g * INZ_GROUP;
        __m256 mx = _mm256_setzero_ps();
        const __m256 absmask = _mm256_castsi256_ps(_mm256_set1_epi32(0x7FFFFFFF));
        for (int j = 0; j < INZ_GROUP; j += 8) mx = _mm256_max_ps(mx, _mm256_and_ps(_mm256_loadu_ps(xg + j), absmask));
        float m8[8];
        _mm256_storeu_ps(m8, mx);
        float amax = 0;
        for (int k = 0; k < 8; k++) amax = fmaxf(amax, m8[k]);
        float s = amax / 127.0f;
        float inv = s > 0 ? 1.0f / s : 0;
        int sum = 0;
        int8_t *q = a->q + g * INZ_GROUP;
        const __m256 vinv = _mm256_set1_ps(inv);
        for (int j = 0; j < INZ_GROUP; j += 8) {
            __m256i qi = _mm256_cvtps_epi32(_mm256_mul_ps(_mm256_loadu_ps(xg + j), vinv));
            int32_t t[8];
            _mm256_storeu_si256((__m256i *)t, qi);
            for (int k = 0; k < 8; k++) { q[j + k] = (int8_t)t[k]; sum += t[k]; }
            a->nb4[(g * INZ_GROUP + j) / 4] = -INZ_LUT_BIAS * (t[0] + t[1] + t[2] + t[3]);
            a->nb4[(g * INZ_GROUP + j) / 4 + 1] = -INZ_LUT_BIAS * (t[4] + t[5] + t[6] + t[7]);
        }
        a->s[g] = s;
        a->bsum[g] = s * (float)sum;
    }
}

/* ------------------------------------------------------------------ */
/* Matvec kernels                                                      */
/* ------------------------------------------------------------------ */

#if defined(__AVXVNNI__) || defined(__AVX512VNNI__)
#define HAVE_VNNI 1
static inline __m256i dot_u8s8(__m256i acc, __m256i u, __m256i s) {
#if defined(__AVX512VNNI__) && defined(__AVX512VL__)
    return _mm256_dpbusd_epi32(acc, u, s);
#else
    return _mm256_dpbusd_avx_epi32(acc, u, s);
#endif
}
#else
#define HAVE_VNNI 0
static inline __m256i dot_u8s8(__m256i acc, __m256i u, __m256i s) {
    // u <= 127 for LUT types so maddubs pairs never saturate; Q8G (u up to 255) uses the split path
    __m256i p = _mm256_maddubs_epi16(u, s);
    return _mm256_add_epi32(acc, _mm256_madd_epi16(p, _mm256_set1_epi16(1)));
}
#endif

const char *inz_kernel_isa(void) {
#if defined(__AVX512VNNI__) && defined(__AVX512VL__)
    return "AVX2+AVX512-VNNI(ymm)";
#elif defined(__AVXVNNI__)
    return "AVX2+AVX-VNNI";
#else
    return "AVX2";
#endif
}

static inline float hsum8(__m256 v) {
    __m128 lo = _mm256_castps256_ps128(v), hi = _mm256_extractf128_ps(v, 1);
    lo = _mm_add_ps(lo, hi);
    lo = _mm_add_ps(lo, _mm_movehl_ps(lo, lo));
    lo = _mm_add_ss(lo, _mm_movehdup_ps(lo));
    return _mm_cvtss_f32(lo);
}

// decode one 64-weight group into two vectors of biased u8 values (weights 0..31, 32..63)
static inline void decode_q4(const uint8_t *q, __m256i lut, __m256i *w0, __m256i *w1) {
    const __m256i m4 = _mm256_set1_epi8(0x0F);
    __m256i v = _mm256_loadu_si256((const __m256i *)q);
    *w0 = _mm256_shuffle_epi8(lut, _mm256_and_si256(v, m4));
    *w1 = _mm256_shuffle_epi8(lut, _mm256_and_si256(_mm256_srli_epi16(v, 4), m4));
}

static inline void decode_q2(const uint8_t *q, __m256i lut, __m256i *w0, __m256i *w1) {
    const __m256i m2 = _mm256_set1_epi8(0x03);
    __m256i v = _mm256_broadcastsi128_si256(_mm_loadu_si128((const __m128i *)q));
    __m256i a = _mm256_srlv_epi64(v, _mm256_set_epi64x(2, 2, 0, 0));
    __m256i b = _mm256_srlv_epi64(v, _mm256_set_epi64x(6, 6, 4, 4));
    *w0 = _mm256_shuffle_epi8(lut, _mm256_and_si256(a, m2));
    *w1 = _mm256_shuffle_epi8(lut, _mm256_and_si256(b, m2));
}

static inline void decode_q3(const uint8_t *q, __m256i lut, __m256i *w0, __m256i *w1) {
    const __m256i m2 = _mm256_set1_epi8(0x03), m1 = _mm256_set1_epi8(0x01);
    __m256i v = _mm256_broadcastsi128_si256(_mm_loadu_si128((const __m128i *)q));
    __m256i a = _mm256_and_si256(_mm256_srlv_epi64(v, _mm256_set_epi64x(2, 2, 0, 0)), m2);
    __m256i b = _mm256_and_si256(_mm256_srlv_epi64(v, _mm256_set_epi64x(6, 6, 4, 4)), m2);
    long long hb;
    memcpy(&hb, q + 16, 8);
    __m256i h = _mm256_set1_epi64x(hb);
    __m256i ha = _mm256_slli_epi16(_mm256_and_si256(_mm256_srlv_epi64(h, _mm256_set_epi64x(3, 2, 1, 0)), m1), 2);
    __m256i hc = _mm256_slli_epi16(_mm256_and_si256(_mm256_srlv_epi64(h, _mm256_set_epi64x(7, 6, 5, 4)), m1), 2);
    *w0 = _mm256_shuffle_epi8(lut, _mm256_or_si256(a, ha));
    *w1 = _mm256_shuffle_epi8(lut, _mm256_or_si256(b, hc));
}

// Matvec for LUT types, two rows at a time. Scales are handled 8 groups at a time in vector
// registers (fp16->fp32 conversion, activation scale and bias correction amortized), so the
// per-group work is only: decode, 2x VPDPBUSD, convert, permute-broadcast scale, FMA.
static inline void decode_q3p(const uint8_t *q, const uint8_t *f, __m256i lut, __m256i *w0, __m256i *w1) {
    const __m256i m2 = _mm256_set1_epi8(0x03), m1 = _mm256_set1_epi8(0x01);
    __m256i v = _mm256_broadcastsi128_si256(_mm_loadu_si128((const __m128i *)q));
    __m256i a = _mm256_and_si256(_mm256_srlv_epi64(v, _mm256_set_epi64x(2, 2, 0, 0)), m2);
    __m256i b = _mm256_and_si256(_mm256_srlv_epi64(v, _mm256_set_epi64x(6, 6, 4, 4)), m2);
    long long hb;
    memcpy(&hb, f, 8);
    __m256i h = _mm256_set1_epi64x(hb);
    __m256i fa = _mm256_and_si256(_mm256_srlv_epi64(h, _mm256_set_epi64x(3, 2, 1, 0)), m1);
    __m256i fb = _mm256_and_si256(_mm256_srlv_epi64(h, _mm256_set_epi64x(7, 6, 5, 4)), m1);
    *w0 = _mm256_shuffle_epi8(lut, _mm256_or_si256(_mm256_add_epi8(a, a), fa));
    *w1 = _mm256_shuffle_epi8(lut, _mm256_or_si256(_mm256_add_epi8(b, b), fb));
}

// uniform decoder signature: (indices, fine plane, lut, out0, out1)
#define DEC_Q4(q, f, lut, a, b) decode_q4(q, lut, a, b)
#define DEC_Q2(q, f, lut, a, b) decode_q2(q, lut, a, b)
#define DEC_Q3(q, f, lut, a, b) decode_q3(q, lut, a, b)
#define DEC_Q3P(q, f, lut, a, b) decode_q3p(q, f, lut, a, b)

#define DEFINE_MV_LUT(NAME, DECODE, GB, LUTOFF)                                                                   \
    static void NAME(const inz_qmat *m, const inz_act *a, float *y, int r0, int r1) {                     \
        const __m256i lut = _mm256_broadcastsi128_si256(_mm_loadu_si128((const __m128i *)(m->lut + LUTOFF)));         \
        const int ng = m->ngroups, ng8 = ng & ~7;                                                         \
        const float bias = (float)INZ_LUT_BIAS;                                                           \
        const __m256i perm[8] = {_mm256_set1_epi32(0), _mm256_set1_epi32(1), _mm256_set1_epi32(2),        \
                                 _mm256_set1_epi32(3), _mm256_set1_epi32(4), _mm256_set1_epi32(5),        \
                                 _mm256_set1_epi32(6), _mm256_set1_epi32(7)};                             \
        int r = r0;                                                                                       \
        for (; r + 2 <= r1; r += 2) {                                                                     \
            const uint8_t *q0 = m->q + (size_t)r * ng * GB, *q1 = q0 + (size_t)ng * GB;                   \
            const uint8_t *f0 = m->f ? m->f + (size_t)r * ng * 8 : NULL, *f1 = f0 ? f0 + (size_t)ng * 8 : NULL; \
            (void)f0; (void)f1;                   \
            const uint16_t *d0 = m->d + (size_t)r * ng, *d1 = d0 + ng;                                    \
            __m256 acc0 = _mm256_setzero_ps(), acc1 = _mm256_setzero_ps();                                \
            __m256 bac0 = _mm256_setzero_ps(), bac1 = _mm256_setzero_ps();                                \
            int g = 0;                                                                                    \
            for (; g < ng8; g += 8) {                                                                     \
                const __m256 sv = _mm256_loadu_ps(a->s + g), bv = _mm256_loadu_ps(a->bsum + g);           \
                const __m256 dv0 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)(d0 + g)));          \
                const __m256 dv1 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)(d1 + g)));          \
                const __m256 ds0 = _mm256_mul_ps(dv0, sv), ds1 = _mm256_mul_ps(dv1, sv);                  \
                bac0 = _mm256_fmadd_ps(dv0, bv, bac0);                                                    \
                bac1 = _mm256_fmadd_ps(dv1, bv, bac1);                                                    \
                for (int k = 0; k < 8; k++) {                                                             \
                    const int gg = g + k;                                                                 \
                    const __m256i x0 = _mm256_load_si256((const __m256i *)(a->q + gg * 64));              \
                    const __m256i x1 = _mm256_load_si256((const __m256i *)(a->q + gg * 64 + 32));         \
                    __m256i u0, u1, v0, v1;                                                               \
                    DECODE(q0 + (size_t)gg * GB, f0 + (size_t)gg * 8, lut, &u0, &u1);                                          \
                    DECODE(q1 + (size_t)gg * GB, f1 + (size_t)gg * 8, lut, &v0, &v1);                                          \
                    __m256i i0 = dot_u8s8(dot_u8s8(_mm256_setzero_si256(), u0, x0), u1, x1);              \
                    __m256i i1 = dot_u8s8(dot_u8s8(_mm256_setzero_si256(), v0, x0), v1, x1);              \
                    acc0 = _mm256_fmadd_ps(_mm256_cvtepi32_ps(i0), _mm256_permutevar8x32_ps(ds0, perm[k]), acc0); \
                    acc1 = _mm256_fmadd_ps(_mm256_cvtepi32_ps(i1), _mm256_permutevar8x32_ps(ds1, perm[k]), acc1); \
                }                                                                                         \
            }                                                                                             \
            float b0 = hsum8(bac0), b1 = hsum8(bac1);                                                     \
            for (; g < ng; g++) {                                                                         \
                const __m256i x0 = _mm256_load_si256((const __m256i *)(a->q + g * 64));                   \
                const __m256i x1 = _mm256_load_si256((const __m256i *)(a->q + g * 64 + 32));              \
                __m256i u0, u1, v0, v1;                                                                   \
                DECODE(q0 + (size_t)g * GB, f0 + (size_t)g * 8, lut, &u0, &u1);                                               \
                DECODE(q1 + (size_t)g * GB, f1 + (size_t)g * 8, lut, &v0, &v1);                                               \
                __m256i i0 = dot_u8s8(dot_u8s8(_mm256_setzero_si256(), u0, x0), u1, x1);                  \
                __m256i i1 = dot_u8s8(dot_u8s8(_mm256_setzero_si256(), v0, x0), v1, x1);                  \
                const float s = a->s[g];                                                                  \
                const float dd0 = _cvtsh_ss(d0[g]), dd1 = _cvtsh_ss(d1[g]);                               \
                acc0 = _mm256_fmadd_ps(_mm256_cvtepi32_ps(i0), _mm256_set1_ps(dd0 * s), acc0);            \
                acc1 = _mm256_fmadd_ps(_mm256_cvtepi32_ps(i1), _mm256_set1_ps(dd1 * s), acc1);            \
                b0 = fmaf(dd0, a->bsum[g], b0);                                                           \
                b1 = fmaf(dd1, a->bsum[g], b1);                                                           \
            }                                                                                             \
            y[r] = hsum8(acc0) - bias * b0;                                                               \
            y[r + 1] = hsum8(acc1) - bias * b1;                                                           \
        }                                                                                                 \
        for (; r < r1; r++) {                                                                             \
            const uint8_t *q0 = m->q + (size_t)r * ng * GB;                                               \
            const uint8_t *f0 = m->f ? m->f + (size_t)r * ng * 8 : NULL; (void)f0;                     \
            const uint16_t *d0 = m->d + (size_t)r * ng;                                                   \
            __m256 acc0 = _mm256_setzero_ps();                                                            \
            float bl[8] = {0};                                                                            \
            for (int g = 0; g < ng; g++) {                                                                \
                const __m256i x0 = _mm256_load_si256((const __m256i *)(a->q + g * 64));                   \
                const __m256i x1 = _mm256_load_si256((const __m256i *)(a->q + g * 64 + 32));              \
                __m256i u0, u1;                                                                           \
                DECODE(q0 + (size_t)g * GB, f0 + (size_t)g * 8, lut, &u0, &u1);                                               \
                __m256i i0 = dot_u8s8(dot_u8s8(_mm256_setzero_si256(), u0, x0), u1, x1);                  \
                const float dd0 = _cvtsh_ss(d0[g]);                                                       \
                acc0 = _mm256_fmadd_ps(_mm256_cvtepi32_ps(i0), _mm256_set1_ps(dd0 * a->s[g]), acc0);      \
                if (g < ng8) bl[g & 7] = fmaf(dd0, a->bsum[g], bl[g & 7]);                                \
            }                                                                                             \
            float b0 = hsum8(_mm256_loadu_ps(bl));                                                        \
            for (int g = ng8; g < ng; g++) b0 = fmaf(_cvtsh_ss(d0[g]), a->bsum[g], b0);                   \
            y[r] = hsum8(acc0) - bias * b0;                                                               \
        }                                                                                                 \
    }

DEFINE_MV_LUT(mv_q4l, DEC_Q4, 32, 0)
DEFINE_MV_LUT(mv_q2l, DEC_Q2, 16, 0)
DEFINE_MV_LUT(mv_q3l, DEC_Q3, 24, 0)
DEFINE_MV_LUT(mv_q3p, DEC_Q3P, 16, 0)
DEFINE_MV_LUT(mv_q3p_coarse, DEC_Q2, 16, 8)

static void mv_q8g(const inz_qmat *m, const inz_act *a, float *y, int r0, int r1) {
    const int ng = m->ngroups;
    for (int r = r0; r < r1; r++) {
        const uint8_t *q0 = m->q + (size_t)r * ng * 64;
        const uint16_t *d0 = m->d + (size_t)r * ng;
        __m256 acc = _mm256_setzero_ps();
        float b0 = 0;
        for (int g = 0; g < ng; g++) {
            const __m256i x0 = _mm256_load_si256((const __m256i *)(a->q + g * 64));
            const __m256i x1 = _mm256_load_si256((const __m256i *)(a->q + g * 64 + 32));
            const __m256i u0 = _mm256_loadu_si256((const __m256i *)(q0 + g * 64));
            const __m256i u1 = _mm256_loadu_si256((const __m256i *)(q0 + g * 64 + 32));
#if HAVE_VNNI
            __m256i i0 = dot_u8s8(dot_u8s8(_mm256_setzero_si256(), u0, x0), u1, x1);
#else
            // u up to 255: undo the +128 bias first and use signed 16-bit products
            const __m256i o = _mm256_set1_epi8((char)0x80);
            __m256i s0 = _mm256_xor_si256(u0, o), s1 = _mm256_xor_si256(u1, o);
            __m256i p0 = _mm256_maddubs_epi16(_mm256_abs_epi8(s0), _mm256_sign_epi8(x0, s0));
            __m256i p1 = _mm256_maddubs_epi16(_mm256_abs_epi8(s1), _mm256_sign_epi8(x1, s1));
            const __m256i ones = _mm256_set1_epi16(1);
            __m256i i0 = _mm256_add_epi32(_mm256_madd_epi16(p0, ones), _mm256_madd_epi16(p1, ones));
#endif
            const float dd = _cvtsh_ss(d0[g]);
            acc = _mm256_fmadd_ps(_mm256_cvtepi32_ps(i0), _mm256_set1_ps(dd * a->s[g]), acc);
#if HAVE_VNNI
            b0 = fmaf(dd, a->bsum[g], b0);
#endif
        }
        y[r] = hsum8(acc) - 128.0f * b0;
    }
}

#if HAVE_VNNI
// multi-token Q8G: each 64-weight group is loaded once for up to 8 activations. Per token the arithmetic is the
// same, in the same order, as mv_q8g, so a token gets bit-identical results alone or in a batch.
static void mm_q8g(const inz_qmat *m, const inz_act *acts, int nt, float *Y, int ldy, int r0, int r1) {
    const int ng = m->ngroups;
    for (int t0 = 0; t0 < nt; t0 += 8) {
        const int tn = nt - t0 < 8 ? nt - t0 : 8;
        for (int r = r0; r < r1; r++) {
            const uint8_t *q0 = m->q + (size_t)r * ng * 64;
            const uint16_t *d0 = m->d + (size_t)r * ng;
            __m256 acc[8];
            float b[8] = {0};
            for (int t = 0; t < 8; t++) acc[t] = _mm256_setzero_ps();
            for (int g = 0; g < ng; g++) {
                const __m256i u0 = _mm256_loadu_si256((const __m256i *)(q0 + g * 64));
                const __m256i u1 = _mm256_loadu_si256((const __m256i *)(q0 + g * 64 + 32));
                const float dd = _cvtsh_ss(d0[g]);
                for (int t = 0; t < tn; t++) {
                    const inz_act *a = &acts[t0 + t];
                    const __m256i x0 = _mm256_load_si256((const __m256i *)(a->q + g * 64));
                    const __m256i x1 = _mm256_load_si256((const __m256i *)(a->q + g * 64 + 32));
                    const __m256i i0 = dot_u8s8(dot_u8s8(_mm256_setzero_si256(), u0, x0), u1, x1);
                    acc[t] = _mm256_fmadd_ps(_mm256_cvtepi32_ps(i0), _mm256_set1_ps(dd * a->s[g]), acc[t]);
                    b[t] = fmaf(dd, a->bsum[g], b[t]);
                }
            }
            for (int t = 0; t < tn; t++) Y[(size_t)(t0 + t) * ldy + r] = hsum8(acc[t]) - 128.0f * b[t];
        }
    }
}
#endif

/* ---- Q4N: NVFP4 repacked exactly ----
 * A 64-weight group holds 32 bytes of e2m1 codes (byte j: weight j in the low nibble, weight 32+j in the high one,
 * as Q4L) and 4 e4m3 scales, one per 16 weights. The LUT maps a code to 64 + 2*e2m1 (an integer in [52, 76]), so
 * weight = (lut - 64) * scale * gscale with gscale = 1 / (2 * global). Each 32-bit lane of VPDPBUSD covers 4 weights
 * of one 16-weight block: the accumulator starts from the activation's -64 * sum4 (the LUT bias), and the lanes of
 * a block are scaled by its e4m3 scale, decoded as an fp16 bit pattern (v << 7) + 0x2000 (positive normal e4m3). */
static const float k_e2m1[8] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};

static inline __m256 q4n_scale4(const uint8_t *s) {  // 4 e4m3 scales (one group) -> f32 lanes 0..3
    int32_t w;
    memcpy(&w, s, 4);
    const __m128i v = _mm_cvtepu8_epi16(_mm_cvtsi32_si128(w));
    return _mm256_cvtph_ps(_mm_add_epi16(_mm_slli_epi16(v, 7), _mm_set1_epi16(0x2000)));
}

static inline __m256 q4n_scale8(const uint8_t *s) {  // 8 e4m3 scales (two groups) -> f32
    const __m128i v = _mm_cvtepu8_epi16(_mm_loadl_epi64((const __m128i *)s));
    return _mm256_cvtph_ps(_mm_add_epi16(_mm_slli_epi16(v, 7), _mm_set1_epi16(0x2000)));
}

static void q4n_header(inz_qhdr *h, int rows, int cols, float gscale) {
    memset(h, 0, sizeof(*h));
    h->magic = QMAGIC;
    h->type = INZ_T_Q4N;
    h->rot_block = 0;
    h->rows = (uint32_t)rows;
    h->cols = (uint32_t)cols;
    h->bits = 4;
    for (int c = 0; c < 16; c++) {
        const int v = (int)(2.0f * k_e2m1[c & 7]);
        h->lut[c] = (uint8_t)(INZ_LUT_BIAS + ((c & 8) ? -v : v));
    }
    memcpy(h->pad2, &gscale, sizeof(float));
}

static bool e4m3_normal_positive(uint8_t v) { return !(v & 0x80) && ((v >> 3) & 15) != 0 && v != 0x7F; }

bool inz_pack_q4n(const uint8_t *packed, const uint8_t *scales, float global_scale, int rows, int cols, void *out) {
    if (cols % INZ_GROUP) inz_die("q4n: cols %d not a multiple of 64", cols);
    const int ng = cols / INZ_GROUP;
    memset(out, 0, inz_qmat_bytes(INZ_T_Q4N, rows, cols));
    q4n_header(out, rows, cols, 0.5f / global_scale);
    uint8_t *q = (uint8_t *)out + sizeof(inz_qhdr), *s8 = q + (size_t)rows * ng * 32;
    bool ok = true;
    for (int r = 0; r < rows; r++) {
        const uint8_t *pr = packed + (size_t)r * (cols / 2), *sr = scales + (size_t)r * (cols / 16);
        for (int g = 0; g < ng; g++) {
            uint8_t *dst = q + ((size_t)r * ng + g) * 32;
            for (int j = 0; j < 32; j++) {
                const int w0 = g * 64 + j, w1 = w0 + 32;
                const uint8_t c0 = (pr[w0 / 2] >> ((w0 & 1) * 4)) & 15, c1 = (pr[w1 / 2] >> ((w1 & 1) * 4)) & 15;
                dst[j] = (uint8_t)(c0 | (c1 << 4));
            }
            for (int b = 0; b < 4; b++) {
                const uint8_t v = sr[g * 4 + b];
                ok &= e4m3_normal_positive(v);
                s8[((size_t)r * ng + g) * 4 + b] = v;
            }
        }
    }
    return ok;
}

static uint8_t e4m3_encode_pos(float x) {  // nearest positive normal e4m3 (clamped to [2^-6, 448])
    if (!(x > 0)) x = 0x1p-6f;
    int e;
    float f = frexpf(x, &e);  // x = f * 2^e, f in [0.5, 1)
    e -= 1;
    int m = (int)lrintf((f * 2.0f - 1.0f) * 8.0f);
    if (m == 8) { m = 0; e++; }
    if (e < -6) { e = -6; m = 0; }
    if (e > 8 || (e == 8 && m > 6)) { e = 8; m = 6; }
    return (uint8_t)(((e + 7) << 3) | m);
}

// NVFP4 quantization of a float matrix (for tests and non-checkpoint tensors): per-tensor global scale chosen
// so the largest block scale lands near the top of the e4m3 range, nearest e2m1 code per weight.
static double quantize_q4n(const float *W, int rows, int cols, void *out) {
    float amax = 0;
    for (size_t i = 0; i < (size_t)rows * cols; i++) amax = fmaxf(amax, fabsf(W[i]));
    const float global = amax > 0 ? 448.0f * 6.0f / amax : 1.0f;
    uint8_t *packed = inz_calloc((size_t)rows * cols / 2, 1), *sc = inz_malloc((size_t)rows * cols / 16);
    double err = 0, en = 0;
    for (int r = 0; r < rows; r++)
        for (int b = 0; b < cols / 16; b++) {
            const float *w = W + (size_t)r * cols + b * 16;
            float bm = 0;
            for (int k = 0; k < 16; k++) bm = fmaxf(bm, fabsf(w[k]));
            const uint8_t s = e4m3_encode_pos(bm * global / 6.0f);
            sc[(size_t)r * (cols / 16) + b] = s;
            const float step = inz_e4m3_to_f32(s) / global;
            for (int k = 0; k < 16; k++) {
                const float t = w[k] / step;
                int best = 0;
                for (int c = 1; c < 8; c++) if (fabsf(fabsf(t) - k_e2m1[c]) < fabsf(fabsf(t) - k_e2m1[best])) best = c;
                const uint8_t code = (uint8_t)(best | (t < 0 && best ? 8 : 0));
                const size_t wi = (size_t)r * cols + b * 16 + k;
                packed[wi / 2] |= (uint8_t)(code << ((wi & 1) * 4));
                const float d = (t < 0 ? -k_e2m1[best] : k_e2m1[best]) * step - w[k];
                err += (double)d * d;
                en += (double)w[k] * w[k];
            }
        }
    inz_pack_q4n(packed, sc, global, rows, cols, out);
    free(packed);
    free(sc);
    return en > 0 ? err / en : 0;
}

static void mv_q4n(const inz_qmat *m, const inz_act *a, float *y, int r0, int r1) {
    const __m256i lut = _mm256_broadcastsi128_si256(_mm_loadu_si128((const __m128i *)m->lut));
    const int ng = m->ngroups;
    const __m256i p01 = _mm256_setr_epi32(0, 0, 0, 0, 1, 1, 1, 1), p23 = _mm256_setr_epi32(2, 2, 2, 2, 3, 3, 3, 3);
    const __m256i p45 = _mm256_setr_epi32(4, 4, 4, 4, 5, 5, 5, 5), p67 = _mm256_setr_epi32(6, 6, 6, 6, 7, 7, 7, 7);
    int r = r0;
    for (; r + 2 <= r1; r += 2) {
        const uint8_t *q0 = m->q + (size_t)r * ng * 32, *q1 = q0 + (size_t)ng * 32;
        const uint8_t *s0 = m->s8 + (size_t)r * ng * 4, *s1 = s0 + (size_t)ng * 4;
        __m256 acc0 = _mm256_setzero_ps(), acc1 = _mm256_setzero_ps();
        for (int g = 0; g + 2 <= ng; g += 2) {
            const __m256 sc0 = q4n_scale8(s0 + g * 4), sc1 = q4n_scale8(s1 + g * 4);
            for (int h = 0; h < 2; h++) {
                const int gg = g + h;
                const __m256i x0 = _mm256_load_si256((const __m256i *)(a->q + gg * 64));
                const __m256i x1 = _mm256_load_si256((const __m256i *)(a->q + gg * 64 + 32));
                const __m256i n0 = _mm256_load_si256((const __m256i *)(a->nb4 + gg * 16));
                const __m256i n1 = _mm256_load_si256((const __m256i *)(a->nb4 + gg * 16 + 8));
                const __m256 xs = _mm256_set1_ps(a->s[gg]);
                const __m256i pa = h ? p45 : p01, pb = h ? p67 : p23;
                __m256i u0, u1, v0, v1;
                decode_q4(q0 + (size_t)gg * 32, lut, &u0, &u1);
                decode_q4(q1 + (size_t)gg * 32, lut, &v0, &v1);
                const __m256i i0 = dot_u8s8(n0, u0, x0), i1 = dot_u8s8(n1, u1, x1);
                const __m256i j0 = dot_u8s8(n0, v0, x0), j1 = dot_u8s8(n1, v1, x1);
                acc0 = _mm256_fmadd_ps(_mm256_cvtepi32_ps(i0), _mm256_mul_ps(_mm256_permutevar8x32_ps(sc0, pa), xs), acc0);
                acc0 = _mm256_fmadd_ps(_mm256_cvtepi32_ps(i1), _mm256_mul_ps(_mm256_permutevar8x32_ps(sc0, pb), xs), acc0);
                acc1 = _mm256_fmadd_ps(_mm256_cvtepi32_ps(j0), _mm256_mul_ps(_mm256_permutevar8x32_ps(sc1, pa), xs), acc1);
                acc1 = _mm256_fmadd_ps(_mm256_cvtepi32_ps(j1), _mm256_mul_ps(_mm256_permutevar8x32_ps(sc1, pb), xs), acc1);
            }
        }
        float t0 = hsum8(acc0), t1 = hsum8(acc1);
        if (ng & 1) {  // last group alone (scalar scales)
            const int g = ng - 1;
            uint8_t idx[64];
            for (int rr = 0; rr < 2; rr++) {
                unpack_group(INZ_T_Q4L, (rr ? q1 : q0) + (size_t)g * 32, idx);
                const uint8_t *ss = (rr ? s1 : s0) + g * 4;
                float acc = 0;
                for (int j = 0; j < 64; j++)
                    acc += (float)((int)m->lut[idx[j]] - INZ_LUT_BIAS) * inz_e4m3_to_f32(ss[j / 16]) * (float)a->q[g * 64 + j];
                if (rr) t1 += acc * a->s[g]; else t0 += acc * a->s[g];
            }
        }
        y[r] = t0 * m->gscale;
        y[r + 1] = t1 * m->gscale;
    }
    for (; r < r1; r++) {  // odd last row: scalar reference path
        float row[16384];
        inz_dequantize_row(m, r, row);
        float acc = 0;
        for (int g = 0; g < ng; g++) {
            float sg = 0;
            for (int j = 0; j < 64; j++) sg += row[g * 64 + j] * (float)a->q[g * 64 + j];
            acc += sg * a->s[g];
        }
        y[r] = acc;
    }
}

static void mm_q4n(const inz_qmat *m, const inz_act *acts, int nt, float *Y, int ldy, int r0, int r1) {
    const __m256i lut = _mm256_broadcastsi128_si256(_mm_loadu_si128((const __m128i *)m->lut));
    const int ng = m->ngroups;
    const __m256i p01 = _mm256_setr_epi32(0, 0, 0, 0, 1, 1, 1, 1), p23 = _mm256_setr_epi32(2, 2, 2, 2, 3, 3, 3, 3);
    for (int t0 = 0; t0 < nt; t0 += 8) {
        const int tn = nt - t0 < 8 ? nt - t0 : 8;
        for (int r = r0; r < r1; r++) {
            const uint8_t *q0 = m->q + (size_t)r * ng * 32, *s0 = m->s8 + (size_t)r * ng * 4;
            __m256 acc[8];
            for (int t = 0; t < 8; t++) acc[t] = _mm256_setzero_ps();
            for (int g = 0; g < ng; g++) {
                __m256i u0, u1;
                decode_q4(q0 + (size_t)g * 32, lut, &u0, &u1);
                const __m256 scg = q4n_scale4(s0 + (size_t)g * 4);
                const __m256 S0 = _mm256_permutevar8x32_ps(scg, p01), S1 = _mm256_permutevar8x32_ps(scg, p23);
                for (int t = 0; t < tn; t++) {
                    const inz_act *a = &acts[t0 + t];
                    const __m256i x0 = _mm256_load_si256((const __m256i *)(a->q + g * 64));
                    const __m256i x1 = _mm256_load_si256((const __m256i *)(a->q + g * 64 + 32));
                    const __m256i i0 = dot_u8s8(_mm256_load_si256((const __m256i *)(a->nb4 + g * 16)), u0, x0);
                    const __m256i i1 = dot_u8s8(_mm256_load_si256((const __m256i *)(a->nb4 + g * 16 + 8)), u1, x1);
                    const __m256 part = _mm256_fmadd_ps(_mm256_cvtepi32_ps(i1), S1, _mm256_mul_ps(_mm256_cvtepi32_ps(i0), S0));
                    acc[t] = _mm256_fmadd_ps(part, _mm256_set1_ps(a->s[g]), acc[t]);
                }
            }
            for (int t = 0; t < tn; t++) Y[(size_t)(t0 + t) * ldy + r] = hsum8(acc[t]) * m->gscale;
        }
    }
}

void inz_matvec_rows(const inz_qmat *m, const inz_act *a, float *y, int r0, int r1) {
    switch (m->type) {
    case INZ_T_Q4N: mv_q4n(m, a, y, r0, r1); break;
    case INZ_T_Q4L: mv_q4l(m, a, y, r0, r1); break;
    case INZ_T_Q2L: mv_q2l(m, a, y, r0, r1); break;
    case INZ_T_Q3L: mv_q3l(m, a, y, r0, r1); break;
    case INZ_T_Q3P: if (m->coarse) mv_q3p_coarse(m, a, y, r0, r1); else mv_q3p(m, a, y, r0, r1); break;
    case INZ_T_Q8G: mv_q8g(m, a, y, r0, r1); break;
    default: inz_die("matvec: unsupported type %d", m->type);
    }
}

/* multi-token: decode a group once, reuse for up to 8 activations at a time */
#define DEFINE_MM_LUT(NAME, DECODE, GB, LUTOFF)                                                                   \
    static void NAME(const inz_qmat *m, const inz_act *acts, int nt, float *Y, int ldy, int r0, int r1) { \
        const __m256i lut = _mm256_broadcastsi128_si256(_mm_loadu_si128((const __m128i *)(m->lut + LUTOFF)));         \
        const int ng = m->ngroups;                                                                        \
        for (int t0 = 0; t0 < nt; t0 += 8) {                                                              \
            const int tn = nt - t0 < 8 ? nt - t0 : 8;                                                     \
            for (int r = r0; r < r1; r++) {                                                               \
                const uint8_t *q0 = m->q + (size_t)r * ng * GB;                                           \
                const uint8_t *f0 = m->f ? m->f + (size_t)r * ng * 8 : NULL; (void)f0;                 \
                const uint16_t *d0 = m->d + (size_t)r * ng;                                               \
                __m256 acc[8];                                                                            \
                /* bias sums: 8 lanes (group g in lane g % 8) for the first ng8 groups, then a scalar tail, \
                   exactly as the single-token kernel adds them, so one token gives bit-identical results */ \
                float bl[8][8] = {{0}}, bt[8] = {0};                                                      \
                const int ng8 = ng & ~7;                                                                  \
                for (int t = 0; t < 8; t++) acc[t] = _mm256_setzero_ps();                                 \
                for (int g = 0; g < ng; g++) {                                                            \
                    __m256i u0, u1;                                                                       \
                    DECODE(q0 + (size_t)g * GB, f0 + (size_t)g * 8, lut, &u0, &u1);                                           \
                    const float dd = _cvtsh_ss(d0[g]);                                                    \
                    for (int t = 0; t < tn; t++) {                                                        \
                        const inz_act *a = &acts[t0 + t];                                                 \
                        const __m256i x0 = _mm256_load_si256((const __m256i *)(a->q + g * 64));           \
                        const __m256i x1 = _mm256_load_si256((const __m256i *)(a->q + g * 64 + 32));      \
                        __m256i i0 = dot_u8s8(dot_u8s8(_mm256_setzero_si256(), u0, x0), u1, x1);          \
                        acc[t] = _mm256_fmadd_ps(_mm256_cvtepi32_ps(i0), _mm256_set1_ps(dd * a->s[g]), acc[t]); \
                        if (g < ng8) bl[t][g & 7] = fmaf(dd, a->bsum[g], bl[t][g & 7]);                   \
                        else bt[t] = fmaf(dd, a->bsum[g], bt[t]);                                         \
                    }                                                                                     \
                }                                                                                         \
                for (int t = 0; t < tn; t++) {                                                            \
                    float b = hsum8(_mm256_loadu_ps(bl[t]));                                              \
                    for (int g = ng8; g < ng; g++) b = fmaf(_cvtsh_ss(d0[g]), acts[t0 + t].bsum[g], b);   \
                    (void)bt;                                                                             \
                    Y[(size_t)(t0 + t) * ldy + r] = hsum8(acc[t]) - (float)INZ_LUT_BIAS * b;              \
                }                                                                                         \
            }                                                                                             \
        }                                                                                                 \
    }

DEFINE_MM_LUT(mm_q4l, DEC_Q4, 32, 0)
DEFINE_MM_LUT(mm_q2l, DEC_Q2, 16, 0)
DEFINE_MM_LUT(mm_q3l, DEC_Q3, 24, 0)
DEFINE_MM_LUT(mm_q3p, DEC_Q3P, 16, 0)
DEFINE_MM_LUT(mm_q3p_coarse, DEC_Q2, 16, 8)

void inz_matmul_rows(const inz_qmat *m, const inz_act *acts, int nt, float *Y, int ldy, int r0, int r1) {
    switch (m->type) {
    case INZ_T_Q4L: mm_q4l(m, acts, nt, Y, ldy, r0, r1); return;
    case INZ_T_Q2L: mm_q2l(m, acts, nt, Y, ldy, r0, r1); return;
    case INZ_T_Q3L: mm_q3l(m, acts, nt, Y, ldy, r0, r1); return;
    case INZ_T_Q3P: if (m->coarse) mm_q3p_coarse(m, acts, nt, Y, ldy, r0, r1); else mm_q3p(m, acts, nt, Y, ldy, r0, r1); return;
    case INZ_T_Q4N: mm_q4n(m, acts, nt, Y, ldy, r0, r1); return;
#if HAVE_VNNI
    case INZ_T_Q8G: mm_q8g(m, acts, nt, Y, ldy, r0, r1); return;
#endif
    default:
        for (int t = 0; t < nt; t++) {
            float tmp[1];
            (void)tmp;
            float *yt = Y + (size_t)t * ldy;
            inz_matvec_rows(m, &acts[t], yt, r0, r1);
        }
    }
}

/* ---- float matvecs (norms, small projections, routers) ---- */

void inz_matvec_f32_rows(const float *W, int cols, const float *x, float *y, int r0, int r1) {
    for (int r = r0; r < r1; r++) {
        const float *w = W + (size_t)r * cols;
        __m256 a0 = _mm256_setzero_ps(), a1 = _mm256_setzero_ps();
        int j = 0;
        for (; j + 16 <= cols; j += 16) {
            a0 = _mm256_fmadd_ps(_mm256_loadu_ps(w + j), _mm256_loadu_ps(x + j), a0);
            a1 = _mm256_fmadd_ps(_mm256_loadu_ps(w + j + 8), _mm256_loadu_ps(x + j + 8), a1);
        }
        float s = hsum8(_mm256_add_ps(a0, a1));
        for (; j < cols; j++) s += w[j] * x[j];
        y[r] = s;
    }
}

void inz_matvec_bf16_rows(const uint16_t *W, int cols, const float *x, float *y, int r0, int r1) {
    for (int r = r0; r < r1; r++) {
        const uint16_t *w = W + (size_t)r * cols;
        __m256 a0 = _mm256_setzero_ps(), a1 = _mm256_setzero_ps();
        int j = 0;
        for (; j + 16 <= cols; j += 16) {
            __m256i h = _mm256_loadu_si256((const __m256i *)(w + j));
            __m256 lo = _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_cvtepu16_epi32(_mm256_castsi256_si128(h)), 16));
            __m256 hi = _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_cvtepu16_epi32(_mm256_extracti128_si256(h, 1)), 16));
            a0 = _mm256_fmadd_ps(lo, _mm256_loadu_ps(x + j), a0);
            a1 = _mm256_fmadd_ps(hi, _mm256_loadu_ps(x + j + 8), a1);
        }
        float s = hsum8(_mm256_add_ps(a0, a1));
        for (; j < cols; j++) s += inz_bf16_to_f32(w[j]) * x[j];
        y[r] = s;
    }
}

// One 8-wide accumulator per row, in the same order as inz_matmul_f16_rows, so one token gives bit-identical
// results either way (decode vs speculative verify). Still bandwidth-bound: 8 FMAs per 16 bytes of weights.
void inz_matvec_f16_rows(const uint16_t *W, int cols, const float *x, float *y, int r0, int r1) {
    for (int r = r0; r < r1; r++) {
        const uint16_t *w = W + (size_t)r * cols;
        __m256 a0 = _mm256_setzero_ps();
        int j = 0;
        for (; j + 8 <= cols; j += 8)
            a0 = _mm256_fmadd_ps(_mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)(w + j))), _mm256_loadu_ps(x + j), a0);
        float s = hsum8(a0);
        for (; j < cols; j++) s += inz_f16_to_f32(w[j]) * x[j];
        y[r] = s;
    }
}

static inline __attribute__((always_inline)) void mm_f16_small(const uint16_t *W, int K, const float *X, int ldx,
                                                              int n, float *Y, int ldy, int r0, int r1) {
    for (int r = r0; r < r1; r++) {
        const uint16_t *w = W + (size_t)r * K;
        __m256 acc[3];
        for (int t = 0; t < n; t++) acc[t] = _mm256_setzero_ps();
        int k = 0;
        for (; k + 8 <= K; k += 8) {
            const __m256 v = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)(w + k)));
            for (int t = 0; t < n; t++)
                acc[t] = _mm256_fmadd_ps(v, _mm256_loadu_ps(X + (size_t)t * ldx + k), acc[t]);
        }
        for (int t = 0; t < n; t++) {
            float sum = hsum8(acc[t]);
            for (int j = k; j < K; j++) sum += inz_f16_to_f32(w[j]) * X[(size_t)t * ldx + j];
            Y[(size_t)t * ldy + r] = sum;
        }
    }
}

// Multi-token F16 matmul for the exact recipe: 3 weight rows x 4 tokens of accumulators in registers, tokens in
// blocks of 32 so the activations of a block stay in L2 while the thread's rows stream past them. K % 8 == 0.
void inz_matmul_f16_rows(const uint16_t *W, int K, const float *X, int ldx, int n, float *Y, int ldy, int r0, int r1) {
    if (n == 2) { mm_f16_small(W, K, X, ldx, 2, Y, ldy, r0, r1); return; }
    if (n == 3) { mm_f16_small(W, K, X, ldx, 3, Y, ldy, r0, r1); return; }
#define CV(p) _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)(p)))
    for (int tb = 0; tb < n; tb += 32) {
        const int te = tb + 32 < n ? tb + 32 : n;
        int r = r0;
        for (; r + 3 <= r1; r += 3) {
            const uint16_t *w0 = W + (size_t)r * K, *w1 = w0 + K, *w2 = w1 + K;
            int t = tb;
            for (; t + 4 <= te; t += 4) {
                const float *x0 = X + (size_t)t * ldx, *x1 = x0 + ldx, *x2 = x1 + ldx, *x3 = x2 + ldx;
                __m256 a00 = _mm256_setzero_ps(), a01 = a00, a02 = a00, a03 = a00, a10 = a00, a11 = a00, a12 = a00, a13 = a00;
                __m256 a20 = a00, a21 = a00, a22 = a00, a23 = a00;
                for (int k = 0; k < K; k += 8) {
                    const __m256 b0 = CV(w0 + k), b1 = CV(w1 + k), b2 = CV(w2 + k);
                    __m256 x = _mm256_loadu_ps(x0 + k);
                    a00 = _mm256_fmadd_ps(b0, x, a00); a10 = _mm256_fmadd_ps(b1, x, a10); a20 = _mm256_fmadd_ps(b2, x, a20);
                    x = _mm256_loadu_ps(x1 + k);
                    a01 = _mm256_fmadd_ps(b0, x, a01); a11 = _mm256_fmadd_ps(b1, x, a11); a21 = _mm256_fmadd_ps(b2, x, a21);
                    x = _mm256_loadu_ps(x2 + k);
                    a02 = _mm256_fmadd_ps(b0, x, a02); a12 = _mm256_fmadd_ps(b1, x, a12); a22 = _mm256_fmadd_ps(b2, x, a22);
                    x = _mm256_loadu_ps(x3 + k);
                    a03 = _mm256_fmadd_ps(b0, x, a03); a13 = _mm256_fmadd_ps(b1, x, a13); a23 = _mm256_fmadd_ps(b2, x, a23);
                }
                float *y = Y + (size_t)t * ldy + r;
                y[0] = hsum8(a00); y[1] = hsum8(a10); y[2] = hsum8(a20); y += ldy;
                y[0] = hsum8(a01); y[1] = hsum8(a11); y[2] = hsum8(a21); y += ldy;
                y[0] = hsum8(a02); y[1] = hsum8(a12); y[2] = hsum8(a22); y += ldy;
                y[0] = hsum8(a03); y[1] = hsum8(a13); y[2] = hsum8(a23);
            }
            for (; t < te; t++) inz_matvec_f16_rows(W, K, X + (size_t)t * ldx, Y + (size_t)t * ldy, r, r + 3);
        }
        for (int t = tb; t < te && r < r1; t++) inz_matvec_f16_rows(W, K, X + (size_t)t * ldx, Y + (size_t)t * ldy, r, r1);
    }
#undef CV
}

/* ------------------------------------------------------------------ */
/* Prefill GEMM: decode a 16-row weight tile once into int8 (VNNI       */
/* layout, scale per row and 256-block), then a 16x6 VPDPBUSD           */
/* micro-kernel keeps 12 int32 accumulators in registers.               */
/* ------------------------------------------------------------------ */

#define GT_ROWS 16
#define GT_TOK 6

void inz_xq_alloc(inz_xq *x, int n, int K) {
    memset(x, 0, sizeof(*x));
    x->cap = n;
    x->K = K;
    x->nkb = (K + INZ_KB - 1) / INZ_KB;
    x->q = inz_aligned(64, (size_t)n * K + 64);
    x->s = inz_aligned(64, sizeof(float) * (size_t)n * x->nkb + 64);
    x->sum = inz_aligned(64, sizeof(int32_t) * (size_t)n * x->nkb + 64);
    x->tmp = inz_aligned(64, sizeof(float) * (size_t)K + 64);
}

void inz_xq_free(inz_xq *x) {
    free(x->q); free(x->s); free(x->sum); free(x->tmp);
    memset(x, 0, sizeof(*x));
}

void inz_xq_begin(inz_xq *x, int n, int K, int rot_block, uint32_t rot_seed) {
    if (n > x->cap) inz_die("xq: capacity exceeded");
    x->n = n;
    x->K = K;
    x->nkb = (K + INZ_KB - 1) / INZ_KB;
    x->rot_block = rot_block;
    x->rot_seed = rot_seed;
}

void inz_xq_quantize(inz_xq *x, const float *X, int ldx, int n, int K, int rot_block, uint32_t rot_seed) {
    inz_xq_begin(x, n, K, rot_block, rot_seed);
    inz_xq_rows(x, X, ldx, 0, n, x->tmp);
}

void inz_xq_rows(inz_xq *x, const float *X, int ldx, int t0, int t1, float *tmp) {
    const int K = x->K, rot_block = x->rot_block;
    const uint32_t rot_seed = x->rot_seed;
    for (int t = t0; t < t1; t++) {
        const float *src = X + (size_t)t * ldx;
        if (rot_block > 1) {
            memcpy(tmp, src, sizeof(float) * K);
            inz_rotate(tmp, K, rot_block, rot_seed);
            src = tmp;
        }
        int8_t *q = x->q + (size_t)t * K;
        for (int b = 0; b < x->nkb; b++) {
            const int k0 = b * INZ_KB, k1 = k0 + INZ_KB < K ? k0 + INZ_KB : K;
            float amax = 0;
            for (int k = k0; k < k1; k++) amax = fmaxf(amax, fabsf(src[k]));
            const float sc = amax / 127.0f, inv = sc > 0 ? 1.0f / sc : 0;
            int32_t sum = 0;
            for (int k = k0; k < k1; k++) { int v = (int)lrintf(src[k] * inv); q[k] = (int8_t)v; sum += v; }
            x->s[(size_t)t * x->nkb + b] = sc;
            x->sum[(size_t)t * x->nkb + b] = sum;
        }
    }
}

size_t inz_gemm_tilebuf_bytes(int K) {
    int nkb = (K + INZ_KB - 1) / INZ_KB;
    // packed tile | row scales | one decoded float row
    // packed tile | row scales | scratch (staging tile + per-group factors; formerly one float row)
    return (size_t)GT_ROWS * K + sizeof(float) * (size_t)GT_ROWS * nkb + sizeof(float) * ((size_t)K + 64) + 4096 +
           (size_t)GT_ROWS * 64 + (size_t)GT_ROWS * (K / INZ_GROUP) * 2 * 4;
}

// Decode rows [r, r+16) into packed u8 [K/4][16][4] (+128 offset) with scales [nkb][16].
// 16x16 transpose of 32-bit elements: src rows of 64 bytes (16 dwords) -> dst[j] = column j (64 bytes)
static inline void tr8x8_epi32(__m256i r[8]) {
    __m256i t[8], u[8];
    for (int k = 0; k < 4; k++) {
        t[2 * k] = _mm256_unpacklo_epi32(r[2 * k], r[2 * k + 1]);
        t[2 * k + 1] = _mm256_unpackhi_epi32(r[2 * k], r[2 * k + 1]);
    }
    for (int k = 0; k < 2; k++) {
        u[4 * k + 0] = _mm256_unpacklo_epi64(t[4 * k + 0], t[4 * k + 2]);
        u[4 * k + 1] = _mm256_unpackhi_epi64(t[4 * k + 0], t[4 * k + 2]);
        u[4 * k + 2] = _mm256_unpacklo_epi64(t[4 * k + 1], t[4 * k + 3]);
        u[4 * k + 3] = _mm256_unpackhi_epi64(t[4 * k + 1], t[4 * k + 3]);
    }
    for (int k = 0; k < 4; k++) {
        r[k] = _mm256_permute2x128_si256(u[k], u[k + 4], 0x20);
        r[k + 4] = _mm256_permute2x128_si256(u[k], u[k + 4], 0x31);
    }
}

static void tr16x16_epi32(const uint8_t *src, uint8_t *dst) {
    for (int bi = 0; bi < 2; bi++)          // source row block (rows 8*bi..)
        for (int bj = 0; bj < 2; bj++) {    // source dword block (dwords 8*bj..)
            __m256i r[8];
            for (int k = 0; k < 8; k++) r[k] = _mm256_load_si256((const __m256i *)(src + (8 * bi + k) * 64 + 32 * bj));
            tr8x8_epi32(r);
            for (int k = 0; k < 8; k++) _mm256_store_si256((__m256i *)(dst + (8 * bj + k) * 64 + 32 * bi), r[k]);
        }
}

// Decode rows [r, r+16) into packed u8 [K/4][16][4] (+128 offset) with scales [nkb][16].
//
// The requantization from (learned level, fp16 group scale) to (int8, per-256-block scale) is folded into a
// per-group 16-entry table: lut_g[i] = round((lut[i] - bias) * d_g / s_block) + 128, built with one PMULHRSW on
// the int16 levels. The PSHUFB decode then yields final u8 values directly; 16 rows x 64 columns are staged and
// transposed (16x16 dwords) into the VNNI layout. The block scale assumes the table's largest |level| is reached
// by the block's largest group scale (true for Lloyd-Max tables, whose outer levels every group uses).
// Q4N tile: every 16-weight block has its own e4m3 scale, so a group needs four folded tables. Two of them fill
// the two 128-bit lanes of one PSHUFB table: the low nibbles of a group are weights 0..31 (blocks 0, 1) and the
// high nibbles weights 32..63 (blocks 2, 3).
static void gemm_decode_tile_q4n(const inz_qmat *m, int r, int nrows, uint8_t *pk, float *ws, float *rowbuf) {
    const int K = m->cols, ng = m->ngroups, nkb = (K + INZ_KB - 1) / INZ_KB, spb = INZ_KB / 16;
    uint8_t *stage = (uint8_t *)(((uintptr_t)rowbuf + 63) & ~(uintptr_t)63);
    int16_t *F = (int16_t *)(stage + GT_ROWS * 64);  // [16][ng*4]
    int16_t lv[16];
    int maxlev = 1;
    for (int c = 0; c < 16; c++) {
        lv[c] = (int16_t)(((int)m->lut[c] - INZ_LUT_BIAS) * 64);
        if (abs((int)m->lut[c] - INZ_LUT_BIAS) > maxlev) maxlev = abs((int)m->lut[c] - INZ_LUT_BIAS);
    }
    const __m256i lev16 = _mm256_loadu_si256((const __m256i *)lv);
    const int nsb = ng * 4;
    for (int i = 0; i < GT_ROWS; i++) {
        if (i >= nrows) {
            for (int b = 0; b < nkb; b++) ws[b * GT_ROWS + i] = 0;
            for (int k = 0; k < nsb; k++) F[i * nsb + k] = 0;
            continue;
        }
        const uint8_t *s = m->s8 + (size_t)(r + i) * nsb;
        for (int b = 0; b < nkb; b++) {
            const int k0 = b * spb, k1 = k0 + spb < nsb ? k0 + spb : nsb;
            float smax = 0;
            for (int k = k0; k < k1; k++) smax = fmaxf(smax, inz_e4m3_to_f32(s[k]));
            const float sc = smax * (float)maxlev / 127.0f;
            ws[b * GT_ROWS + i] = sc * m->gscale;
            const float inv = sc > 0 ? 512.0f / sc : 0;
            for (int k = k0; k < k1; k++) F[i * nsb + k] = (int16_t)lrintf(inz_e4m3_to_f32(s[k]) * inv);
        }
    }
    const __m256i c128 = _mm256_set1_epi16(128), m4 = _mm256_set1_epi8(0x0F);
    for (int g = 0; g < ng; g++) {
        for (int i = 0; i < GT_ROWS; i++) {
            __m256i u0, u1;
            if (i >= nrows) u0 = u1 = _mm256_set1_epi8((char)0x80);
            else {
                const int16_t *f = F + i * nsb + g * 4;
                const __m256i t0 = _mm256_add_epi16(_mm256_mulhrs_epi16(lev16, _mm256_set1_epi16(f[0])), c128);
                const __m256i t1 = _mm256_add_epi16(_mm256_mulhrs_epi16(lev16, _mm256_set1_epi16(f[1])), c128);
                const __m256i t2 = _mm256_add_epi16(_mm256_mulhrs_epi16(lev16, _mm256_set1_epi16(f[2])), c128);
                const __m256i t3 = _mm256_add_epi16(_mm256_mulhrs_epi16(lev16, _mm256_set1_epi16(f[3])), c128);
                const __m256i l01 = _mm256_permute4x64_epi64(_mm256_packus_epi16(t0, t1), 0xD8);
                const __m256i l23 = _mm256_permute4x64_epi64(_mm256_packus_epi16(t2, t3), 0xD8);
                const __m256i v = _mm256_loadu_si256((const __m256i *)(m->q + ((size_t)(r + i) * ng + g) * 32));
                u0 = _mm256_shuffle_epi8(l01, _mm256_and_si256(v, m4));
                u1 = _mm256_shuffle_epi8(l23, _mm256_and_si256(_mm256_srli_epi16(v, 4), m4));
            }
            _mm256_store_si256((__m256i *)(stage + i * 64), u0);
            _mm256_store_si256((__m256i *)(stage + i * 64 + 32), u1);
        }
        tr16x16_epi32(stage, pk + (size_t)g * 16 * GT_ROWS * 4);
    }
}

static void gemm_decode_tile(const inz_qmat *m, int r, int nrows, uint8_t *pk, float *ws, float *rowbuf) {
    if (m->type == INZ_T_Q4N) { gemm_decode_tile_q4n(m, r, nrows, pk, ws, rowbuf); return; }
    const int K = m->cols, ng = m->ngroups, nkb = (K + INZ_KB - 1) / INZ_KB, gpb = INZ_KB / INZ_GROUP;
    const bool q8 = m->type == INZ_T_Q8G;
    const int bias = q8 ? 128 : INZ_LUT_BIAS;
    const int lutoff = m->type == INZ_T_Q3P && m->coarse ? 8 : 0;
    const int nlev = q8 ? 0 : m->type == INZ_T_Q4L ? 16 : (m->type == INZ_T_Q2L || lutoff) ? 4 : 8;
    // scratch: staging tile (16 x 64 B) + per-(row, group) int16 factors
    uint8_t *stage = (uint8_t *)(((uintptr_t)rowbuf + 63) & ~(uintptr_t)63);
    int16_t *F = (int16_t *)(stage + GT_ROWS * 64);
    int maxlev = 127;
    __m256i lo16 = _mm256_setzero_si256(), hi16 = lo16;
    if (!q8) {
        int16_t lv[16] = {0};
        maxlev = 1;
        for (int i = 0; i < nlev; i++) {
            lv[i] = (int16_t)((int)m->lut[lutoff + i] - bias);
            if (abs(lv[i]) > maxlev) maxlev = abs(lv[i]);
        }
        int16_t a[16], b[16];
        for (int i = 0; i < 8; i++) { a[i] = a[i + 8] = (int16_t)(lv[i] * 64); b[i] = b[i + 8] = (int16_t)(lv[8 + i] * 64); }
        lo16 = _mm256_loadu_si256((const __m256i *)a);
        hi16 = _mm256_loadu_si256((const __m256i *)b);
    }
    // block scales and per-group factors F = round(512 * d_g / s_block)
    for (int i = 0; i < GT_ROWS; i++) {
        if (i >= nrows) {
            for (int b = 0; b < nkb; b++) ws[b * GT_ROWS + i] = 0;
            for (int g = 0; g < ng; g++) F[i * ng + g] = 0;
            continue;
        }
        const uint16_t *d = m->d + (size_t)(r + i) * ng;
        for (int b = 0; b < nkb; b++) {
            const int g0 = b * gpb, g1 = g0 + gpb < ng ? g0 + gpb : ng;
            float dmax = 0;
            for (int g = g0; g < g1; g++) dmax = fmaxf(dmax, fabsf(inz_f16_to_f32(d[g])));
            const float sc = dmax * (float)maxlev / 127.0f;
            ws[b * GT_ROWS + i] = sc;
            const float inv = sc > 0 ? 512.0f / sc : 0;
            for (int g = g0; g < g1; g++) F[i * ng + g] = (int16_t)lrintf(inz_f16_to_f32(d[g]) * inv);
        }
    }
    const __m256i c128 = _mm256_set1_epi16(128);
    for (int g = 0; g < ng; g++) {
        for (int i = 0; i < GT_ROWS; i++) {
            __m256i u0, u1;
            if (i >= nrows) {
                u0 = u1 = _mm256_set1_epi8((char)0x80);
            } else if (q8) {
                const uint8_t *q = m->q + ((size_t)(r + i) * ng + g) * 64;
                const __m256i f = _mm256_set1_epi16(F[i * ng + g]);
                __m256i v[4];
                for (int h = 0; h < 4; h++) {
                    __m256i w = _mm256_cvtepu8_epi16(_mm_loadu_si128((const __m128i *)(q + 16 * h)));
                    w = _mm256_slli_epi16(_mm256_sub_epi16(w, c128), 6);
                    v[h] = _mm256_add_epi16(_mm256_mulhrs_epi16(w, f), c128);
                }
                u0 = _mm256_permute4x64_epi64(_mm256_packus_epi16(v[0], v[1]), 0xD8);
                u1 = _mm256_permute4x64_epi64(_mm256_packus_epi16(v[2], v[3]), 0xD8);
            } else {
                const size_t gi = (size_t)(r + i) * ng + g;
                const __m256i f = _mm256_set1_epi16(F[i * ng + g]);
                const __m256i l = _mm256_add_epi16(_mm256_mulhrs_epi16(lo16, f), c128);
                const __m256i h = nlev > 8 ? _mm256_add_epi16(_mm256_mulhrs_epi16(hi16, f), c128) : l;
                const __m256i lut = _mm256_packus_epi16(l, h);  // per lane: entries 0..7 | 8..15
                const uint8_t *q = m->q + gi * m->qbytes;
                switch (m->type) {
                case INZ_T_Q4L: decode_q4(q, lut, &u0, &u1); break;
                case INZ_T_Q3L: decode_q3(q, lut, &u0, &u1); break;
                case INZ_T_Q2L: decode_q2(q, lut, &u0, &u1); break;
                default:
                    if (m->coarse) decode_q2(q, lut, &u0, &u1);
                    else decode_q3p(q, m->f + gi * 8, lut, &u0, &u1);
                    break;
                }
            }
            _mm256_store_si256((__m256i *)(stage + i * 64), u0);
            _mm256_store_si256((__m256i *)(stage + i * 64 + 32), u1);
        }
        tr16x16_epi32(stage, pk + (size_t)g * 16 * GT_ROWS * 4);
    }
}

void inz_gemm_rows(const inz_qmat *m, const inz_xq *x, float *Y, int ldy, int r0, int r1, uint8_t *tilebuf) {
    const int K = m->cols, nkb = x->nkb;
    uint8_t *pk = tilebuf;
    float *ws = (float *)(tilebuf + (size_t)GT_ROWS * K);
    float *rowbuf = (float *)(((uintptr_t)(ws + (size_t)GT_ROWS * nkb) + 63) & ~(uintptr_t)63);
    const __m256 c128 = _mm256_set1_ps(128.0f);
    for (int r = r0; r < r1; r += GT_ROWS) {
        const int nrows = r1 - r < GT_ROWS ? r1 - r : GT_ROWS;
        gemm_decode_tile(m, r, nrows, pk, ws, rowbuf);
        for (int t0 = 0; t0 < x->n; t0 += GT_TOK) {
            const int nt = x->n - t0 < GT_TOK ? x->n - t0 : GT_TOK;
            __m256 f[GT_TOK][2];
            for (int t = 0; t < GT_TOK; t++) f[t][0] = f[t][1] = _mm256_setzero_ps();
            for (int b = 0; b < nkb; b++) {
                const int k0 = b * INZ_KB, k1 = k0 + INZ_KB < K ? k0 + INZ_KB : K;
                __m256i a00 = _mm256_setzero_si256(), a01 = a00, a10 = a00, a11 = a00, a20 = a00, a21 = a00;
                __m256i a30 = a00, a31 = a00, a40 = a00, a41 = a00, a50 = a00, a51 = a00;
                const int8_t *xq0 = x->q + (size_t)t0 * K;
                const int8_t *xq1 = xq0 + (nt > 1 ? K : 0), *xq2 = xq0 + (size_t)(nt > 2 ? 2 : 0) * K;
                const int8_t *xq3 = xq0 + (size_t)(nt > 3 ? 3 : 0) * K, *xq4 = xq0 + (size_t)(nt > 4 ? 4 : 0) * K;
                const int8_t *xq5 = xq0 + (size_t)(nt > 5 ? 5 : 0) * K;
                for (int k = k0; k < k1; k += 4) {
                    const uint8_t *wp = pk + (size_t)(k >> 2) * GT_ROWS * 4;
                    const __m256i w0 = _mm256_load_si256((const __m256i *)wp);
                    const __m256i w1 = _mm256_load_si256((const __m256i *)(wp + 32));
                    __m256i bx;
#define GSTEP(acc0, acc1, xp)                                                  \
                    bx = _mm256_set1_epi32(*(const int32_t *)((xp) + k));      \
                    acc0 = dot_u8s8(acc0, w0, bx);                             \
                    acc1 = dot_u8s8(acc1, w1, bx);
                    GSTEP(a00, a01, xq0) GSTEP(a10, a11, xq1) GSTEP(a20, a21, xq2)
                    GSTEP(a30, a31, xq3) GSTEP(a40, a41, xq4) GSTEP(a50, a51, xq5)
#undef GSTEP
                }
                const __m256 s0 = _mm256_load_ps(ws + b * GT_ROWS), s1 = _mm256_load_ps(ws + b * GT_ROWS + 8);
                __m256i acc[GT_TOK][2] = {{a00, a01}, {a10, a11}, {a20, a21}, {a30, a31}, {a40, a41}, {a50, a51}};
                for (int t = 0; t < nt; t++) {
                    const float xs = x->s[(size_t)(t0 + t) * nkb + b];
                    const __m256 corr = _mm256_mul_ps(c128, _mm256_set1_ps((float)x->sum[(size_t)(t0 + t) * nkb + b]));
                    const __m256 vx = _mm256_set1_ps(xs);
                    f[t][0] = _mm256_fmadd_ps(_mm256_sub_ps(_mm256_cvtepi32_ps(acc[t][0]), corr), _mm256_mul_ps(s0, vx), f[t][0]);
                    f[t][1] = _mm256_fmadd_ps(_mm256_sub_ps(_mm256_cvtepi32_ps(acc[t][1]), corr), _mm256_mul_ps(s1, vx), f[t][1]);
                }
            }
            for (int t = 0; t < nt; t++) {
                float out[16];
                _mm256_storeu_ps(out, f[t][0]);
                _mm256_storeu_ps(out + 8, f[t][1]);
                float *y = Y + (size_t)(t0 + t) * ldy + r;
                for (int i = 0; i < nrows; i++) y[i] = out[i];
            }
        }
    }
}

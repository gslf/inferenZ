// inferenZ quantization formats ("LUT quants").
//
// Every weight matrix W[rows][K] is stored row-major in groups of 64 input columns.
// A group holds one fp16 scale d and 64 b-bit indices. An index selects one of 2^b
// levels from a per-tensor lookup table learned offline (weighted Lloyd-Max over the
// importance matrix), so the grid is non-uniform but decodes with a single PSHUFB.
// Optionally the input dimension is pre-rotated by a randomized Hadamard transform
// (block size rot_block), which Gaussianizes groups and removes outliers; the runtime
// applies the same transform to the activation once per input vector.
//
// LUT entries are stored biased (u8 in [1,127], bias INZ_LUT_BIAS) so the integer dot
// product maps directly to VPDPBUSD (u8 x s8); the bias is removed with per-group
// activation sums. Activations are int8 with one float scale per 64-group.
#ifndef INZ_QUANT_H
#define INZ_QUANT_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#define INZ_GROUP 64
#define INZ_LUT_BIAS 64

// tensor type ids inside the GGUF (>= 100 to stay clear of ggml ids)
enum {
    INZ_T_F32 = 0,      // ggml compatible
    INZ_T_F16 = 1,      // ggml compatible
    INZ_T_BF16 = 30,    // ggml compatible
    INZ_T_Q2L = 100,    // 2-bit LUT, 2.25 bpw
    INZ_T_Q3L = 101,    // 3-bit LUT, 3.25 bpw
    INZ_T_Q4L = 102,    // 4-bit LUT, 4.25 bpw
    INZ_T_Q8G = 103,    // int8 symmetric, 8.25 bpw
    INZ_T_Q3P = 106,    // 3-bit LUT, planar "Matryoshka": 2-bit coarse plane + scales first, 1-bit fine plane last
    INZ_T_Q4N = 107,    // NVFP4 repacked exactly: 4-bit e2m1 codes, one e4m3 scale per 16, per-tensor scale (4.5 bpw)
    INZ_T_FP8 = 110,    // raw e4m3 bytes without scales (row-gathered tables, e.g. the n-gram embedding)
    INZ_T_EXPERTS = 120 // per-layer packed routed experts (see model.h)
};

// header stored in front of every quantized matrix payload (64 bytes)
typedef struct {
    uint32_t magic;      // 'INZQ'
    uint16_t type;       // INZ_T_*
    uint16_t rot_block;  // 0 = no rotation, else Hadamard block (power of 2, divides K)
    uint32_t rows, cols;
    uint32_t rot_seed;   // random sign seed
    uint8_t bits;
    uint8_t pad[3];
    uint8_t lut[32];     // biased levels, lut[0 .. 2^bits-1] used (LUT types only)
    uint8_t pad2[8];
} inz_qhdr;

typedef struct {
    int type;
    int bits;
    int rows, cols;
    int ngroups;          // cols / 64
    int rot_block;
    uint32_t rot_seed;
    const uint8_t *lut;   // 32 bytes (lut[0..2^bits-1])
    const uint8_t *q;     // packed indices, rows * ngroups * qbytes
    const uint16_t *d;    // fp16 scales, rows * ngroups
    size_t qbytes;        // bytes of indices per group (Q3P: coarse plane only)
    const uint8_t *f;     // Q3P: fine plane (8 bytes per group), rows * ngroups * 8
    bool coarse;          // Q3P: fine plane absent -> decode with the 4-level coarse LUT (lut[8..11])
    const uint8_t *s8;    // Q4N: e4m3 scales, rows * ngroups * 4 (one per 16 weights)
    float gscale;         // Q4N: weight = (lut[code] - 64) * e4m3 * gscale
} inz_qmat;

// Q4N: repack an NVFP4 tensor (compressed-tensors layout: two e2m1 codes per byte, low nibble first; one e4m3
// scale per 16 inputs; w = e2m1 * scale / global) without changing any value. Returns false when a scale is not
// a positive normal e4m3 number (the kernels decode scales with two integer ops that assume it).
bool inz_pack_q4n(const uint8_t *packed, const uint8_t *scales, float global_scale, int rows, int cols, void *out);

// Q3P helpers: bytes of the coarse part (indices + scales) and of the fine plane, excluding the header.
size_t inz_q3p_coarse_bytes(int rows, int cols);
size_t inz_q3p_fine_bytes(int rows, int cols);
// Bind a Q3P matrix from separate parts (blob layout keeps all coarse parts before all fine planes).
void inz_qmat_view_q3p(inz_qmat *m, const void *hdr, const uint8_t *coarse, const uint8_t *fine);

size_t inz_qtype_group_bytes(int type);       // index bytes per group (without scale)
int inz_qtype_bits(int type);
size_t inz_qmat_bytes(int type, int rows, int cols);  // total payload incl. header
size_t inz_mat_bytes(int type, int rows, int cols);   // any storage type: F32/F16/BF16/FP8 raw, quantized with header
const char *inz_qtype_name(int type);
int inz_qtype_from_name(const char *s);

// Bind a view over a serialized payload (header + q + d).
bool inz_qmat_view(inz_qmat *m, const void *payload);

// ---------- quantization (offline) ----------
typedef struct {
    int type;
    int rot_block;          // 0 = auto (largest power of 2 <= 256 dividing cols), -1 = off
    uint32_t rot_seed;
    const float *imatrix;   // optional per-column importance (length cols), in the ORIGINAL basis
    int lut_iters;          // Lloyd-Max iterations (default 5)
    int lut_sample_groups;  // groups sampled for Lloyd-Max (default 8192)
    const float *fixed_lut; // optional fixed levels (integer units -63..63, sorted), skips learning
    int threads;            // 0 = OpenMP default, 1 = single thread (caller parallelizes)
} inz_quant_params;

// Quantize W (rows x cols, row-major f32) into out (inz_qmat_bytes bytes). Returns mean
// weighted squared error relative to weighted energy (for diagnostics).
double inz_quantize(const float *W, int rows, int cols, const inz_quant_params *p, void *out);
// Learn LUT levels on rotated rows (used to share one LUT across all experts of a layer).
void inz_learn_lut_rotated(const float *Wr, int rows, int cols, const float *imp, int bits, int iters, int sample_groups,
                           float *Lout);
int inz_auto_rot_block(int cols);
void inz_dequantize_row(const inz_qmat *m, int row, float *out);  // in the ROTATED basis
void inz_dequantize_row_orig(const inz_qmat *m, int row, float *out); // undo rotation

// ---------- runtime ----------
// Rotation: in-place randomized Hadamard on a vector of length n (blocks of size `block`).
void inz_rotate(float *x, int n, int block, uint32_t seed);
void inz_unrotate(float *x, int n, int block, uint32_t seed);

typedef struct {
    int n, ng;
    int8_t *q;        // n
    float *s;         // ng: per-group scale
    float *bsum;      // ng: s * sum(q)  (multiplied by the format bias to undo it)
    float *xrot;      // scratch for rotation, n
    int32_t *nb4;     // n/4: -64 * (sum of each 4 consecutive q), the LUT bias at the granularity of VPDPBUSD lanes (Q4N)
    int rot_block;    // rotation this activation was prepared for (0 = none)
    uint32_t rot_seed;
} inz_act;

void inz_act_alloc(inz_act *a, int n);
void inz_act_free(inz_act *a);
// prepare activation for matrices with the given rotation (block 0 = none)
void inz_act_quantize(inz_act *a, const float *x, int n, int rot_block, uint32_t rot_seed);

// y[r] = W[r] . x for rows [r0, r1)
void inz_matvec_rows(const inz_qmat *m, const inz_act *a, float *y, int r0, int r1);
// float32 / f16 / bf16 matvec for small dense tensors
void inz_matvec_f32_rows(const float *W, int cols, const float *x, float *y, int r0, int r1);
void inz_matvec_bf16_rows(const uint16_t *W, int cols, const float *x, float *y, int r0, int r1);
void inz_matvec_f16_rows(const uint16_t *W, int cols, const float *x, float *y, int r0, int r1);
// Y[t*ldy + r] = W[r] . X[t*ldx] for t < n, rows [r0, r1); cols % 8 == 0
void inz_matmul_f16_rows(const uint16_t *W, int cols, const float *X, int ldx, int n, float *Y, int ldy, int r0, int r1);

// multi-token variant: Y[t][r] = W[r] . X[t]  for t in [0,nt), rows [r0,r1)
// decodes each weight group once and reuses it for all nt activations (prefill / verify).
void inz_matmul_rows(const inz_qmat *m, const inz_act *acts, int nt, float *Y, int ldy, int r0, int r1);

// ---------- prefill GEMM (many tokens) ----------
// Activations for GEMM: int8 per token with one scale per 256-element block.
#define INZ_KB 256
typedef struct {
    int n, K, nkb, cap;
    int8_t *q;        // [n][K]
    float *s;         // [n][nkb]
    int32_t *sum;     // [n][nkb]  sum of q in the block (for the u8 offset correction)
    int rot_block;
    uint32_t rot_seed;
    float *tmp;       // rotation scratch [K]
} inz_xq;

void inz_xq_alloc(inz_xq *x, int n, int K);
void inz_xq_free(inz_xq *x);
// set shape/rotation, then quantize rows [t0,t1) (tmp: K floats scratch); lets callers parallelize over tokens
void inz_xq_begin(inz_xq *x, int n, int K, int rot_block, uint32_t rot_seed);
void inz_xq_rows(inz_xq *x, const float *X, int ldx, int t0, int t1, float *tmp);
// quantize rows X[t*ldx .. +K) for t < n (rotated as the target matrix expects)
void inz_xq_quantize(inz_xq *x, const float *X, int ldx, int n, int K, int rot_block, uint32_t rot_seed);
// Y[t*ldy + r] = W[r] . X_t for rows [r0, r1) and all n tokens (r0, r1 multiples of 16 except at the end)
void inz_gemm_rows(const inz_qmat *m, const inz_xq *x, float *Y, int ldy, int r0, int r1, uint8_t *tilebuf);
size_t inz_gemm_tilebuf_bytes(int K);

const char *inz_kernel_isa(void);

#endif

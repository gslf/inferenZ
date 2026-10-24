// Engine internals shared by engine.c and model_fwd.c (Qwen3.8-Flash-Next).
#ifndef INZ_ENGINE_INTERNAL_H
#define INZ_ENGINE_INTERNAL_H

#include <stdbool.h>
#include <stdint.h>

#include "engine.h"
#include "expert_cache.h"
#include "gguf.h"
#include "model.h"
#include "quant.h"
#include "threadpool.h"

// A weight matrix in any storage type: F16 / F32 (exact) or an inferenZ quantized format.
typedef struct {
    int type, rows, cols;
    inz_qmat q;              // quantized types
    const uint16_t *h;       // F16
    const float *f;          // F32
    int imx;                 // importance-matrix collection: entry index + 1 (0 = off)
} lin;

// Importance matrix (INZ_IMATRIX=file): mean squared input of every column of every matrix, for the quantizer.
typedef struct { char name[48]; int cols; uint64_t n; double *acc; } imx_entry;

// hyper-connection (gated residual) weights around one block
typedef struct {
    const float *norm;       // (1 + w) folded, 4H
    lin down, up;            // 320 x 4H, 4H x 320
    const float *inject;     // 4 x 4H (NULL for the final mixer)
} hcw;

typedef struct {
    hcw ahc, mhc;
    // Gated DeltaNet
    lin dn_qkv, dn_z, dn_out;
    const float *dn_ab;      // [96][H]: a rows, then b rows
    const float *dn_conv;    // [10240][4]
    float dn_negA[Q_DN_NV];  // -exp(A_log)
    const float *dn_dt, *dn_norm;
    // QSA attention
    lin at_q, at_kv, at_idx, at_o;
    const float *q_norm, *k_norm, *iq_norm, *ik_norm;
    // MoE
    lin router, sh_gu, sh_down;
    const float *sh_gate;
} q_layer;

typedef struct {
    bool present;
    lin key, value;
    const float *norm_key, *norm_query, *norm_conv, *conv;
    const uint8_t *table;    // FP8 rows of Q_NG_DIM bytes (mmap of the GGUF)
    uint64_t rows;
    float scale;             // per-tensor FP8 scale of the table
    int64_t mult[3];
    int64_t prime[Q_NG_HEADS], offset[Q_NG_HEADS];
} q_ple;

// MTP head: fc projections into a residual, one QSA + MoE layer, its own mixer; the LM head is shared.
typedef struct {
    bool present;
    lin fc_emb, fc_hid;
    const float *norm_emb, *norm_hid;  // (1 + w) folded: H, 4H
    hcw mixer;
    q_layer L;
} q_mtp;

#define Q_SPEC_MAX 3         // draft tokens per speculative step

// Recurrent and attention state of one sequence.
typedef struct {
    float *S;                // [Q_N_DN][Q_DN_NV][DK][DV] DeltaNet states
    float *conv;             // [Q_N_DN][3][Q_DN_CONV] last 3 conv inputs (oldest first)
    uint16_t *k, *v;         // [Q_N_ATTN + 1][ctx][Q_KV] f16 (the last one is the MTP layer's)
    float *ik;               // [Q_N_ATTN + 1][ctx][Q_IDX_HD] raw index keys
    float *bk;               // [Q_N_ATTN + 1][ctx/4][Q_IDX_HD] pooled block keys (normalized, roped)
    float *ple_conv;         // [9][Q_HCW] last 9 normalized PLE values (oldest first)
} q_state;

#define Q_MAX_T 512          // tokens per forward call (prefill chunk)

struct inz_engine {
    inz_opts opt;
    gguf_file *g;
    tokenizer *tok;
    tp_pool *tp;
    int n_layer;
    uint8_t *dense_mem;
    size_t dense_bytes;
    q_layer L[Q_N_LAYER];
    hcw mixer;
    q_ple ple;
    q_mtp mtp;
    // embeddings (rows read on demand) and LM head
    inz_qmat embed;
    int embed_type;          // INZ_T_F16 or a quantized type
    int embed_fd;
    uint64_t embed_offset;
    uint8_t *embed_hdr;
    lin head;                // exact head: in RAM, or (two-stage) only its geometry, rows read from the file
    lin head_screen;         // two-stage head: low-bit copy that scores the whole vocabulary
    bool two_stage;
    int head_fd;
    uint64_t head_off;       // file offset of the exact head's payload
    inz_qhdr head_hdr;       // header of a quantized exact head (geometry for row reads)
    double head_sigma;       // running RMS error of the screen on rescored rows (sets the candidate margin)
    uint64_t head_rows, head_cands;
    // experts
    ec_cache *ec;
    bool ec_all_resident;
    int32_t pred[Q_N_LAYER + 1][Q_TOPK];  // lookahead: experts predicted for each layer at the previous one
    bool pred_valid[Q_N_LAYER + 1];
    int expert_type;
    uint64_t blob_bytes, gu_bytes;
    // sequence state
    int ctx;
    q_state st;
    int n_past;
    int32_t *hist;
    q_state ck;              // recurrent state checkpoint (S, conv, ple_conv) taken at position ck_pos
    int ck_pos;              // -1: none
    float *ck_x;             // the final residual of position ck_pos - 1 (MTP input), 4H
    q_state snap[Q_SPEC_MAX];  // recurrent state after token t of a verify batch, for t < snap_n
    int snap_n;
    float *x_last;           // final residual (after layer 47) of position n_past - 1, 4H
    int mtp_valid;           // MTP cache slots [0, mtp_valid) are filled
    uint64_t state_bytes;    // attention cache + recurrent state, for the memory report
    // scratch for one forward call
    float *X, *Xn, *xin, *y, *inj, *tmp, *tmp2, *big;
    float *X2;               // MTP residual [Q_MAX_T][4H]
    float *Xm;               // MTP input rows [Q_MAX_T][4H]
    float *ex_x, *ex_gu, *ex_d;  // routed expert batch
    float *moe_out;          // [Q_MAX_T][Q_TOPK][H] weighted expert outputs, summed in rank order
    void *eb_items;          // decode expert batch (model_fwd.c)
    float *eb_gu, *eb_d;
    inz_act *eb_acts, *eb_xa;
    float *dn_ab, *dn_g, *dn_beta;  // DeltaNet a | b projections, decays, update strengths [Q_MAX_T][..]
    void *units;             // [Q_N_EXPERT] tokens routed to each expert (model_fwd.c)
    inz_act *acts;           // [Q_MAX_T] activation scratch (up to 4H wide)
    float **att_scr;         // per-thread attention scratch
    int32_t *att_sel;        // [Q_MAX_T][selected tokens] QSA selection
    int *att_nsel;
    float e4m3[256];         // FP8 decode table
    // prefill GEMM scratch
    uint8_t **gemm_tb;
    float **gemm_tmp;
    inz_xq gemm_x;
    // importance-matrix collection (INZ_IMATRIX=file)
    const char *imx_path;
    imx_entry *imx;
    int n_imx, cap_imx;
    double *imx_exp;         // [layers + MTP][512][H + FF]: experts' gate|up and down inputs
    uint64_t *imx_exp_n;     // [layers + MTP][512]
    int imx_layer;           // layer of the MoE block running
    const char *state;
    double t_loaded;
    char model_name[64];
    const char *chat_template;
};

void q_forward(inz_engine *e, const int32_t *tokens, int n, const int *logits_rows, int n_logits, float *out_logits);
void q_scratch_alloc(inz_engine *e);
void q_scratch_free(inz_engine *e);
void q_state_reset(inz_engine *e);
// bind a GGUF tensor as a lin
bool q_bind_lin(lin *m, int type, int rows, int cols, const uint8_t *data);
uint64_t q_lin_bytes(const lin *W);
int q_imx_register(inz_engine *e, const char *name, int cols);
void q_imx_write(inz_engine *e);
void q_head(inz_engine *e, const hcw *mixer, const float *X, int n, const int *logits_rows, int n_logits, float *out_logits);
void q_mtp_forward(inz_engine *e, const float *X, int ldx, const int32_t *next, int n, int p0, const int *logits_rows,
                   int n_logits, float *out_logits);

#endif

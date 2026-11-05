// Engine internals shared by engine.c and model_fwd.c (Laguna S 2.1).
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

// A weight matrix in any storage type: F16 / F32 (exact), F8T (exact FP8) or an inferenZ quantized format.
typedef struct {
    int type, rows, cols;
    inz_qmat q;              // quantized types and F8T
    const uint16_t *h;       // F16
    const float *f;          // F32
    int imx;                 // importance-matrix collection: entry index + 1 (0 = off)
} lin;

// Importance matrix (INZ_IMATRIX=file): mean squared input of every column of every matrix, for the quantizer.
typedef struct { char name[48]; int cols; uint64_t n; double *acc; } imx_entry;

// The routed experts of one blob class (the layers the official checkpoint quantized, the layers it kept in BF16)
typedef struct {
    ec_cache *ec;
    int type;                // storage type of the expert matrices
    uint64_t blob_bytes, gu_bytes;
    bool all_resident;
} q_xclass;

typedef struct {
    int nh;                  // query heads (48 full, 72 sliding)
    const float *attn_norm, *mlp_norm;   // H
    // attention
    lin at_q, at_k, at_v, at_o, at_g;    // q nh*128 x H, k/v 1024 x H, o H x nh*128, g nh x H
    const float *q_norm, *k_norm;        // 128
    // dense MLP (layer 0)
    lin d_gu, d_down;        // 2*12288 x H (gate rows, then up rows), H x 12288
    // MoE
    lin router;              // 256 x H, scores in f32
    const float *router_bias;  // e_score_correction_bias: shifts the selection, not the weights
    lin sh_gu, sh_down;      // 2*1024 x H, H x 1024
    q_xclass *xc;            // expert class of this layer
    int xl;                  // layer index inside that class's cache
} q_layer;

// MTP: Laguna has none; the engine keeps the hooks (always absent)
typedef struct { bool present; } q_mtp;

#define Q_SPEC_MAX 3         // draft tokens per speculative step (unused without an MTP head)

// Attention caches of one sequence: f16 K/V, positional. Full layers keep every position; sliding layers a ring of
// Q_RING positions (slot = position % Q_RING).
typedef struct {
    uint16_t *kf, *vf;       // [Q_N_FULL][ctx][Q_KV]
    uint16_t *ks, *vs;       // [Q_N_LAYER - Q_N_FULL][Q_RING][Q_KV]
} q_state;

#define Q_MAX_T 2048         // tokens per forward call (prefill chunk)

struct inz_engine {
    inz_opts opt;
    gguf_file *g;
    tokenizer *tok;
    tp_pool *tp;
    int n_layer;
    uint8_t *dense_mem;
    size_t dense_bytes;
    q_layer L[Q_N_LAYER];
    const float *norm_f;
    q_mtp mtp;
    // embeddings (rows read on demand) and LM head
    inz_qmat embed;
    int embed_type;          // INZ_T_F16 or a quantized type
    int embed_fd;
    uint64_t embed_offset;
    uint8_t *embed_hdr;
    lin head;
    // experts: two blob classes, each with its own cache
    q_xclass xc[2];
    int n_xc;
    int n_moe;               // MoE layers of the (possibly truncated) model
    int32_t pred[Q_N_LAYER][Q_TOPK];  // lookahead: experts predicted for each layer at the previous one
    bool pred_valid[Q_N_LAYER];
    // sequence state
    int ctx;
    q_state st;
    int n_past;
    int32_t *hist;
    int ck_pos;              // checkpoint: a position the caches can rewind to (-1: none)
    float *ck_x;             // unused (no MTP)
    int snap_n;              // unused (no recurrent state)
    float *x_last;           // final residual of position n_past - 1, H
    int mtp_valid;
    uint64_t state_bytes;    // attention caches, for the memory report
    // scratch for one forward call
    float *X, *Xn, *y, *tmp, *tmp2, *big;
    float *X2;               // final-norm rows for the head
    float *Xm;
    float *ex_x, *ex_gu, *ex_d;  // routed expert batch
    float *moe_out;          // [T][TOPK][H] weighted expert outputs, summed in rank order
    void *eb_items;          // decode expert batch (model_fwd.c)
    float *eb_gu, *eb_d;
    inz_act *eb_acts, *eb_xa;
    void *units;             // [Q_N_EXPERT] tokens routed to each expert (model_fwd.c)
    inz_act *acts;           // [Q_MAX_T] activation scratch (up to the widest quantized input)
    float **att_scr;         // per-thread attention scratch
    float rope_full[Q_ROPE_FULL_DIM / 2], rope_swa[Q_HD / 2];  // inverse frequencies
    // prefill GEMM scratch
    uint8_t **gemm_tb;
    float **gemm_tmp;
    inz_xq gemm_x;
    // importance-matrix collection (INZ_IMATRIX=file)
    const char *imx_path;
    imx_entry *imx;
    int n_imx, cap_imx;
    double *imx_exp;         // [MoE layers][256][H + FF]: experts' gate|up and down inputs
    uint64_t *imx_exp_n;     // [MoE layers][256]
    int imx_layer;           // MoE index of the block running
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
// LM head on rows of hidden states that are already normalized
void q_head(inz_engine *e, const float *Hn, int n, const int *logits_rows, int n_logits, float *out_logits);
// MTP hook (Laguna has no MTP head: never called)
void q_mtp_forward(inz_engine *e, const float *H, int ldh, bool residual, const int32_t *next, int n, int p0,
                   const int *logits_rows, int n_logits, float *out_logits);
// the YaRN / default RoPE inverse frequencies, computed as transformers does (float32)
void q_rope_init(inz_engine *e);

#endif

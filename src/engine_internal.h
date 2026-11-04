// Engine internals shared by engine.c and model_fwd.c (Nemotron 3.5 Super).
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

// One block: x += mixer(rmsnorm(x) * norm). Only the fields of its kind are bound.
typedef struct {
    int kind;                // QB_MAMBA, QB_MOE, QB_ATTN
    int idx;                 // index among the blocks of its kind (state slot, attention cache, expert cache layer)
    const float *norm;       // H
    // Mamba-2
    lin mb_in, mb_out;       // in_proj (z | x B C | dt) 18560 x H, out_proj H x 8192
    const float *mb_conv;    // [10240][4]
    const float *mb_conv_b;  // [10240]
    float mb_negA[Q_MB_NH];  // -exp(A_log)
    const float *mb_D, *mb_dt, *mb_norm;  // [128], [128], [8192]
    // attention
    lin at_q, at_k, at_v, at_o;
    // MoE
    lin router;              // 512 x H, scores in f32
    const float *router_bias;  // e_score_correction_bias: shifts the selection, not the weights
    lin lat_in, lat_out;     // 1024 x H, H x 1024
    lin sh_up, sh_down;      // 5376 x H, H x 5376
} q_layer;

// MTP: eh_proj of [enorm(embed(next)) | hnorm(hidden)], an attention block and a MoE block, a final norm; the LM
// head is shared.
typedef struct {
    bool present;
    lin eh;                  // H x 2H
    const float *enorm, *hnorm, *final_norm;
    q_layer at, moe;
} q_mtp;

#define Q_SPEC_MAX 3         // draft tokens per speculative step

// Recurrent and attention state of one sequence.
typedef struct {
    float *S;                // [Q_N_MAMBA][NH][HD][DS] SSM states
    float *conv;             // [Q_N_MAMBA][3][Q_MB_CONV] last 3 conv inputs (oldest first)
    uint16_t *k, *v;         // [Q_N_ATTN + 1][ctx][Q_KV] f16 (the last one is the MTP layer's)
} q_state;

#define Q_MAX_T 2048          // tokens per forward call (prefill chunk)

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
    // experts
    ec_cache *ec;
    bool ec_all_resident;
    int n_moe;               // MoE blocks of the (possibly truncated) model
    int32_t pred[Q_N_MOE + 1][Q_TOPK];  // lookahead: experts predicted for each MoE block at the previous one
    bool pred_valid[Q_N_MOE + 1];
    int expert_type;
    uint64_t blob_bytes, up_bytes;
    // sequence state
    int ctx;
    q_state st;
    int n_past;
    int32_t *hist;
    q_state ck;              // recurrent state checkpoint (S, conv) taken at position ck_pos
    int ck_pos;              // -1: none
    float *ck_x;             // the final residual of position ck_pos - 1 (MTP input), H
    q_state snap[Q_SPEC_MAX];  // recurrent state after token t of a verify batch, for t < snap_n
    int snap_n;
    float *x_last;           // final residual (after the last block) of position n_past - 1, H
    int mtp_valid;           // MTP cache slots [0, mtp_valid) are filled
    uint64_t state_bytes;    // attention cache + recurrent state, for the memory report
    // scratch for one forward call
    float *X, *Xn, *y, *tmp, *tmp2, *big;
    float *X2;               // MTP output hidden [Q_MAX_T][H] (input of a chained draft step)
    float *Xm;               // MTP input rows [Q_MAX_T][H]
    float *xlat, *moe_lat;   // MoE latent input [T][LAT], weighted latent outputs [T][TOPK][LAT]
    float *ex_x, *ex_up, *ex_d;  // routed expert batch
    void *eb_items;          // decode expert batch (model_fwd.c)
    float *eb_up, *eb_d;
    inz_act *eb_acts, *eb_xa;
    float *mb_dt;            // [Q_MAX_T][NH] Mamba step sizes
    void *units;             // [Q_N_EXPERT] tokens routed to each expert (model_fwd.c)
    inz_act *acts;           // [Q_MAX_T] activation scratch (up to Q_MB_INNER wide)
    float **att_scr;         // per-thread attention scratch
    // prefill GEMM scratch
    uint8_t **gemm_tb;
    float **gemm_tmp;
    inz_xq gemm_x;
    // importance-matrix collection (INZ_IMATRIX=file)
    const char *imx_path;
    imx_entry *imx;
    int n_imx, cap_imx;
    double *imx_exp;         // [MoE blocks + MTP][512][LAT + FF]: experts' up and down inputs
    uint64_t *imx_exp_n;     // [MoE blocks + MTP][512]
    int imx_layer;           // expert cache layer of the MoE block running
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
// MTP slots p0 .. p0+n-1: rows of H (main residuals when `residual`, else hidden states of a previous MTP step) with
// the tokens that follow them. Leaves the MTP output hidden states in e->X2.
void q_mtp_forward(inz_engine *e, const float *H, int ldh, bool residual, const int32_t *next, int n, int p0,
                   const int *logits_rows, int n_logits, float *out_logits);

#endif

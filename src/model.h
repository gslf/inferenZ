// NVIDIA Nemotron 3.5 Super (120B-A12B, the language model of Nemotron-3.5-Super-VL): fixed shapes and the
// inferenZ GGUF tensor layout (see docs/MODEL.md). The engine is specialized for exactly this model; shapes are
// compile-time constants and are verified against the GGUF metadata at load time.
#ifndef INZ_MODEL_H
#define INZ_MODEL_H

#include <stdint.h>

#define Q_ARCH "nemotron-3.5-super"
#define Q_N_LAYER 88                       // blocks: 40 Mamba-2, 40 MoE, 8 attention
#define Q_H 4096                           // hidden size
#define Q_VOCAB 131072
#define Q_IM_END 11                        // <|im_end|>: end of an assistant turn
#define Q_EOS 2                            // </s>
#define Q_RMS_EPS 1e-5f

// Block kinds, in order ('M' Mamba-2, 'E' MoE, '*' attention). Every block is x += mixer(rmsnorm(x)).
#define Q_PATTERN "MEMEMEM*EMEMEMEM*EMEMEMEM*EMEMEMEMEM*EMEMEMEMEM*EMEMEMEMEM*EMEMEMEMEM*EMEMEMEM*EMEMEMEME"
enum { QB_MAMBA = 'M', QB_MOE = 'E', QB_ATTN = '*' };
static inline int q_kind(int l) { return Q_PATTERN[l]; }
#define Q_N_MAMBA 40
#define Q_N_MOE 40
#define Q_N_ATTN 8

// Mamba-2 (SSD): 128 heads of 64 channels, 8 groups of B/C with a 128-wide state
#define Q_MB_NH 128
#define Q_MB_HD 64
#define Q_MB_INNER (Q_MB_NH * Q_MB_HD)     // 8192
#define Q_MB_NG 8
#define Q_MB_DS 128                        // state size
#define Q_MB_CONV (Q_MB_INNER + 2 * Q_MB_NG * Q_MB_DS)  // 10240 conv channels: x | B | C
#define Q_MB_KS 4                          // conv kernel
#define Q_MB_PROJ (Q_MB_INNER + Q_MB_CONV + Q_MB_NH)    // 18560 in_proj rows: z | x B C | dt
#define Q_MB_NORM_GROUP (Q_MB_INNER / Q_MB_NG)          // 1024: gated RMSNorm group

// Attention: grouped-query, no positional encoding
#define Q_NH 32
#define Q_NKV 2
#define Q_HD 128
#define Q_QD (Q_NH * Q_HD)                 // 4096
#define Q_KV (Q_NKV * Q_HD)                // 256

// LatentMoE: the router reads the hidden state; routed experts work in a 1024-wide latent space
#define Q_N_EXPERT 512
#define Q_TOPK 22
#define Q_FF 2688                          // routed expert: up 2688 x 1024, relu^2, down 1024 x 2688
#define Q_LAT 1024
#define Q_SHARED_FF 5376                   // shared expert on the hidden state: up, relu^2, down
#define Q_ROUTED_SCALE 5.0f

// MTP: one attention block and one MoE block, fed by eh_proj([enorm(embed(t+1)) | hnorm(h_t)])
#define Q_MTP_LAYERS 1

// One routed expert blob: up (FF x LAT), then down (LAT x FF), 4 KiB aligned.
#endif

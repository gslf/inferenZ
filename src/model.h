// Poolside Laguna S 2.1 (118B-A8B): fixed shapes and the inferenZ GGUF tensor layout (see docs/MODEL.md). The
// engine is specialized for exactly this model; shapes are compile-time constants and are verified against the GGUF
// metadata at load time.
#ifndef INZ_MODEL_H
#define INZ_MODEL_H

#include <stdint.h>

#define Q_ARCH "laguna-s-2.1"
#define Q_N_LAYER 48
#define Q_H 3072                           // hidden size
#define Q_VOCAB 100352
#define Q_IM_END 24                        // </assistant>: end of an assistant turn
#define Q_EOS 2                            // 〈|EOS|〉
#define Q_RMS_EPS 1e-6f

// Attention in every layer: grouped-query, 8 KV heads of 128. One layer in four (l % 4 == 0) attends to the whole
// context with 48 query heads and YaRN RoPE on half of each head; the others attend to the last 512 positions with
// 72 query heads and plain RoPE on the whole head. Every head's output is scaled by softplus(g_proj . x).
static inline int q_is_full(int l) { return l % 4 == 0; }
#define Q_N_FULL (Q_N_LAYER / 4)            // 12
static inline int q_nh(int l) { return q_is_full(l) ? 48 : 72; }
#define Q_NH_MAX 72
#define Q_NKV 8
#define Q_HD 128
#define Q_QD_MAX (Q_NH_MAX * Q_HD)          // 9216
#define Q_KV (Q_NKV * Q_HD)                 // 1024
#define Q_WINDOW 512                        // sliding window, current position included
#define Q_RING 4096                         // positions kept by the sliding layers' caches (window + rewind room)
// RoPE: full layers YaRN (theta 5e5, factor 128 over 8192 positions, beta 32/1) on the first 64 dims, the rotated
// part scaled by attention_factor; sliding layers theta 1e4 on all 128 dims
#define Q_ROPE_FULL_DIM 64
#define Q_ROPE_FULL_THETA 500000.0
#define Q_ROPE_FULL_FACTOR 128.0
#define Q_ROPE_FULL_ORIG 8192.0
#define Q_ROPE_FULL_ATTN 1.4852030263919618f
#define Q_ROPE_SWA_THETA 10000.0

// MLP: layer 0 dense SwiGLU, layers 1..47 MoE: 256 SwiGLU experts, top-10 by sigmoid score + selection bias, weights
// normalized and scaled by 2.5, plus one shared SwiGLU expert
#define Q_DENSE_FF 12288
#define Q_N_EXPERT 256
#define Q_TOPK 10
#define Q_FF 1024
#define Q_SHARED_FF 1024
#define Q_ROUTED_SCALE 2.5f
#define Q_N_MOE (Q_N_LAYER - 1)             // 47
static inline int q_moe_index(int l) { return l - 1; }

// No MTP head (the official drafter, DFlash, is a separate model)
#define Q_MTP_LAYERS 0

// One routed expert blob: gate|up rows (2*FF x H), then down (H x FF), 4 KiB aligned. Two blob classes (the official
// NVFP4 layers and the layers it keeps in BF16) live in two expert caches.
#endif

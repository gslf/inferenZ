// Qwen3.8-Flash-Next: fixed shapes and the inferenZ GGUF tensor layout (see docs/MODEL.md).
// The engine is specialized for exactly this model; shapes are compile-time constants and
// are verified against the GGUF metadata at load time.
#ifndef INZ_MODEL_H
#define INZ_MODEL_H

#include <stdint.h>

#define Q_ARCH "qwen3.8-flash-next"
#define Q_N_LAYER 48
#define Q_H 2560                           // hidden size
#define Q_HC 4                             // hyper-connection streams
#define Q_HCW (Q_HC * Q_H)                 // residual width 10240
#define Q_HC_RANK 320
#define Q_VOCAB 248320
#define Q_EOS 248044                       // <|endoftext|>: also the n-gram segment separator
#define Q_IM_END 248046
#define Q_RMS_EPS 1e-6f

// layer kinds: Gated DeltaNet, except every 4th layer (QSA attention)
static inline int q_is_attn(int l) { return l % 4 == 3; }
#define Q_N_ATTN (Q_N_LAYER / 4)            // 12
#define Q_N_DN (Q_N_LAYER - Q_N_ATTN)       // 36
static inline int q_attn_index(int l) { return l / 4; }       // for l % 4 == 3
static inline int q_dn_index(int l) { return l - l / 4; }     // for l % 4 != 3 (l/4 attention layers before it)

// Gated DeltaNet
#define Q_DN_NK 16
#define Q_DN_NV 48
#define Q_DN_DK 128
#define Q_DN_DV 128
#define Q_DN_KEY (Q_DN_NK * Q_DN_DK)        // 2048
#define Q_DN_VAL (Q_DN_NV * Q_DN_DV)        // 6144
#define Q_DN_CONV (2 * Q_DN_KEY + Q_DN_VAL) // 10240 conv channels (q | k | v)
#define Q_DN_KS 4                           // conv kernel

// QSA attention
#define Q_NH 24
#define Q_NKV 2
#define Q_HD 256
#define Q_ROT 64                            // rotary dims (partial factor 0.25)
#define Q_ROPE_THETA 10000000.0
#define Q_QPROJ (Q_NH * Q_HD * 2)           // 12288: q and output gate, interleaved per head
#define Q_KV (Q_NKV * Q_HD)                 // 512
#define Q_IDX_NH 4
#define Q_IDX_HD 128
#define Q_IDX_PROJ ((Q_IDX_NH + 1) * Q_IDX_HD)  // 640: 4 query heads | 1 key head
#define Q_IDX_BLOCK 4                       // compress ratio
#define Q_IDX_TOPK 512                      // blocks attended (budget 2048 tokens)

// MoE
#define Q_N_EXPERT 512
#define Q_TOPK 10
#define Q_FF 640
#define Q_SHARED_FF 640

// Per-layer n-gram embedding (layer 1)
#define Q_PLE_LAYER 1
#define Q_NG_HEADS 16                       // (ngram_size - 1) * heads_per_ngram
#define Q_NG_DIM 160                        // per head
#define Q_NG_BASE 20000000
#define Q_NG_SEED 1234
#define Q_PLE_CONV_K 4
#define Q_PLE_CONV_DIL 3

// n-gram hashing parameters, computed exactly as the official code does (verified against the checkpoint's
// buffers at conversion): three odd SplitMix64 multipliers, and per head the (h+1)-th prime >= Q_NG_BASE with its
// row offset in the concatenated table.
static inline uint64_t q_splitmix64(uint64_t v) {
    v += 0x9E3779B97F4A7C15ull;
    v = (v ^ (v >> 30)) * 0xBF58476D1CE4E5B9ull;
    v = (v ^ (v >> 27)) * 0x94D049BB133111EBull;
    return v ^ (v >> 31);
}
static inline int q_is_prime(int64_t v) {
    if (v < 2) return 0;
    if (v % 2 == 0) return v == 2;
    for (int64_t d = 3; d * d <= v; d += 2)
        if (v % d == 0) return 0;
    return 1;
}
static inline void q_ngram_params(int64_t mult[3], int64_t prime[Q_NG_HEADS], int64_t offset[Q_NG_HEADS]) {
    const uint64_t half = (uint64_t)(INT64_MAX / Q_VOCAB) / 2;
    const uint64_t base_seed = Q_NG_SEED;  // + 10007 * ple_layer_index (0: the only PLE layer)
    for (int i = 0; i < 3; i++) mult[i] = (int64_t)(2 * (q_splitmix64(base_seed + 0x9E3779B97F4A7C15ull * (uint64_t)(i + 1)) % half) + 1);
    int64_t p = Q_NG_BASE - 1, total = 0;
    for (int h = 0; h < Q_NG_HEADS; h++) {
        do p++; while (!q_is_prime(p));
        prime[h] = p;
        offset[h] = total;
        total += p;
    }
}

// MTP
#define Q_MTP_LAYERS 1

// One routed expert blob: gate|up rows (2*FF x H), then down (H x FF), 4 KiB aligned.
#endif

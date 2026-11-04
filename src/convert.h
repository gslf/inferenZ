#ifndef INZ_CONVERT_H
#define INZ_CONVERT_H

#include <stdbool.h>
#include <stdint.h>

typedef struct {
    const char *src_dir;
    const char *out_path;
    // storage types (INZ_T_*). The defaults copy the official checkpoint exactly: F8T for its FP8 tensors, Q4N for
    // its NVFP4 tensors, F16 for its BF16 matrices (BF16 has 7 mantissa bits, F16 has 10).
    int t_mamba;       // Mamba-2 in/out projections (FP8)
    int t_attn;        // attention projections (FP8)
    int t_shared;      // shared expert (NVFP4)
    int t_latent;      // LatentMoE projections (BF16)
    int t_router;      // MoE router (BF16)
    int t_mtp;         // MTP input projection and attention (BF16)
    int t_experts;     // routed experts (NVFP4)
    int t_head, t_embed;
    int max_layers;    // truncate (for tests)
    const char *imatrix;  // importance matrix (INZ_IMATRIX=file inz dump ...): weights the quantization error
    int threads;
} convert_opts;

void convert_opts_default(convert_opts *o);
int convert_run(const convert_opts *o);
uint64_t convert_expert_blob_bytes(int type);

#endif

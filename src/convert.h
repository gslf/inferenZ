#ifndef INZ_CONVERT_H
#define INZ_CONVERT_H

#include <stdbool.h>
#include <stdint.h>

typedef struct {
    const char *src_dir;
    const char *out_path;
    // storage types (INZ_T_*). NVFP4 experts are always repacked exactly as Q4N; the checkpoint's BF16 tensors get the
    // recipe's formats (F16 keeps them practically exact: BF16 has 7 mantissa bits, F16 has 10).
    int t_experts;        // NVFP4 routed experts (Q4N)
    int t_experts_bf16;   // routed experts the checkpoint keeps in BF16 (layers 40-47)
    int t_attn;           // attention projections
    int t_dense;          // layer 0 MLP
    int t_shared;         // shared experts
    int t_router;         // MoE router
    int t_head, t_embed;
    int max_layers;    // truncate (for tests)
    bool dense_only;   // no experts: a file of dense tensors for INZ_OVERRIDE studies
    const char *imatrix;  // importance matrix (INZ_IMATRIX=file inz dump ...): weights the quantization error
    int threads;
} convert_opts;

void convert_opts_default(convert_opts *o);
int convert_run(const convert_opts *o);
uint64_t convert_expert_blob_bytes(int type);

#endif

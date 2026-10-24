#ifndef INZ_CONVERT_H
#define INZ_CONVERT_H

#include <stdbool.h>
#include <stdint.h>

typedef struct {
    const char *src_dir;
    const char *out_path;
    // storage types (INZ_T_*): F16 keeps the BF16 source practically exact
    int t_dense;       // DeltaNet and QSA projections
    int t_hc;          // hyper-connection mixers
    int t_shared;      // shared expert
    int t_router;      // MoE router
    int t_ple;         // n-gram embedding projections (layer 1)
    int t_experts;     // routed experts
    int t_head, t_head_screen, t_embed;
    int max_layers;    // truncate (for tests)
    bool rot8;         // Hadamard-rotate Q8G layer tensors (spreads activation outliers before int8)
    bool no_ngram;     // skip the 51 GB n-gram table (only for truncated test models without layer 1)
    const char *imatrix;  // importance matrix (INZ_IMATRIX=file inz dump ...): weights the quantization error
    int threads;
} convert_opts;

void convert_opts_default(convert_opts *o);
int convert_run(const convert_opts *o);
uint64_t convert_expert_blob_bytes(int type);

#endif

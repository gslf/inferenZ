// Diagnostics: per-stage timing with byte accounting (roofline), per-layer breakdown,
// rolling speed windows, routing analytics, and a Chrome/Perfetto trace recorder.
//
// Everything is cheap enough to stay on in production: a stage costs two TSC reads.
#ifndef INZ_DIAG_H
#define INZ_DIAG_H

#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <x86intrin.h>

#include "json.h"

typedef enum {
    DG_EMBED,
    DG_ATTN_NORM,
    DG_ATTN_QKV,        // q/k/v + index q/k projections
    DG_ATTN_INDEX,      // MSA block scoring / top-k
    DG_ATTN_CORE,       // softmax(QK)V over selected blocks
    DG_ATTN_OUT,        // o_proj
    DG_FFN_NORM,
    DG_ROUTER,
    DG_PREFETCH,        // router lookahead + io submission
    DG_EXPERT_HIT,      // routed experts computed from RAM
    DG_EXPERT_WAIT,     // stalled on SSD
    DG_EXPERT_MISS,     // routed experts computed after arrival
    DG_SHARED,          // shared expert
    DG_DENSE_FFN,       // layers 0-2
    DG_LM_HEAD,
    DG_SAMPLE,
    DG_DRAFT,           // DSpark drafter
    DG_VERIFY_EXTRA,    // speculative bookkeeping
    DG_OTHER,
    DG_N_STAGES
} diag_stage;

typedef enum { PH_PREFILL, PH_DECODE, PH_VERIFY, PH_DRAFT, DG_N_PHASES } diag_phase;

#define DIAG_MAX_LAYERS 64
#define DIAG_WINDOW 128

typedef struct {
    uint64_t cycles;   // TSC cycles spent
    uint64_t calls;
    uint64_t bytes;    // weight/KV bytes streamed from RAM by this stage
    uint64_t ssd_bytes;
    uint64_t flops;    // multiply-adds * 2
} diag_counter;

typedef struct {
    // configuration / environment
    double tsc_hz;
    double peak_ram_gbs;     // measured at startup
    double peak_ssd_gbs;
    // global stage counters and per-layer counters
    diag_phase phase;
    diag_counter stage[DG_N_PHASES][DG_N_STAGES];
    diag_counter layer[DIAG_MAX_LAYERS][DG_N_STAGES];   // decode + verify only
    uint64_t phase_tokens[DG_N_PHASES];
    // token accounting
    uint64_t tokens_decoded, tokens_prefilled, tokens_drafted, tokens_accepted, verify_steps;
    uint64_t requests;
    // rolling windows (decode)
    double win_t[DIAG_WINDOW];   // seconds per emitted token
    int win_n, win_pos;
    double last_token_time;
    double prefill_tps_last, decode_tps_last, ttft_last;
    // routing analytics
    int n_layers, n_experts, topk;
    uint64_t *sel_hist;          // [layers * experts]
    uint64_t adj_pairs[DIAG_MAX_LAYERS];
    double adj_overlap[DIAG_MAX_LAYERS];
    int32_t *prev_sel;           // [layers * topk]
    bool *prev_valid;
    // miss policy decisions
    uint64_t miss_skipped, miss_rerouted, miss_deferred, miss_waited;
    uint64_t pred_total, pred_hits;  // router lookahead: predicted experts that were actually selected
    uint64_t pred_needed, pred_covered; // selected experts that had been predicted
    // routing trace capture (for offline cache simulation)
    FILE *route_trace;
} diag_state;

extern diag_state g_diag;

void diag_init(int n_layers, int n_experts, int topk);
void diag_reset(void);
static inline uint64_t diag_tsc(void) { return __rdtsc(); }

// stage timing: uint64_t t = diag_tsc(); ...; diag_add(DG_X, layer, t, bytes, flops);
void diag_add(diag_stage s, int layer, uint64_t t0, uint64_t bytes, uint64_t flops);
static inline void diag_set_phase(diag_phase p) { g_diag.phase = p; }
const char *diag_phase_name(diag_phase p);
void diag_add_ssd(diag_stage s, int layer, uint64_t bytes);
void diag_token_emitted(void);              // decode token produced
void diag_request_start(void);  // a new request: the idle gap before it is not an inter-token interval
void diag_routing(int layer, const int32_t *sel, const float *w, int k, int64_t pos);
const char *diag_stage_name(diag_stage s);

// Chrome trace recorder (Perfetto compatible): off unless armed.
void trace_arm(int n_tokens);               // record the next n decode tokens
bool trace_active(void);
void trace_event(const char *name, int tid, uint64_t t0, uint64_t t1, int layer);
void trace_token_boundary(void);
char *trace_json(size_t *len);              // malloc'ed Chrome trace JSON of the last capture

// Snapshot as JSON (for /v1/diag and the web monitor).
void diag_write_json(jw *w);
// Prometheus text exposition.
char *diag_prometheus(void);

// Measure peak RAM read bandwidth with the engine's pinned thread pool (~0.5 s).
struct tp_pool;
double diag_probe_ram(struct tp_pool *tp);

#endif

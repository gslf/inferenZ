// Public runtime API of the inferenZ engine (one model: Qwen3.8-Flash-Next).
#ifndef INZ_ENGINE_H
#define INZ_ENGINE_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "json.h"
#include "sampler.h"
#include "tokenizer.h"

typedef struct {
    const char *model_path;     // converted .gguf
    int threads;                // compute threads (0 = auto: P + E cores)
    int ctx;                    // max context tokens
    uint64_t cache_bytes;       // expert cache budget (0 = auto)
    double ram_reserve_gb;      // RAM left to the OS when sizing the cache automatically (0 = max(10 GB, 30%))
    int sqpoll_cpu;             // io_uring SQPOLL cpu: -2 auto (LP-E core), -1 off
    float cache_decay;          // LRFU decay per token (half-life 1/decay tokens)
    bool mlock;
    bool spec;                  // MTP speculative decoding (when the model has the MTP layer)
    int spec_k;                 // draft tokens per step (1..3)
    bool exact_head;            // ignore the screening head: full exact LM head in RAM (validation)
    int lookahead;              // decode: prefetch the experts the next layer's router predicts (0 = off)
    const char *hotlist;        // expert warm-start scores file: path, "auto" or "off"
    const char *route_trace;    // write routing trace for offline cache simulation (optional)
    int verbose;
} inz_opts;

void inz_opts_default(inz_opts *o);

typedef struct inz_engine inz_engine;

inz_engine *engine_open(const inz_opts *o, char *err, size_t errlen);
void engine_close(inz_engine *e);
tokenizer *engine_tokenizer(inz_engine *e);
const char *engine_model_name(inz_engine *e);
const char *engine_chat_template(inz_engine *e);
int engine_eos(inz_engine *e);      // <|im_end|>: end of an assistant turn
int engine_eot(inz_engine *e);      // <|endoftext|>

typedef struct {
    sampler_params sp;
    int max_tokens;
    const int32_t *stop_ids;
    int n_stop;
    bool spec;           // allow speculative decoding for this request
} gen_opts;

typedef struct {
    int prompt_tokens, cached_tokens, gen_tokens;
    double t_prefill, t_decode, ttft;
    int drafted, accepted;
    const char *finish_reason;   // "stop", "length", "abort"
} gen_result;

// Called for every generated token. Return false to stop generation.
typedef bool (*gen_cb)(void *ud, int token);

// Generate a continuation of `prompt`. The engine keeps the state of the previous call and reuses the longest
// common token prefix it can (chat clients resend the whole history): the attention cache is positional, and the
// recurrent DeltaNet state is checkpointed before the last prompt token of every request.
int engine_generate(inz_engine *e, const int32_t *prompt, int n_prompt, const gen_opts *g, gen_cb cb, void *ud,
                    gen_result *res);

// Evaluate tokens from an empty context and return logits for the last `last` positions (0 = all) as a
// malloc'ed [last][vocab] array. With `stepwise`, those positions go through the decode path one at a time.
float *engine_eval_logits_ex(inz_engine *e, const int32_t *tokens, int n, int last, bool stepwise);
float *engine_eval_logits(inz_engine *e, const int32_t *tokens, int n);
// validation of the speculative rewind (see engine.c); same output as engine_eval_logits_ex(..., last, false)
float *engine_eval_snapshots(inz_engine *e, const int32_t *tokens, int n, int last);
int engine_vocab(void);

// Diagnostics snapshot (JSON object written into w): state, diag, cache, memory, config.
void engine_diag_json(inz_engine *e, jw *w);
const char *engine_state(inz_engine *e);   // "idle", "prefill", "decode", "loading"

#endif

// Token sampling: temperature, top-k, top-p, min-p, repetition/presence/frequency penalties.
#ifndef INZ_SAMPLER_H
#define INZ_SAMPLER_H

#include <stdint.h>

typedef struct {
    float temperature;   // 0 => greedy
    float top_p;         // 1 => off
    int top_k;           // 0 => off
    float min_p;         // 0 => off
    float repeat_penalty;    // 1 => off (llama.cpp style, applied to last repeat_last_n tokens)
    int repeat_last_n;
    float presence_penalty;  // OpenAI style
    float frequency_penalty; // OpenAI style
    uint64_t seed;
} sampler_params;

typedef struct {
    sampler_params p;
    uint64_t rng;
    int vocab;
    // scratch
    int *idx;
    float *prob;
} sampler;

void sampler_default(sampler_params *p);
void sampler_init(sampler *s, const sampler_params *p, int vocab);
void sampler_free(sampler *s);
// logits are modified in place. history used for penalties (may be NULL).
int sampler_sample(sampler *s, float *logits, const int32_t *history, int n_history);
// Final sampling distribution (sparse, sorted by probability) after penalties, temperature, top-k,
// top-p and min-p. Returns the number of entries; *idx/*prob point into the sampler's buffers.
int sampler_dist(sampler *s, float *logits, const int32_t *history, int n_history, const int **idx, const float **prob);
// probability of `tok` under the same transformation (used for speculative acceptance)
float sampler_prob_of(sampler *s, const float *logits, int tok);
int argmax_f32(const float *x, int n);
// Descending values, lower token ID first on ties. k <= 0 selects all n.
int topk_f32(const float *x, int n, int k, int *idx);
// log-softmax helpers for diagnostics
float logsumexp_f32(const float *x, int n);
uint64_t sampler_rand_u64(sampler *s);
float sampler_rand_f(sampler *s);

#endif

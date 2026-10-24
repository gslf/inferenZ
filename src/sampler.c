#define _GNU_SOURCE
#include "sampler.h"

#include <immintrin.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

void sampler_default(sampler_params *p) {
    memset(p, 0, sizeof(*p));
    p->temperature = 0.7f;
    p->top_p = 0.95f;
    p->top_k = 40;
    p->min_p = 0.0f;
    p->repeat_penalty = 1.0f;
    p->repeat_last_n = 64;
    p->seed = 0x1234abcdULL;
}

void sampler_init(sampler *s, const sampler_params *p, int vocab) {
    s->p = *p;
    s->rng = p->seed ? p->seed : 0x9E3779B97F4A7C15ULL;
    s->vocab = vocab;
    s->idx = malloc(sizeof(int) * vocab);
    s->prob = malloc(sizeof(float) * vocab);
}

void sampler_free(sampler *s) {
    free(s->idx);
    free(s->prob);
    s->idx = NULL;
    s->prob = NULL;
}

uint64_t sampler_rand_u64(sampler *s) {
    // xorshift64*
    s->rng ^= s->rng >> 12;
    s->rng ^= s->rng << 25;
    s->rng ^= s->rng >> 27;
    return s->rng * 0x2545F4914F6CDD1DULL;
}

float sampler_rand_f(sampler *s) { return (float)((sampler_rand_u64(s) >> 40) * (1.0 / 16777216.0)); }

int argmax_f32(const float *x, int n) {
    int best = 0;
    float bv = x[0];
    int i = 0;
    __m256 vmax = _mm256_set1_ps(-INFINITY);
    for (; i + 8 <= n; i += 8) vmax = _mm256_max_ps(vmax, _mm256_loadu_ps(x + i));
    float m[8];
    _mm256_storeu_ps(m, vmax);
    for (int k = 0; k < 8; k++) if (m[k] > bv) bv = m[k];
    for (; i < n; i++) if (x[i] > bv) bv = x[i];
    for (i = 0; i < n; i++) if (x[i] == bv) { best = i; break; }
    return best;
}

float logsumexp_f32(const float *x, int n) {
    float m = x[argmax_f32(x, n)];
    double s = 0;
    for (int i = 0; i < n; i++) s += exp((double)(x[i] - m));
    return m + (float)log(s);
}

static void apply_penalties(sampler *s, float *logits, const int32_t *hist, int nh) {
    const sampler_params *p = &s->p;
    if (!hist || nh <= 0) return;
    if (p->repeat_penalty != 1.0f && p->repeat_penalty > 0) {
        int start = nh - p->repeat_last_n;
        if (start < 0 || p->repeat_last_n <= 0) start = 0;
        for (int i = start; i < nh; i++) {
            int t = hist[i];
            if (t < 0 || t >= s->vocab) continue;
            float l = logits[t];
            logits[t] = l > 0 ? l / p->repeat_penalty : l * p->repeat_penalty;
        }
    }
    if (p->presence_penalty != 0 || p->frequency_penalty != 0) {
        // count occurrences using prob[] as scratch (zeroed lazily)
        for (int i = 0; i < nh; i++) if (hist[i] >= 0 && hist[i] < s->vocab) s->prob[hist[i]] = 0;
        for (int i = 0; i < nh; i++) if (hist[i] >= 0 && hist[i] < s->vocab) s->prob[hist[i]] += 1;
        for (int i = 0; i < nh; i++) {
            int t = hist[i];
            if (t < 0 || t >= s->vocab || s->prob[t] < 0) continue;
            logits[t] -= p->presence_penalty + p->frequency_penalty * s->prob[t];
            s->prob[t] = -1;  // mark applied
        }
    }
}

static int cmp_desc(const void *a, const void *b, void *ctx) {
    const float *pr = ctx;
    float x = pr[*(const int *)a], y = pr[*(const int *)b];
    return x < y ? 1 : x > y ? -1 : 0;
}

// Candidates for sampling: the K highest logits (K = top_k, capped at 4096), sorted descending.
// Selection is O(vocab): a 1024-bin histogram of logits finds the threshold of the K-th value, then only the
// survivors are sorted. The softmax mass outside the candidates is returned in *tail_mass (relative to
// exp(max/T)), so top-p is computed against the full distribution, not only against the survivors.
static int select_top(sampler *s, const float *logits, int k, float inv_t, double *tail_mass) {
    const int n = s->vocab;
    float mx = logits[argmax_f32(logits, n)];
    float mn = mx - 40.0f / (inv_t > 0 ? inv_t : 1.0f);
    if (k <= 0 || k > 4096) k = 4096;
    enum { NB = 1024 };
    int hist[NB] = {0};
    const float scale = (NB - 1) / (mx - mn);
    for (int i = 0; i < n; i++) {
        float v = logits[i];
        if (v < mn) continue;
        hist[(int)((mx - v) * scale)]++;
    }
    int acc = 0, cut = NB - 1;
    for (int b = 0; b < NB; b++) { acc += hist[b]; if (acc >= k) { cut = b; break; } }
    const float th = mx - (float)(cut + 1) / scale;
    int c = 0;
    double tail = 0;
    for (int i = 0; i < n; i++) {
        float v = logits[i];
        if (v >= th && c < 8192) s->idx[c++] = i;
        else if (tail_mass && v >= mn) tail += exp((double)(v - mx) * inv_t);
    }
    qsort_r(s->idx, c, sizeof(int), cmp_desc, (void *)logits);
    if (c > k) {
        if (tail_mass) for (int i = k; i < c; i++) tail += exp((double)(logits[s->idx[i]] - mx) * inv_t);
        c = k;
    }
    if (tail_mass) *tail_mass = tail;
    return c;
}

int sampler_sample(sampler *s, float *logits, const int32_t *hist, int nh) {
    const int *idx;
    const float *prob;
    int c = sampler_dist(s, logits, hist, nh, &idx, &prob);
    double r = sampler_rand_f(s), acc = 0;
    for (int i = 0; i < c; i++) {
        acc += prob[i];
        if (r < acc) return idx[i];
    }
    return idx[c - 1];
}

int sampler_dist(sampler *s, float *logits, const int32_t *hist, int nh, const int **idx, const float **prob) {
    apply_penalties(s, logits, hist, nh);
    const sampler_params *p = &s->p;
    if (p->temperature <= 0.0f) {
        s->idx[0] = argmax_f32(logits, s->vocab);
        s->prob[0] = 1.0f;
        *idx = s->idx;
        *prob = s->prob;
        return 1;
    }
    const float inv_t = 1.0f / p->temperature;
    double tail = 0;
    int c = select_top(s, logits, p->top_k > 0 ? p->top_k : 0, inv_t, p->top_k > 0 ? NULL : &tail);
    float m = logits[s->idx[0]];
    double sum = tail;  // relative to the whole vocabulary: top-p must account for the tail
    for (int i = 0; i < c; i++) { s->prob[i] = expf((logits[s->idx[i]] - m) * inv_t); sum += s->prob[i]; }
    for (int i = 0; i < c; i++) s->prob[i] = (float)(s->prob[i] / sum);
    if (p->min_p > 0) {
        float th = s->prob[0] * p->min_p;
        int k = 1;
        while (k < c && s->prob[k] >= th) k++;
        c = k;
    }
    if (p->top_p > 0 && p->top_p < 1.0f) {
        double acc = 0;
        int k = 0;
        for (; k < c; k++) { acc += s->prob[k]; if (acc >= p->top_p) { k++; break; } }
        if (k < 1) k = 1;
        if (k < c) c = k;
    }
    double tot = 0;
    for (int i = 0; i < c; i++) tot += s->prob[i];
    for (int i = 0; i < c; i++) s->prob[i] = (float)(s->prob[i] / tot);
    *idx = s->idx;
    *prob = s->prob;
    return c;
}

float sampler_prob_of(sampler *s, const float *logits, int tok) {
    float t = s->p.temperature > 0 ? s->p.temperature : 1.0f;
    float m = logits[argmax_f32(logits, s->vocab)];
    double sum = 0;
    for (int i = 0; i < s->vocab; i++) sum += exp((logits[i] - m) / t);
    return (float)(exp((logits[tok] - m) / t) / sum);
}

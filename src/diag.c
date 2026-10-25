#define _GNU_SOURCE
#include "diag.h"

#include <immintrin.h>
#include <pthread.h>
#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <time.h>

#include "util.h"

diag_state g_diag;

static const char *stage_names[DG_N_STAGES] = {
    "embed", "attn_norm", "attn_qkv", "attn_index", "attn_core", "attn_out", "ffn_norm", "router",
    "prefetch", "expert_hit", "expert_wait", "expert_miss", "shared_expert", "dense_ffn", "lm_head",
    "sample", "draft", "verify_extra", "other"};

const char *diag_stage_name(diag_stage s) { return s < DG_N_STAGES ? stage_names[s] : "?"; }

static const char *phase_names[DG_N_PHASES] = {"prefill", "decode", "verify", "draft"};
const char *diag_phase_name(diag_phase p) { return p < DG_N_PHASES ? phase_names[p] : "?"; }

static double measure_tsc_hz(void) {
    double t0 = inz_now();
    uint64_t c0 = __rdtsc();
    while (inz_now() - t0 < 0.05) _mm_pause();
    double t1 = inz_now();
    uint64_t c1 = __rdtsc();
    return (double)(c1 - c0) / (t1 - t0);
}

void diag_init(int n_layers, int n_experts, int topk) {
    free(g_diag.sel_hist);
    free(g_diag.prev_sel);
    free(g_diag.prev_valid);
    double ram = g_diag.peak_ram_gbs, ssd = g_diag.peak_ssd_gbs;
    memset(&g_diag, 0, sizeof(g_diag));
    g_diag.peak_ram_gbs = ram;
    g_diag.peak_ssd_gbs = ssd;
    g_diag.tsc_hz = measure_tsc_hz();
    g_diag.n_layers = n_layers;
    g_diag.n_experts = n_experts;
    g_diag.topk = topk;
    g_diag.sel_hist = inz_calloc((size_t)n_layers * n_experts, sizeof(uint64_t));
    g_diag.prev_sel = inz_calloc((size_t)n_layers * topk, sizeof(int32_t));
    g_diag.prev_valid = inz_calloc((size_t)n_layers, sizeof(bool));
}

void diag_reset(void) {
    memset(g_diag.stage, 0, sizeof(g_diag.stage));
    memset(g_diag.layer, 0, sizeof(g_diag.layer));
    memset(g_diag.phase_tokens, 0, sizeof(g_diag.phase_tokens));
    g_diag.tokens_decoded = g_diag.tokens_prefilled = g_diag.tokens_drafted = g_diag.tokens_accepted = 0;
    g_diag.verify_steps = 0;
    g_diag.win_n = g_diag.win_pos = 0;
    memset(g_diag.adj_pairs, 0, sizeof(g_diag.adj_pairs));
    memset(g_diag.adj_overlap, 0, sizeof(g_diag.adj_overlap));
    if (g_diag.sel_hist) memset(g_diag.sel_hist, 0, sizeof(uint64_t) * (size_t)g_diag.n_layers * g_diag.n_experts);
    g_diag.miss_skipped = g_diag.miss_rerouted = g_diag.miss_deferred = g_diag.miss_waited = 0;
    g_diag.pred_total = g_diag.pred_hits = g_diag.pred_needed = g_diag.pred_covered = 0;
}

void diag_add(diag_stage s, int layer, uint64_t t0, uint64_t bytes, uint64_t flops) {
    uint64_t t1 = __rdtsc();
    uint64_t dt = t1 - t0;
    diag_counter *c = &g_diag.stage[g_diag.phase][s];
    c->cycles += dt;
    c->calls++;
    c->bytes += bytes;
    c->flops += flops;
    if (layer >= 0 && layer < DIAG_MAX_LAYERS && (g_diag.phase == PH_DECODE || g_diag.phase == PH_VERIFY)) {
        diag_counter *l = &g_diag.layer[layer][s];
        l->cycles += dt;
        l->calls++;
        l->bytes += bytes;
        l->flops += flops;
    }
    if (trace_active()) trace_event(stage_names[s], 0, t0, t1, layer);
}

void diag_add_ssd(diag_stage s, int layer, uint64_t bytes) {
    g_diag.stage[g_diag.phase][s].ssd_bytes += bytes;
    if (layer >= 0 && layer < DIAG_MAX_LAYERS && (g_diag.phase == PH_DECODE || g_diag.phase == PH_VERIFY))
        g_diag.layer[layer][s].ssd_bytes += bytes;
}

void diag_token_emitted(void) {
    double now = inz_now();
    if (g_diag.last_token_time > 0) {
        g_diag.win_t[g_diag.win_pos] = now - g_diag.last_token_time;
        g_diag.win_pos = (g_diag.win_pos + 1) % DIAG_WINDOW;
        if (g_diag.win_n < DIAG_WINDOW) g_diag.win_n++;
    }
    g_diag.last_token_time = now;
    g_diag.tokens_decoded++;
    trace_token_boundary();
}

void diag_routing(int layer, const int32_t *sel, const float *w, int k, int64_t pos) {
    if (layer < 0 || layer >= g_diag.n_layers || !g_diag.sel_hist) return;
    for (int i = 0; i < k; i++)
        if (sel[i] >= 0 && sel[i] < g_diag.n_experts) g_diag.sel_hist[(size_t)layer * g_diag.n_experts + sel[i]]++;
    int32_t *prev = g_diag.prev_sel + (size_t)layer * g_diag.topk;
    if (g_diag.prev_valid[layer] && k == g_diag.topk) {
        int inter = 0;
        for (int a = 0; a < k; a++)
            for (int b = 0; b < k; b++)
                if (prev[a] == sel[b]) { inter++; break; }
        g_diag.adj_pairs[layer]++;
        g_diag.adj_overlap[layer] += (double)inter / k;
    }
    if (k == g_diag.topk) {
        memcpy(prev, sel, sizeof(int32_t) * k);
        g_diag.prev_valid[layer] = true;
    }
    if (g_diag.route_trace) {
        // "pos layer e1 .. ek | w1 .. wk" (weights as used by the FFN, including the routed scale)
        fprintf(g_diag.route_trace, "%lld %d", (long long)pos, layer);
        for (int i = 0; i < k; i++) fprintf(g_diag.route_trace, " %d", sel[i]);
        fputs(" |", g_diag.route_trace);
        for (int i = 0; i < k; i++) fprintf(g_diag.route_trace, " %.4f", w[i]);
        fputc('\n', g_diag.route_trace);
    }
}

/* ---------------- trace ---------------- */

typedef struct {
    const char *name;
    uint64_t t0, t1;
    int16_t tid, layer;
} tev;

#define TRACE_CAP (1 << 20)
static tev *g_trace;
static atomic_int g_trace_n;
static int g_trace_tokens_left;
static bool g_trace_on;
static uint64_t g_trace_t0;
static pthread_mutex_t g_trace_mu = PTHREAD_MUTEX_INITIALIZER;
static char *g_trace_last;
static size_t g_trace_last_len;
static atomic_int g_trace_pending;

void trace_arm(int n_tokens) {
    atomic_store(&g_trace_pending, n_tokens > 0 ? n_tokens : 8);
}

bool trace_active(void) { return g_trace_on; }

void trace_event(const char *name, int tid, uint64_t t0, uint64_t t1, int layer) {
    if (!g_trace_on) return;
    int i = atomic_fetch_add(&g_trace_n, 1);
    if (i >= TRACE_CAP) return;
    g_trace[i] = (tev){name, t0, t1, (int16_t)tid, (int16_t)layer};
}

static void trace_finish(void) {
    int n = atomic_load(&g_trace_n);
    if (n > TRACE_CAP) n = TRACE_CAP;
    jw w;
    jw_init(&w);
    jw_obj(&w);
    jw_key(&w, "traceEvents");
    jw_arr(&w);
    const double us = 1e6 / (g_diag.tsc_hz > 0 ? g_diag.tsc_hz : 1e9);
    for (int i = 0; i < n; i++) {
        tev *e = &g_trace[i];
        jw_obj(&w);
        jw_kstr(&w, "name", e->name);
        jw_kstr(&w, "ph", "X");
        jw_kint(&w, "pid", 1);
        jw_kint(&w, "tid", e->tid);
        jw_knum(&w, "ts", (double)(e->t0 - g_trace_t0) * us);
        jw_knum(&w, "dur", (double)(e->t1 - e->t0) * us);
        if (e->layer >= 0) {
            jw_key(&w, "args");
            jw_obj(&w);
            jw_kint(&w, "layer", e->layer);
            jw_obj_end(&w);
        }
        jw_obj_end(&w);
    }
    jw_arr_end(&w);
    jw_kstr(&w, "displayTimeUnit", "ms");
    jw_obj_end(&w);
    pthread_mutex_lock(&g_trace_mu);
    free(g_trace_last);
    g_trace_last = w.buf;
    g_trace_last_len = w.len;
    pthread_mutex_unlock(&g_trace_mu);
}

void trace_token_boundary(void) {
    int pending = atomic_exchange(&g_trace_pending, 0);
    if (pending) {
        if (!g_trace) g_trace = inz_malloc(sizeof(tev) * TRACE_CAP);
        atomic_store(&g_trace_n, 0);
        g_trace_tokens_left = pending;
        g_trace_t0 = __rdtsc();
        g_trace_on = true;
        return;
    }
    if (!g_trace_on) return;
    if (--g_trace_tokens_left <= 0) {
        g_trace_on = false;
        trace_finish();
    }
}

char *trace_json(size_t *len) {
    pthread_mutex_lock(&g_trace_mu);
    char *r = NULL;
    if (g_trace_last) {
        r = inz_malloc(g_trace_last_len + 1);
        memcpy(r, g_trace_last, g_trace_last_len + 1);
        *len = g_trace_last_len;
    }
    pthread_mutex_unlock(&g_trace_mu);
    return r;
}

/* ---------------- snapshots ---------------- */

// Tokens per second over the last `last` tokens, extended backwards until the window spans at least 2 s:
// speculative decoding emits accepted drafts in bursts, and a few near-zero intervals alone would read as
// hundreds of tokens per second.
static double win_tps(int last) {
    if (g_diag.win_n == 0) return 0;
    int n = 0;
    double s = 0;
    while (n < g_diag.win_n && (n < last || s < 2.0)) {
        s += g_diag.win_t[(g_diag.win_pos - 1 - n + DIAG_WINDOW) % DIAG_WINDOW];
        n++;
    }
    return s > 0.25 ? n / s : 0;
}

void diag_request_start(void) { g_diag.last_token_time = 0; }

void diag_write_json(jw *w) {
    const double hz = g_diag.tsc_hz > 0 ? g_diag.tsc_hz : 1e9;
    jw_obj(w);
    jw_knum(w, "peak_ram_gbs", g_diag.peak_ram_gbs);
    jw_knum(w, "peak_ssd_gbs", g_diag.peak_ssd_gbs);
    jw_kint(w, "tokens_decoded", (int64_t)g_diag.tokens_decoded);
    jw_kint(w, "tokens_prefilled", (int64_t)g_diag.tokens_prefilled);
    jw_kint(w, "tokens_drafted", (int64_t)g_diag.tokens_drafted);
    jw_kint(w, "tokens_accepted", (int64_t)g_diag.tokens_accepted);
    jw_kint(w, "verify_steps", (int64_t)g_diag.verify_steps);
    jw_knum(w, "decode_tps_8", win_tps(8));
    jw_knum(w, "decode_tps_64", win_tps(64));
    jw_knum(w, "prefill_tps_last", g_diag.prefill_tps_last);
    jw_knum(w, "ttft_last", g_diag.ttft_last);
    jw_key(w, "phases");
    jw_obj(w);
    for (int ph = 0; ph < DG_N_PHASES; ph++) {
        double total = 0;
        for (int s = 0; s < DG_N_STAGES; s++) total += (double)g_diag.stage[ph][s].cycles;
        if (total == 0) continue;
        jw_key(w, phase_names[ph]);
        jw_obj(w);
        jw_knum(w, "seconds", total / hz);
        jw_kint(w, "tokens", (int64_t)g_diag.phase_tokens[ph]);
        jw_key(w, "stages");
        jw_arr(w);
        for (int s = 0; s < DG_N_STAGES; s++) {
            diag_counter *c = &g_diag.stage[ph][s];
            if (!c->calls) continue;
            double sec = (double)c->cycles / hz;
            jw_obj(w);
            jw_kstr(w, "name", stage_names[s]);
            jw_knum(w, "seconds", sec);
            jw_knum(w, "share", (double)c->cycles / total);
            jw_kint(w, "calls", (int64_t)c->calls);
            jw_knum(w, "ms_per_token", g_diag.phase_tokens[ph] ? sec * 1e3 / g_diag.phase_tokens[ph] : 0);
            jw_knum(w, "gb", (double)c->bytes / 1e9);
            jw_knum(w, "ssd_gb", (double)c->ssd_bytes / 1e9);
            jw_knum(w, "gbs", sec > 0 ? (double)c->bytes / sec / 1e9 : 0);
            jw_knum(w, "roofline", sec > 0 && g_diag.peak_ram_gbs > 0 ? (double)c->bytes / sec / 1e9 / g_diag.peak_ram_gbs : 0);
            jw_knum(w, "gflops", sec > 0 ? (double)c->flops / sec / 1e9 : 0);
            jw_obj_end(w);
        }
        jw_arr_end(w);
        jw_obj_end(w);
    }
    jw_obj_end(w);
    // per-layer: time share of attention / experts / wait
    jw_key(w, "layers");
    jw_arr(w);
    for (int l = 0; l < g_diag.n_layers && l < DIAG_MAX_LAYERS; l++) {
        double attn = 0, exp_ = 0, wait = 0, other = 0;
        for (int s = 0; s < DG_N_STAGES; s++) {
            double sec = (double)g_diag.layer[l][s].cycles / hz;
            if (s >= DG_ATTN_NORM && s <= DG_ATTN_OUT) attn += sec;
            else if (s == DG_EXPERT_HIT || s == DG_EXPERT_MISS) exp_ += sec;
            else if (s == DG_EXPERT_WAIT) wait += sec;
            else other += sec;
        }
        jw_obj(w);
        jw_knum(w, "attn", attn);
        jw_knum(w, "experts", exp_);
        jw_knum(w, "wait", wait);
        jw_knum(w, "other", other);
        jw_knum(w, "adj_overlap", g_diag.adj_pairs[l] ? g_diag.adj_overlap[l] / g_diag.adj_pairs[l] : 0);
        jw_obj_end(w);
    }
    jw_arr_end(w);
    jw_key(w, "miss_policy");
    jw_obj(w);
    jw_kint(w, "waited", (int64_t)g_diag.miss_waited);
    jw_kint(w, "skipped", (int64_t)g_diag.miss_skipped);
    jw_kint(w, "rerouted", (int64_t)g_diag.miss_rerouted);
    jw_kint(w, "deferred", (int64_t)g_diag.miss_deferred);
    jw_obj_end(w);
    jw_key(w, "lookahead");
    jw_obj(w);
    jw_knum(w, "precision", g_diag.pred_total ? (double)g_diag.pred_hits / g_diag.pred_total : 0);
    jw_knum(w, "recall", g_diag.pred_needed ? (double)g_diag.pred_covered / g_diag.pred_needed : 0);
    jw_obj_end(w);
    jw_obj_end(w);
}

char *diag_prometheus(void) {
    const double hz = g_diag.tsc_hz > 0 ? g_diag.tsc_hz : 1e9;
    size_t cap = 16384, n = 0;
    char *b = inz_malloc(cap);
#define P(...) do { int k = snprintf(b + n, cap - n, __VA_ARGS__); if (k > 0) n += (size_t)k; if (n >= cap - 512) { cap *= 2; b = realloc(b, cap); } } while (0)
    P("# HELP inz_tokens_total Tokens processed\n# TYPE inz_tokens_total counter\n");
    P("inz_tokens_total{kind=\"decode\"} %llu\n", (unsigned long long)g_diag.tokens_decoded);
    P("inz_tokens_total{kind=\"prefill\"} %llu\n", (unsigned long long)g_diag.tokens_prefilled);
    P("inz_tokens_total{kind=\"drafted\"} %llu\n", (unsigned long long)g_diag.tokens_drafted);
    P("inz_tokens_total{kind=\"accepted\"} %llu\n", (unsigned long long)g_diag.tokens_accepted);
    P("# TYPE inz_decode_tps gauge\ninz_decode_tps %.3f\n", win_tps(64));
    P("# TYPE inz_stage_seconds_total counter\n");
    for (int ph = 0; ph < DG_N_PHASES; ph++)
        for (int s = 0; s < DG_N_STAGES; s++)
            if (g_diag.stage[ph][s].calls)
                P("inz_stage_seconds_total{phase=\"%s\",stage=\"%s\"} %.6f\n", phase_names[ph], stage_names[s],
                  (double)g_diag.stage[ph][s].cycles / hz);
    P("# TYPE inz_stage_bytes_total counter\n");
    for (int ph = 0; ph < DG_N_PHASES; ph++)
        for (int s = 0; s < DG_N_STAGES; s++)
            if (g_diag.stage[ph][s].calls)
                P("inz_stage_bytes_total{phase=\"%s\",stage=\"%s\",src=\"ram\"} %llu\n"
                  "inz_stage_bytes_total{phase=\"%s\",stage=\"%s\",src=\"ssd\"} %llu\n",
                  phase_names[ph], stage_names[s], (unsigned long long)g_diag.stage[ph][s].bytes, phase_names[ph],
                  stage_names[s], (unsigned long long)g_diag.stage[ph][s].ssd_bytes);
#undef P
    return b;
}

/* ---------------- bandwidth probe ---------------- */

#include "threadpool.h"

typedef struct { const uint8_t *p; double sink[256]; } probe_ctx;

static void probe_task(void *ctx, int64_t b, int64_t e, int tid) {
    probe_ctx *pc = ctx;
    __m256i a0 = _mm256_setzero_si256(), a1 = a0, a2 = a0, a3 = a0;
    for (int64_t c = b; c < e; c++) {
        const uint8_t *p = pc->p + (size_t)c * (1u << 20);
        for (size_t i = 0; i < (1u << 20); i += 128) {
            a0 = _mm256_add_epi64(a0, _mm256_load_si256((const __m256i *)(p + i)));
            a1 = _mm256_add_epi64(a1, _mm256_load_si256((const __m256i *)(p + i + 32)));
            a2 = _mm256_add_epi64(a2, _mm256_load_si256((const __m256i *)(p + i + 64)));
            a3 = _mm256_add_epi64(a3, _mm256_load_si256((const __m256i *)(p + i + 96)));
        }
    }
    a0 = _mm256_add_epi64(_mm256_add_epi64(a0, a1), _mm256_add_epi64(a2, a3));
    pc->sink[tid & 255] += (double)_mm256_extract_epi64(a0, 0);
}

double diag_probe_ram(struct tp_pool *tp) {
    const int chunks = 768;  // 768 MiB working set, far beyond the LLC
    uint8_t *buf = inz_mmap_anon((size_t)chunks << 20, true);
    memset(buf, 1, (size_t)chunks << 20);
    probe_ctx pc = {.p = buf};
    double best = 0;
    for (int rep = 0; rep < 4; rep++) {
        double t0 = inz_now();
        tp_for(tp, chunks, 4, probe_task, &pc);
        double gbs = (double)chunks * (1u << 20) / (inz_now() - t0) / 1e9;
        if (gbs > best) best = gbs;
    }
    inz_munmap(buf, (size_t)chunks << 20);
    g_diag.peak_ram_gbs = best;
    return best;
}

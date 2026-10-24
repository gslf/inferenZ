#define _GNU_SOURCE
#include "expert_cache.h"

#include <errno.h>
#include <fcntl.h>
#include <liburing.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <unistd.h>

#include "util.h"

enum { S_EMPTY = 0, S_LOADING = 1, S_READY = 2 };
enum { PQ_DEMAND = 0, PQ_PREFETCH = 1 };

typedef struct {
    int32_t key;          // layer * n_expert + expert, -1 when empty
    uint8_t state;
    uint8_t level;        // level currently valid in the slot (0, COARSE, FULL)
    uint8_t want;         // level the in-flight read will reach
    uint8_t prefetched;   // loaded by a prefetch and not yet used
    uint8_t miss_counted; // demand miss already accounted by ec_request
    uint8_t queued;       // waiting in a submission queue
    int16_t pins;
    double t_submit;
} ec_slot;

typedef struct {
    int32_t *v;
    int head, tail, cap;
} fifo;

struct ec_cache {
    int fd;
    int n_layer, n_expert;
    uint64_t blob, coarse;
    uint64_t *off;
    uint8_t *arena;
    uint64_t arena_bytes;
    uint32_t n_slots;
    int32_t *slot_of;     // [L*E] slot index or -1
    ec_slot *slots;
    int32_t *free_list;
    uint32_t n_free;
    // LRFU state (kept for every key, also when not resident = frequency memory)
    float *crf;
    uint32_t *crf_tick;
    float lambda;
    uint32_t tick;
    uint64_t *use_count;
    // io
    struct io_uring ring;
    bool ring_ok, fixed_bufs;
    int qd, inflight;
    fifo pq[2];
    uint64_t rng;
    ec_stats st;
};

static void fifo_init(fifo *f, int cap) { f->v = inz_malloc(sizeof(int32_t) * cap); f->cap = cap; f->head = f->tail = 0; }
static int fifo_len(const fifo *f) { return (f->tail - f->head + f->cap) % f->cap; }
static void fifo_push(fifo *f, int32_t x) {
    if (fifo_len(f) == f->cap - 1) return;
    f->v[f->tail] = x;
    f->tail = (f->tail + 1) % f->cap;
}
static int32_t fifo_pop(fifo *f) {
    int32_t x = f->v[f->head];
    f->head = (f->head + 1) % f->cap;
    return x;
}

static uint64_t xs(uint64_t *s) { *s ^= *s << 13; *s ^= *s >> 7; *s ^= *s << 17; return *s; }

static float crf_now(const ec_cache *c, int32_t key) {
    uint32_t dt = c->tick - c->crf_tick[key];
    if (dt == 0 || c->lambda == 0) return c->crf[key];
    return c->crf[key] * exp2f(-c->lambda * (float)dt);
}

static void crf_touch(ec_cache *c, int32_t key, float w) {
    c->crf[key] = crf_now(c, key) + (w > 0 ? w : 0.05f);
    c->crf_tick[key] = c->tick;
    c->use_count[key]++;
}

ec_cache *ec_open(const ec_config *cfg, char *err, int errlen) {
    if (cfg->blob_bytes % 4096 || cfg->coarse_bytes % 4096) { snprintf(err, errlen, "expert blob sizes must be 4 KiB aligned"); return NULL; }
    ec_cache *c = inz_calloc(1, sizeof(*c));
    c->n_layer = cfg->n_layer;
    c->n_expert = cfg->n_expert;
    c->blob = cfg->blob_bytes;
    c->coarse = cfg->coarse_bytes;
    size_t nk = (size_t)cfg->n_layer * cfg->n_expert;
    c->off = inz_malloc(nk * sizeof(uint64_t));
    memcpy(c->off, cfg->offset, nk * sizeof(uint64_t));
    for (size_t i = 0; i < nk; i++)
        if (c->off[i] % 4096) { snprintf(err, errlen, "expert offset not 4 KiB aligned"); free(c->off); free(c); return NULL; }
    c->fd = open(cfg->path, O_RDONLY | O_DIRECT);
    if (c->fd < 0) {
        c->fd = open(cfg->path, O_RDONLY);
        if (c->fd < 0) { snprintf(err, errlen, "open %s: %s", cfg->path, strerror(errno)); free(c->off); free(c); return NULL; }
        LOGI("expert cache: O_DIRECT unavailable, using buffered reads");
    }
    uint64_t want = cfg->all_resident ? (uint64_t)nk * c->blob : cfg->cache_bytes;
    uint64_t slots = want / c->blob;
    if (slots > nk) slots = nk;
    if (slots < (uint64_t)cfg->n_expert) slots = (uint64_t)cfg->n_expert;
    c->n_slots = (uint32_t)slots;
    c->arena_bytes = slots * c->blob;
    c->arena = inz_mmap_anon(c->arena_bytes, true);
    c->slots = inz_calloc(slots, sizeof(ec_slot));
    c->free_list = inz_malloc(sizeof(int32_t) * slots);
    for (uint32_t i = 0; i < slots; i++) {
        c->slots[i].key = -1;
        c->free_list[i] = (int32_t)(slots - 1 - i);
    }
    c->n_free = (uint32_t)slots;
    c->slot_of = inz_malloc(sizeof(int32_t) * nk);
    for (size_t i = 0; i < nk; i++) c->slot_of[i] = -1;
    c->crf = inz_calloc(nk, sizeof(float));
    c->crf_tick = inz_calloc(nk, sizeof(uint32_t));
    c->use_count = inz_calloc(nk, sizeof(uint64_t));
    c->lambda = cfg->decay > 0 ? cfg->decay : 0.002f;
    c->qd = cfg->queue_depth > 0 ? cfg->queue_depth : 64;
    c->rng = 0x2545F4914F6CDD1Dull;
    fifo_init(&c->pq[0], (int)slots + 2);
    fifo_init(&c->pq[1], (int)slots + 2);
    c->st.layer_hits = inz_calloc(cfg->n_layer, sizeof(uint64_t));
    c->st.layer_misses = inz_calloc(cfg->n_layer, sizeof(uint64_t));
    c->st.slots = slots;

    struct io_uring_params p;
    memset(&p, 0, sizeof(p));
    if (cfg->sqpoll_cpu >= 0) {
        p.flags = IORING_SETUP_SQPOLL | IORING_SETUP_SQ_AFF;
        p.sq_thread_cpu = (unsigned)cfg->sqpoll_cpu;
        p.sq_thread_idle = 50;
    }
    int r = io_uring_queue_init_params((unsigned)c->qd, &c->ring, &p);
    if (r < 0 && cfg->sqpoll_cpu >= 0) {
        memset(&p, 0, sizeof(p));
        r = io_uring_queue_init_params((unsigned)c->qd, &c->ring, &p);
    }
    if (r < 0) { snprintf(err, errlen, "io_uring init: %s", strerror(-r)); ec_close(c); return NULL; }
    c->ring_ok = true;
    io_uring_register_files(&c->ring, &c->fd, 1);
    // fixed buffers need RLIMIT_MEMLOCK; plain reads are used otherwise
    uint64_t per = (1ull << 30) / c->blob * c->blob;
    if (per == 0) per = c->blob;
    uint32_t nb = (uint32_t)((c->arena_bytes + per - 1) / per);
    struct iovec *iov = inz_malloc(sizeof(struct iovec) * nb);
    for (uint32_t i = 0; i < nb; i++) {
        iov[i].iov_base = c->arena + (uint64_t)i * per;
        uint64_t rem = c->arena_bytes - (uint64_t)i * per;
        iov[i].iov_len = rem < per ? rem : per;
    }
    c->fixed_bufs = io_uring_register_buffers(&c->ring, iov, nb) == 0;
    free(iov);
    LOGI("expert cache: %u slots x %.2f MiB = %.2f GiB (%s%s%s)", c->n_slots, c->blob / 1048576.0,
         c->arena_bytes / 1073741824.0, cfg->sqpoll_cpu >= 0 && (p.flags & IORING_SETUP_SQPOLL) ? "sqpoll, " : "",
         c->fixed_bufs ? "fixed buffers" : "plain buffers", c->coarse ? ", two-level planar experts" : "");
    return c;
}

void ec_close(ec_cache *c) {
    if (!c) return;
    if (c->ring_ok) {
        while (c->inflight > 0) {
            struct io_uring_cqe *cqe;
            if (io_uring_wait_cqe(&c->ring, &cqe) < 0) break;
            io_uring_cqe_seen(&c->ring, cqe);
            c->inflight--;
        }
        io_uring_queue_exit(&c->ring);
    }
    if (c->fd >= 0) close(c->fd);
    inz_munmap(c->arena, c->arena_bytes);
    free(c->slots); free(c->free_list); free(c->slot_of); free(c->crf); free(c->crf_tick);
    free(c->use_count); free(c->off); free(c->pq[0].v); free(c->pq[1].v);
    free(c->st.layer_hits); free(c->st.layer_misses);
    free(c);
}

void ec_tick(ec_cache *c) { c->tick++; }

static int lat_bucket(double s) {
    double us = s * 1e6;
    int b = 0;
    while (us >= 1.0 && b < 15) { us *= 0.5; b++; }
    return b;
}

// byte range of the read that takes a slot from `level` to `want`
static void read_range(const ec_cache *c, const ec_slot *s, uint64_t *off, uint64_t *len) {
    if (s->level == EC_COARSE && s->want == EC_FULL) { *off = c->coarse; *len = c->blob - c->coarse; }
    else if (s->want == EC_COARSE && c->coarse) { *off = 0; *len = c->coarse; }
    else { *off = 0; *len = c->blob; }
}

static void complete(ec_cache *c, struct io_uring_cqe *cqe) {
    uint64_t data = io_uring_cqe_get_data64(cqe);
    if (data == UINT64_MAX) { c->inflight--; return; }  // bandwidth probe read
    int32_t slot = (int32_t)data;
    ec_slot *s = &c->slots[slot];
    uint64_t off, len;
    read_range(c, s, &off, &len);
    if (cqe->res != (int)len) {
        // short read / error: retry synchronously so correctness never depends on io_uring
        uint8_t *dst = c->arena + (uint64_t)slot * c->blob + off;
        ssize_t got = pread(c->fd, dst, len, (off_t)(c->off[s->key] + off));
        if (got != (ssize_t)len) inz_die("expert read failed (key %d): %s", s->key, strerror(errno));
    }
    double lat = inz_now() - s->t_submit;
    c->st.read_lat_sum_s += lat;
    c->st.lat_hist[lat_bucket(lat)]++;
    s->level = s->want;
    s->state = S_READY;
    c->inflight--;
}

static void submit_slot(ec_cache *c, int32_t slot) {
    ec_slot *s = &c->slots[slot];
    uint64_t off, len;
    read_range(c, s, &off, &len);
    struct io_uring_sqe *sqe = io_uring_get_sqe(&c->ring);
    uint8_t *dst = c->arena + (uint64_t)slot * c->blob + off;
    if (c->fixed_bufs) {
        uint64_t per = (1ull << 30) / c->blob * c->blob;
        io_uring_prep_read_fixed(sqe, 0, dst, (unsigned)len, c->off[s->key] + off, (int)(((uint64_t)slot * c->blob) / per));
    } else {
        io_uring_prep_read(sqe, 0, dst, (unsigned)len, c->off[s->key] + off);
    }
    sqe->flags |= IOSQE_FIXED_FILE;
    io_uring_sqe_set_data64(sqe, (uint64_t)slot);
    s->t_submit = inz_now();
    s->queued = 0;
    c->inflight++;
    c->st.reads++;
    c->st.bytes_read += len;
    if (len != c->blob) { if (s->level == EC_COARSE) c->st.upgrades++; else c->st.coarse_reads++; }
}

static void pump(ec_cache *c) {
    int n = 0;
    while (c->inflight < c->qd) {
        int32_t slot;
        if (fifo_len(&c->pq[PQ_DEMAND])) slot = fifo_pop(&c->pq[PQ_DEMAND]);
        else if (fifo_len(&c->pq[PQ_PREFETCH])) slot = fifo_pop(&c->pq[PQ_PREFETCH]);
        else break;
        if (c->slots[slot].state != S_LOADING || !c->slots[slot].queued) continue;  // already submitted
        submit_slot(c, slot);
        n++;
    }
    if (n) io_uring_submit(&c->ring);
}

static void reap_one(ec_cache *c, bool block) {
    struct io_uring_cqe *cqe;
    int r = block ? io_uring_wait_cqe(&c->ring, &cqe) : io_uring_peek_cqe(&c->ring, &cqe);
    if (r == 0) {
        complete(c, cqe);
        io_uring_cqe_seen(&c->ring, cqe);
    }
}

void ec_poll(ec_cache *c) {
    struct io_uring_cqe *cqe;
    while (io_uring_peek_cqe(&c->ring, &cqe) == 0) {
        complete(c, cqe);
        io_uring_cqe_seen(&c->ring, cqe);
    }
    pump(c);
}

// Redis-style approximated eviction: sample random slots, evict the lowest decayed score.
static int32_t pick_victim(ec_cache *c) {
    int32_t best = -1;
    float best_score = INFINITY;
    const int samples = 24;
    int valid = 0;
    for (int tries = 0; tries < samples * 8 && valid < samples; tries++) {
        int32_t s = (int32_t)(xs(&c->rng) % c->n_slots);
        ec_slot *sl = &c->slots[s];
        if (sl->state != S_READY || sl->pins > 0) continue;
        valid++;
        float sc = crf_now(c, sl->key);
        if (sl->prefetched) sc *= 0.5f;   // never-used prefetches are cheap to drop
        if (sc < best_score) { best_score = sc; best = s; }
    }
    if (best < 0) {
        for (uint32_t s = 0; s < c->n_slots; s++) {
            ec_slot *sl = &c->slots[s];
            if (sl->state != S_READY || sl->pins > 0) continue;
            float sc = crf_now(c, sl->key);
            if (sc < best_score) { best_score = sc; best = (int32_t)s; }
        }
    }
    return best;
}

static int32_t alloc_slot(ec_cache *c, bool demand) {
    if (c->n_free) return c->free_list[--c->n_free];
    for (int attempt = 0; attempt < 1000000; attempt++) {
        int32_t v = pick_victim(c);
        if (v >= 0) {
            ec_slot *s = &c->slots[v];
            if (s->prefetched) c->st.prefetch_wasted++;
            c->slot_of[s->key] = -1;
            memset(s, 0, sizeof(*s));
            s->key = -1;
            c->st.evictions++;
            return v;
        }
        if (!demand) return -1;
        if (c->inflight == 0) inz_die("expert cache too small: all slots pinned");
        reap_one(c, true);
    }
    return -1;
}

static int32_t start_load(ec_cache *c, int32_t key, bool demand, int level) {
    int32_t slot = alloc_slot(c, demand);
    if (slot < 0) return -1;
    ec_slot *s = &c->slots[slot];
    memset(s, 0, sizeof(*s));
    s->key = key;
    s->state = S_LOADING;
    s->want = (uint8_t)(c->coarse ? level : EC_FULL);
    s->prefetched = !demand;
    s->queued = 1;
    c->slot_of[key] = slot;
    fifo_push(&c->pq[demand ? PQ_DEMAND : PQ_PREFETCH], slot);
    return slot;
}

// Bring a resident coarse slot to full level (reads only the fine planes).
static void start_upgrade(ec_cache *c, int32_t slot, bool demand) {
    ec_slot *s = &c->slots[slot];
    if (s->state != S_READY || s->level >= EC_FULL || !c->coarse) return;
    s->state = S_LOADING;
    s->want = EC_FULL;
    s->queued = 1;
    fifo_push(&c->pq[demand ? PQ_DEMAND : PQ_PREFETCH], slot);
}

bool ec_prefetch(ec_cache *c, int layer, int expert, float weight, int level) {
    int32_t key = layer * c->n_expert + expert;
    (void)weight;
    int32_t slot = c->slot_of[key];
    if (slot >= 0) {
        ec_slot *s = &c->slots[slot];
        if (s->state == S_READY && s->level >= level) return true;
        if (s->state == S_READY && level == EC_FULL) { start_upgrade(c, slot, false); pump(c); }
        return false;
    }
    if (start_load(c, key, false, level) >= 0) c->st.prefetch_issued++;
    pump(c);
    return false;
}

int ec_resident_level(const ec_cache *c, int layer, int expert) {
    int32_t slot = c->slot_of[layer * c->n_expert + expert];
    if (slot < 0) return 0;
    const ec_slot *s = &c->slots[slot];
    // a slot being upgraded keeps its coarse data valid
    if (s->state == S_READY || (s->state == S_LOADING && s->level == EC_COARSE)) return s->level;
    return 0;
}

const uint8_t *ec_try_acquire(ec_cache *c, int layer, int expert, float weight, int min_level, int *level_out) {
    int32_t key = layer * c->n_expert + expert;
    int32_t slot = c->slot_of[key];
    if (slot < 0) return NULL;
    ec_slot *s = &c->slots[slot];
    int lvl = s->state == S_READY ? s->level : (s->state == S_LOADING && s->level == EC_COARSE ? EC_COARSE : 0);
    if (lvl < min_level || lvl == 0) return NULL;
    if (s->miss_counted) s->miss_counted = 0;          // arrived after a counted miss: not a new hit
    else {
        c->st.lookups++;
        c->st.hits++;
        c->st.layer_hits[layer]++;
        crf_touch(c, key, weight);
    }
    if (s->prefetched) { c->st.prefetch_useful++; s->prefetched = 0; }
    s->pins++;
    if (level_out) *level_out = lvl;
    return c->arena + (uint64_t)slot * c->blob;
}

void ec_request(ec_cache *c, int layer, int expert, float weight, int level) {
    int32_t key = layer * c->n_expert + expert;
    int32_t slot = c->slot_of[key];
    if (!c->coarse) level = EC_FULL;
    if (slot >= 0 && c->slots[slot].state == S_READY && c->slots[slot].level >= level) return;
    c->st.lookups++;
    c->st.misses++;
    c->st.layer_misses[layer]++;
    crf_touch(c, key, weight);
    if (slot >= 0) {
        ec_slot *s = &c->slots[slot];
        if (s->prefetched) { c->st.prefetch_late++; s->prefetched = 0; }
        if (s->state == S_READY) start_upgrade(c, slot, true);
        else if (s->queued) fifo_push(&c->pq[PQ_DEMAND], slot);  // promote a queued prefetch
        else if (s->want < level) { /* in flight at coarse: upgraded after it lands */ }
    } else {
        slot = start_load(c, key, true, level);
    }
    if (slot >= 0) c->slots[slot].miss_counted = 1;
    pump(c);
}

const uint8_t *ec_acquire(ec_cache *c, int layer, int expert, float weight, int level, int *level_out) {
    int32_t key = layer * c->n_expert + expert;
    if (!c->coarse) level = EC_FULL;
    const uint8_t *p = ec_try_acquire(c, layer, expert, weight, level, level_out);
    if (p) return p;
    ec_request(c, layer, expert, weight, level);
    double t0 = inz_now();
    for (;;) {
        int32_t slot = c->slot_of[key];
        ec_slot *s = &c->slots[slot];
        if (s->state == S_READY && s->level < level) start_upgrade(c, slot, true);
        if (s->state == S_READY && s->level >= level) break;
        pump(c);
        reap_one(c, true);
    }
    c->st.stall_s += inz_now() - t0;
    return ec_try_acquire(c, layer, expert, weight, level, level_out);
}

bool ec_wait_any(ec_cache *c) {
    pump(c);
    if (c->inflight == 0) return false;
    double t0 = inz_now();
    reap_one(c, true);
    c->st.stall_s += inz_now() - t0;
    ec_poll(c);
    return true;
}

void ec_release(ec_cache *c, int layer, int expert) {
    int32_t slot = c->slot_of[layer * c->n_expert + expert];
    if (slot >= 0 && c->slots[slot].pins > 0) c->slots[slot].pins--;
}

const ec_stats *ec_get_stats(const ec_cache *c) {
    ec_stats *st = (ec_stats *)&c->st;
    uint64_t res = 0;
    for (uint32_t i = 0; i < c->n_slots; i++) res += c->slots[i].state == S_READY;
    st->resident = res;
    return st;
}

void ec_reset_stats(ec_cache *c) {
    uint64_t *lh = c->st.layer_hits, *lm = c->st.layer_misses;
    memset(lh, 0, sizeof(uint64_t) * c->n_layer);
    memset(lm, 0, sizeof(uint64_t) * c->n_layer);
    uint64_t slots = c->st.slots;
    memset(&c->st, 0, sizeof(c->st));
    c->st.layer_hits = lh;
    c->st.layer_misses = lm;
    c->st.slots = slots;
}

void ec_usage_counts(const ec_cache *c, uint64_t *out) {
    memcpy(out, c->use_count, sizeof(uint64_t) * (size_t)c->n_layer * c->n_expert);
}

int ec_preload(ec_cache *c, const int32_t *pairs, int n_pairs) {
    int loaded = 0;
    for (int i = 0; i < n_pairs; i++) {
        int32_t key = pairs[2 * i] * c->n_expert + pairs[2 * i + 1];
        if (c->slot_of[key] >= 0) continue;
        if (!c->n_free) break;  // preload never evicts
        int32_t slot = start_load(c, key, false, EC_FULL);
        c->slots[slot].prefetched = 0;
        c->crf[key] = 1.0f + (float)(n_pairs - i) / (float)n_pairs;
        c->crf_tick[key] = c->tick;
        loaded++;
        if ((loaded & 63) == 0) ec_poll(c);
    }
    while (c->inflight > 0 || fifo_len(&c->pq[0]) || fifo_len(&c->pq[1])) {
        pump(c);
        if (c->inflight) reap_one(c, true);
    }
    return loaded;
}

uint64_t ec_slot_count(const ec_cache *c) { return c->n_slots; }

double ec_probe_bandwidth(ec_cache *c, int n) {
    if ((uint32_t)n > c->n_free) n = (int)c->n_free;
    if (n <= 0) return 0;
    const uint64_t nk = (uint64_t)c->n_layer * c->n_expert;
    double t0 = inz_now();
    int submitted = 0, done = 0;
    uint64_t r = 0x9E3779B97F4A7C15ull;
    while (done < n) {
        while (submitted < n && submitted - done < c->qd) {
            struct io_uring_sqe *sqe = io_uring_get_sqe(&c->ring);
            uint64_t key = xs(&r) % nk;
            uint8_t *dst = c->arena + (uint64_t)submitted * c->blob;
            io_uring_prep_read(sqe, 0, dst, (unsigned)c->blob, c->off[key]);
            sqe->flags |= IOSQE_FIXED_FILE;
            io_uring_sqe_set_data64(sqe, UINT64_MAX);
            submitted++;
            c->inflight++;
        }
        io_uring_submit(&c->ring);
        struct io_uring_cqe *cqe;
        if (io_uring_wait_cqe(&c->ring, &cqe) == 0) { io_uring_cqe_seen(&c->ring, cqe); done++; c->inflight--; }
    }
    return (double)n * c->blob / (inz_now() - t0) / 1e9;
}

/* ---------------- warm start ---------------- */

typedef struct { uint32_t magic, n_layer, n_expert, tick; } score_hdr;

bool ec_save_scores(const ec_cache *c, const char *path) {
    FILE *f = fopen(path, "wb");
    if (!f) return false;
    score_hdr h = {0x534E5A49u, (uint32_t)c->n_layer, (uint32_t)c->n_expert, c->tick};
    size_t nk = (size_t)c->n_layer * c->n_expert;
    float *sc = inz_malloc(sizeof(float) * nk);
    for (size_t k = 0; k < nk; k++) sc[k] = crf_now(c, (int32_t)k);
    bool ok = fwrite(&h, sizeof(h), 1, f) == 1 && fwrite(sc, sizeof(float), nk, f) == nk &&
              fwrite(c->use_count, sizeof(uint64_t), nk, f) == nk;
    free(sc);
    return fclose(f) == 0 && ok;
}

static const float *g_sort_scores;
static int cmp_score_desc(const void *a, const void *b) {
    float x = g_sort_scores[*(const int32_t *)a], y = g_sort_scores[*(const int32_t *)b];
    return x < y ? 1 : x > y ? -1 : 0;
}

int ec_load_and_preload(ec_cache *c, const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) return 0;
    score_hdr h;
    size_t nk = (size_t)c->n_layer * c->n_expert;
    if (fread(&h, sizeof(h), 1, f) != 1 || h.magic != 0x534E5A49u || h.n_layer != (uint32_t)c->n_layer ||
        h.n_expert != (uint32_t)c->n_expert) { fclose(f); return 0; }
    float *sc = inz_malloc(sizeof(float) * nk);
    bool ok = fread(sc, sizeof(float), nk, f) == nk && fread(c->use_count, sizeof(uint64_t), nk, f) == nk;
    fclose(f);
    if (!ok) { free(sc); return 0; }
    int32_t *order = inz_malloc(sizeof(int32_t) * nk);
    for (size_t k = 0; k < nk; k++) order[k] = (int32_t)k;
    g_sort_scores = sc;
    qsort(order, nk, sizeof(int32_t), cmp_score_desc);
    int n = (int)(c->n_free < nk ? c->n_free : nk);
    int32_t *pairs = inz_malloc(sizeof(int32_t) * 2 * (size_t)n);
    int np = 0;
    for (int i = 0; i < n; i++) {
        if (sc[order[i]] <= 0) break;
        pairs[2 * np] = order[i] / c->n_expert;
        pairs[2 * np + 1] = order[i] % c->n_expert;
        np++;
    }
    int loaded = ec_preload(c, pairs, np);
    for (size_t k = 0; k < nk; k++) { c->crf[k] = sc[k]; c->crf_tick[k] = c->tick; }
    free(pairs);
    free(order);
    free(sc);
    return loaded;
}

// SSD-streamed MoE expert cache.
//
// Every routed expert (gate+up+down of one expert in one layer) is stored in the GGUF as a single
// 4 KiB aligned blob, so a miss costs exactly one O_DIRECT read. With planar (Q3P) experts the blob
// starts with all coarse parts and ends with the fine planes, so an expert can be loaded at two
// levels: COARSE (one read of the first part, ~69% of the bytes) or FULL; a coarse slot is upgraded
// later by reading only the fine planes.
//
// Resident experts live in a huge-page arena split into fixed size slots. Reads go through io_uring.
// Eviction uses a decayed-frequency score (LRFU) approximated Redis-style: sample a few random slots
// and evict the lowest score.
#ifndef INZ_EXPERT_CACHE_H
#define INZ_EXPERT_CACHE_H

#include <stdbool.h>
#include <stdint.h>

enum { EC_COARSE = 1, EC_FULL = 2 };

typedef struct ec_cache ec_cache;

typedef struct {
    const char *path;          // GGUF file
    int n_layer;               // number of MoE layers indexed 0..n_layer-1
    int n_expert;              // experts per layer
    const uint64_t *offset;    // [n_layer * n_expert] absolute file offsets (4 KiB aligned)
    uint64_t blob_bytes;       // bytes per expert blob (uniform, multiple of 4096)
    uint64_t coarse_bytes;     // >0: blob prefix that forms the coarse level (multiple of 4096)
    uint64_t cache_bytes;      // arena budget
    int queue_depth;           // io_uring depth (default 64)
    int sqpoll_cpu;            // >=0: use SQPOLL thread pinned there; -1: plain submit
    float decay;               // LRFU decay per tick (0..1+). 0 => pure LFU, large => LRU-like
    bool all_resident;         // load every expert at start (model fits in RAM)
} ec_config;

typedef struct {
    uint64_t lookups, hits, misses;      // demand accesses (counted once, at routing time)
    uint64_t coarse_reads, upgrades;     // partial reads (planar experts)
    uint64_t prefetch_issued, prefetch_useful, prefetch_wasted, prefetch_late;
    uint64_t bytes_read, reads;
    uint64_t evictions;
    double stall_s;                      // time compute waited on SSD
    double read_lat_sum_s;               // sum of per-read latency
    uint64_t lat_hist[16];               // log2 buckets of microseconds
    uint64_t *layer_hits, *layer_misses;
    uint64_t resident, slots;
} ec_stats;

ec_cache *ec_open(const ec_config *cfg, char *err, int errlen);
void ec_close(ec_cache *c);

void ec_tick(ec_cache *c, int tokens);
uint64_t ec_read_bytes(const ec_cache *c);

// Ask for an expert ahead of use at the given level. Returns true if already resident at that level.
bool ec_prefetch(ec_cache *c, int layer, int expert, float weight, int level);
// Non-blocking: if resident at >= min_level, pin it and return the blob (counts a hit), else NULL.
// *level_out receives the resident level.
const uint8_t *ec_try_acquire(ec_cache *c, int layer, int expert, float weight, int min_level, int *level_out);
// Make sure a read is in flight (or done) so the expert reaches `level`, counting a demand miss.
void ec_request(ec_cache *c, int layer, int expert, float weight, int level);
// Blocking acquire at >= level. Pins the slot until ec_release.
const uint8_t *ec_acquire(ec_cache *c, int layer, int expert, float weight, int level, int *level_out);
void ec_release(ec_cache *c, int layer, int expert);
// Block until at least one in-flight read completes. Returns false if nothing is in flight.
bool ec_wait_any(ec_cache *c);
// Reap completed reads without blocking.
void ec_poll(ec_cache *c);

int ec_resident_level(const ec_cache *c, int layer, int expert);  // 0 when absent or loading
static inline bool ec_is_resident(const ec_cache *c, int layer, int expert) { return ec_resident_level(c, layer, expert) > 0; }
const ec_stats *ec_get_stats(const ec_cache *c);
void ec_reset_stats(ec_cache *c);
void ec_usage_counts(const ec_cache *c, uint64_t *out);
// Preload (layer, expert) pairs at full level, highest priority first. Never evicts. Returns loaded count.
int ec_preload(ec_cache *c, const int32_t *pairs, int n_pairs);
uint64_t ec_slot_count(const ec_cache *c);
double ec_probe_bandwidth(ec_cache *c, int n);
bool ec_save_scores(const ec_cache *c, const char *path);
int ec_load_and_preload(ec_cache *c, const char *path);

#endif

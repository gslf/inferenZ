// Hybrid-CPU aware thread pool.
//  - workers pinned to cores, ordered fastest-first (P, E, LP-E)
//  - dynamic chunking through an atomic cursor: fast cores simply take more chunks,
//    so heterogeneous P/E speed never creates stragglers
//  - low latency dispatch: spin with UMWAIT (WAITPKG) when available, then futex sleep
#ifndef INZ_THREADPOOL_H
#define INZ_THREADPOOL_H

#include <stdbool.h>
#include <stdint.h>

typedef void (*tp_fn)(void *ctx, int64_t begin, int64_t end, int tid);

typedef struct tp_pool tp_pool;

typedef struct {
    int n_cpu;
    int cpu_id[256];
    int cpu_class[256];   // 0 = performance, 1 = efficient, 2 = low-power efficient
    int max_khz[256];
    int first_sibling[256]; // 1 if this cpu is the first SMT thread of its core
    int n_p, n_e, n_lp, n_p_cores;
} tp_topology;

// One cpu per physical P/E core (no SMT siblings, no LP-E): the default compute set.
int tp_default_cpus(const tp_topology *t, int *cpus, int max);

void tp_detect_topology(tp_topology *t);

// cpus: list of logical cpus to use for workers (index 0 = main thread's cpu). n<=0 => auto.
tp_pool *tp_create(int n_threads, const int *cpus);
void tp_destroy(tp_pool *p);
int tp_size(const tp_pool *p);
// run fn over [0,n) split in chunks of `grain` items; main thread participates.
void tp_for(tp_pool *p, int64_t n, int64_t grain, tp_fn fn, void *ctx);
// run fn once per thread (begin=tid, end=tid+1)
void tp_each(tp_pool *p, tp_fn fn, void *ctx);
// statistics for diagnostics
typedef struct {
    uint64_t jobs;
    double busy_s[256];   // time each thread spent inside fn
    uint64_t chunks[256]; // chunks taken by each thread
} tp_stats;
const tp_stats *tp_get_stats(const tp_pool *p);
void tp_reset_stats(tp_pool *p);
const int *tp_cpus(const tp_pool *p);
bool tp_has_waitpkg(void);

#endif

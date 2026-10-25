#define _GNU_SOURCE
#include "threadpool.h"

#include <cpuid.h>
#include <immintrin.h>
#include <linux/futex.h>
#include <pthread.h>
#include <sched.h>
#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/syscall.h>
#include <time.h>
#include <unistd.h>

#define TP_MAX 256

struct tp_pool {
    int n;
    int cpus[TP_MAX];
    pthread_t th[TP_MAX];
    _Alignas(64) atomic_uint gen;      // job generation, workers wait for change
    _Alignas(64) atomic_llong cursor;  // next item to hand out
    _Alignas(64) atomic_int pending;   // workers still running current job
    _Alignas(64) atomic_int quit;
    int64_t n_items, grain;
    tp_fn fn;
    void *ctx;
    bool each;
    tp_stats stats;
    bool collect;
};

static double tnow(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

static int g_waitpkg = -1;
bool tp_has_waitpkg(void) {
    if (g_waitpkg < 0) {
        unsigned a, b, c, d;
        g_waitpkg = __get_cpuid_count(7, 0, &a, &b, &c, &d) ? (int)((c >> 5) & 1) : 0;
    }
    return g_waitpkg;
}

static long futex(atomic_uint *addr, int op, unsigned val) {
    return syscall(SYS_futex, (unsigned *)addr, op, val, NULL, NULL, 0);
}

static int read_int_file(const char *path, int def) {
    FILE *f = fopen(path, "r");
    if (!f) return def;
    int v = def;
    if (fscanf(f, "%d", &v) != 1) v = def;
    fclose(f);
    return v;
}

void tp_detect_topology(tp_topology *t) {
    memset(t, 0, sizeof(*t));
    int ncpu = (int)sysconf(_SC_NPROCESSORS_ONLN);
    if (ncpu > TP_MAX) ncpu = TP_MAX;
    int maxf = 0, minf = 1 << 30;
    for (int i = 0; i < ncpu; i++) {
        char p[128];
        snprintf(p, sizeof(p), "/sys/devices/system/cpu/cpu%d/cpufreq/cpuinfo_max_freq", i);
        t->cpu_id[i] = i;
        t->max_khz[i] = read_int_file(p, 0);
        if (t->max_khz[i] > maxf) maxf = t->max_khz[i];
        if (t->max_khz[i] && t->max_khz[i] < minf) minf = t->max_khz[i];
    }
    t->n_cpu = ncpu;
    for (int i = 0; i < ncpu; i++) {
        char p[160];
        snprintf(p, sizeof(p), "/sys/devices/system/cpu/cpu%d/topology/thread_siblings_list", i);
        int first = read_int_file(p, i);  // "0-1" or "0,1" -> fscanf reads the first number
        t->first_sibling[i] = first == i;
    }
    // Classify by max frequency tiers: top tier = P, lowest tier (if clearly lower) = LP-E.
    for (int i = 0; i < ncpu; i++) {
        int f = t->max_khz[i];
        if (f == 0 || f >= maxf * 0.95) t->cpu_class[i] = 0;
        else if (f <= minf * 1.05 && minf < maxf * 0.6) t->cpu_class[i] = 2;
        else t->cpu_class[i] = 1;
        if (t->cpu_class[i] == 0) { t->n_p++; if (t->first_sibling[i]) t->n_p_cores++; }
        else if (t->cpu_class[i] == 1) t->n_e++;
        else t->n_lp++;
    }
}

int tp_default_cpus(const tp_topology *t, int *cpus, int max) {
    int k = 0;
    for (int cls = 0; cls < 2; cls++)
        for (int i = 0; i < t->n_cpu && k < max; i++)
            if (t->cpu_class[i] == cls && t->first_sibling[i]) cpus[k++] = t->cpu_id[i];
    if (k == 0)
        for (int i = 0; i < t->n_cpu && k < max; i++) cpus[k++] = t->cpu_id[i];
    return k;
}

static void pin_self(int cpu) {
    if (cpu < 0) return;
    cpu_set_t s;
    CPU_ZERO(&s);
    CPU_SET(cpu, &s);
    pthread_setaffinity_np(pthread_self(), sizeof(s), &s);
}

static void run_job(tp_pool *p, int tid) {
    double t0 = p->collect ? tnow() : 0;
    uint64_t chunks = 0;
    if (p->each) {
        p->fn(p->ctx, tid, tid + 1, tid);
        chunks = 1;
    } else {
        const int64_t n = p->n_items, g = p->grain;
        for (;;) {
            int64_t b = atomic_fetch_add_explicit(&p->cursor, g, memory_order_relaxed);
            if (b >= n) break;
            int64_t e = b + g < n ? b + g : n;
            p->fn(p->ctx, b, e, tid);
            chunks++;
        }
    }
    if (p->collect) {
        p->stats.busy_s[tid] += tnow() - t0;
        p->stats.chunks[tid] += chunks;
    }
}

typedef struct { tp_pool *p; int tid; } warg;

static void *worker(void *arg) {
    warg *w = arg;
    tp_pool *p = w->p;
    int tid = w->tid;
    free(w);
    pin_self(p->cpus[tid]);
    unsigned seen = 0;  // generation at creation: jobs dispatched before this thread runs are still seen
    const bool wp = tp_has_waitpkg();
    for (;;) {
        // spin phase: ~200us worth of polling, low power with UMWAIT
        unsigned g;
        int spins = 0;
        for (;;) {
            g = atomic_load_explicit(&p->gen, memory_order_acquire);
            if (g != seen || atomic_load_explicit(&p->quit, memory_order_relaxed)) break;
            if (wp) {
                _umonitor((void *)&p->gen);
                g = atomic_load_explicit(&p->gen, memory_order_acquire);
                if (g != seen) break;
                // ~20k TSC cycles max per wait (C0.1 state, fast wake)
                _umwait(1, __rdtsc() + 20000);
                if (++spins > 40) break;
            } else {
                _mm_pause();
                if (++spins > 20000) break;
            }
        }
        if (atomic_load(&p->quit)) return NULL;
        if (g == seen) {
            futex(&p->gen, FUTEX_WAIT_PRIVATE, seen);
            continue;
        }
        seen = g;
        run_job(p, tid);
        if (atomic_fetch_sub_explicit(&p->pending, 1, memory_order_acq_rel) == 1) {
            // last worker: nothing else to do, main thread spins on pending
        }
    }
}

tp_pool *tp_create(int n_threads, const int *cpus) {
    tp_pool *p = aligned_alloc(64, sizeof(*p));
    if (!p) return NULL;
    memset(p, 0, sizeof(*p));
    tp_topology topo;
    tp_detect_topology(&topo);
    if (n_threads <= 0) n_threads = topo.n_cpu;
    if (n_threads > TP_MAX) n_threads = TP_MAX;
    p->n = n_threads;
    if (cpus) {
        memcpy(p->cpus, cpus, sizeof(int) * n_threads);
    } else {
        int k = tp_default_cpus(&topo, p->cpus, n_threads);
        for (int cls = 0; cls < 3 && k < n_threads; cls++)
            for (int i = 0; i < topo.n_cpu && k < n_threads; i++) {
                bool used = false;
                for (int j = 0; j < k; j++) used |= p->cpus[j] == topo.cpu_id[i];
                if (topo.cpu_class[i] == cls && !used) p->cpus[k++] = topo.cpu_id[i];
            }
        for (; k < n_threads; k++) p->cpus[k] = -1;
    }
    p->collect = true;
    pin_self(p->cpus[0]);
    for (int i = 1; i < n_threads; i++) {
        warg *w = malloc(sizeof(*w));
        w->p = p;
        w->tid = i;
        pthread_create(&p->th[i], NULL, worker, w);
    }
    return p;
}

void tp_destroy(tp_pool *p) {
    if (!p) return;
    atomic_store(&p->quit, 1);
    atomic_fetch_add(&p->gen, 1);
    futex(&p->gen, FUTEX_WAKE_PRIVATE, 1 << 30);
    for (int i = 1; i < p->n; i++) pthread_join(p->th[i], NULL);
    free(p);
}

int tp_size(const tp_pool *p) { return p->n; }
const int *tp_cpus(const tp_pool *p) { return p->cpus; }

static void dispatch(tp_pool *p) {
    p->stats.jobs++;
    if (p->n == 1) { run_job(p, 0); return; }
    atomic_store_explicit(&p->pending, p->n - 1, memory_order_relaxed);
    atomic_fetch_add_explicit(&p->gen, 1, memory_order_release);
    futex(&p->gen, FUTEX_WAKE_PRIVATE, 1 << 30);
    run_job(p, 0);
    while (atomic_load_explicit(&p->pending, memory_order_acquire) != 0) _mm_pause();
}

void tp_for(tp_pool *p, int64_t n, int64_t grain, tp_fn fn, void *ctx) {
    if (n <= 0) return;
    if (grain < 1) grain = 1;
    p->fn = fn;
    p->ctx = ctx;
    p->n_items = n;
    p->grain = grain;
    p->each = false;
    atomic_store_explicit(&p->cursor, 0, memory_order_relaxed);
    if (n <= grain) {  // too small to be worth waking anyone
        double t0 = p->collect ? tnow() : 0;
        fn(ctx, 0, n, 0);
        if (p->collect) { p->stats.busy_s[0] += tnow() - t0; p->stats.chunks[0]++; }
        return;
    }
    dispatch(p);
}

void tp_each(tp_pool *p, tp_fn fn, void *ctx) {
    p->fn = fn;
    p->ctx = ctx;
    p->each = true;
    dispatch(p);
}

const tp_stats *tp_get_stats(const tp_pool *p) { return &p->stats; }
void tp_reset_stats(tp_pool *p) { memset(&p->stats, 0, sizeof(p->stats)); }

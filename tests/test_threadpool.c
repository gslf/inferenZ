#include <assert.h>
#include <stdatomic.h>
#include <stdio.h>
#include <unistd.h>
#include "../src/threadpool.h"

static void count(void *ctx, int64_t b, int64_t e, int tid) {
    atomic_int *visits = ctx;
    for (int64_t i = b; i < e; i++) atomic_fetch_add(&visits[i], 1);
}

int main(void) {
    enum { N = 257, RUNS = 10000 };
    atomic_int visits[N] = {0};
    tp_pool *p = tp_create(4, NULL);
    assert(p);
    assert((uintptr_t)p % 64 == 0);
    tp_topology topology;
    tp_detect_topology(&topology);
    int expected[4];
    int n = tp_default_cpus(&topology, expected, 4);
    for (int i = 0; i < n; i++) assert(tp_cpus(p)[i] == expected[i]);
    for (int i = 0; i < RUNS; i++) {
        if (i % 100 == 0) usleep(2000);
        tp_for(p, N, 1 + i % 65, count, visits);
    }
    for (int i = 0; i < N; i++) assert(atomic_load(&visits[i]) == RUNS);
    tp_each(p, count, visits);
    for (int i = 0; i < N; i++) assert(atomic_load(&visits[i]) == RUNS + (i < 4));
    const tp_stats *s = tp_get_stats(p);
    assert(s->jobs == RUNS + 1);
    for (int i = 0; i < 4; i++) assert(s->chunks[i] > 0);
    tp_reset_stats(p);
    assert(tp_get_stats(p)->jobs == 0);
    tp_destroy(p);
    puts("Thread pool: no lost or duplicate work across idle wakeups");
}

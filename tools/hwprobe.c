// hwprobe: measure the three numbers that bound MoE decode speed on this box.
//   1. DRAM read bandwidth per core class (P / E / LP-E) and combined
//   2. NVMe random-read bandwidth with O_DIRECT + io_uring at several block sizes / depths
//   3. int8 VNNI dot-product throughput per core class
// Build: cc -O3 -march=native -pthread -o hwprobe tools/hwprobe.c -luring
#define _GNU_SOURCE
#include <errno.h>
#include <fcntl.h>
#include <immintrin.h>
#include <liburing.h>
#include <pthread.h>
#include <sched.h>
#include <stdatomic.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

static double now(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

/* ---------------- memory bandwidth ---------------- */
typedef struct {
    int cpu;
    const uint8_t *buf;
    size_t bytes;
    int iters;
    atomic_int *start;
    double secs;
    double sink;
} bw_arg;

static void pin(int cpu) {
    if (cpu < 0) return;
    cpu_set_t s;
    CPU_ZERO(&s);
    CPU_SET(cpu, &s);
    pthread_setaffinity_np(pthread_self(), sizeof(s), &s);
}

static void *bw_worker(void *p) {
    bw_arg *a = p;
    pin(a->cpu);
    while (!atomic_load(a->start)) {}
    __m256i acc0 = _mm256_setzero_si256(), acc1 = acc0, acc2 = acc0, acc3 = acc0;
    double t0 = now();
    for (int it = 0; it < a->iters; it++) {
        const __m256i *v = (const __m256i *)a->buf;
        size_t n = a->bytes / 32;
        for (size_t i = 0; i < n; i += 4) {
            acc0 = _mm256_add_epi64(acc0, _mm256_load_si256(v + i));
            acc1 = _mm256_add_epi64(acc1, _mm256_load_si256(v + i + 1));
            acc2 = _mm256_add_epi64(acc2, _mm256_load_si256(v + i + 2));
            acc3 = _mm256_add_epi64(acc3, _mm256_load_si256(v + i + 3));
        }
    }
    a->secs = now() - t0;
    acc0 = _mm256_add_epi64(_mm256_add_epi64(acc0, acc1), _mm256_add_epi64(acc2, acc3));
    a->sink = (double)_mm256_extract_epi64(acc0, 0);
    return NULL;
}

static double run_bw(const int *cpus, int n, uint8_t *buf, size_t per_thread) {
    pthread_t th[64];
    bw_arg args[64];
    atomic_int start = 0;
    for (int i = 0; i < n; i++) {
        args[i] = (bw_arg){.cpu = cpus[i], .buf = buf + (size_t)i * per_thread,
                           .bytes = per_thread, .iters = 4, .start = &start};
        pthread_create(&th[i], NULL, bw_worker, &args[i]);
    }
    usleep(20000);
    double t0 = now();
    atomic_store(&start, 1);
    for (int i = 0; i < n; i++) pthread_join(th[i], NULL);
    double wall = now() - t0;
    return (double)per_thread * n * 4 / wall / 1e9;
}

/* ---------------- VNNI compute ---------------- */
typedef struct { int cpu; atomic_int *start; double gops; } vn_arg;

static void *vnni_worker(void *p) {
    vn_arg *a = p;
    pin(a->cpu);
    while (!atomic_load(a->start)) {}
    __m256i x = _mm256_set1_epi8(3), y = _mm256_set1_epi8(5);
    __m256i c0 = _mm256_setzero_si256(), c1 = c0, c2 = c0, c3 = c0, c4 = c0, c5 = c0, c6 = c0, c7 = c0;
    const long N = 50 * 1000 * 1000;
    double t0 = now();
    for (long i = 0; i < N; i++) {
        c0 = _mm256_dpbusd_avx_epi32(c0, x, y);
        c1 = _mm256_dpbusd_avx_epi32(c1, x, y);
        c2 = _mm256_dpbusd_avx_epi32(c2, x, y);
        c3 = _mm256_dpbusd_avx_epi32(c3, x, y);
        c4 = _mm256_dpbusd_avx_epi32(c4, x, y);
        c5 = _mm256_dpbusd_avx_epi32(c5, x, y);
        c6 = _mm256_dpbusd_avx_epi32(c6, x, y);
        c7 = _mm256_dpbusd_avx_epi32(c7, x, y);
    }
    double s = now() - t0;
    c0 = _mm256_add_epi32(_mm256_add_epi32(_mm256_add_epi32(c0, c1), _mm256_add_epi32(c2, c3)),
                          _mm256_add_epi32(_mm256_add_epi32(c4, c5), _mm256_add_epi32(c6, c7)));
    volatile int sink = _mm256_extract_epi32(c0, 0);
    (void)sink;
    a->gops = (double)N * 8 * 32 * 2 / s / 1e9; /* 32 int8 MACs per dpbusd, 2 ops each */
    return NULL;
}

static double run_vnni(const int *cpus, int n) {
    pthread_t th[64];
    vn_arg args[64];
    atomic_int start = 0;
    for (int i = 0; i < n; i++) {
        args[i] = (vn_arg){.cpu = cpus[i], .start = &start};
        pthread_create(&th[i], NULL, vnni_worker, &args[i]);
    }
    usleep(20000);
    atomic_store(&start, 1);
    double tot = 0;
    for (int i = 0; i < n; i++) { pthread_join(th[i], NULL); tot += args[i].gops; }
    return tot;
}

/* ---------------- SSD random read ---------------- */
static uint64_t rng = 0x9E3779B97F4A7C15ull;
static uint64_t xr(void) { rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17; return rng; }

static double run_ssd(int fd, uint64_t fsize, size_t bs, int qd, double seconds, double *lat_us) {
    struct io_uring ring;
    if (io_uring_queue_init(qd, &ring, 0) < 0) { perror("io_uring"); return 0; }
    void **bufs = calloc(qd, sizeof(void *));
    for (int i = 0; i < qd; i++) bufs[i] = aligned_alloc(4096, bs);
    uint64_t nblk = fsize / bs;
    uint64_t done = 0;
    double lat_sum = 0;
    double *submit_t = calloc(qd, sizeof(double));
    double t0 = now();
    for (int i = 0; i < qd; i++) {
        struct io_uring_sqe *sqe = io_uring_get_sqe(&ring);
        io_uring_prep_read(sqe, fd, bufs[i], bs, (xr() % nblk) * bs);
        io_uring_sqe_set_data64(sqe, (uint64_t)i);
        submit_t[i] = now();
    }
    io_uring_submit(&ring);
    while (now() - t0 < seconds) {
        struct io_uring_cqe *cqe;
        if (io_uring_wait_cqe(&ring, &cqe) < 0) break;
        int slot = (int)io_uring_cqe_get_data64(cqe);
        if (cqe->res != (int)bs) { fprintf(stderr, "short read %d\n", cqe->res); }
        io_uring_cqe_seen(&ring, cqe);
        lat_sum += now() - submit_t[slot];
        done++;
        struct io_uring_sqe *sqe = io_uring_get_sqe(&ring);
        io_uring_prep_read(sqe, fd, bufs[slot], bs, (xr() % nblk) * bs);
        io_uring_sqe_set_data64(sqe, (uint64_t)slot);
        submit_t[slot] = now();
        io_uring_submit(&ring);
    }
    double el = now() - t0;
    /* drain */
    for (int i = 0; i < qd; i++) {
        struct io_uring_cqe *cqe;
        if (io_uring_wait_cqe(&ring, &cqe) < 0) break;
        io_uring_cqe_seen(&ring, cqe);
    }
    io_uring_queue_exit(&ring);
    for (int i = 0; i < qd; i++) free(bufs[i]);
    free(bufs);
    free(submit_t);
    *lat_us = done ? lat_sum / done * 1e6 : 0;
    return (double)done * bs / el / 1e9;
}

int main(int argc, char **argv) {
    const char *ssd_file = argc > 1 ? argv[1] : NULL;
    uint64_t ssd_gb = argc > 2 ? strtoull(argv[2], NULL, 10) : 8;

    int P[] = {0, 2}, PHT[] = {0, 1, 2, 3}, E[] = {4, 5, 6, 7, 8, 9, 10, 11}, LP[] = {12, 13};
    int PE[] = {0, 2, 4, 5, 6, 7, 8, 9, 10, 11};
    int ALL[] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13};
    int PEHT[] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};
    struct { const char *name; int *c; int n; } sets[] = {
        {"1 P-core", P, 1}, {"2 P-cores", P, 2}, {"2P+HT (4)", PHT, 4}, {"1 E-core", E, 1},
        {"8 E-cores", E, 8}, {"2 LP-E", LP, 2}, {"2P+8E (10)", PE, 10}, {"2P+HT+8E (12)", PEHT, 12},
        {"all 14", ALL, 14},
    };
    const int nsets = sizeof(sets) / sizeof(sets[0]);

    size_t per_thread = 256ull << 20;
    uint8_t *buf = mmap(NULL, per_thread * 14, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    madvise(buf, per_thread * 14, MADV_HUGEPAGE);
    memset(buf, 1, per_thread * 14);

    printf("== DRAM read bandwidth (AVX2 loads, 256 MiB/thread, 4 passes) ==\n");
    for (int s = 0; s < nsets; s++)
        printf("  %-16s %7.1f GB/s\n", sets[s].name, run_bw(sets[s].c, sets[s].n, buf, per_thread));

    printf("== int8 VNNI (vpdpbusd ymm) peak ==\n");
    for (int s = 0; s < nsets; s++)
        printf("  %-16s %7.1f GOPS\n", sets[s].name, run_vnni(sets[s].c, sets[s].n));
    munmap(buf, per_thread * 14);

    if (!ssd_file) return 0;
    struct stat st;
    if (stat(ssd_file, &st) != 0 || (uint64_t)st.st_size < (ssd_gb << 30)) {
        printf("== creating %llu GiB test file %s ==\n", (unsigned long long)ssd_gb, ssd_file);
        int wfd = open(ssd_file, O_WRONLY | O_CREAT | O_TRUNC, 0644);
        size_t chunk = 64 << 20;
        uint8_t *w = aligned_alloc(4096, chunk);
        for (size_t i = 0; i < chunk; i++) w[i] = (uint8_t)xr();
        for (uint64_t off = 0; off < (ssd_gb << 30); off += chunk)
            if (write(wfd, w, chunk) != (ssize_t)chunk) { perror("write"); return 1; }
        fsync(wfd);
        close(wfd);
        free(w);
    }
    int fd = open(ssd_file, O_RDONLY | O_DIRECT);
    if (fd < 0) { perror("open O_DIRECT"); return 1; }
    fstat(fd, &st);
    printf("== NVMe random read, O_DIRECT + io_uring, file %.1f GiB ==\n", st.st_size / 1073741824.0);
    size_t bss[] = {64 << 10, 256 << 10, 1 << 20, 2 << 20, 4 << 20};
    int qds[] = {1, 4, 16, 32, 64};
    for (size_t b = 0; b < sizeof(bss) / sizeof(bss[0]); b++)
        for (size_t q = 0; q < sizeof(qds) / sizeof(qds[0]); q++) {
            double lat;
            double gbs = run_ssd(fd, st.st_size, bss[b], qds[q], 2.0, &lat);
            printf("  bs=%5zuK qd=%2d  %6.2f GB/s  avg lat %8.1f us\n", bss[b] >> 10, qds[q], gbs, lat);
        }
    close(fd);
    return 0;
}

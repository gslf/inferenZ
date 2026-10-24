#define _GNU_SOURCE
#include "util.h"

#include <errno.h>
#include <immintrin.h>
#include <stdlib.h>
#include <sys/mman.h>
#include <unistd.h>

int inz_log_level = 1;

void inz_log(int level, const char *fmt, ...) {
    if (level > inz_log_level) return;
    va_list ap;
    va_start(ap, fmt);
    fputs("inz: ", stderr);
    vfprintf(stderr, fmt, ap);
    fputc('\n', stderr);
    va_end(ap);
}

void inz_die(const char *fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    fputs("inz: fatal: ", stderr);
    vfprintf(stderr, fmt, ap);
    fputc('\n', stderr);
    va_end(ap);
    exit(1);
}

void *inz_malloc(size_t n) {
    void *p = malloc(n ? n : 1);
    if (!p) inz_die("out of memory (%zu bytes)", n);
    return p;
}

void *inz_calloc(size_t n, size_t sz) {
    void *p = calloc(n ? n : 1, sz ? sz : 1);
    if (!p) inz_die("out of memory (%zu x %zu bytes)", n, sz);
    return p;
}

void *inz_mmap_anon(size_t n, bool huge) {
    void *p = mmap(NULL, n, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (p == MAP_FAILED) inz_die("mmap %zu bytes: %s", n, strerror(errno));
    if (huge) madvise(p, n, MADV_HUGEPAGE);
    return p;
}

void inz_munmap(void *p, size_t n) { if (p) munmap(p, n); }

void *inz_aligned(size_t align, size_t n) {
    // always free()-able; large blocks are 2 MiB aligned so transparent huge pages can back them
    if (n >= (4u << 20) && align < (2u << 20)) align = 2u << 20;
    size_t sz = INZ_ALIGN_UP(n ? n : 1, align);
    void *p = aligned_alloc(align, sz);
    if (!p) inz_die("out of memory (%zu bytes)", n);
    if (sz >= (4u << 20)) madvise(p, sz, MADV_HUGEPAGE);
    memset(p, 0, sz);
    return p;
}

char *inz_read_file(const char *path, size_t *len) {
    FILE *f = fopen(path, "rb");
    if (!f) return NULL;
    fseek(f, 0, SEEK_END);
    long n = ftell(f);
    fseek(f, 0, SEEK_SET);
    char *b = inz_malloc((size_t)n + 1);
    if (fread(b, 1, (size_t)n, f) != (size_t)n) { fclose(f); free(b); return NULL; }
    b[n] = 0;
    fclose(f);
    if (len) *len = (size_t)n;
    return b;
}

char *inz_strdup(const char *s) {
    size_t n = strlen(s);
    char *d = inz_malloc(n + 1);
    memcpy(d, s, n + 1);
    return d;
}

uint64_t inz_mem_available(void) {
    FILE *f = fopen("/proc/meminfo", "r");
    if (!f) return 0;
    char line[256];
    unsigned long long kb = 0;
    while (fgets(line, sizeof(line), f))
        if (sscanf(line, "MemAvailable: %llu kB", &kb) == 1) break;
    fclose(f);
    return (uint64_t)kb * 1024;
}

uint64_t inz_mem_total(void) {
    FILE *f = fopen("/proc/meminfo", "r");
    if (!f) return 0;
    char line[256];
    unsigned long long kb = 0;
    while (fgets(line, sizeof(line), f))
        if (sscanf(line, "MemTotal: %llu kB", &kb) == 1) break;
    fclose(f);
    return (uint64_t)kb * 1024;
}

uint64_t inz_rss(void) {
    FILE *f = fopen("/proc/self/statm", "r");
    if (!f) return 0;
    unsigned long long sz = 0, res = 0;
    if (fscanf(f, "%llu %llu", &sz, &res) != 2) res = 0;
    fclose(f);
    return (uint64_t)res * (uint64_t)sysconf(_SC_PAGESIZE);
}

float inz_f16_to_f32(uint16_t h) { return _cvtsh_ss(h); }
uint16_t inz_f32_to_f16(float f) { return _cvtss_sh(f, 0); }

float inz_e4m3_to_f32(uint8_t x) {
    uint32_t s = x >> 7, e = (x >> 3) & 0xF, m = x & 7;
    float v;
    if (e == 0) v = (float)m / 8.0f / 64.0f;               // subnormal: m/8 * 2^-6
    else if (e == 15 && m == 7) v = __builtin_nanf("");
    else v = (1.0f + (float)m / 8.0f) * __builtin_ldexpf(1.0f, (int)e - 7);
    return s ? -v : v;
}

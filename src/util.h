// Shared helpers: logging, timing, allocation, half/bfloat conversions.
#ifndef INZ_UTIL_H
#define INZ_UTIL_H

#include <stdarg.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <time.h>

#define INZ_VERSION "0.1.0"

#define INZ_MIN(a, b) ((a) < (b) ? (a) : (b))
#define INZ_MAX(a, b) ((a) > (b) ? (a) : (b))
#define INZ_ALIGN_UP(x, a) (((x) + (a) - 1) / (a) * (a))
#define INZ_LIKELY(x) __builtin_expect(!!(x), 1)
#define INZ_UNLIKELY(x) __builtin_expect(!!(x), 0)

static inline double inz_now(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

static inline uint64_t inz_now_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ull + (uint64_t)ts.tv_nsec;
}

extern int inz_log_level;  // 0 quiet, 1 info, 2 verbose, 3 debug
void inz_log(int level, const char *fmt, ...) __attribute__((format(printf, 2, 3)));
#define LOGI(...) inz_log(1, __VA_ARGS__)
#define LOGV(...) inz_log(2, __VA_ARGS__)
#define LOGD(...) inz_log(3, __VA_ARGS__)
void inz_die(const char *fmt, ...) __attribute__((format(printf, 1, 2), noreturn));

void *inz_malloc(size_t n);
void *inz_calloc(size_t n, size_t sz);
void *inz_aligned(size_t align, size_t n);  // zeroed, huge-page advised for big sizes
void *inz_mmap_anon(size_t n, bool huge);   // anonymous mapping (zeroed)
void inz_munmap(void *p, size_t n);
char *inz_read_file(const char *path, size_t *len);
char *inz_strdup(const char *s);
uint64_t inz_mem_available(void);           // bytes, from /proc/meminfo
uint64_t inz_mem_total(void);               // bytes, from /proc/meminfo
uint64_t inz_rss(void);                     // resident set size, bytes

static inline float inz_bf16_to_f32(uint16_t h) {
    union { uint32_t u; float f; } v = {.u = (uint32_t)h << 16};
    return v.f;
}
static inline uint16_t inz_f32_to_bf16(float f) {
    union { uint32_t u; float f; } v = {.f = f};
    uint32_t r = v.u + 0x7FFF + ((v.u >> 16) & 1);
    return (uint16_t)(r >> 16);
}
float inz_f16_to_f32(uint16_t h);
uint16_t inz_f32_to_f16(float f);
// FP8 e4m3 (OCP, "fn": no inf, max 448)
float inz_e4m3_to_f32(uint8_t x);

#endif

// GGUF v3 reader (mmap based) and streaming writer.
// inferenZ uses standard GGUF containers with its own tensor type ids
// (>= INZ_TYPE_BASE) for the custom quantization formats.
#ifndef INZ_GGUF_H
#define INZ_GGUF_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>

enum {
    GGUF_T_U8 = 0, GGUF_T_I8 = 1, GGUF_T_U16 = 2, GGUF_T_I16 = 3, GGUF_T_U32 = 4, GGUF_T_I32 = 5,
    GGUF_T_F32 = 6, GGUF_T_BOOL = 7, GGUF_T_STR = 8, GGUF_T_ARR = 9, GGUF_T_U64 = 10, GGUF_T_I64 = 11,
    GGUF_T_F64 = 12,
};

typedef struct {
    char *key;
    uint32_t type;
    uint32_t arr_type;   // for arrays
    uint64_t arr_n;      // for arrays
    const uint8_t *data; // pointer into the mapped file (scalars / array payload)
    union { uint64_t u; int64_t i; double f; bool b; } v; // decoded scalar
    char *s;             // decoded string (GGUF_T_STR)
} gguf_kv;

typedef struct {
    char *name;
    uint32_t n_dims;
    uint64_t ne[4];
    uint32_t type;
    uint64_t offset;   // relative to data section
    uint64_t nbytes;   // derived: next tensor offset or computed size
    const uint8_t *data; // absolute pointer in mapping
} gguf_tensor;

typedef struct {
    int fd;
    uint8_t *map;
    size_t size;
    uint32_t version;
    uint64_t n_kv, n_tensors;
    gguf_kv *kv;
    gguf_tensor *tensors;
    uint64_t alignment;
    uint64_t data_offset;  // absolute file offset of tensor data
    char *path;
} gguf_file;

gguf_file *gguf_open(const char *path, char *err, size_t errlen);
void gguf_close(gguf_file *g);
const gguf_kv *gguf_find_kv(const gguf_file *g, const char *key);
const gguf_tensor *gguf_find_tensor(const gguf_file *g, const char *name);
int64_t gguf_get_int(const gguf_file *g, const char *key, int64_t def);
double gguf_get_f(const gguf_file *g, const char *key, double def);
const char *gguf_get_str(const gguf_file *g, const char *key, const char *def);
// array of strings: returns count and fills offsets lazily (pointer walk)
uint64_t gguf_arr_len(const gguf_kv *kv);
// iterate string array: *pp starts at kv->data; returns string len and sets *s
uint64_t gguf_arr_next_str(const uint8_t **pp, const char **s);
const char *gguf_type_name(uint32_t t);

// ---------------- writer ----------------
typedef struct {
    char *name;
    uint32_t n_dims;
    uint64_t ne[4];
    uint32_t type;
    uint64_t nbytes;
    uint64_t offset;
} gguf_wtensor;

typedef struct {
    FILE *fp;
    uint8_t *meta; size_t meta_len, meta_cap; uint64_t n_kv;
    gguf_wtensor *t; uint64_t n_t, cap_t;
    uint64_t alignment;
    uint64_t data_start;     // absolute offset where data begins (after header written)
    uint64_t written;        // bytes of tensor data written so far (relative)
    uint64_t next_tensor;    // index of the next tensor to write data for
    char *path;
} gguf_writer;

gguf_writer *gguf_writer_new(const char *path, uint64_t alignment);
void gguf_w_u32(gguf_writer *w, const char *key, uint32_t v);
void gguf_w_i32(gguf_writer *w, const char *key, int32_t v);
void gguf_w_u64(gguf_writer *w, const char *key, uint64_t v);
void gguf_w_f32(gguf_writer *w, const char *key, float v);
void gguf_w_bool(gguf_writer *w, const char *key, bool v);
void gguf_w_str(gguf_writer *w, const char *key, const char *v);
void gguf_w_arr_str(gguf_writer *w, const char *key, const char *const *v, uint64_t n);
void gguf_w_arr_i32(gguf_writer *w, const char *key, const int32_t *v, uint64_t n);
void gguf_w_arr_f32(gguf_writer *w, const char *key, const float *v, uint64_t n);
void gguf_w_arr_i64(gguf_writer *w, const char *key, const int64_t *v, uint64_t n);
void gguf_w_arr_u8(gguf_writer *w, const char *key, const uint8_t *v, uint64_t n);
// declare all tensors first (sizes must be known), then write header, then data in order
void gguf_w_tensor(gguf_writer *w, const char *name, uint32_t type, uint32_t n_dims, const uint64_t *ne, uint64_t nbytes);
bool gguf_w_header(gguf_writer *w);
bool gguf_w_data(gguf_writer *w, const void *data, uint64_t nbytes); // data for next tensor (may be called in pieces)
bool gguf_w_tensor_done(gguf_writer *w);  // pad to alignment, advance
bool gguf_writer_close(gguf_writer *w);

#endif

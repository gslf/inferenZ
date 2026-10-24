#define _GNU_SOURCE
#include "gguf.h"

#include <errno.h>
#include <fcntl.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#define GGUF_MAGIC 0x46554747u

typedef struct {
    const uint8_t *p, *end;
    bool ok;
} rd;

static uint64_t rd_u64(rd *r) {
    if (r->end - r->p < 8) { r->ok = false; return 0; }
    uint64_t v; memcpy(&v, r->p, 8); r->p += 8; return v;
}
static uint32_t rd_u32(rd *r) {
    if (r->end - r->p < 4) { r->ok = false; return 0; }
    uint32_t v; memcpy(&v, r->p, 4); r->p += 4; return v;
}
static char *rd_str(rd *r) {
    uint64_t n = rd_u64(r);
    if (!r->ok || (uint64_t)(r->end - r->p) < n) { r->ok = false; return NULL; }
    char *s = malloc(n + 1);
    memcpy(s, r->p, n);
    s[n] = 0;
    r->p += n;
    return s;
}

static size_t scalar_size(uint32_t t) {
    switch (t) {
    case GGUF_T_U8: case GGUF_T_I8: case GGUF_T_BOOL: return 1;
    case GGUF_T_U16: case GGUF_T_I16: return 2;
    case GGUF_T_U32: case GGUF_T_I32: case GGUF_T_F32: return 4;
    case GGUF_T_U64: case GGUF_T_I64: case GGUF_T_F64: return 8;
    default: return 0;
    }
}

static bool skip_value(rd *r, uint32_t t) {
    if (t == GGUF_T_STR) {
        uint64_t n = rd_u64(r);
        if (!r->ok || (uint64_t)(r->end - r->p) < n) return r->ok = false;
        r->p += n;
        return true;
    }
    if (t == GGUF_T_ARR) {
        uint32_t at = rd_u32(r);
        uint64_t n = rd_u64(r);
        if (!r->ok) return false;
        if (at == GGUF_T_STR || at == GGUF_T_ARR) {
            for (uint64_t i = 0; i < n; i++) if (!skip_value(r, at)) return false;
            return true;
        }
        size_t es = scalar_size(at);
        if (!es || (uint64_t)(r->end - r->p) / es < n) return r->ok = false;
        r->p += es * n;
        return true;
    }
    size_t es = scalar_size(t);
    if (!es || (size_t)(r->end - r->p) < es) return r->ok = false;
    r->p += es;
    return true;
}

static int cmp_off(const void *a, const void *b) {
    const gguf_tensor *x = *(const gguf_tensor *const *)a, *y = *(const gguf_tensor *const *)b;
    return x->offset < y->offset ? -1 : x->offset > y->offset;
}

gguf_file *gguf_open(const char *path, char *err, size_t errlen) {
    int fd = open(path, O_RDONLY);
    if (fd < 0) { snprintf(err, errlen, "open %s: %s", path, strerror(errno)); return NULL; }
    struct stat st;
    fstat(fd, &st);
    uint8_t *map = mmap(NULL, st.st_size, PROT_READ, MAP_SHARED, fd, 0);
    if (map == MAP_FAILED) { snprintf(err, errlen, "mmap: %s", strerror(errno)); close(fd); return NULL; }
    // Header and metadata are read sequentially once; weights are accessed with explicit policies later.
    madvise(map, st.st_size, MADV_RANDOM);

    gguf_file *g = calloc(1, sizeof(*g));
    g->fd = fd; g->map = map; g->size = st.st_size; g->path = strdup(path);
    rd r = {.p = map, .end = map + st.st_size, .ok = true};
    if (rd_u32(&r) != GGUF_MAGIC) { snprintf(err, errlen, "not a GGUF file"); goto fail; }
    g->version = rd_u32(&r);
    if (g->version < 2 || g->version > 3) { snprintf(err, errlen, "unsupported GGUF version %u", g->version); goto fail; }
    g->n_tensors = rd_u64(&r);
    g->n_kv = rd_u64(&r);
    if (!r.ok || g->n_kv > (1u << 24) || g->n_tensors > (1u << 24)) { snprintf(err, errlen, "bad header"); goto fail; }
    g->kv = calloc(g->n_kv ? g->n_kv : 1, sizeof(gguf_kv));
    g->alignment = 32;
    for (uint64_t i = 0; i < g->n_kv; i++) {
        gguf_kv *kv = &g->kv[i];
        kv->key = rd_str(&r);
        kv->type = rd_u32(&r);
        if (!r.ok) { snprintf(err, errlen, "truncated metadata"); goto fail; }
        kv->data = r.p;
        if (kv->type == GGUF_T_STR) {
            kv->s = rd_str(&r);
        } else if (kv->type == GGUF_T_ARR) {
            const uint8_t *start = r.p;
            kv->arr_type = rd_u32(&r);
            kv->arr_n = rd_u64(&r);
            kv->data = r.p;
            r.p = start;
            if (!skip_value(&r, GGUF_T_ARR)) { snprintf(err, errlen, "bad array %s", kv->key); goto fail; }
        } else {
            size_t es = scalar_size(kv->type);
            if (!es) { snprintf(err, errlen, "bad kv type"); goto fail; }
            uint64_t u = 0;
            memcpy(&u, r.p, es);
            switch (kv->type) {
            case GGUF_T_I8: kv->v.i = (int8_t)u; break;
            case GGUF_T_I16: kv->v.i = (int16_t)u; break;
            case GGUF_T_I32: kv->v.i = (int32_t)u; break;
            case GGUF_T_I64: kv->v.i = (int64_t)u; break;
            case GGUF_T_F32: { float f; memcpy(&f, r.p, 4); kv->v.f = f; break; }
            case GGUF_T_F64: { double f; memcpy(&f, r.p, 8); kv->v.f = f; break; }
            case GGUF_T_BOOL: kv->v.b = u != 0; break;
            default: kv->v.u = u; break;
            }
            r.p += es;
        }
        if (!r.ok) { snprintf(err, errlen, "truncated metadata"); goto fail; }
        if (!strcmp(kv->key, "general.alignment")) g->alignment = kv->v.u ? kv->v.u : 32;
    }
    g->tensors = calloc(g->n_tensors ? g->n_tensors : 1, sizeof(gguf_tensor));
    for (uint64_t i = 0; i < g->n_tensors; i++) {
        gguf_tensor *t = &g->tensors[i];
        t->name = rd_str(&r);
        t->n_dims = rd_u32(&r);
        if (t->n_dims > 4) { snprintf(err, errlen, "tensor %s has %u dims", t->name, t->n_dims); goto fail; }
        for (int d = 0; d < 4; d++) t->ne[d] = 1;
        for (uint32_t d = 0; d < t->n_dims; d++) t->ne[d] = rd_u64(&r);
        t->type = rd_u32(&r);
        t->offset = rd_u64(&r);
        if (!r.ok) { snprintf(err, errlen, "truncated tensor info"); goto fail; }
    }
    uint64_t hdr = (uint64_t)(r.p - map);
    g->data_offset = (hdr + g->alignment - 1) / g->alignment * g->alignment;
    // derive byte sizes from offsets
    gguf_tensor **ord = malloc(sizeof(gguf_tensor *) * (g->n_tensors + 1));
    for (uint64_t i = 0; i < g->n_tensors; i++) ord[i] = &g->tensors[i];
    qsort(ord, g->n_tensors, sizeof(*ord), cmp_off);
    for (uint64_t i = 0; i < g->n_tensors; i++) {
        uint64_t end = i + 1 < g->n_tensors ? ord[i + 1]->offset : g->size - g->data_offset;
        ord[i]->nbytes = end - ord[i]->offset;
        ord[i]->data = map + g->data_offset + ord[i]->offset;
        if (g->data_offset + ord[i]->offset > g->size) { free(ord); snprintf(err, errlen, "tensor beyond EOF"); goto fail; }
    }
    free(ord);
    return g;
fail:
    gguf_close(g);
    return NULL;
}

void gguf_close(gguf_file *g) {
    if (!g) return;
    for (uint64_t i = 0; i < g->n_kv && g->kv; i++) { free(g->kv[i].key); free(g->kv[i].s); }
    for (uint64_t i = 0; i < g->n_tensors && g->tensors; i++) free(g->tensors[i].name);
    free(g->kv);
    free(g->tensors);
    if (g->map) munmap(g->map, g->size);
    if (g->fd >= 0) close(g->fd);
    free(g->path);
    free(g);
}

const gguf_kv *gguf_find_kv(const gguf_file *g, const char *key) {
    for (uint64_t i = 0; i < g->n_kv; i++)
        if (g->kv[i].key && !strcmp(g->kv[i].key, key)) return &g->kv[i];
    return NULL;
}

const gguf_tensor *gguf_find_tensor(const gguf_file *g, const char *name) {
    for (uint64_t i = 0; i < g->n_tensors; i++)
        if (!strcmp(g->tensors[i].name, name)) return &g->tensors[i];
    return NULL;
}

int64_t gguf_get_int(const gguf_file *g, const char *key, int64_t def) {
    const gguf_kv *kv = gguf_find_kv(g, key);
    if (!kv) return def;
    switch (kv->type) {
    case GGUF_T_I8: case GGUF_T_I16: case GGUF_T_I32: case GGUF_T_I64: return kv->v.i;
    case GGUF_T_U8: case GGUF_T_U16: case GGUF_T_U32: case GGUF_T_U64: return (int64_t)kv->v.u;
    case GGUF_T_BOOL: return kv->v.b;
    case GGUF_T_F32: case GGUF_T_F64: return (int64_t)kv->v.f;
    default: return def;
    }
}

double gguf_get_f(const gguf_file *g, const char *key, double def) {
    const gguf_kv *kv = gguf_find_kv(g, key);
    if (!kv) return def;
    if (kv->type == GGUF_T_F32 || kv->type == GGUF_T_F64) return kv->v.f;
    return (double)gguf_get_int(g, key, (int64_t)def);
}

const char *gguf_get_str(const gguf_file *g, const char *key, const char *def) {
    const gguf_kv *kv = gguf_find_kv(g, key);
    return kv && kv->type == GGUF_T_STR ? kv->s : def;
}

uint64_t gguf_arr_len(const gguf_kv *kv) { return kv && kv->type == GGUF_T_ARR ? kv->arr_n : 0; }

uint64_t gguf_arr_next_str(const uint8_t **pp, const char **s) {
    uint64_t n;
    memcpy(&n, *pp, 8);
    *s = (const char *)(*pp + 8);
    *pp += 8 + n;
    return n;
}

/* ---------------- writer ---------------- */

static void mput(gguf_writer *w, const void *p, size_t n) {
    if (w->meta_len + n > w->meta_cap) {
        size_t nc = w->meta_cap ? w->meta_cap * 2 : 4096;
        while (nc < w->meta_len + n) nc *= 2;
        w->meta = realloc(w->meta, nc);
        w->meta_cap = nc;
    }
    memcpy(w->meta + w->meta_len, p, n);
    w->meta_len += n;
}
static void mput_u32(gguf_writer *w, uint32_t v) { mput(w, &v, 4); }
static void mput_u64(gguf_writer *w, uint64_t v) { mput(w, &v, 8); }
static void mput_str(gguf_writer *w, const char *s) { uint64_t n = strlen(s); mput_u64(w, n); mput(w, s, n); }
static void mkey(gguf_writer *w, const char *k, uint32_t t) { mput_str(w, k); mput_u32(w, t); w->n_kv++; }

gguf_writer *gguf_writer_new(const char *path, uint64_t alignment) {
    gguf_writer *w = calloc(1, sizeof(*w));
    w->fp = fopen(path, "wb");
    if (!w->fp) { free(w); return NULL; }
    w->alignment = alignment ? alignment : 32;
    w->path = strdup(path);
    gguf_w_u32(w, "general.alignment", (uint32_t)w->alignment);
    return w;
}

void gguf_w_u32(gguf_writer *w, const char *k, uint32_t v) { mkey(w, k, GGUF_T_U32); mput_u32(w, v); }
void gguf_w_i32(gguf_writer *w, const char *k, int32_t v) { mkey(w, k, GGUF_T_I32); mput(w, &v, 4); }
void gguf_w_u64(gguf_writer *w, const char *k, uint64_t v) { mkey(w, k, GGUF_T_U64); mput_u64(w, v); }
void gguf_w_f32(gguf_writer *w, const char *k, float v) { mkey(w, k, GGUF_T_F32); mput(w, &v, 4); }
void gguf_w_bool(gguf_writer *w, const char *k, bool v) { mkey(w, k, GGUF_T_BOOL); uint8_t b = v; mput(w, &b, 1); }
void gguf_w_str(gguf_writer *w, const char *k, const char *v) { mkey(w, k, GGUF_T_STR); mput_str(w, v); }
void gguf_w_arr_str(gguf_writer *w, const char *k, const char *const *v, uint64_t n) {
    mkey(w, k, GGUF_T_ARR); mput_u32(w, GGUF_T_STR); mput_u64(w, n);
    for (uint64_t i = 0; i < n; i++) mput_str(w, v[i] ? v[i] : "");
}
void gguf_w_arr_i32(gguf_writer *w, const char *k, const int32_t *v, uint64_t n) {
    mkey(w, k, GGUF_T_ARR); mput_u32(w, GGUF_T_I32); mput_u64(w, n); mput(w, v, n * 4);
}
void gguf_w_arr_f32(gguf_writer *w, const char *k, const float *v, uint64_t n) {
    mkey(w, k, GGUF_T_ARR); mput_u32(w, GGUF_T_F32); mput_u64(w, n); mput(w, v, n * 4);
}
void gguf_w_arr_i64(gguf_writer *w, const char *k, const int64_t *v, uint64_t n) {
    mkey(w, k, GGUF_T_ARR); mput_u32(w, GGUF_T_I64); mput_u64(w, n); mput(w, v, n * 8);
}
void gguf_w_arr_u8(gguf_writer *w, const char *k, const uint8_t *v, uint64_t n) {
    mkey(w, k, GGUF_T_ARR); mput_u32(w, GGUF_T_U8); mput_u64(w, n); mput(w, v, n);
}

void gguf_w_tensor(gguf_writer *w, const char *name, uint32_t type, uint32_t n_dims, const uint64_t *ne, uint64_t nbytes) {
    if (w->n_t == w->cap_t) {
        w->cap_t = w->cap_t ? w->cap_t * 2 : 256;
        w->t = realloc(w->t, w->cap_t * sizeof(*w->t));
    }
    gguf_wtensor *t = &w->t[w->n_t++];
    memset(t, 0, sizeof(*t));
    t->name = strdup(name);
    t->type = type;
    t->n_dims = n_dims;
    for (uint32_t d = 0; d < n_dims; d++) t->ne[d] = ne[d];
    t->nbytes = nbytes;
}

static bool pad_to(FILE *fp, uint64_t pos, uint64_t align) {
    static const uint8_t zeros[4096];
    uint64_t target = (pos + align - 1) / align * align;
    while (pos < target) {
        uint64_t n = target - pos;
        if (n > sizeof(zeros)) n = sizeof(zeros);
        if (fwrite(zeros, 1, n, fp) != n) return false;
        pos += n;
    }
    return true;
}

bool gguf_w_header(gguf_writer *w) {
    uint64_t off = 0;
    for (uint64_t i = 0; i < w->n_t; i++) {
        w->t[i].offset = off;
        off += (w->t[i].nbytes + w->alignment - 1) / w->alignment * w->alignment;
    }
    uint32_t magic = GGUF_MAGIC, ver = 3;
    fwrite(&magic, 4, 1, w->fp);
    fwrite(&ver, 4, 1, w->fp);
    fwrite(&w->n_t, 8, 1, w->fp);
    fwrite(&w->n_kv, 8, 1, w->fp);
    fwrite(w->meta, 1, w->meta_len, w->fp);
    uint64_t pos = 24 + w->meta_len;
    for (uint64_t i = 0; i < w->n_t; i++) {
        gguf_wtensor *t = &w->t[i];
        uint64_t n = strlen(t->name);
        fwrite(&n, 8, 1, w->fp);
        fwrite(t->name, 1, n, w->fp);
        fwrite(&t->n_dims, 4, 1, w->fp);
        fwrite(t->ne, 8, t->n_dims, w->fp);
        fwrite(&t->type, 4, 1, w->fp);
        fwrite(&t->offset, 8, 1, w->fp);
        pos += 8 + n + 4 + 8ull * t->n_dims + 4 + 8;
    }
    if (!pad_to(w->fp, pos, w->alignment)) return false;
    w->data_start = (pos + w->alignment - 1) / w->alignment * w->alignment;
    return !ferror(w->fp);
}

bool gguf_w_data(gguf_writer *w, const void *data, uint64_t nbytes) {
    if (w->next_tensor >= w->n_t) return false;
    if (fwrite(data, 1, nbytes, w->fp) != nbytes) return false;
    w->written += nbytes;
    return true;
}

bool gguf_w_tensor_done(gguf_writer *w) {
    gguf_wtensor *t = &w->t[w->next_tensor];
    uint64_t expect_end = t->offset + t->nbytes;
    if (w->written != expect_end) {
        fprintf(stderr, "gguf writer: tensor %s wrote %llu bytes, declared %llu\n", t->name,
                (unsigned long long)(w->written - t->offset), (unsigned long long)t->nbytes);
        return false;
    }
    if (!pad_to(w->fp, w->written, w->alignment)) return false;
    w->written = (w->written + w->alignment - 1) / w->alignment * w->alignment;
    w->next_tensor++;
    return true;
}

bool gguf_writer_close(gguf_writer *w) {
    bool ok = w->next_tensor == w->n_t && !ferror(w->fp);
    if (fclose(w->fp) != 0) ok = false;
    for (uint64_t i = 0; i < w->n_t; i++) free(w->t[i].name);
    free(w->t);
    free(w->meta);
    free(w->path);
    free(w);
    return ok;
}

const char *gguf_type_name(uint32_t t) {
    static const char *names[] = {"F32", "F16", "Q4_0", "Q4_1", "?", "?", "Q5_0", "Q5_1", "Q8_0", "Q8_1",
                                  "Q2_K", "Q3_K", "Q4_K", "Q5_K", "Q6_K", "Q8_K", "IQ2_XXS", "IQ2_XS",
                                  "IQ3_XXS", "IQ1_S", "IQ4_NL", "IQ3_S", "IQ2_S", "IQ4_XS", "I8", "I16",
                                  "I32", "I64", "F64", "IQ1_M", "BF16"};
    if (t < sizeof(names) / sizeof(names[0])) return names[t];
    return "INZ";
}

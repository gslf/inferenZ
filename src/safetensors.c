#define _GNU_SOURCE
#include "safetensors.h"

#include <errno.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "json.h"
#include "util.h"

static st_dtype parse_dtype(const char *s) {
    if (!strcmp(s, "F32")) return ST_F32;
    if (!strcmp(s, "F16")) return ST_F16;
    if (!strcmp(s, "BF16")) return ST_BF16;
    if (!strcmp(s, "F8_E4M3")) return ST_F8_E4M3;
    if (!strcmp(s, "U8")) return ST_U8;
    if (!strcmp(s, "I8")) return ST_I8;
    if (!strcmp(s, "I32")) return ST_I32;
    if (!strcmp(s, "I64")) return ST_I64;
    if (!strcmp(s, "F64")) return ST_F64;
    return ST_UNKNOWN;
}

const char *st_dtype_name(st_dtype d) {
    static const char *n[] = {"F32", "F16", "BF16", "F8_E4M3", "U8", "I8", "I32", "I64", "F64", "?"};
    return n[d];
}

static uint32_t hstr(const char *s) {
    uint32_t h = 2166136261u;
    while (*s) { h ^= (unsigned char)*s++; h *= 16777619u; }
    return h;
}

static int add_file(st_model *m, const char *path) {
    for (int i = 0; i < m->nfiles; i++) if (!strcmp(m->paths[i], path)) return i;
    m->paths = realloc(m->paths, sizeof(char *) * (m->nfiles + 1));
    m->fds = realloc(m->fds, sizeof(int) * (m->nfiles + 1));
    m->paths[m->nfiles] = inz_strdup(path);
    m->fds[m->nfiles] = open(path, O_RDONLY);
    return m->nfiles++;
}

static bool load_header(st_model *m, int fi, char *err, size_t errlen) {
    uint64_t n;
    if (pread(m->fds[fi], &n, 8, 0) != 8 || n > (256u << 20)) {
        snprintf(err, errlen, "%s: bad safetensors header", m->paths[fi]);
        return false;
    }
    char *h = inz_malloc(n + 1);
    if (pread(m->fds[fi], h, n, 8) != (ssize_t)n) { free(h); snprintf(err, errlen, "short header read"); return false; }
    h[n] = 0;
    jval *root = json_parse(h, n, err, errlen);
    free(h);
    if (!root) return false;
    for (jval *t = root->child; t; t = t->next) {
        if (!strcmp(t->key, "__metadata__")) continue;
        m->tensors = realloc(m->tensors, sizeof(st_tensor) * (m->ntensors + 1));
        st_tensor *x = &m->tensors[m->ntensors++];
        memset(x, 0, sizeof(*x));
        x->name = inz_strdup(t->key);
        x->dtype = parse_dtype(json_str(json_get(t, "dtype"), "?"));
        jval *sh = json_get(t, "shape");
        x->ndim = sh ? (int)sh->len : 0;
        for (int d = 0; d < x->ndim && d < 4; d++) x->shape[d] = json_int(json_at(sh, (size_t)d), 0);
        jval *off = json_get(t, "data_offsets");
        uint64_t s = (uint64_t)json_int(json_at(off, 0), 0), e = (uint64_t)json_int(json_at(off, 1), 0);
        x->file = fi;
        x->start = 8 + n + s;
        x->nbytes = e - s;
    }
    json_free(root);
    return true;
}

static int cmp_u64pair(const void *a, const void *b) {
    const uint64_t *x = a, *y = b;
    return x[0] < y[0] ? -1 : x[0] > y[0];
}

void st_refresh(st_model *m) {
    if (!m->nfiles) return;
    char dir[4096];
    snprintf(dir, sizeof(dir), "%s", m->paths[0]);
    char *slash = strrchr(dir, '/');
    if (slash) *slash = 0; else snprintf(dir, sizeof(dir), ".");
    char jp[4200];
    snprintf(jp, sizeof(jp), "%s/.hfget.done", dir);
    FILE *f = fopen(jp, "r");
    if (!f) { m->have_journal = false; return; }
    m->have_journal = true;
    for (int i = 0; i < m->nfiles; i++) { free(m->ranges[i]); m->ranges[i] = NULL; m->nranges[i] = 0; }
    int *cap = inz_calloc((size_t)m->nfiles, sizeof(int));
    char name[1024];
    unsigned long long s, e;
    while (fscanf(f, "%1023s %llu %llu", name, &s, &e) == 3) {
        int fi = -1;
        for (int i = 0; i < m->nfiles; i++) {
            const char *b = strrchr(m->paths[i], '/');
            b = b ? b + 1 : m->paths[i];
            if (!strcmp(b, name)) { fi = i; break; }
        }
        if (fi < 0) continue;
        if (m->nranges[fi] == cap[fi]) {
            cap[fi] = cap[fi] ? cap[fi] * 2 : 256;
            m->ranges[fi] = realloc(m->ranges[fi], sizeof(uint64_t) * 2 * cap[fi]);
        }
        m->ranges[fi][2 * m->nranges[fi]] = s;
        m->ranges[fi][2 * m->nranges[fi] + 1] = e;
        m->nranges[fi]++;
    }
    fclose(f);
    free(cap);
    for (int i = 0; i < m->nfiles; i++) {
        if (!m->nranges[i]) continue;
        qsort(m->ranges[i], (size_t)m->nranges[i], 16, cmp_u64pair);
        int w = 0;
        for (int k = 0; k < m->nranges[i]; k++) {
            uint64_t a = m->ranges[i][2 * k], b = m->ranges[i][2 * k + 1];
            if (w && a <= m->ranges[i][2 * (w - 1) + 1]) {
                if (b > m->ranges[i][2 * (w - 1) + 1]) m->ranges[i][2 * (w - 1) + 1] = b;
            } else {
                m->ranges[i][2 * w] = a;
                m->ranges[i][2 * w + 1] = b;
                w++;
            }
        }
        m->nranges[i] = w;
    }
}

st_model *st_open(const char *dir, char *err, size_t errlen) {
    st_model *m = inz_calloc(1, sizeof(*m));
    char p[4096];
    snprintf(p, sizeof(p), "%s/model.safetensors.index.json", dir);
    size_t len;
    char *txt = inz_read_file(p, &len);
    if (txt) {
        jval *root = json_parse(txt, len, err, errlen);
        free(txt);
        if (!root) { free(m); return NULL; }
        jval *wm = json_get(root, "weight_map");
        for (jval *e = wm ? wm->child : NULL; e; e = e->next) {
            char fp[4096];
            snprintf(fp, sizeof(fp), "%s/%s", dir, e->str);
            add_file(m, fp);
        }
        json_free(root);
    } else {
        snprintf(p, sizeof(p), "%s/model.safetensors", dir);
        add_file(m, p);
    }
    for (int i = 0; i < m->nfiles; i++) {
        if (m->fds[i] < 0) { snprintf(err, errlen, "open %s: %s", m->paths[i], strerror(errno)); st_close(m); return NULL; }
        if (!load_header(m, i, err, errlen)) { st_close(m); return NULL; }
    }
    uint32_t sz = 1;
    while (sz < (uint32_t)m->ntensors * 2) sz <<= 1;
    m->hmask = sz - 1;
    m->htab = inz_malloc(sizeof(int32_t) * sz);
    for (uint32_t i = 0; i < sz; i++) m->htab[i] = -1;
    for (int i = 0; i < m->ntensors; i++) {
        uint32_t h = hstr(m->tensors[i].name) & m->hmask;
        while (m->htab[h] >= 0) h = (h + 1) & m->hmask;
        m->htab[h] = i;
    }
    m->ranges = inz_calloc((size_t)m->nfiles, sizeof(uint64_t *));
    m->nranges = inz_calloc((size_t)m->nfiles, sizeof(int));
    st_refresh(m);
    return m;
}

void st_close(st_model *m) {
    if (!m) return;
    for (int i = 0; i < m->nfiles; i++) {
        if (m->fds[i] >= 0) close(m->fds[i]);
        free(m->paths[i]);
        if (m->ranges) free(m->ranges[i]);
    }
    for (int i = 0; i < m->ntensors; i++) free(m->tensors[i].name);
    free(m->paths); free(m->fds); free(m->ranges); free(m->nranges); free(m->tensors); free(m->htab);
    free(m);
}

const st_tensor *st_find(const st_model *m, const char *name) {
    uint32_t h = hstr(name) & m->hmask;
    while (m->htab[h] >= 0) {
        if (!strcmp(m->tensors[m->htab[h]].name, name)) return &m->tensors[m->htab[h]];
        h = (h + 1) & m->hmask;
    }
    return NULL;
}

bool st_available(const st_model *m, const st_tensor *t) {
    if (!m->have_journal) return true;
    const uint64_t *r = m->ranges[t->file];
    int n = m->nranges[t->file];
    uint64_t s = t->start, e = t->start + t->nbytes;
    if (t->nbytes == 0) return true;
    int lo = 0, hi = n - 1;
    while (lo <= hi) {
        int mid = (lo + hi) >> 1;
        if (r[2 * mid] > s) hi = mid - 1;
        else lo = mid + 1;
    }
    if (hi < 0) return false;
    return r[2 * hi] <= s && r[2 * hi + 1] >= e;
}

bool st_read(const st_model *m, const st_tensor *t, void *dst) {
    uint64_t done = 0;
    while (done < t->nbytes) {
        ssize_t r = pread(m->fds[t->file], (char *)dst + done, t->nbytes - done, (off_t)(t->start + done));
        if (r <= 0) return false;
        done += (uint64_t)r;
    }
    return true;
}

int64_t st_numel(const st_tensor *t) {
    int64_t n = 1;
    for (int d = 0; d < t->ndim; d++) n *= t->shape[d];
    return n;
}

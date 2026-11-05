// inz convert: the official Laguna S 2.1 NVFP4 checkpoint (compressed-tensors: NVFP4 routed experts in layers 1-39,
// BF16 everything else, including the experts of layers 40-47) -> inferenZ GGUF.
//
// The converter streams: it declares every tensor up front, then writes them in order, waiting for each source
// tensor to finish downloading (tools/hfget.py journal) when needed. The NVFP4 experts are repacked exactly as Q4N
// (gate and up share their global scale, so they fuse into one gate|up matrix). The BF16 tensors are stored in the
// recipe's formats (default: 8-bit Q8G for the large matrices, F16 for the router and the embeddings). Routed experts
// are written as one 4 KiB-aligned blob per expert (gate|up, down), so the runtime fetches a missed expert with one
// O_DIRECT read.
#define _GNU_SOURCE
#include "convert.h"

#include <math.h>
#include <omp.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "gguf.h"
#include "model.h"
#include "quant.h"
#include "safetensors.h"
#include "tokenizer.h"
#include "util.h"

#define SRC "model."

/* ------------------------------------------------------------------ */
/* Source decoding                                                     */
/* ------------------------------------------------------------------ */

static void wait_available(st_model *m, const st_tensor *t) {
    int waited = 0;
    while (!st_available(m, t)) {
        if (waited % 60 == 0) fprintf(stderr, "inz convert: waiting for %s to download...\n", t->name);
        sleep(5);
        waited += 5;
        st_refresh(m);
    }
}

static const st_tensor *need(st_model *m, const char *fmt, ...) __attribute__((format(printf, 2, 3)));
static const st_tensor *need(st_model *m, const char *fmt, ...) {
    char name[512];
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(name, sizeof(name), fmt, ap);
    va_end(ap);
    const st_tensor *t = st_find(m, name);
    if (!t) inz_die("convert: missing source tensor %s", name);
    wait_available(m, t);
    return t;
}

static uint8_t *read_all(st_model *m, const st_tensor *t) {
    uint8_t *b = inz_malloc(t->nbytes ? t->nbytes : 1);
    if (!st_read(m, t, b)) inz_die("convert: read failed for %s", t->name);
    return b;
}

static float e4m3_lut[256];
static const float k_e2m1[16] = {0, 0.5f, 1, 1.5f, 2, 3, 4, 6, -0.0f, -0.5f, -1, -1.5f, -2, -3, -4, -6};

// the per-tensor F32 scalar `<base>suffix` of a modelopt tensor `<base>weight`
static float sibling_scalar(st_model *m, const char *weight_name, const char *suffix) {
    char n[600];
    const size_t bl = strlen(weight_name) - strlen("weight");
    snprintf(n, sizeof(n), "%.*s%s", (int)bl, weight_name, suffix);
    const st_tensor *t = need(m, "%s", n);
    if (t->dtype != ST_F32 || st_numel(t) != 1) inz_die("convert: %s is not an F32 scalar", n);
    float v;
    if (!st_read(m, t, &v)) inz_die("convert: read failed for %s", n);
    return v;
}

static const st_tensor *sibling(st_model *m, const char *weight_name, const char *suffix) {
    char n[600];
    const size_t bl = strlen(weight_name) - strlen("weight");
    snprintf(n, sizeof(n), "%.*s%s", (int)bl, weight_name, suffix);
    return need(m, "%s", n);
}

// Any source tensor as f32 [rows][cols]: BF16/F16/F32, FP8 with its per-tensor weight_scale, or NVFP4 (two e2m1
// codes per byte, an e4m3 weight_scale per 16 inputs, a per-tensor weight_scale_2).
static float *load(st_model *m, const char *name, int64_t *rows_out, int64_t *cols_out) {
    const st_tensor *t = need(m, "%s", name);
    const int64_t n = st_numel(t);
    int64_t rows = t->ndim >= 2 ? t->shape[0] : 1, cols = t->ndim >= 2 ? n / t->shape[0] : n;
    uint8_t *b = read_all(m, t);
    float *f;
    switch (t->dtype) {
    case ST_F32: f = inz_malloc(sizeof(float) * (size_t)n); memcpy(f, b, sizeof(float) * (size_t)n); break;
    case ST_BF16: f = inz_malloc(sizeof(float) * (size_t)n); for (int64_t i = 0; i < n; i++) f[i] = inz_bf16_to_f32(((uint16_t *)b)[i]); break;
    case ST_F16: f = inz_malloc(sizeof(float) * (size_t)n); for (int64_t i = 0; i < n; i++) f[i] = inz_f16_to_f32(((uint16_t *)b)[i]); break;
    case ST_F8_E4M3: {
        const float s = sibling_scalar(m, name, "weight_scale");
        f = inz_malloc(sizeof(float) * (size_t)n);
        for (int64_t i = 0; i < n; i++) f[i] = e4m3_lut[b[i]] * s;
        break;
    }
    case ST_U8: {
        const st_tensor *st = sibling(m, name, "weight_scale");
        const float s2 = sibling_scalar(m, name, "weight_scale_2");
        cols *= 2;
        if (st->dtype != ST_F8_E4M3 || st_numel(st) != rows * cols / 16) inz_die("convert: %s has unexpected NVFP4 scales", name);
        uint8_t *sc = read_all(m, st);
        f = inz_malloc(sizeof(float) * (size_t)(rows * cols));
        for (int64_t r = 0; r < rows; r++)
            for (int64_t c = 0; c < cols; c++) {
                const uint8_t code = (b[(r * cols + c) / 2] >> ((c & 1) * 4)) & 15;
                f[r * cols + c] = k_e2m1[code] * e4m3_lut[sc[(r * cols + c) / 16]] * s2;
            }
        free(sc);
        break;
    }
    default: inz_die("convert: %s has unexpected dtype %s", name, st_dtype_name(t->dtype));
    }
    free(b);
    if (rows_out) *rows_out = rows;
    if (cols_out) *cols_out = cols;
    return f;
}

/* ------------------------------------------------------------------ */
/* Importance matrix (from INZ_IMATRIX=file inz dump ...)              */
/* ------------------------------------------------------------------ */

typedef struct { char *name; uint32_t cols; float *v; } imx_item;
static imx_item *g_imx;
static size_t g_nimx;

static int imx_cmp(const void *a, const void *b) { return strcmp(((const imx_item *)a)->name, ((const imx_item *)b)->name); }

static void imx_load(const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) inz_die("convert: cannot open imatrix %s", path);
    char magic[8];
    uint32_t count;
    if (fread(magic, 1, 8, f) != 8 || memcmp(magic, "INZIMX01", 8) || fread(&count, 4, 1, f) != 1) inz_die("convert: %s is not an imatrix", path);
    g_imx = inz_calloc(count, sizeof(imx_item));
    for (uint32_t i = 0; i < count; i++) {
        uint16_t len;
        uint64_t n;
        if (fread(&len, 2, 1, f) != 1) inz_die("convert: truncated imatrix");
        g_imx[i].name = inz_malloc(len + 1u);
        if (fread(g_imx[i].name, 1, len, f) != len || fread(&g_imx[i].cols, 4, 1, f) != 1 || fread(&n, 8, 1, f) != 1)
            inz_die("convert: truncated imatrix");
        g_imx[i].name[len] = 0;
        g_imx[i].v = inz_malloc(sizeof(float) * g_imx[i].cols);
        if (fread(g_imx[i].v, 4, g_imx[i].cols, f) != g_imx[i].cols) inz_die("convert: truncated imatrix");
    }
    fclose(f);
    g_nimx = count;
    qsort(g_imx, g_nimx, sizeof(imx_item), imx_cmp);
    fprintf(stderr, "inz convert: imatrix with %u entries\n", count);
}

// importance of the columns of tensor `name` (GGUF name), or NULL
static const float *imx_get(const char *name, int cols) {
    if (!g_nimx || !name) return NULL;
    imx_item key = {(char *)name, 0, NULL};
    const imx_item *it = bsearch(&key, g_imx, g_nimx, sizeof(imx_item), imx_cmp);
    return it && it->cols == (uint32_t)cols ? it->v : NULL;
}

/* ------------------------------------------------------------------ */
/* Recipe and writer                                                   */
/* ------------------------------------------------------------------ */

void convert_opts_default(convert_opts *o) {
    memset(o, 0, sizeof(*o));
    o->t_attn = INZ_T_Q8G;
    o->t_dense = INZ_T_Q8G;
    o->t_shared = INZ_T_Q8G;
    o->t_router = INZ_T_F16;
    o->t_experts = INZ_T_Q4N;
    o->t_experts_bf16 = INZ_T_Q8G;
    o->t_head = INZ_T_Q8G;
    o->t_embed = INZ_T_F16;
    o->max_layers = Q_N_LAYER;
}

typedef struct {
    gguf_writer *w;
    st_model *m;
    const convert_opts *o;
    double t0;
} cvt;

static uint64_t mat_bytes(int type, int rows, int cols) { return inz_mat_bytes(type, rows, cols); }

static void declare_mat(cvt *c, const char *name, int type, int rows, int cols) {
    uint64_t ne[2] = {(uint64_t)cols, (uint64_t)rows};
    gguf_w_tensor(c->w, name, (uint32_t)type, 2, ne, mat_bytes(type, rows, cols));
}

static void declare_f32(cvt *c, const char *name, int n) {
    uint64_t ne[1] = {(uint64_t)n};
    gguf_w_tensor(c->w, name, INZ_T_F32, 1, ne, (uint64_t)n * 4);
}

uint64_t convert_expert_blob_bytes(int type) {
    uint64_t b = mat_bytes(type, 2 * Q_FF, Q_H) + mat_bytes(type, Q_H, Q_FF);
    return INZ_ALIGN_UP(b, 4096);
}

static void put(cvt *c, const void *data, uint64_t n) {
    if (!gguf_w_data(c->w, data, n) || !gguf_w_tensor_done(c->w)) inz_die("convert: write failed");
}

// Encode W (rows x cols, f32) as `type` into out (mat_bytes bytes). imp: optional per-column importance (mean squared
// input), which weights the quantization error.
static void encode(const float *W, int rows, int cols, int type, int threads, const float *imp, void *out) {
    if (type == INZ_T_F16) {
        uint16_t *h = out;
        for (size_t i = 0; i < (size_t)rows * cols; i++) {
            if (fabsf(W[i]) > 65504.0f) inz_die("convert: a weight (%g) overflows F16", W[i]);
            h[i] = inz_f32_to_f16(W[i]);
        }
        return;
    }
    if (type == INZ_T_F32) { memcpy(out, W, sizeof(float) * (size_t)rows * cols); return; }
    if (type == INZ_T_F8T) inz_die("convert: F8T keeps FP8 checkpoint tensors only");
    inz_quant_params p = {.type = type, .rot_block = type == INZ_T_Q8G ? inz_auto_rot_block(cols) : 0, .threads = threads, .imatrix = imp};
    inz_quantize(W, rows, cols, &p, out);
}

// Source tensor `src` as `type` into out. FP8 -> F8T and NVFP4 -> Q4N are exact copies; anything else goes
// through f32. dst: the GGUF name (for the importance matrix).
static void convert_mat(st_model *m, const char *src, const char *dst, int type, int rows, int cols, int threads, void *out) {
    const st_tensor *t = need(m, "%s", src);
    if (type == INZ_T_F8T && t->dtype == ST_F8_E4M3) {
        if (t->ndim != 2 || t->shape[0] != rows || t->shape[1] != cols) inz_die("convert: %s is not %dx%d", src, rows, cols);
        uint8_t *b = read_all(m, t);
        for (int64_t i = 0; i < (int64_t)rows * cols; i++)
            if ((b[i] & 0x7F) == 0x7F) inz_die("convert: %s holds a NaN", src);
        inz_pack_f8t(b, sibling_scalar(m, src, "weight_scale"), rows, cols, out);
        free(b);
        return;
    }
    if (type == INZ_T_Q4N && t->dtype == ST_U8) {
        if (t->ndim != 2 || t->shape[0] != rows || t->shape[1] * 2 != cols) inz_die("convert: %s is not %dx%d NVFP4", src, rows, cols);
        const st_tensor *st = sibling(m, src, "weight_scale");
        if (st->dtype != ST_F8_E4M3 || st_numel(st) != (int64_t)rows * cols / 16) inz_die("convert: %s has unexpected scales", src);
        uint8_t *b = read_all(m, t), *sc = read_all(m, st);
        if (!inz_pack_q4n_mul(b, sc, sibling_scalar(m, src, "weight_scale_2"), rows, cols, out))
            inz_die("convert: %s has a negative or NaN block scale", src);
        free(b);
        free(sc);
        return;
    }
    int64_t r, cc;
    float *W = load(m, src, &r, &cc);
    if (r * cc != (int64_t)rows * cols) inz_die("convert: %s has %lldx%lld, expected %dx%d", src, (long long)r, (long long)cc, rows, cols);
    encode(W, rows, cols, type, threads, imx_get(dst, cols), out);
    free(W);
}

static void write_mat(cvt *c, const char *src, const char *dst, int type, int rows, int cols) {
    const uint64_t nb = mat_bytes(type, rows, cols);
    void *buf = inz_aligned(64, nb);
    convert_mat(c->m, src, dst, type, rows, cols, c->o->threads, buf);
    put(c, buf, nb);
    free(buf);
}

static void write_vec(cvt *c, const char *src, int n) {
    int64_t r, cc;
    float *f = load(c->m, src, &r, &cc);
    if (r * cc != n) inz_die("convert: %s has %lld elements, expected %d", src, (long long)(r * cc), n);
    put(c, f, (uint64_t)n * 4);
    free(f);
}

static void progress(cvt *c, const char *what) {
    fprintf(stderr, "inz convert: [%7.1fs] %s\n", inz_now() - c->t0, what);
}

/* ------------------------------------------------------------------ */
/* Layers                                                              */
/* ------------------------------------------------------------------ */

static void declare_layer(cvt *c, int l) {
    const convert_opts *o = c->o;
    char n[96];
    const int nh = q_nh(l);
#define D(suffix) (snprintf(n, sizeof(n), "blk.%d." suffix, l), n)
    declare_f32(c, D("attn_norm"), Q_H);
    declare_f32(c, D("mlp_norm"), Q_H);
    declare_mat(c, D("at.q"), o->t_attn, nh * Q_HD, Q_H);
    declare_mat(c, D("at.k"), o->t_attn, Q_KV, Q_H);
    declare_mat(c, D("at.v"), o->t_attn, Q_KV, Q_H);
    declare_mat(c, D("at.o"), o->t_attn, Q_H, nh * Q_HD);
    declare_mat(c, D("at.g"), o->t_attn, nh, Q_H);
    declare_f32(c, D("at.q_norm"), Q_HD);
    declare_f32(c, D("at.k_norm"), Q_HD);
    if (l == 0) {
        declare_mat(c, D("mlp.gate_up"), o->t_dense, 2 * Q_DENSE_FF, Q_H);
        declare_mat(c, D("mlp.down"), o->t_dense, Q_H, Q_DENSE_FF);
    } else {
        declare_mat(c, D("router"), o->t_router, Q_N_EXPERT, Q_H);
        declare_f32(c, D("router_bias"), Q_N_EXPERT);
        declare_mat(c, D("sh.gate_up"), o->t_shared, 2 * Q_SHARED_FF, Q_H);
        declare_mat(c, D("sh.down"), o->t_shared, Q_H, Q_SHARED_FF);
    }
#undef D
}

// gate and up rows fused into one matrix
static void write_gate_up(cvt *c, const char *gsrc, const char *usrc, const char *dst, int type, int ff) {
    int64_t r, cc;
    float *G = load(c->m, gsrc, &r, &cc), *U = load(c->m, usrc, &r, &cc);
    float *GU = inz_malloc(sizeof(float) * 2 * (size_t)ff * Q_H);
    memcpy(GU, G, sizeof(float) * (size_t)ff * Q_H);
    memcpy(GU + (size_t)ff * Q_H, U, sizeof(float) * (size_t)ff * Q_H);
    const uint64_t nb = mat_bytes(type, 2 * ff, Q_H);
    void *buf = inz_aligned(64, nb);
    encode(GU, 2 * ff, Q_H, type, c->o->threads, imx_get(dst, Q_H), buf);
    put(c, buf, nb);
    free(buf); free(G); free(U); free(GU);
}

static void write_layer(cvt *c, int l) {
    const convert_opts *o = c->o;
    char n[256], n2[256], d[96];
    const int nh = q_nh(l);
#define S(suffix) (snprintf(n, sizeof(n), SRC "layers.%d." suffix, l), n)
#define S2(suffix) (snprintf(n2, sizeof(n2), SRC "layers.%d." suffix, l), n2)
#define D(suffix) (snprintf(d, sizeof(d), "blk.%d." suffix, l), d)
    write_vec(c, S("input_layernorm.weight"), Q_H);
    write_vec(c, S("post_attention_layernorm.weight"), Q_H);
    write_mat(c, S("self_attn.q_proj.weight"), D("at.q"), o->t_attn, nh * Q_HD, Q_H);
    write_mat(c, S("self_attn.k_proj.weight"), D("at.k"), o->t_attn, Q_KV, Q_H);
    write_mat(c, S("self_attn.v_proj.weight"), D("at.v"), o->t_attn, Q_KV, Q_H);
    write_mat(c, S("self_attn.o_proj.weight"), D("at.o"), o->t_attn, Q_H, nh * Q_HD);
    write_mat(c, S("self_attn.g_proj.weight"), D("at.g"), o->t_attn, nh, Q_H);
    write_vec(c, S("self_attn.q_norm.weight"), Q_HD);
    write_vec(c, S("self_attn.k_norm.weight"), Q_HD);
    if (l == 0) {
        write_gate_up(c, S("mlp.gate_proj.weight"), S2("mlp.up_proj.weight"), D("mlp.gate_up"), o->t_dense, Q_DENSE_FF);
        write_mat(c, S("mlp.down_proj.weight"), D("mlp.down"), o->t_dense, Q_H, Q_DENSE_FF);
    } else {
        write_mat(c, S("mlp.gate.weight"), D("router"), o->t_router, Q_N_EXPERT, Q_H);
        write_vec(c, S("mlp.experts.e_score_correction_bias"), Q_N_EXPERT);
        write_gate_up(c, S("mlp.shared_expert.gate_proj.weight"), S2("mlp.shared_expert.up_proj.weight"), D("sh.gate_up"),
                      o->t_shared, Q_SHARED_FF);
        write_mat(c, S("mlp.shared_expert.down_proj.weight"), D("sh.down"), o->t_shared, Q_H, Q_SHARED_FF);
    }
#undef S
#undef S2
#undef D
}

// One routed expert into out (blob layout: gate|up, then down). A compressed-tensors NVFP4 expert (weight_packed,
// weight_scale per 16, weight_global_scale shared by gate and up) is repacked exactly as Q4N; a BF16 expert is
// encoded as `type`.
static void write_expert(st_model *m, const char *base, int e, int type, const char *dbase, void *out) {
    char a[512], b[512], dn[96];
    snprintf(a, sizeof(a), "%s%d.gate_proj.weight_packed", base, e);
    if (st_find(m, a)) {
        if (type != INZ_T_Q4N) inz_die("convert: NVFP4 experts can only be repacked as Q4N");
        static const char *nm[3] = {"gate_proj", "up_proj", "down_proj"};
        uint8_t *pk[3], *sc[3];
        float gs[3];
        for (int w = 0; w < 3; w++) {
            const st_tensor *t = need(m, "%s%d.%s.weight_packed", base, e, nm[w]);
            const st_tensor *ts = need(m, "%s%d.%s.weight_scale", base, e, nm[w]);
            const st_tensor *tg = need(m, "%s%d.%s.weight_global_scale", base, e, nm[w]);
            if (t->dtype != ST_U8 || ts->dtype != ST_F8_E4M3 || tg->dtype != ST_F32 || st_numel(tg) != 1)
                inz_die("convert: %s is not compressed-tensors NVFP4", t->name);
            pk[w] = read_all(m, t);
            sc[w] = read_all(m, ts);
            if (!st_read(m, tg, &gs[w])) inz_die("convert: read failed for %s", tg->name);
        }
        if (gs[0] != gs[1]) inz_die("convert: %sexperts.%d gate and up have different global scales", base, e);
        // gate rows, then up rows: same global scale, so one Q4N matrix holds both exactly
        const size_t pkr = (size_t)Q_FF * Q_H / 2, scr = (size_t)Q_FF * Q_H / 16;
        uint8_t *gpk = inz_malloc(2 * pkr), *gsc = inz_malloc(2 * scr);
        memcpy(gpk, pk[0], pkr); memcpy(gpk + pkr, pk[1], pkr);
        memcpy(gsc, sc[0], scr); memcpy(gsc + scr, sc[1], scr);
        if (!inz_pack_q4n(gpk, gsc, gs[0], 2 * Q_FF, Q_H, out) ||
            !inz_pack_q4n(pk[2], sc[2], gs[2], Q_H, Q_FF, (uint8_t *)out + mat_bytes(INZ_T_Q4N, 2 * Q_FF, Q_H)))
            inz_die("convert: %sexperts.%d has a negative or NaN block scale", base, e);
        free(gpk); free(gsc);
        for (int w = 0; w < 3; w++) { free(pk[w]); free(sc[w]); }
        return;
    }
    int64_t r, cc;
    snprintf(a, sizeof(a), "%s%d.gate_proj.weight", base, e);
    snprintf(b, sizeof(b), "%s%d.up_proj.weight", base, e);
    float *G = load(m, a, &r, &cc), *U = load(m, b, &r, &cc);
    float *GU = inz_malloc(sizeof(float) * 2 * Q_FF * Q_H);
    memcpy(GU, G, sizeof(float) * Q_FF * Q_H);
    memcpy(GU + (size_t)Q_FF * Q_H, U, sizeof(float) * Q_FF * Q_H);
    snprintf(dn, sizeof(dn), "%s%d.gate_up", dbase, e);
    encode(GU, 2 * Q_FF, Q_H, type, 1, imx_get(dn, Q_H), out);
    snprintf(a, sizeof(a), "%s%d.down_proj.weight", base, e);
    float *D = load(m, a, &r, &cc);
    snprintf(dn, sizeof(dn), "%s%d.down", dbase, e);
    encode(D, Q_H, Q_FF, type, 1, imx_get(dn, Q_FF), (uint8_t *)out + mat_bytes(type, 2 * Q_FF, Q_H));
    free(G); free(U); free(GU); free(D);
}

/* ------------------------------------------------------------------ */
/* Conversion                                                          */
/* ------------------------------------------------------------------ */

int convert_run(const convert_opts *o) {
    for (int i = 0; i < 256; i++) e4m3_lut[i] = inz_e4m3_to_f32((uint8_t)i);
    if (o->imatrix) imx_load(o->imatrix);
    char err[512];
    st_model *m = st_open(o->src_dir, err, sizeof(err));
    if (!m) { fprintf(stderr, "inz convert: %s\n", err); return 1; }
    char tpath[4096];
    snprintf(tpath, sizeof(tpath), "%s/tokenizer.json", o->src_dir);
    tokenizer *tok = tok_load_hf_json(tpath, err, sizeof(err));
    if (!tok) { fprintf(stderr, "inz convert: tokenizer: %s\n", err); return 1; }
    snprintf(tpath, sizeof(tpath), "%s/chat_template.jinja", o->src_dir);
    char *tmpl = inz_read_file(tpath, NULL);

    cvt c = {.m = m, .o = o, .t0 = inz_now()};
    c.w = gguf_writer_new(o->out_path, 4096);
    if (!c.w) { fprintf(stderr, "inz convert: cannot create %s\n", o->out_path); return 1; }
    const int NL = o->max_layers > 0 && o->max_layers < Q_N_LAYER ? o->max_layers : Q_N_LAYER;
    // expert storage per layer: NVFP4 layers keep Q4N, the checkpoint's BF16 layers get t_experts_bf16
    int32_t xtype[Q_N_LAYER] = {0};
    for (int l = 1; l < NL; l++) {
        char name[256];
        snprintf(name, sizeof(name), SRC "layers.%d.mlp.experts.0.gate_proj.weight_packed", l);
        xtype[l] = st_find(m, name) ? o->t_experts : o->t_experts_bf16;
    }

    // ---------------- metadata ----------------
    gguf_w_str(c.w, "general.architecture", Q_ARCH);
    gguf_w_str(c.w, "general.name", "Poolside Laguna S 2.1 (inferenZ)");
    gguf_w_str(c.w, "inz.version", INZ_VERSION);
    gguf_w_u32(c.w, "inz.n_layer", (uint32_t)NL);
    gguf_w_u32(c.w, "inz.n_layer_full", Q_N_LAYER);
    gguf_w_u32(c.w, "inz.n_embd", Q_H);
    gguf_w_u32(c.w, "inz.n_expert", Q_N_EXPERT);
    gguf_w_arr_i32(c.w, "inz.expert_types", xtype, (uint64_t)NL);
    char recipe[256];
    snprintf(recipe, sizeof(recipe), "experts=%s experts_bf16=%s attn=%s dense=%s shared=%s router=%s head=%s embed=%s%s",
             inz_qtype_name(o->t_experts), inz_qtype_name(o->t_experts_bf16), inz_qtype_name(o->t_attn), inz_qtype_name(o->t_dense),
             inz_qtype_name(o->t_shared), inz_qtype_name(o->t_router), inz_qtype_name(o->t_head), inz_qtype_name(o->t_embed),
             o->imatrix ? " imatrix" : "");
    gguf_w_str(c.w, "inz.recipe", recipe);
    if (tmpl) gguf_w_str(c.w, "tokenizer.chat_template", tmpl);
    gguf_w_u32(c.w, "tokenizer.split", (uint32_t)tok_get_split(tok));
    tok_export te;
    tok_get_export(tok, &te);
    gguf_w_u32(c.w, "tokenizer.n_vocab", (uint32_t)te.n_vocab);
    {
        uint64_t total = 0;
        for (int i = 0; i < te.n_vocab; i++) total += te.piece_len[i];
        uint8_t *blob = inz_malloc(total + 1);
        int32_t *lens = inz_malloc(sizeof(int32_t) * te.n_vocab);
        uint64_t off = 0;
        for (int i = 0; i < te.n_vocab; i++) {
            lens[i] = te.pieces[i] ? (int32_t)te.piece_len[i] : -1;
            if (te.pieces[i]) { memcpy(blob + off, te.pieces[i], te.piece_len[i]); off += te.piece_len[i]; }
        }
        gguf_w_arr_u8(c.w, "tokenizer.piece_bytes", blob, off);
        gguf_w_arr_i32(c.w, "tokenizer.piece_len", lens, (uint64_t)te.n_vocab);
        free(blob);
        free(lens);
        gguf_w_arr_i32(c.w, "tokenizer.merge_a", te.merge_a, (uint64_t)te.n_merges);
        gguf_w_arr_i32(c.w, "tokenizer.merge_b", te.merge_b, (uint64_t)te.n_merges);
        gguf_w_arr_i32(c.w, "tokenizer.merge_res", te.merge_res, (uint64_t)te.n_merges);
        gguf_w_arr_i32(c.w, "tokenizer.added_ids", te.added_ids, (uint64_t)te.n_added);
        gguf_w_arr_str(c.w, "tokenizer.added_str", te.added_str, (uint64_t)te.n_added);
        gguf_w_arr_u8(c.w, "tokenizer.added_special", te.added_special, (uint64_t)te.n_added);
    }

    // ---------------- declarations ----------------
    char name[256];
    declare_mat(&c, "token_embd.weight", o->t_embed, Q_VOCAB, Q_H);
    declare_mat(&c, "output.weight", o->t_head, Q_VOCAB, Q_H);
    declare_f32(&c, "output_norm", Q_H);
    for (int l = 0; l < NL; l++) declare_layer(&c, l);
    // experts last: they arrive last from the download and dominate the file
    for (int l = 1; l < NL; l++) {
        const uint64_t blob = convert_expert_blob_bytes(xtype[l]);
        uint64_t ne[2] = {blob, Q_N_EXPERT};
        snprintf(name, sizeof(name), "blk.%d.experts", l);
        gguf_w_tensor(c.w, name, INZ_T_EXPERTS, 2, ne, blob * Q_N_EXPERT);
    }
    if (!gguf_w_header(c.w)) inz_die("convert: header write failed");
    progress(&c, "header written");

    // ---------------- data ----------------
    write_mat(&c, SRC "embed_tokens.weight", NULL, o->t_embed, Q_VOCAB, Q_H);
    write_mat(&c, "lm_head.weight", "output.weight", o->t_head, Q_VOCAB, Q_H);
    write_vec(&c, SRC "norm.weight", Q_H);
    progress(&c, "embeddings and head");
    for (int l = 0; l < NL; l++) {
        write_layer(&c, l);
        if (l % 8 == 7 || l == NL - 1) {
            char msg[64];
            snprintf(msg, sizeof(msg), "layers 0-%d dense part", l);
            progress(&c, msg);
        }
    }

    // ---------------- experts ----------------
    uint8_t *layer_buf = inz_aligned(4096, convert_expert_blob_bytes(o->t_experts_bf16 > 0 ? o->t_experts_bf16 : o->t_experts) * Q_N_EXPERT +
                                               convert_expert_blob_bytes(o->t_experts) * Q_N_EXPERT);
    for (int l = 1; l < NL; l++) {
        const uint64_t blob = convert_expert_blob_bytes(xtype[l]);
        char base[128], dbase[32];
        snprintf(base, sizeof(base), SRC "layers.%d.mlp.experts.", l);
        snprintf(dbase, sizeof(dbase), "blk.%d.experts.", l);
        // wait for the whole layer first (the reads below run in parallel)
        for (int e = 0; e < Q_N_EXPERT; e++) {
            static const char *nm[3] = {"gate_proj", "up_proj", "down_proj"};
            for (int w = 0; w < 3; w++) {
                char t[512];
                snprintf(t, sizeof(t), "%s%d.%s.weight_packed", base, e, nm[w]);
                if (st_find(m, t)) {
                    need(m, "%s", t);
                    need(m, "%s%d.%s.weight_scale", base, e, nm[w]);
                    need(m, "%s%d.%s.weight_global_scale", base, e, nm[w]);
                } else need(m, "%s%d.%s.weight", base, e, nm[w]);
            }
        }
        memset(layer_buf, 0, blob * Q_N_EXPERT);
        int nthreads = o->threads > 0 ? o->threads : omp_get_max_threads();
        #pragma omp parallel for num_threads(nthreads) schedule(dynamic, 1)
        for (int e = 0; e < Q_N_EXPERT; e++) write_expert(m, base, e, xtype[l], dbase, layer_buf + (size_t)e * blob);
        put(&c, layer_buf, blob * Q_N_EXPERT);
        char msg[80];
        snprintf(msg, sizeof(msg), "layer %d experts (%s, %.2f MiB/expert)", l, inz_qtype_name(xtype[l]), blob / 1048576.0);
        progress(&c, msg);
    }
    free(layer_buf);
    if (!gguf_writer_close(c.w)) inz_die("convert: finalize failed");
    progress(&c, "done");
    tok_free(tok);
    free(tmpl);
    st_close(m);
    return 0;
}

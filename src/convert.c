// inz convert: the official Nemotron 3.5 Super VL NVFP4 checkpoint (modelopt mixed precision: NVFP4 experts and
// shared experts, FP8 Mamba and attention projections and LM head, BF16 rest) -> inferenZ GGUF. The vision tower is
// not converted: the engine runs the language model.
//
// The converter streams: it declares every tensor up front, then writes them in order, waiting for each source
// tensor to finish downloading (tools/hfget.py journal) when needed. By default every tensor keeps its exact
// checkpoint values: NVFP4 is repacked as Q4N, FP8 copied as F8T, BF16 matrices stored as F16 (F16 has more
// mantissa bits; only magnitudes below 6e-5 lose relative precision, by less than 3e-8 in absolute terms). Routed
// experts are written as one 4 KiB-aligned blob per expert (up, down), so the runtime fetches a missed expert with
// one O_DIRECT read.
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

#define SRC "language_model.backbone."
#define SRC_MTP "language_model.mtp."

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
    o->t_mamba = INZ_T_F8T;
    o->t_attn = INZ_T_F8T;
    o->t_shared = INZ_T_Q4N;
    o->t_latent = INZ_T_F16;
    o->t_router = INZ_T_F16;
    o->t_mtp = INZ_T_F16;
    o->t_experts = INZ_T_Q4N;
    o->t_head = INZ_T_F8T;
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
    uint64_t b = mat_bytes(type, Q_FF, Q_LAT) + mat_bytes(type, Q_LAT, Q_FF);
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
/* Blocks                                                              */
/* ------------------------------------------------------------------ */

// dst: "blk.N", "mtp.at" or "mtp.moe"; t_attn: attention storage (FP8 in the model, BF16 in the MTP layer)
static void declare_block(cvt *c, const char *dst, int kind, int t_attn) {
    const convert_opts *o = c->o;
    char n[96];
#define D(suffix) (snprintf(n, sizeof(n), "%s." suffix, dst), n)
    declare_f32(c, D("norm"), Q_H);
    if (kind == QB_MAMBA) {
        declare_mat(c, D("mb.in"), o->t_mamba, Q_MB_PROJ, Q_H);
        declare_mat(c, D("mb.out"), o->t_mamba, Q_H, Q_MB_INNER);
        declare_mat(c, D("mb.conv"), INZ_T_F32, Q_MB_CONV, Q_MB_KS);
        declare_f32(c, D("mb.conv_bias"), Q_MB_CONV);
        declare_f32(c, D("mb.A_log"), Q_MB_NH);
        declare_f32(c, D("mb.D"), Q_MB_NH);
        declare_f32(c, D("mb.dt_bias"), Q_MB_NH);
        declare_f32(c, D("mb.norm"), Q_MB_INNER);
    } else if (kind == QB_ATTN) {
        declare_mat(c, D("at.q"), t_attn, Q_QD, Q_H);
        declare_mat(c, D("at.k"), t_attn, Q_KV, Q_H);
        declare_mat(c, D("at.v"), t_attn, Q_KV, Q_H);
        declare_mat(c, D("at.o"), t_attn, Q_H, Q_QD);
    } else {
        declare_mat(c, D("router"), o->t_router, Q_N_EXPERT, Q_H);
        declare_f32(c, D("router_bias"), Q_N_EXPERT);
        declare_mat(c, D("lat_in"), o->t_latent, Q_LAT, Q_H);
        declare_mat(c, D("lat_out"), o->t_latent, Q_H, Q_LAT);
        declare_mat(c, D("sh.up"), o->t_shared, Q_SHARED_FF, Q_H);
        declare_mat(c, D("sh.down"), o->t_shared, Q_H, Q_SHARED_FF);
    }
#undef D
}

// src: the checkpoint prefix of the block ("language_model.backbone.layers.N." or "language_model.mtp.layers.0.")
static void write_block(cvt *c, const char *src, const char *dst, int kind, int t_attn) {
    const convert_opts *o = c->o;
    char n[256], d[96];
#define S(suffix) (snprintf(n, sizeof(n), "%s" suffix, src), n)
#define D(suffix) (snprintf(d, sizeof(d), "%s." suffix, dst), d)
    write_vec(c, S("norm.weight"), Q_H);
    if (kind == QB_MAMBA) {
        write_mat(c, S("mixer.in_proj.weight"), D("mb.in"), o->t_mamba, Q_MB_PROJ, Q_H);
        write_mat(c, S("mixer.out_proj.weight"), D("mb.out"), o->t_mamba, Q_H, Q_MB_INNER);
        write_mat(c, S("mixer.conv1d.weight"), NULL, INZ_T_F32, Q_MB_CONV, Q_MB_KS);
        write_vec(c, S("mixer.conv1d.bias"), Q_MB_CONV);
        write_vec(c, S("mixer.A_log"), Q_MB_NH);
        write_vec(c, S("mixer.D"), Q_MB_NH);
        write_vec(c, S("mixer.dt_bias"), Q_MB_NH);
        write_vec(c, S("mixer.norm.weight"), Q_MB_INNER);
    } else if (kind == QB_ATTN) {
        write_mat(c, S("mixer.q_proj.weight"), D("at.q"), t_attn, Q_QD, Q_H);
        write_mat(c, S("mixer.k_proj.weight"), D("at.k"), t_attn, Q_KV, Q_H);
        write_mat(c, S("mixer.v_proj.weight"), D("at.v"), t_attn, Q_KV, Q_H);
        write_mat(c, S("mixer.o_proj.weight"), D("at.o"), t_attn, Q_H, Q_QD);
    } else {
        write_mat(c, S("mixer.gate.weight"), D("router"), o->t_router, Q_N_EXPERT, Q_H);
        write_vec(c, S("mixer.gate.e_score_correction_bias"), Q_N_EXPERT);
        write_mat(c, S("mixer.fc1_latent_proj.weight"), D("lat_in"), o->t_latent, Q_LAT, Q_H);
        write_mat(c, S("mixer.fc2_latent_proj.weight"), D("lat_out"), o->t_latent, Q_H, Q_LAT);
        write_mat(c, S("mixer.shared_experts.up_proj.weight"), D("sh.up"), o->t_shared, Q_SHARED_FF, Q_H);
        write_mat(c, S("mixer.shared_experts.down_proj.weight"), D("sh.down"), o->t_shared, Q_H, Q_SHARED_FF);
    }
#undef S
#undef D
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
    int n_moe = 0;
    for (int l = 0; l < NL; l++) n_moe += q_kind(l) == QB_MOE;

    // ---------------- metadata ----------------
    gguf_w_str(c.w, "general.architecture", Q_ARCH);
    gguf_w_str(c.w, "general.name", "NVIDIA Nemotron 3.5 Super 120B-A12B (inferenZ)");
    gguf_w_str(c.w, "inz.version", INZ_VERSION);
    gguf_w_u32(c.w, "inz.n_layer", (uint32_t)NL);
    gguf_w_u32(c.w, "inz.n_layer_full", Q_N_LAYER);
    gguf_w_u32(c.w, "inz.n_embd", Q_H);
    gguf_w_u32(c.w, "inz.n_expert", Q_N_EXPERT);
    gguf_w_u32(c.w, "inz.expert_type", (uint32_t)o->t_experts);
    gguf_w_u64(c.w, "inz.expert_blob_bytes", convert_expert_blob_bytes(o->t_experts));
    char recipe[256];
    snprintf(recipe, sizeof(recipe), "mamba=%s attn=%s shared=%s latent=%s router=%s mtp=%s experts=%s head=%s embed=%s%s",
             inz_qtype_name(o->t_mamba), inz_qtype_name(o->t_attn), inz_qtype_name(o->t_shared), inz_qtype_name(o->t_latent),
             inz_qtype_name(o->t_router), inz_qtype_name(o->t_mtp), inz_qtype_name(o->t_experts), inz_qtype_name(o->t_head),
             inz_qtype_name(o->t_embed), o->imatrix ? " imatrix" : "");
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
    // MTP: an attention block and a MoE block for speculative decoding ("mtp.*", experts "mtp.experts")
    const bool with_mtp = NL == Q_N_LAYER && st_find(m, SRC_MTP "layers.0.eh_proj.weight");
    char name[256];
    declare_mat(&c, "token_embd.weight", o->t_embed, Q_VOCAB, Q_H);
    declare_mat(&c, "output.weight", o->t_head, Q_VOCAB, Q_H);
    declare_f32(&c, "output_norm", Q_H);
    for (int l = 0; l < NL; l++) {
        snprintf(name, sizeof(name), "blk.%d", l);
        declare_block(&c, name, q_kind(l), o->t_attn);
    }
    if (with_mtp) {
        declare_mat(&c, "mtp.eh", o->t_mtp, Q_H, 2 * Q_H);
        declare_f32(&c, "mtp.enorm", Q_H);
        declare_f32(&c, "mtp.hnorm", Q_H);
        declare_f32(&c, "mtp.final_norm", Q_H);
        declare_block(&c, "mtp.at", QB_ATTN, o->t_mtp);
        declare_block(&c, "mtp.moe", QB_MOE, o->t_mtp);
        gguf_w_u32(c.w, "inz.mtp_layers", Q_MTP_LAYERS);
    }
    // experts last: they arrive last from the download and dominate the file
    const uint64_t blob = convert_expert_blob_bytes(o->t_experts);
    for (int k = 0; k <= n_moe; k++) {
        if (k == n_moe && !with_mtp) break;
        uint64_t ne[2] = {blob, Q_N_EXPERT};
        if (k < n_moe) snprintf(name, sizeof(name), "moe.%d.experts", k);
        else snprintf(name, sizeof(name), "mtp.experts");
        gguf_w_tensor(c.w, name, INZ_T_EXPERTS, 2, ne, blob * Q_N_EXPERT);
    }
    if (!gguf_w_header(c.w)) inz_die("convert: header write failed");
    progress(&c, "header written");

    // ---------------- data ----------------
    write_mat(&c, SRC "embeddings.weight", NULL, o->t_embed, Q_VOCAB, Q_H);
    write_mat(&c, "language_model.lm_head.weight", "output.weight", o->t_head, Q_VOCAB, Q_H);
    write_vec(&c, SRC "norm_f.weight", Q_H);
    progress(&c, "embeddings and head");
    for (int l = 0; l < NL; l++) {
        char src[128], dst[32];
        snprintf(src, sizeof(src), SRC "layers.%d.", l);
        snprintf(dst, sizeof(dst), "blk.%d", l);
        write_block(&c, src, dst, q_kind(l), o->t_attn);
        if (l % 8 == 7 || l == NL - 1) {
            char msg[64];
            snprintf(msg, sizeof(msg), "blocks 0-%d dense part", l);
            progress(&c, msg);
        }
    }
    if (with_mtp) {
        write_mat(&c, SRC_MTP "layers.0.eh_proj.weight", "mtp.eh", o->t_mtp, Q_H, 2 * Q_H);
        write_vec(&c, SRC_MTP "layers.0.enorm.weight", Q_H);
        write_vec(&c, SRC_MTP "layers.0.hnorm.weight", Q_H);
        write_vec(&c, SRC_MTP "layers.1.final_layernorm.weight", Q_H);
        write_block(&c, SRC_MTP "layers.0.", "mtp.at", QB_ATTN, o->t_mtp);
        write_block(&c, SRC_MTP "layers.1.", "mtp.moe", QB_MOE, o->t_mtp);
        progress(&c, "MTP dense part");
    }

    // ---------------- experts ----------------
    const int et = o->t_experts;
    const uint64_t b_up = mat_bytes(et, Q_FF, Q_LAT);
    uint8_t *layer_buf = inz_aligned(4096, blob * Q_N_EXPERT);
    for (int k = 0, l = 0; k <= n_moe; k++) {
        if (k == n_moe && !with_mtp) break;
        char base[128], dbase[32];
        if (k < n_moe) {
            while (q_kind(l) != QB_MOE) l++;
            snprintf(base, sizeof(base), SRC "layers.%d.mixer.experts.", l++);
            snprintf(dbase, sizeof(dbase), "moe.%d.experts.", k);
        } else {
            snprintf(base, sizeof(base), SRC_MTP "layers.1.mixer.experts.");
            snprintf(dbase, sizeof(dbase), "mtp.experts.");
        }
        // wait for the whole layer first (the reads below run in parallel)
        static const char *nm[2] = {"up_proj", "down_proj"};
        for (int e = 0; e < Q_N_EXPERT; e++)
            for (int w = 0; w < 2; w++) {
                need(m, "%s%d.%s.weight", base, e, nm[w]);
                need(m, "%s%d.%s.weight_scale", base, e, nm[w]);
                need(m, "%s%d.%s.weight_scale_2", base, e, nm[w]);
            }
        memset(layer_buf, 0, blob * Q_N_EXPERT);
        int nthreads = o->threads > 0 ? o->threads : omp_get_max_threads();
        #pragma omp parallel for num_threads(nthreads) schedule(dynamic, 1)
        for (int e = 0; e < Q_N_EXPERT; e++) {
            char src[512], dst[64];
            uint8_t *out = layer_buf + (size_t)e * blob;
            snprintf(src, sizeof(src), "%s%d.up_proj.weight", base, e);
            snprintf(dst, sizeof(dst), "%s%d.up", dbase, e);
            convert_mat(m, src, dst, et, Q_FF, Q_LAT, 1, out);
            snprintf(src, sizeof(src), "%s%d.down_proj.weight", base, e);
            snprintf(dst, sizeof(dst), "%s%d.down", dbase, e);
            convert_mat(m, src, dst, et, Q_LAT, Q_FF, 1, out + b_up);
        }
        put(&c, layer_buf, blob * Q_N_EXPERT);
        char msg[64];
        if (k < n_moe) snprintf(msg, sizeof(msg), "MoE block %d experts (%.2f MiB/expert)", k, blob / 1048576.0);
        else snprintf(msg, sizeof(msg), "MTP experts");
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

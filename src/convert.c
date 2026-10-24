// inz convert: official Qwen3.8-Flash-Next safetensors (FP8 repository: block-scaled E4M3 experts, BF16 rest)
// -> inferenZ GGUF.
//
// The converter streams: it declares every tensor up front, then writes them in order, waiting for each
// source tensor to finish downloading (tools/hfget.py journal) when needed. Routed experts are written as one
// 4 KiB-aligned blob per expert (gate|up, down), so the runtime fetches a missed expert with one O_DIRECT read.
// BF16 sources stored as F16 keep their exact values (BF16 has 7 mantissa bits, F16 has 10).
#define _GNU_SOURCE
#include "convert.h"

#include <math.h>
#include <omp.h>
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

#define SRC "model.language_model."

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

// Any float-like source tensor as f32. FP8 E4M3 weights are multiplied by their BF16 weight_scale_inv,
// one per 128x128 block.
static float *load(st_model *m, const char *name, int64_t *rows_out, int64_t *cols_out) {
    const st_tensor *t = need(m, "%s", name);
    const int64_t n = st_numel(t);
    const int64_t rows = t->ndim >= 2 ? t->shape[0] : 1, cols = t->ndim >= 2 ? n / t->shape[0] : n;
    uint8_t *b = read_all(m, t);
    float *f = inz_malloc(sizeof(float) * (size_t)n);
    switch (t->dtype) {
    case ST_F32: memcpy(f, b, sizeof(float) * (size_t)n); break;
    case ST_BF16: for (int64_t i = 0; i < n; i++) f[i] = inz_bf16_to_f32(((uint16_t *)b)[i]); break;
    case ST_F16: for (int64_t i = 0; i < n; i++) f[i] = inz_f16_to_f32(((uint16_t *)b)[i]); break;
    case ST_F8_E4M3: {
        char sname[600];
        snprintf(sname, sizeof(sname), "%s_scale_inv", name);
        const st_tensor *st = st_find(m, sname);
        float *sc = NULL;
        int64_t sr = 1, scn = 1;
        if (st) sc = load(m, sname, &sr, &scn);
        const int64_t br = st ? (rows + sr - 1) / sr : rows, bc = st ? (cols + scn - 1) / scn : cols;
        for (int64_t r = 0; r < rows; r++)
            for (int64_t c = 0; c < cols; c++) {
                float v = e4m3_lut[b[r * cols + c]];
                if (sc) v *= sc[(r / br) * scn + c / bc];
                f[r * cols + c] = v;
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
    o->t_dense = INZ_T_F16;
    o->t_hc = INZ_T_F16;
    o->t_shared = INZ_T_F16;
    o->t_router = INZ_T_F16;
    o->t_ple = INZ_T_F16;
    o->t_experts = INZ_T_Q8G;
    o->t_head = INZ_T_Q8G;
    o->t_head_screen = 0;     // none: the exact head is read whole (--screen Q4L for the two-stage head)
    o->t_embed = INZ_T_Q8G;
    o->max_layers = Q_N_LAYER;
    o->rot8 = true;
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

// Encode W (rows x cols, f32) as `type` into out (mat_bytes bytes). rot8: Hadamard-rotate Q8G as well.
// imp: optional per-column importance (mean squared input), which weights the quantization error.
static void encode(const float *W, int rows, int cols, int type, bool rot8, int threads, const float *imp, void *out) {
    if (type == INZ_T_F16) {
        uint16_t *h = out;
        for (size_t i = 0; i < (size_t)rows * cols; i++) h[i] = inz_f32_to_f16(W[i]);
        return;
    }
    if (type == INZ_T_F32) { memcpy(out, W, sizeof(float) * (size_t)rows * cols); return; }
    inz_quant_params p = {.type = type, .rot_block = type == INZ_T_Q8G ? (rot8 ? inz_auto_rot_block(cols) : -1) : 0,
                          .threads = threads, .imatrix = imp};
    inz_quantize(W, rows, cols, &p, out);
}

// dst: the GGUF name (for the importance matrix)
static void write_mat(cvt *c, const char *src, const char *dst, int type, int rows, int cols, bool rot8) {
    int64_t r, cc;
    float *W = load(c->m, src, &r, &cc);
    if (r * cc != (int64_t)rows * cols) inz_die("convert: %s has %lldx%lld, expected %dx%d", src, (long long)r, (long long)cc, rows, cols);
    const uint64_t nb = mat_bytes(type, rows, cols);
    void *buf = inz_aligned(64, nb);
    encode(W, rows, cols, type, rot8, c->o->threads, imx_get(dst, cols), buf);
    put(c, buf, nb);
    free(buf);
    free(W);
}

// fuse two sources with the same input into one matrix (rows of a, then rows of b)
static void write_fused(cvt *c, const char *sa, const char *sb, const char *dst, int type, int rows_each, int cols, bool rot8) {
    int64_t r, cc;
    float *A = load(c->m, sa, &r, &cc);
    float *B = load(c->m, sb, &r, &cc);
    float *AB = inz_malloc(sizeof(float) * 2 * (size_t)rows_each * cols);
    memcpy(AB, A, sizeof(float) * (size_t)rows_each * cols);
    memcpy(AB + (size_t)rows_each * cols, B, sizeof(float) * (size_t)rows_each * cols);
    const uint64_t nb = mat_bytes(type, 2 * rows_each, cols);
    void *buf = inz_aligned(64, nb);
    encode(AB, 2 * rows_each, cols, type, rot8, c->o->threads, imx_get(dst, cols), buf);
    put(c, buf, nb);
    free(buf); free(A); free(B); free(AB);
}

// small f32 tensor; zero_centered: the model multiplies by (1 + w), which is folded in here
static void write_vec(cvt *c, const char *src, int n, bool zero_centered) {
    int64_t r, cc;
    float *f = load(c->m, src, &r, &cc);
    if (r * cc != n) inz_die("convert: %s has %lld elements, expected %d", src, (long long)(r * cc), n);
    if (zero_centered) for (int i = 0; i < n; i++) f[i] += 1.0f;
    put(c, f, (uint64_t)n * 4);
    free(f);
}

static void progress(cvt *c, const char *what) {
    fprintf(stderr, "inz convert: [%7.1fs] %s\n", inz_now() - c->t0, what);
}

/* ------------------------------------------------------------------ */
/* One decoder layer (the 48 of the model and the MTP layer)            */
/* ------------------------------------------------------------------ */

static void declare_hc(cvt *c, const char *k, bool inject) {
    char n[96];
    snprintf(n, sizeof(n), "%s.norm", k); declare_f32(c, n, Q_HCW);
    snprintf(n, sizeof(n), "%s.down", k); declare_mat(c, n, c->o->t_hc, Q_HC_RANK, Q_HCW);
    snprintf(n, sizeof(n), "%s.up", k); declare_mat(c, n, c->o->t_hc, Q_HCW, Q_HC_RANK);
    if (inject) { snprintf(n, sizeof(n), "%s.inject", k); declare_mat(c, n, INZ_T_F32, Q_HC, Q_HCW); }
}

// dst: "blk.N" or "mtp.blk"
static void declare_block(cvt *c, const char *dst, bool attn, bool ple) {
    const convert_opts *o = c->o;
    char n[96];
#define D(suffix) (snprintf(n, sizeof(n), "%s." suffix, dst), n)
    declare_hc(c, D("ahc"), true);
    declare_hc(c, D("mhc"), true);
    if (attn) {
        declare_mat(c, D("at.q"), o->t_dense, Q_QPROJ, Q_H);
        declare_mat(c, D("at.kv"), o->t_dense, 2 * Q_KV, Q_H);
        declare_mat(c, D("at.idx"), o->t_dense, Q_IDX_PROJ, Q_H);
        declare_mat(c, D("at.o"), o->t_dense, Q_H, Q_NH * Q_HD);
        declare_f32(c, D("at.q_norm"), Q_HD);
        declare_f32(c, D("at.k_norm"), Q_HD);
        declare_f32(c, D("at.iq_norm"), Q_IDX_HD);
        declare_f32(c, D("at.ik_norm"), Q_IDX_HD);
    } else {
        declare_mat(c, D("dn.qkv"), o->t_dense, Q_DN_CONV, Q_H);
        declare_mat(c, D("dn.z"), o->t_dense, Q_DN_VAL, Q_H);
        declare_mat(c, D("dn.ab"), INZ_T_F32, 2 * Q_DN_NV, Q_H);
        declare_mat(c, D("dn.conv"), INZ_T_F32, Q_DN_CONV, Q_DN_KS);
        declare_f32(c, D("dn.A_log"), Q_DN_NV);
        declare_f32(c, D("dn.dt_bias"), Q_DN_NV);
        declare_f32(c, D("dn.norm"), Q_DN_DV);
        declare_mat(c, D("dn.out"), o->t_dense, Q_H, Q_DN_VAL);
    }
    declare_mat(c, D("router"), o->t_router, Q_N_EXPERT, Q_H);
    declare_mat(c, D("sh.gate_up"), o->t_shared, 2 * Q_SHARED_FF, Q_H);
    declare_mat(c, D("sh.down"), o->t_shared, Q_H, Q_SHARED_FF);
    declare_f32(c, D("sh.gate"), Q_H);
#undef D
    if (ple) {
        declare_mat(c, "ple.key", o->t_ple, Q_HCW, Q_H);
        declare_mat(c, "ple.value", o->t_ple, Q_H, Q_H);
        declare_f32(c, "ple.norm_key", Q_HCW);
        declare_f32(c, "ple.norm_query", Q_HCW);
        declare_f32(c, "ple.norm_conv", Q_HCW);
        declare_mat(c, "ple.conv", INZ_T_F32, Q_HCW, Q_PLE_CONV_K);
    }
}

// src: the checkpoint prefix of the module ("model.language_model.layers.N." or "mtp.layers.0.")
// dst: the GGUF prefix ("blk.N.ahc", "mixer", ...)
static void write_hc(cvt *c, const char *src, const char *dst, bool inject) {
    const convert_opts *o = c->o;
    char n[256], d[96];
    snprintf(n, sizeof(n), "%shc_norm.weight", src); write_vec(c, n, Q_HCW, true);
    snprintf(n, sizeof(n), "%sinput_mix_weight_down.weight", src); snprintf(d, sizeof(d), "%s.down", dst);
    write_mat(c, n, d, o->t_hc, Q_HC_RANK, Q_HCW, o->rot8);
    snprintf(n, sizeof(n), "%sinput_mix_weight_up.weight", src); snprintf(d, sizeof(d), "%s.up", dst);
    write_mat(c, n, d, o->t_hc, Q_HCW, Q_HC_RANK, o->rot8);
    if (inject) { snprintf(n, sizeof(n), "%sblock_inject_weight.weight", src); write_mat(c, n, NULL, INZ_T_F32, Q_HC, Q_HCW, false); }
}

static void write_block(cvt *c, const char *src, const char *dst, bool attn, bool ple) {
    const convert_opts *o = c->o;
    char n[256], n2[256], d[96];
#define S(suffix) (snprintf(n, sizeof(n), "%s" suffix, src), n)
#define S2(suffix) (snprintf(n2, sizeof(n2), "%s" suffix, src), n2)
#define D(suffix) (snprintf(d, sizeof(d), "%s." suffix, dst), d)
    write_hc(c, S("attn_hyper_connection."), D("ahc"), true);
    write_hc(c, S("mlp_hyper_connection."), D("mhc"), true);
    if (attn) {
        write_mat(c, S("self_attn.q_proj.weight"), D("at.q"), o->t_dense, Q_QPROJ, Q_H, o->rot8);
        write_fused(c, S("self_attn.k_proj.weight"), S2("self_attn.v_proj.weight"), D("at.kv"), o->t_dense, Q_KV, Q_H, o->rot8);
        write_mat(c, S("self_attn.indexer.index_qk_proj.weight"), D("at.idx"), o->t_dense, Q_IDX_PROJ, Q_H, o->rot8);
        write_mat(c, S("self_attn.o_proj.weight"), D("at.o"), o->t_dense, Q_H, Q_NH * Q_HD, o->rot8);
        write_vec(c, S("self_attn.q_norm.weight"), Q_HD, true);
        write_vec(c, S("self_attn.k_norm.weight"), Q_HD, true);
        write_vec(c, S("self_attn.indexer.q_layernorm.weight"), Q_IDX_HD, true);
        write_vec(c, S("self_attn.indexer.k_layernorm.weight"), Q_IDX_HD, true);
    } else {
        write_mat(c, S("linear_attn.in_proj_qkv.weight"), D("dn.qkv"), o->t_dense, Q_DN_CONV, Q_H, o->rot8);
        write_mat(c, S("linear_attn.in_proj_z.weight"), D("dn.z"), o->t_dense, Q_DN_VAL, Q_H, o->rot8);
        // a | b fused and kept f32: they set every head's decay and update strength
        write_fused(c, S("linear_attn.in_proj_a.weight"), S2("linear_attn.in_proj_b.weight"), NULL, INZ_T_F32, Q_DN_NV, Q_H, false);
        write_mat(c, S("linear_attn.conv1d.weight"), NULL, INZ_T_F32, Q_DN_CONV, Q_DN_KS, false);
        write_vec(c, S("linear_attn.A_log"), Q_DN_NV, false);
        write_vec(c, S("linear_attn.dt_bias"), Q_DN_NV, false);
        write_vec(c, S("linear_attn.norm.weight"), Q_DN_DV, false);
        write_mat(c, S("linear_attn.out_proj.weight"), D("dn.out"), o->t_dense, Q_H, Q_DN_VAL, o->rot8);
    }
    write_mat(c, S("mlp.gate.weight"), D("router"), o->t_router, Q_N_EXPERT, Q_H, o->rot8);
    write_fused(c, S("mlp.shared_expert.gate_proj.weight"), S2("mlp.shared_expert.up_proj.weight"), D("sh.gate_up"), o->t_shared,
                Q_SHARED_FF, Q_H, o->rot8);
    write_mat(c, S("mlp.shared_expert.down_proj.weight"), D("sh.down"), o->t_shared, Q_H, Q_SHARED_FF, o->rot8);
    write_vec(c, S("mlp.shared_expert_gate.weight"), Q_H, false);
    if (ple) {
        write_mat(c, S("ple.key_proj.weight"), "ple.key", o->t_ple, Q_HCW, Q_H, o->rot8);
        write_mat(c, S("ple.value_proj.weight"), "ple.value", o->t_ple, Q_H, Q_H, o->rot8);
        write_vec(c, S("ple.norm_key.weight"), Q_HCW, true);
        write_vec(c, S("ple.norm_query.weight"), Q_HCW, true);
        write_vec(c, S("ple.norm_conv.weight"), Q_HCW, true);
        write_mat(c, S("ple.conv1d.weight"), NULL, INZ_T_F32, Q_HCW, Q_PLE_CONV_K, false);
    }
#undef S
#undef S2
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
    const bool with_ngram = NL > Q_PLE_LAYER && !o->no_ngram;

    // ---------------- metadata ----------------
    gguf_w_str(c.w, "general.architecture", Q_ARCH);
    gguf_w_str(c.w, "general.name", "Qwen3.8-Flash-Next (inferenZ)");
    gguf_w_str(c.w, "inz.version", INZ_VERSION);
    gguf_w_u32(c.w, "inz.n_layer", (uint32_t)NL);
    gguf_w_u32(c.w, "inz.n_layer_full", Q_N_LAYER);
    gguf_w_u32(c.w, "inz.n_embd", Q_H);
    gguf_w_u32(c.w, "inz.n_expert", Q_N_EXPERT);
    gguf_w_u32(c.w, "inz.expert_type", (uint32_t)o->t_experts);
    gguf_w_u64(c.w, "inz.expert_blob_bytes", convert_expert_blob_bytes(o->t_experts));
    char recipe[256];
    snprintf(recipe, sizeof(recipe), "dense=%s hc=%s shared=%s router=%s ple=%s experts=%s head=%s screen=%s embed=%s rot8=%d%s",
             inz_qtype_name(o->t_dense), inz_qtype_name(o->t_hc), inz_qtype_name(o->t_shared), inz_qtype_name(o->t_router),
             inz_qtype_name(o->t_ple), inz_qtype_name(o->t_experts), inz_qtype_name(o->t_head),
             o->t_head_screen > 0 ? inz_qtype_name(o->t_head_screen) : "none", inz_qtype_name(o->t_embed), o->rot8,
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
    // MTP: one more decoder layer (QSA + MoE) for speculative decoding, stored as "mtp.blk" (+ experts "mtp.experts")
    const bool with_mtp = NL == Q_N_LAYER && st_find(m, "mtp.fc_embedding.weight");
    char name[256];
#define DN(fmt, ...) (snprintf(name, sizeof(name), fmt, __VA_ARGS__), name)
    declare_mat(&c, "token_embd.weight", o->t_embed, Q_VOCAB, Q_H);
    declare_mat(&c, "output.weight", o->t_head, Q_VOCAB, Q_H);
    if (o->t_head_screen > 0) declare_mat(&c, "output_screen.weight", o->t_head_screen, Q_VOCAB, Q_H);
    declare_hc(&c, "mixer", false);
    for (int l = 0; l < NL; l++) {
        char blk[32];
        snprintf(blk, sizeof(blk), "blk.%d", l);
        declare_block(&c, blk, q_is_attn(l), l == Q_PLE_LAYER && with_ngram);
    }
    if (with_mtp) {
        declare_mat(&c, "mtp.fc_emb", o->t_dense, Q_H, Q_H);
        declare_mat(&c, "mtp.fc_hid", o->t_dense, Q_H, Q_H);
        declare_f32(&c, "mtp.norm_emb", Q_H);
        declare_f32(&c, "mtp.norm_hid", Q_HCW);
        declare_hc(&c, "mtp.mixer", false);
        declare_block(&c, "mtp.blk", true, false);
        gguf_w_u32(c.w, "inz.mtp_layers", Q_MTP_LAYERS);
    }
    // n-gram table: contiguous shards of raw FP8 rows
    int ng_shards = 0;
    if (with_ngram) {
        uint64_t ng_rows = 0;
        for (;; ng_shards++) {
            const st_tensor *t = st_find(m, DN(SRC "layers.%d.ple.ple_embedding.ngram_embedding.shard_%d.weight", Q_PLE_LAYER, ng_shards));
            if (!t) break;
            if (t->dtype != ST_F8_E4M3 || t->shape[1] != Q_NG_DIM) inz_die("convert: unexpected n-gram shard %s", t->name);
            ng_rows += (uint64_t)t->shape[0];
        }
        gguf_w_u32(c.w, "inz.ngram_shards", (uint32_t)ng_shards);
        uint64_t ne[2] = {Q_NG_DIM, ng_rows};
        gguf_w_tensor(c.w, "ngram.table", INZ_T_FP8, 2, ne, ng_rows * Q_NG_DIM);
        declare_f32(&c, "ngram.scale", 1);  // FP8Embedding: every row is multiplied by this per-tensor scale
    }
    // experts last: they arrive last from the download and dominate the file
    const uint64_t blob = convert_expert_blob_bytes(o->t_experts);
    for (int l = 0; l <= NL; l++) {
        if (l == NL && !with_mtp) break;
        uint64_t ne[2] = {blob, Q_N_EXPERT};
        gguf_w_tensor(c.w, l < NL ? DN("blk.%d.experts", l) : "mtp.experts", INZ_T_EXPERTS, 2, ne, blob * Q_N_EXPERT);
    }
    if (!gguf_w_header(c.w)) inz_die("convert: header write failed");
    progress(&c, "header written");

    // ---------------- data ----------------
    {
        // embeddings and head are row-addressed at runtime: never rotated
        int64_t r, cc;
        float *E = load(m, SRC "embed_tokens.weight", &r, &cc);
        void *buf = inz_aligned(64, mat_bytes(o->t_embed, Q_VOCAB, Q_H));
        encode(E, Q_VOCAB, Q_H, o->t_embed, false, o->threads, NULL, buf);
        put(&c, buf, mat_bytes(o->t_embed, Q_VOCAB, Q_H));
        free(buf);
        free(E);
        float *Hd = load(m, "lm_head.weight", &r, &cc);
        buf = inz_aligned(64, mat_bytes(o->t_head, Q_VOCAB, Q_H));
        encode(Hd, Q_VOCAB, Q_H, o->t_head, false, o->threads, imx_get("output.weight", Q_H), buf);
        put(&c, buf, mat_bytes(o->t_head, Q_VOCAB, Q_H));
        free(buf);
        if (o->t_head_screen > 0) {
            buf = inz_aligned(64, mat_bytes(o->t_head_screen, Q_VOCAB, Q_H));
            encode(Hd, Q_VOCAB, Q_H, o->t_head_screen, false, o->threads, imx_get("output.weight", Q_H), buf);
            put(&c, buf, mat_bytes(o->t_head_screen, Q_VOCAB, Q_H));
            free(buf);
        }
        free(Hd);
        progress(&c, "embeddings and head");
    }
    write_hc(&c, SRC "hyper_connection_mixer.", "mixer", false);
    for (int l = 0; l < NL; l++) {
        char src[128], dst[32];
        snprintf(src, sizeof(src), SRC "layers.%d.", l);
        snprintf(dst, sizeof(dst), "blk.%d", l);
        write_block(&c, src, dst, q_is_attn(l), l == Q_PLE_LAYER && with_ngram);
        char msg[64];
        snprintf(msg, sizeof(msg), "layer %d dense part", l);
        progress(&c, msg);
    }
    if (with_mtp) {
        write_mat(&c, "mtp.fc_embedding.weight", "mtp.fc_emb", o->t_dense, Q_H, Q_H, o->rot8);
        write_mat(&c, "mtp.fc_hidden.weight", "mtp.fc_hid", o->t_dense, Q_H, Q_H, o->rot8);
        write_vec(&c, "mtp.pre_fc_norm_embedding.weight", Q_H, true);
        write_vec(&c, "mtp.pre_fc_norm_hidden.weight", Q_HCW, true);
        write_hc(&c, "mtp.hyper_connection_mixer.", "mtp.mixer", false);
        write_block(&c, "mtp.layers.0.", "mtp.blk", true, false);
        progress(&c, "MTP dense part");
    }
    if (with_ngram) {
        // the runtime recomputes the hashing parameters: check them against the checkpoint's buffers
        int64_t mult[3], prime[Q_NG_HEADS], offset[Q_NG_HEADS];
        q_ngram_params(mult, prime, offset);
        static const char *buf_name[3] = {"layer_multipliers", "ngram_heads_vocab_sizes", "ngram_heads_offsets"};
        const int64_t *want[3] = {mult, prime, offset};
        const int want_n[3] = {3, Q_NG_HEADS, Q_NG_HEADS};
        for (int b = 0; b < 3; b++) {
            const st_tensor *t = need(m, SRC "layers.%d.ple.ple_embedding.%s", Q_PLE_LAYER, buf_name[b]);
            if (t->dtype != ST_I64 || st_numel(t) != want_n[b]) inz_die("convert: unexpected %s", t->name);
            int64_t *v = (int64_t *)read_all(m, t);
            for (int i = 0; i < want_n[b]; i++)
                if (v[i] != want[b][i]) inz_die("convert: n-gram %s[%d] = %lld, expected %lld", buf_name[b], i, (long long)v[i], (long long)want[b][i]);
            free(v);
        }
        for (int s = 0; s < ng_shards; s++) {
            const st_tensor *t = need(m, SRC "layers.%d.ple.ple_embedding.ngram_embedding.shard_%d.weight", Q_PLE_LAYER, s);
            uint8_t *b = read_all(m, t);
            if (!gguf_w_data(c.w, b, t->nbytes)) inz_die("convert: write failed");
            free(b);
            if (s % 16 == 15) {
                char msg[64];
                snprintf(msg, sizeof(msg), "n-gram table %d/%d shards", s + 1, ng_shards);
                progress(&c, msg);
            }
        }
        if (!gguf_w_tensor_done(c.w)) inz_die("convert: write failed");
        write_vec(&c, SRC "layers.1.ple.ple_embedding.ngram_embedding.weight_scale", 1, false);
    }

    // ---------------- experts ----------------
    const int et = o->t_experts;
    const uint64_t b_gu = mat_bytes(et, 2 * Q_FF, Q_H);
    uint8_t *layer_buf = inz_aligned(4096, blob * Q_N_EXPERT);
    for (int l = 0; l <= NL; l++) {
        if (l == NL && !with_mtp) break;
        char base[128], dbase[32];
        if (l < NL) snprintf(base, sizeof(base), SRC "layers.%d.mlp.experts.", l);
        else snprintf(base, sizeof(base), "mtp.layers.0.mlp.experts.");
        if (l < NL) snprintf(dbase, sizeof(dbase), "blk.%d.experts.", l);
        else snprintf(dbase, sizeof(dbase), "mtp.experts.");
        // wait for the whole layer first (the reads below run in parallel)
        static const char *nm[3] = {"gate_proj", "up_proj", "down_proj"};
        for (int e = 0; e < Q_N_EXPERT; e++)
            for (int w = 0; w < 3; w++) {
                need(m, "%s%d.%s.weight", base, e, nm[w]);
                need(m, "%s%d.%s.weight_scale_inv", base, e, nm[w]);
            }
        memset(layer_buf, 0, blob * Q_N_EXPERT);
        int nthreads = o->threads > 0 ? o->threads : omp_get_max_threads();
        #pragma omp parallel for num_threads(nthreads) schedule(dynamic, 1)
        for (int e = 0; e < Q_N_EXPERT; e++) {
            char nb[512];
            int64_t r, cc;
            snprintf(nb, sizeof(nb), "%s%d.gate_proj.weight", base, e);
            float *G = load(m, nb, &r, &cc);
            snprintf(nb, sizeof(nb), "%s%d.up_proj.weight", base, e);
            float *U = load(m, nb, &r, &cc);
            snprintf(nb, sizeof(nb), "%s%d.down_proj.weight", base, e);
            float *D = load(m, nb, &r, &cc);
            float *GU = inz_malloc(sizeof(float) * 2 * Q_FF * Q_H);
            memcpy(GU, G, sizeof(float) * Q_FF * Q_H);
            memcpy(GU + (size_t)Q_FF * Q_H, U, sizeof(float) * Q_FF * Q_H);
            uint8_t *dst = layer_buf + (size_t)e * blob;
            char in1[64], in2[64];
            snprintf(in1, sizeof(in1), "%s%d.gate_up", dbase, e);
            snprintf(in2, sizeof(in2), "%s%d.down", dbase, e);
            encode(GU, 2 * Q_FF, Q_H, et, o->rot8, 1, imx_get(in1, Q_H), dst);
            encode(D, Q_H, Q_FF, et, o->rot8, 1, imx_get(in2, Q_FF), dst + b_gu);
            free(G); free(U); free(D); free(GU);
        }
        put(&c, layer_buf, blob * Q_N_EXPERT);
        char msg[64];
        if (l < NL) snprintf(msg, sizeof(msg), "layer %d experts (%.2f MiB/expert)", l, blob / 1048576.0);
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

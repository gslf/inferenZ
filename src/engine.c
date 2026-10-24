// Engine lifecycle (load, memory budgeting, expert cache) and the generation loop.
#define _GNU_SOURCE
#include "engine_internal.h"

#include <fcntl.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include "diag.h"
#include "util.h"

void inz_opts_default(inz_opts *o) {
    memset(o, 0, sizeof(*o));
    o->model_path = "models/qwen3.8-flash-next-q4q8.inz.gguf";  // the recommended recipe (README)
    o->ctx = 32768;
    o->ram_reserve_gb = 0;    // auto: max(10 GB, 30% of RAM)
    o->cache_decay = 0.01f;
    o->sqpoll_cpu = -2;       // auto: a low-power E-core
    o->spec = true;
    o->spec_k = 1;
    o->lookahead = 1;
    o->hotlist = "auto";      // ~/.cache/inferenz/<model file>.<size>.scores
}

// Per-model warm-start file: scores of one model must never seed another one.
static const char *hotlist_path(const char *opt, const char *model, uint64_t size) {
    static char hot[1024];
    if (!opt || !*opt || !strcmp(opt, "off")) return NULL;
    if (strcmp(opt, "auto")) return opt;
    const char *home = getenv("HOME");
    if (!home) return NULL;
    snprintf(hot, sizeof(hot), "%s/.cache/inferenz", home);
    mkdir(hot, 0755);
    const char *base = strrchr(model, '/');
    base = base ? base + 1 : model;
    snprintf(hot, sizeof(hot), "%s/.cache/inferenz/%s.%llu.scores", home, base, (unsigned long long)size);
    return hot;
}

static int build_tokenizer(inz_engine *e) {
    gguf_file *g = e->g;
    const gguf_kv *pb = gguf_find_kv(g, "tokenizer.piece_bytes"), *pl = gguf_find_kv(g, "tokenizer.piece_len");
    const gguf_kv *ma = gguf_find_kv(g, "tokenizer.merge_a"), *mb = gguf_find_kv(g, "tokenizer.merge_b"),
                  *mr = gguf_find_kv(g, "tokenizer.merge_res");
    const gguf_kv *ai = gguf_find_kv(g, "tokenizer.added_ids"), *as = gguf_find_kv(g, "tokenizer.added_str"),
                  *asp = gguf_find_kv(g, "tokenizer.added_special");
    if (!pb || !pl || !ma || !mb || !mr || !ai || !as || !asp) return -1;
    int nv = (int)pl->arr_n;
    const int32_t *lens = (const int32_t *)pl->data;
    const char **pieces = inz_calloc((size_t)nv, sizeof(char *));
    uint32_t *plen = inz_calloc((size_t)nv, sizeof(uint32_t));
    const char *blob = (const char *)pb->data;
    uint64_t off = 0;
    for (int i = 0; i < nv; i++) {
        if (lens[i] < 0) continue;
        pieces[i] = blob + off;
        plen[i] = (uint32_t)lens[i];
        off += (uint64_t)lens[i];
    }
    int na = (int)ai->arr_n;
    const char **astr = inz_calloc((size_t)na + 1, sizeof(char *));
    char **owned = inz_calloc((size_t)na + 1, sizeof(char *));
    const uint8_t *p = as->data;
    for (int i = 0; i < na; i++) {
        const char *s;
        uint64_t n = gguf_arr_next_str(&p, &s);
        owned[i] = inz_malloc(n + 1);
        memcpy(owned[i], s, n);
        owned[i][n] = 0;
        astr[i] = owned[i];
    }
    e->tok = tok_create(nv, pieces, plen, (int)ma->arr_n, (const int32_t *)ma->data, (const int32_t *)mb->data,
                        (const int32_t *)mr->data, na, (const int32_t *)ai->data, astr, (const uint8_t *)asp->data);
    tok_set_split(e->tok, (int)gguf_get_int(g, "tokenizer.split", TOK_SPLIT_QWEN));
    for (int i = 0; i < na; i++) free(owned[i]);
    free(owned);
    free(astr);
    free(pieces);
    free(plen);
    return 0;
}

/* ------------------------------------------------------------------ */
/* Dense weights                                                       */
/* ------------------------------------------------------------------ */

// Dense tensors are copied into one anonymous arena so the page cache can be dropped and the OS never evicts
// weights that every token needs. Routed experts, the embedding table and the n-gram table stay in the file.
typedef struct { const gguf_tensor *t; size_t off; } dense_item;
typedef struct { inz_engine *e; dense_item *items; int n; } dense_set;

static bool resident(const gguf_file *g, const gguf_tensor *t, bool exact_head) {
    if (t->type == INZ_T_EXPERTS || !strcmp(t->name, "token_embd.weight") || !strcmp(t->name, "ngram.table")) return false;
    const bool screen = gguf_find_tensor(g, "output_screen.weight") != NULL;
    if (!strcmp(t->name, "output_screen.weight")) return !exact_head;
    if (!strcmp(t->name, "output.weight")) return exact_head || !screen;  // two-stage: exact rows read on demand
    return true;
}

static const uint8_t *dense(const dense_set *d, const char *fmt, int l, const gguf_tensor **tout) {
    char name[128];
    snprintf(name, sizeof(name), fmt, l);
    for (int k = 0; k < d->n; k++)
        if (!strcmp(d->items[k].t->name, name)) {
            if (tout) *tout = d->items[k].t;
            return d->e->dense_mem + d->items[k].off;
        }
    inz_die("model: missing tensor %s", name);
}

static const float *dvec(const dense_set *d, const char *fmt, int l, uint64_t n) {
    const gguf_tensor *t;
    const float *p = (const float *)dense(d, fmt, l, &t);
    uint64_t ne = 1;
    for (uint32_t i = 0; i < t->n_dims; i++) ne *= t->ne[i];
    if (t->type != INZ_T_F32 || ne != n) inz_die("model: %s has an unexpected type or size", t->name);
    return p;
}

static void dlin(const dense_set *d, lin *m, const char *fmt, int l, int rows, int cols) {
    const gguf_tensor *t;
    const uint8_t *p = dense(d, fmt, l, &t);
    if (t->ne[0] != (uint64_t)cols || t->ne[1] != (uint64_t)rows || !q_bind_lin(m, (int)t->type, rows, cols, p))
        inz_die("model: %s is not a %dx%d matrix", t->name, rows, cols);
    if (d->e->imx_path) m->imx = q_imx_register(d->e, t->name, cols);
}

static void bind_hc(const dense_set *d, hcw *w, const char *prefix, bool inject) {
    char f[64];
    snprintf(f, sizeof(f), "%s.norm", prefix);
    w->norm = dvec(d, f, 0, Q_HCW);
    snprintf(f, sizeof(f), "%s.down", prefix);
    dlin(d, &w->down, f, 0, Q_HC_RANK, Q_HCW);
    snprintf(f, sizeof(f), "%s.up", prefix);
    dlin(d, &w->up, f, 0, Q_HCW, Q_HC_RANK);
    w->inject = NULL;
    if (inject) {
        snprintf(f, sizeof(f), "%s.inject", prefix);
        w->inject = dvec(d, f, 0, (uint64_t)Q_HC * Q_HCW);
    }
}

// one decoder layer: "blk.N" or "mtp.blk"
static void bind_block(const dense_set *d, q_layer *L, const char *pfx, bool attn) {
    char f[64];
#define N(suffix) (snprintf(f, sizeof(f), "%s." suffix, pfx), f)
    bind_hc(d, &L->ahc, N("ahc"), true);
    bind_hc(d, &L->mhc, N("mhc"), true);
    if (attn) {
        dlin(d, &L->at_q, N("at.q"), 0, Q_QPROJ, Q_H);
        dlin(d, &L->at_kv, N("at.kv"), 0, 2 * Q_KV, Q_H);
        dlin(d, &L->at_idx, N("at.idx"), 0, Q_IDX_PROJ, Q_H);
        dlin(d, &L->at_o, N("at.o"), 0, Q_H, Q_NH * Q_HD);
        L->q_norm = dvec(d, N("at.q_norm"), 0, Q_HD);
        L->k_norm = dvec(d, N("at.k_norm"), 0, Q_HD);
        L->iq_norm = dvec(d, N("at.iq_norm"), 0, Q_IDX_HD);
        L->ik_norm = dvec(d, N("at.ik_norm"), 0, Q_IDX_HD);
    } else {
        dlin(d, &L->dn_qkv, N("dn.qkv"), 0, Q_DN_CONV, Q_H);
        dlin(d, &L->dn_z, N("dn.z"), 0, Q_DN_VAL, Q_H);
        dlin(d, &L->dn_out, N("dn.out"), 0, Q_H, Q_DN_VAL);
        L->dn_ab = dvec(d, N("dn.ab"), 0, 2ull * Q_DN_NV * Q_H);
        L->dn_conv = dvec(d, N("dn.conv"), 0, (uint64_t)Q_DN_CONV * Q_DN_KS);
        const float *A_log = dvec(d, N("dn.A_log"), 0, Q_DN_NV);
        for (int h = 0; h < Q_DN_NV; h++) L->dn_negA[h] = -expf(A_log[h]);
        L->dn_dt = dvec(d, N("dn.dt_bias"), 0, Q_DN_NV);
        L->dn_norm = dvec(d, N("dn.norm"), 0, Q_DN_DV);
    }
    dlin(d, &L->router, N("router"), 0, Q_N_EXPERT, Q_H);
    dlin(d, &L->sh_gu, N("sh.gate_up"), 0, 2 * Q_SHARED_FF, Q_H);
    dlin(d, &L->sh_down, N("sh.down"), 0, Q_H, Q_SHARED_FF);
    L->sh_gate = dvec(d, N("sh.gate"), 0, Q_H);
#undef N
}

// Recipe studies: INZ_OVERRIDE="other.gguf:pattern1,pattern2" takes the dense tensors whose name contains a
// pattern from another conversion of the same model (e.g. exact DeltaNet projections into a 4-bit recipe).
static gguf_file *g_ovr;
static char g_ovr_pat[16][32];
static int g_n_ovr;

static const gguf_tensor *override_of(const gguf_tensor *t) {
    for (int k = 0; k < g_n_ovr; k++)
        if (strstr(t->name, g_ovr_pat[k])) {
            const gguf_tensor *o = gguf_find_tensor(g_ovr, t->name);
            if (!o) inz_die("override: %s has no tensor %s", g_ovr->path, t->name);
            return o;
        }
    return NULL;
}

static void override_open(void) {
    const char *spec = getenv("INZ_OVERRIDE");
    if (!spec || !*spec || g_ovr) return;
    char path[1024];
    const char *colon = strrchr(spec, ':');
    if (!colon) inz_die("INZ_OVERRIDE must be file.gguf:pattern[,pattern...]");
    snprintf(path, sizeof(path), "%.*s", (int)(colon - spec), spec);
    char err[256];
    g_ovr = gguf_open(path, err, sizeof(err));
    if (!g_ovr) inz_die("override: %s", err);
    for (const char *p = colon + 1; *p && g_n_ovr < 16;) {
        const char *c = strchr(p, ',');
        const size_t l = c ? (size_t)(c - p) : strlen(p);
        snprintf(g_ovr_pat[g_n_ovr++], sizeof(g_ovr_pat[0]), "%.*s", (int)l, p);
        p += l + (c ? 1 : 0);
    }
    LOGI("override: dense tensors matching %d pattern(s) come from %s", g_n_ovr, path);
}

static void load_dense(inz_engine *e) {
    gguf_file *g = e->g;
    override_open();
    dense_set d = {e, inz_malloc(sizeof(dense_item) * g->n_tensors), 0};
    size_t total = 0;
    for (uint64_t i = 0; i < g->n_tensors; i++) {
        const gguf_tensor *t = &g->tensors[i];
        if (!resident(g, t, e->opt.exact_head)) continue;
        const gguf_tensor *o = g_ovr ? override_of(t) : NULL;
        d.items[d.n].t = o ? o : t;
        d.items[d.n].off = total;
        total += INZ_ALIGN_UP(d.items[d.n].t->nbytes, 64);
        d.n++;
    }
    e->dense_bytes = total;
    e->dense_mem = inz_mmap_anon(total, true);
    int fd = open(g->path, O_RDONLY), ofd = g_ovr ? open(g_ovr->path, O_RDONLY) : -1;
    const double t0 = inz_now();
    for (int i = 0; i < d.n; i++) {
        const gguf_tensor *t = d.items[i].t;
        const bool ov = g_ovr && t >= g_ovr->tensors && t < g_ovr->tensors + g_ovr->n_tensors;
        const gguf_file *src = ov ? g_ovr : g;
        const int sfd = ov ? ofd : fd;
        uint64_t foff = src->data_offset + t->offset, done = 0;
        while (done < t->nbytes) {
            ssize_t r = pread(sfd, e->dense_mem + d.items[i].off + done, t->nbytes - done, (off_t)(foff + done));
            if (r <= 0) inz_die("model: read failed for %s", t->name);
            done += (uint64_t)r;
        }
        posix_fadvise(sfd, (off_t)foff, (off_t)t->nbytes, POSIX_FADV_DONTNEED);
    }
    close(fd);
    if (ofd >= 0) close(ofd);
    if (e->opt.mlock && mlock(e->dense_mem, total) != 0) LOGI("mlock of dense weights failed (RLIMIT_MEMLOCK); continuing");
    LOGI("dense weights: %.2f GiB loaded in %.1f s", total / 1073741824.0, inz_now() - t0);

    bind_hc(&d, &e->mixer, "mixer", false);
    for (int l = 0; l < e->n_layer; l++) {
        char pfx[16];
        snprintf(pfx, sizeof(pfx), "blk.%d", l);
        bind_block(&d, &e->L[l], pfx, q_is_attn(l));
    }
    q_mtp *M = &e->mtp;
    M->present = gguf_find_tensor(g, "mtp.fc_emb") != NULL;
    if (M->present) {
        dlin(&d, &M->fc_emb, "mtp.fc_emb", 0, Q_H, Q_H);
        dlin(&d, &M->fc_hid, "mtp.fc_hid", 0, Q_H, Q_H);
        M->norm_emb = dvec(&d, "mtp.norm_emb", 0, Q_H);
        M->norm_hid = dvec(&d, "mtp.norm_hid", 0, Q_HCW);
        bind_hc(&d, &M->mixer, "mtp.mixer", false);
        bind_block(&d, &M->L, "mtp.blk", true);
        LOGI("MTP layer loaded: speculative decoding available");
    }
    if (gguf_find_tensor(g, "output_screen.weight") && !e->opt.exact_head) {
        // two-stage head: the screen in RAM, exact rows from the file (page cache) for the candidates only
        dlin(&d, &e->head_screen, "output_screen.weight", 0, Q_VOCAB, Q_H);
        const gguf_tensor *t = gguf_find_tensor(g, "output.weight");
        e->two_stage = true;
        e->head_fd = open(g->path, O_RDONLY);
        e->head_off = g->data_offset + t->offset;
        e->head.type = (int)t->type;
        e->head.rows = Q_VOCAB;
        e->head.cols = Q_H;
        if (t->type != INZ_T_F16) {
            if (pread(e->head_fd, &e->head_hdr, sizeof(e->head_hdr), (off_t)e->head_off) != (ssize_t)sizeof(e->head_hdr) ||
                !inz_qmat_view(&e->head.q, &e->head_hdr) || e->head.q.rot_block)
                inz_die("model: output.weight must be F16 or an unrotated quantized matrix");
        }
        e->head_sigma = 0.5;
        LOGI("two-stage LM head: %s screen in RAM, exact %s rows on demand", inz_qtype_name(e->head_screen.type),
             inz_qtype_name(e->head.type));
    } else {
        dlin(&d, &e->head, "output.weight", 0, Q_VOCAB, Q_H);
    }
    // per-layer n-gram embedding: projections in RAM, the 51 GB table mapped from the file
    const gguf_tensor *tab = gguf_find_tensor(g, "ngram.table");
    q_ple *P = &e->ple;
    P->present = tab && e->n_layer > Q_PLE_LAYER;
    if (P->present) {
        dlin(&d, &P->key, "ple.key", 0, Q_HCW, Q_H);
        dlin(&d, &P->value, "ple.value", 0, Q_H, Q_H);
        P->norm_key = dvec(&d, "ple.norm_key", 0, Q_HCW);
        P->norm_query = dvec(&d, "ple.norm_query", 0, Q_HCW);
        P->norm_conv = dvec(&d, "ple.norm_conv", 0, Q_HCW);
        P->conv = dvec(&d, "ple.conv", 0, (uint64_t)Q_HCW * Q_PLE_CONV_K);
        P->scale = dvec(&d, "ngram.scale", 0, 1)[0];
        P->table = tab->data;
        P->rows = tab->ne[1];
        q_ngram_params(P->mult, P->prime, P->offset);
        if (P->offset[Q_NG_HEADS - 1] + P->prime[Q_NG_HEADS - 1] > (int64_t)P->rows || tab->ne[0] != Q_NG_DIM)
            inz_die("model: n-gram table has %llu rows, the hashing needs %lld", (unsigned long long)P->rows,
                    (long long)(P->offset[Q_NG_HEADS - 1] + P->prime[Q_NG_HEADS - 1]));
        // 16 random 160-byte rows per token: no readahead
        const uintptr_t a = (uintptr_t)P->table & ~(uintptr_t)4095;
        madvise((void *)a, (size_t)((uintptr_t)P->table + tab->nbytes - a), MADV_RANDOM);
        LOGI("n-gram table: %.1f GiB mapped (%llu rows, scale %.3g)", tab->nbytes / 1073741824.0, (unsigned long long)P->rows, P->scale);
    } else if (e->n_layer > Q_PLE_LAYER) {
        inz_die("model: the n-gram table is missing (converted with --no-ngram?)");
    }
    free(d.items);
}

static void setup_embed(inz_engine *e) {
    const gguf_tensor *t = gguf_find_tensor(e->g, "token_embd.weight");
    if (!t) inz_die("model: missing token_embd.weight");
    e->embed_fd = open(e->g->path, O_RDONLY);
    e->embed_offset = e->g->data_offset + t->offset;
    e->embed_type = (int)t->type;
    if (t->type == INZ_T_F16) {
        if (t->ne[0] != Q_H || t->ne[1] != Q_VOCAB) inz_die("model: token_embd.weight is not %dx%d", Q_VOCAB, Q_H);
        return;
    }
    e->embed_hdr = inz_malloc(sizeof(inz_qhdr));
    if (pread(e->embed_fd, e->embed_hdr, sizeof(inz_qhdr), (off_t)e->embed_offset) != (ssize_t)sizeof(inz_qhdr))
        inz_die("model: cannot read embedding header");
    // geometry only: q/d offsets relative to the header locate rows for pread
    if (!inz_qmat_view(&e->embed, e->embed_hdr) || e->embed.rows != Q_VOCAB || e->embed.cols != Q_H || e->embed.rot_block)
        inz_die("model: token_embd.weight must be an unrotated quantized %dx%d matrix", Q_VOCAB, Q_H);
}

static void setup_experts(inz_engine *e, uint64_t other_bytes) {
    gguf_file *g = e->g;
    e->expert_type = (int)gguf_get_int(g, "inz.expert_type", INZ_T_Q8G);
    e->blob_bytes = (uint64_t)gguf_get_int(g, "inz.expert_blob_bytes", 0);
    e->gu_bytes = inz_mat_bytes(e->expert_type, 2 * Q_FF, Q_H);
    if (e->gu_bytes + inz_mat_bytes(e->expert_type, Q_H, Q_FF) > e->blob_bytes || e->blob_bytes % 4096)
        inz_die("model: expert blob size %llu does not fit type %s", (unsigned long long)e->blob_bytes, inz_qtype_name(e->expert_type));
    const int nl = e->n_layer + (e->mtp.present ? 1 : 0);  // the MTP layer's experts are layer n_layer
    uint64_t *off = inz_malloc(sizeof(uint64_t) * (size_t)nl * Q_N_EXPERT);
    for (int l = 0; l < nl; l++) {
        char name[64];
        if (l < e->n_layer) snprintf(name, sizeof(name), "blk.%d.experts", l);
        else snprintf(name, sizeof(name), "mtp.experts");
        const gguf_tensor *t = gguf_find_tensor(g, name);
        if (!t || t->nbytes < e->blob_bytes * Q_N_EXPERT) inz_die("model: missing or short %s", name);
        for (int x = 0; x < Q_N_EXPERT; x++) off[(size_t)l * Q_N_EXPERT + x] = g->data_offset + t->offset + (uint64_t)x * e->blob_bytes;
    }
    const uint64_t total = (uint64_t)nl * Q_N_EXPERT * e->blob_bytes;
    uint64_t budget = e->opt.cache_bytes;
    if (!budget) {
        // The engine's whole footprint stays within MemTotal - reserve. The reserve keeps the desktop and the page
        // cache out of swap (measured on the 62 GB laptop: sizing from MemAvailable minus 3 GB swapped 12 GB of
        // other programs out and once got the server OOM-killed). MemAvailable alone is not a safe base: right
        // after another program was swapped out it looks free.
        uint64_t reserve = (uint64_t)(e->opt.ram_reserve_gb * 1073741824.0);
        const uint64_t ram = inz_mem_total(), rss = inz_rss(), avail = inz_mem_available();
        if (!reserve) {
            reserve = ram / 10 * 3;
            if (reserve < (10ull << 30)) reserve = 10ull << 30;
        }
        const uint64_t cap = ram > reserve + rss + other_bytes ? ram - reserve - rss - other_bytes : 0;
        const uint64_t now = avail > other_bytes + (2ull << 30) ? avail - other_bytes - (2ull << 30) : 0;
        budget = cap < now ? cap : now;
        if (budget < 2ull * Q_N_EXPERT * e->blob_bytes) budget = 2ull * Q_N_EXPERT * e->blob_bytes;
    }
    const bool all = budget >= total;
    e->ec_all_resident = all;
    ec_config cfg = {.path = g->path, .n_layer = nl, .n_expert = Q_N_EXPERT, .offset = off, .blob_bytes = e->blob_bytes,
                     .cache_bytes = all ? total : budget, .queue_depth = 64, .sqpoll_cpu = e->opt.sqpoll_cpu,
                     .decay = e->opt.cache_decay, .all_resident = all};
    char err[256];
    e->ec = ec_open(&cfg, err, sizeof(err));
    free(off);
    if (!e->ec) inz_die("expert cache: %s", err);
    LOGI("experts: %s, %.2f MiB each, %d layers x %d = %.1f GiB on SSD, cache %.1f GiB (%.0f%%)", inz_qtype_name(e->expert_type),
         e->blob_bytes / 1048576.0, nl, Q_N_EXPERT, total / 1073741824.0, ec_slot_count(e->ec) * (double)e->blob_bytes / 1073741824.0,
         100.0 * ec_slot_count(e->ec) / (double)(nl * Q_N_EXPERT));
    if (!all) {
        g_diag.peak_ssd_gbs = ec_probe_bandwidth(e->ec, 24);
        LOGI("SSD expert-read bandwidth probe: %.2f GB/s", g_diag.peak_ssd_gbs);
        if (e->opt.hotlist) {
            const double t0 = inz_now();
            const int n = ec_load_and_preload(e->ec, e->opt.hotlist);
            if (n) LOGI("warm start: preloaded %d experts from %s in %.1f s", n, e->opt.hotlist, inz_now() - t0);
        }
    } else {
        int32_t *pairs = inz_malloc(sizeof(int32_t) * 2 * (size_t)nl * Q_N_EXPERT);
        int k = 0;
        for (int l = 0; l < nl; l++)
            for (int x = 0; x < Q_N_EXPERT; x++) { pairs[2 * k] = l; pairs[2 * k + 1] = x; k++; }
        const double t0 = inz_now();
        ec_preload(e->ec, pairs, k);
        LOGI("all experts resident (%.1f s)", inz_now() - t0);
        free(pairs);
    }
}

/* ------------------------------------------------------------------ */
/* Sequence state                                                      */
/* ------------------------------------------------------------------ */

static const size_t S_FLOATS = (size_t)Q_N_DN * Q_DN_NV * Q_DN_DK * Q_DN_DV;
static const size_t CONV_FLOATS = (size_t)Q_N_DN * (Q_DN_KS - 1) * Q_DN_CONV;
static const size_t PLE_FLOATS = 9 * (size_t)Q_HCW;
#define N_ATT (Q_N_ATTN + 1)  // the model's QSA layers + the MTP layer

static void rec_alloc(q_state *q) {
    q->S = inz_mmap_anon(S_FLOATS * 4, true);
    q->conv = inz_mmap_anon(CONV_FLOATS * 4, false);
    q->ple_conv = inz_mmap_anon(PLE_FLOATS * 4, false);
}

static void rec_free(q_state *q) {
    inz_munmap(q->S, S_FLOATS * 4);
    inz_munmap(q->conv, CONV_FLOATS * 4);
    inz_munmap(q->ple_conv, PLE_FLOATS * 4);
}

static void rec_copy(q_state *dst, const q_state *src) {
    memcpy(dst->S, src->S, S_FLOATS * 4);
    memcpy(dst->conv, src->conv, CONV_FLOATS * 4);
    memcpy(dst->ple_conv, src->ple_conv, PLE_FLOATS * 4);
}

static void state_alloc(inz_engine *e) {
    q_state *st = &e->st;
    const size_t ctx = (size_t)e->ctx;
    rec_alloc(st);
    // pages are touched lazily: an unused context or MTP layer costs no memory
    st->k = inz_mmap_anon((size_t)N_ATT * ctx * Q_KV * 2, true);
    st->v = inz_mmap_anon((size_t)N_ATT * ctx * Q_KV * 2, true);
    st->ik = inz_mmap_anon((size_t)N_ATT * ctx * Q_IDX_HD * 4, true);
    st->bk = inz_mmap_anon((size_t)N_ATT * (ctx / Q_IDX_BLOCK) * Q_IDX_HD * 4, true);
    rec_alloc(&e->ck);
    for (int i = 0; i < Q_SPEC_MAX; i++) rec_alloc(&e->snap[i]);
    e->ck_x = inz_aligned(64, sizeof(float) * Q_HCW);
    e->x_last = inz_aligned(64, sizeof(float) * Q_HCW);
    e->Xm = inz_aligned(64, sizeof(float) * (size_t)Q_MAX_T * Q_HCW);
    e->ck_pos = -1;
    e->hist = inz_malloc(sizeof(int32_t) * ctx);
    e->state_bytes = 2 * (S_FLOATS + CONV_FLOATS + PLE_FLOATS) * 4 + (size_t)N_ATT * ctx * (2 * Q_KV * 2 + Q_IDX_HD * 4 + Q_IDX_HD);
}

static void state_free(inz_engine *e) {
    q_state *st = &e->st;
    const size_t ctx = (size_t)e->ctx;
    rec_free(st);
    inz_munmap(st->k, (size_t)N_ATT * ctx * Q_KV * 2);
    inz_munmap(st->v, (size_t)N_ATT * ctx * Q_KV * 2);
    inz_munmap(st->ik, (size_t)N_ATT * ctx * Q_IDX_HD * 4);
    inz_munmap(st->bk, (size_t)N_ATT * (ctx / Q_IDX_BLOCK) * Q_IDX_HD * 4);
    rec_free(&e->ck);
    for (int i = 0; i < Q_SPEC_MAX; i++) rec_free(&e->snap[i]);
    free(e->ck_x);
    free(e->x_last);
    free(e->Xm);
    free(e->hist);
}

// The attention caches are positional (entries below a position stay valid when the sequence is rewound there);
// the recurrent state is not, so it is saved before the last prompt token and restored to rewind.
static void checkpoint_save(inz_engine *e) {
    rec_copy(&e->ck, &e->st);
    memcpy(e->ck_x, e->x_last, sizeof(float) * Q_HCW);
    e->ck_pos = e->n_past;
}

static void checkpoint_restore(inz_engine *e) {
    rec_copy(&e->st, &e->ck);
    memcpy(e->x_last, e->ck_x, sizeof(float) * Q_HCW);
    e->n_past = e->ck_pos;
    // MTP slot p needs the token at p + 1: slots from ck_pos - 1 on are recomputed
    if (e->mtp_valid > e->ck_pos - 1) e->mtp_valid = e->ck_pos > 0 ? e->ck_pos - 1 : 0;
}

/* ------------------------------------------------------------------ */
/* Open / close                                                        */
/* ------------------------------------------------------------------ */

inz_engine *engine_open(const inz_opts *o, char *err, size_t errlen) {
    inz_engine *e = inz_calloc(1, sizeof(*e));
    e->opt = *o;
    e->state = "loading";
    e->g = gguf_open(o->model_path, err, errlen);
    if (!e->g) { free(e); return NULL; }
    const char *arch = gguf_get_str(e->g, "general.architecture", "");
    if (strcmp(arch, Q_ARCH)) {
        snprintf(err, errlen, "%s: architecture '%s' is not %s (inferenZ runs one model)", o->model_path, arch, Q_ARCH);
        gguf_close(e->g);
        free(e);
        return NULL;
    }
    e->n_layer = (int)gguf_get_int(e->g, "inz.n_layer", Q_N_LAYER);
    if (e->n_layer < 1 || e->n_layer > Q_N_LAYER || gguf_get_int(e->g, "inz.n_embd", 0) != Q_H) {
        snprintf(err, errlen, "%s: unexpected shapes (n_layer %d)", o->model_path, e->n_layer);
        gguf_close(e->g);
        free(e);
        return NULL;
    }
    e->opt.hotlist = hotlist_path(o->hotlist, o->model_path, e->g->size);
    e->imx_path = getenv("INZ_IMATRIX");
    if (e->imx_path && !*e->imx_path) e->imx_path = NULL;
    snprintf(e->model_name, sizeof(e->model_name), "qwen3.8-flash-next");
    e->chat_template = gguf_get_str(e->g, "tokenizer.chat_template", NULL);
    if (build_tokenizer(e) != 0) { snprintf(err, errlen, "tokenizer metadata missing"); gguf_close(e->g); free(e); return NULL; }
    // threads: P + E cores by default (LP-E cores serve I/O and HTTP)
    tp_topology topo;
    tp_detect_topology(&topo);
    int cpus[256], nt;
    if (o->threads > 0) { nt = o->threads; e->tp = tp_create(nt, NULL); }
    else { nt = tp_default_cpus(&topo, cpus, 256); e->tp = tp_create(nt, cpus); }
    LOGI("threads: %d of %d cpus (P %d threads on %d cores, E %d, LP-E %d kept for I/O), kernels %s", nt, topo.n_cpu,
         topo.n_p, topo.n_p_cores, topo.n_e, topo.n_lp, inz_kernel_isa());
    // io_uring submission thread on the last low-power core when there is one
    if (e->opt.sqpoll_cpu == -2) {
        e->opt.sqpoll_cpu = -1;
        for (int i = topo.n_cpu - 1; i >= 0; i--) if (topo.cpu_class[i] == 2) { e->opt.sqpoll_cpu = topo.cpu_id[i]; break; }
    }
    diag_probe_ram(e->tp);
    diag_init(Q_N_LAYER + 1, Q_N_EXPERT, Q_TOPK);  // + the MTP layer
    if (o->route_trace) g_diag.route_trace = fopen(o->route_trace, "w");
    LOGI("RAM bandwidth probe: %.1f GB/s", g_diag.peak_ram_gbs);
    load_dense(e);
    setup_embed(e);
    e->ctx = o->ctx > 0 ? (o->ctx + Q_IDX_BLOCK - 1) / Q_IDX_BLOCK * Q_IDX_BLOCK : 32768;
    state_alloc(e);
    q_scratch_alloc(e);
    q_state_reset(e);
    if (e->imx_path) {
        const size_t nl = (size_t)e->n_layer + (e->mtp.present ? 1 : 0);
        e->imx_exp = inz_calloc(nl * Q_N_EXPERT * (Q_H + Q_FF), sizeof(double));
        e->imx_exp_n = inz_calloc(nl * Q_N_EXPERT, sizeof(uint64_t));
        LOGI("imatrix: collecting input statistics into %s", e->imx_path);
    }
    // state pages are touched lazily; budget the full context so the cache never starves long chats
    setup_experts(e, e->state_bytes + (2ull << 30) + (e->imx_path ? (uint64_t)Q_N_LAYER * Q_N_EXPERT * (Q_H + Q_FF) * 8 : 0));
    e->t_loaded = inz_now();
    e->state = "idle";
    return e;
}

void engine_close(inz_engine *e) {
    if (!e) return;
    if (e->imx_path) q_imx_write(e);
    if (e->ec && e->opt.hotlist && ec_save_scores(e->ec, e->opt.hotlist)) LOGI("expert scores saved to %s", e->opt.hotlist);
    ec_close(e->ec);
    q_scratch_free(e);
    state_free(e);
    inz_munmap(e->dense_mem, e->dense_bytes);
    if (e->embed_fd > 0) close(e->embed_fd);
    if (e->head_fd > 0) close(e->head_fd);
    free(e->embed_hdr);
    tok_free(e->tok);
    tp_destroy(e->tp);
    if (g_diag.route_trace) fclose(g_diag.route_trace);
    gguf_close(e->g);
    free(e);
}

tokenizer *engine_tokenizer(inz_engine *e) { return e->tok; }
const char *engine_model_name(inz_engine *e) { return e->model_name; }
const char *engine_chat_template(inz_engine *e) { return e->chat_template; }
int engine_eos(inz_engine *e) { (void)e; return Q_IM_END; }
int engine_eot(inz_engine *e) { (void)e; return Q_EOS; }
const char *engine_state(inz_engine *e) { return e->state; }
int engine_vocab(void) { return Q_VOCAB; }

/* ------------------------------------------------------------------ */
/* Generation                                                          */
/* ------------------------------------------------------------------ */

static bool is_stop(const gen_opts *g, int tok) {
    for (int i = 0; i < g->n_stop; i++) if (g->stop_ids[i] == tok) return true;
    return false;
}

static bool use_mtp(const inz_engine *e) { return e->mtp.present && e->opt.spec; }

// After a main forward of positions [p0, p0 + n) (rows in e->X): fill the MTP slots [p0 - 1, p0 + n - 1), whose
// next tokens are now known. x_prev is the final residual of position p0 - 1. The MTP cache stays one slot
// behind the main one: the last slot waits for the next token.
static void mtp_fill(inz_engine *e, int p0, int n, const float *x_prev) {
    if (!use_mtp(e) || e->mtp_valid < 0) return;
    const int first = p0 > 0 ? p0 - 1 : 0, last = p0 + n - 1;  // slots [first, last)
    if (last <= first) return;
    if (e->mtp_valid != first) {               // a gap (state restored from elsewhere): MTP stays off until a reset
        LOGI("MTP cache has a gap (%d filled, %d needed): speculative decoding off until the next reset", e->mtp_valid, first);
        e->mtp_valid = -1;
        return;
    }
    float *X = e->Xm;
    int m = 0;
    if (p0 > 0) memcpy(X + (size_t)m++ * Q_HCW, x_prev, sizeof(float) * Q_HCW);
    memcpy(X + (size_t)m * Q_HCW, e->X, sizeof(float) * Q_HCW * (size_t)(n - 1));
    m += n - 1;
    q_mtp_forward(e, X, Q_HCW, e->hist + first + 1, m, first, NULL, 0, NULL);
    e->mtp_valid = last;
}

// one main forward of tokens at e->n_past with MTP bookkeeping
static void forward_tracked(inz_engine *e, const int32_t *tok, int n, const int *rows, int nr, float *logits) {
    float xp[Q_HCW];
    const int p0 = e->n_past;
    if (use_mtp(e) && p0 > 0) memcpy(xp, e->x_last, sizeof(xp));
    q_forward(e, tok, n, rows, nr, logits);
    mtp_fill(e, p0, n, xp);
}

static void prefill(inz_engine *e, const int32_t *toks, int from, int to) {
    for (int i = from; i < to;) {
        const int n = to - i < Q_MAX_T ? to - i : Q_MAX_T;
        forward_tracked(e, toks + i, n, NULL, 0, NULL);
        g_diag.tokens_prefilled += (uint64_t)n;
        g_diag.phase_tokens[PH_PREFILL] += (uint64_t)n;
        i += n;
    }
}

typedef struct { int n; int idx[64]; float prob[64]; float *dense; } dist;  // a sampling distribution, dense view

// Speculative sampling (Leviathan et al.): the drafted token d ~ q is accepted with probability min(1, p(d)/q(d));
// on rejection the token is drawn from normalize(max(0, p - q)). The output follows p exactly.
static bool spec_accept(sampler *smp, const float *p_dense, const int *pidx, int pn, const float *q_dense, int d, int *out) {
    const float pd = p_dense[d], qd = q_dense[d];
    if (qd > 0 && sampler_rand_f(smp) * qd < pd) return true;
    double tot = 0;
    for (int i = 0; i < pn; i++) { const float r = p_dense[pidx[i]] - q_dense[pidx[i]]; if (r > 0) tot += r; }
    if (tot <= 0) { *out = pidx[0]; return false; }
    double u = sampler_rand_f(smp) * tot, acc = 0;
    *out = pidx[pn - 1];
    for (int i = 0; i < pn; i++) {
        const float r = p_dense[pidx[i]] - q_dense[pidx[i]];
        if (r <= 0) continue;
        acc += r;
        if (u < acc) { *out = pidx[i]; break; }
    }
    return false;
}

int engine_generate(inz_engine *e, const int32_t *prompt, int n_prompt, const gen_opts *g, gen_cb cb, void *ud,
                    gen_result *res) {
    memset(res, 0, sizeof(*res));
    res->prompt_tokens = n_prompt;
    if (n_prompt <= 0) { res->finish_reason = "stop"; return 0; }
    if (n_prompt >= e->ctx) { res->finish_reason = "length"; return -1; }
    g_diag.requests++;
    diag_request_start();
    // reuse the longest common prefix: continue the current state when it is a prefix of the prompt, else rewind
    // to the checkpoint when that lies inside the common part, else start over
    int keep = 0;
    while (keep < e->n_past && keep < n_prompt && e->hist[keep] == prompt[keep]) keep++;
    if (!(keep == e->n_past && keep < n_prompt)) {
        if (e->ck_pos >= 0 && e->ck_pos <= keep && e->ck_pos < n_prompt) checkpoint_restore(e);
        else q_state_reset(e);
    }
    if (use_mtp(e) && e->n_past > 0 && e->mtp_valid != e->n_past - 1) q_state_reset(e);  // regain the MTP cache
    res->cached_tokens = e->n_past;
    const double t0 = inz_now();
    e->state = "prefill";
    diag_set_phase(PH_PREFILL);
    const int K = g->spec && use_mtp(e) ? (e->opt.spec_k < 1 ? 1 : e->opt.spec_k > Q_SPEC_MAX ? Q_SPEC_MAX : e->opt.spec_k) : 0;
    float *logits = inz_malloc(sizeof(float) * (size_t)(Q_SPEC_MAX + 1) * Q_VOCAB);
    float *qlog = inz_malloc(sizeof(float) * (size_t)Q_SPEC_MAX * Q_VOCAB);
    float *pd = inz_calloc(Q_VOCAB, sizeof(float)), *qd = inz_calloc((size_t)Q_SPEC_MAX * Q_VOCAB, sizeof(float));
    prefill(e, prompt, e->n_past, n_prompt - 1);
    checkpoint_save(e);
    int rows[Q_SPEC_MAX + 1] = {0, 1, 2, 3};
    forward_tracked(e, prompt + n_prompt - 1, 1, rows, 1, logits);
    g_diag.tokens_prefilled++;
    g_diag.phase_tokens[PH_PREFILL]++;
    res->t_prefill = inz_now() - t0;
    g_diag.prefill_tps_last = res->t_prefill > 0 ? (n_prompt - res->cached_tokens) / res->t_prefill : 0;
    e->state = "decode";
    diag_set_phase(PH_DECODE);
    sampler smp;
    sampler_init(&smp, &g->sp, Q_VOCAB);
    const double td = inz_now();
    g_diag.last_token_time = td;
    res->finish_reason = "length";
    int hf = e->n_past > 256 ? e->n_past - 256 : 0;
    uint64_t ts = diag_tsc();
    int tok = sampler_sample(&smp, logits, e->hist + hf, e->n_past - hf);
    diag_add(DG_SAMPLE, -1, ts, 0, 0);
    res->ttft = inz_now() - t0;
    g_diag.ttft_last = res->ttft;
    int m = 1;  // rows of the last main forward in e->X: positions n_past - m .. n_past - 1, their MTP slots pending
    for (;;) {
        if (is_stop(g, tok)) { res->finish_reason = "stop"; break; }
        res->gen_tokens++;
        diag_token_emitted();
        if (!cb(ud, tok)) { res->finish_reason = "stop"; break; }
        if (res->gen_tokens >= g->max_tokens) { res->finish_reason = "length"; break; }
        const int P = e->n_past;
        if (P + 1 >= e->ctx) { res->finish_reason = "length"; break; }
        int k = K;
        if (k > g->max_tokens - res->gen_tokens) k = g->max_tokens - res->gen_tokens;
        if (P + k + 1 >= e->ctx) k = 0;
        int32_t vt[Q_SPEC_MAX + 1];
        vt[0] = tok;
        // MTP slots P - m .. P - 1 now have their next tokens (the last one is tok); the last row drafts d0,
        // chained MTP steps on the MTP's own residual draft d1, d2
        if (use_mtp(e) && e->mtp_valid == P - m) {
            const uint64_t tdr = diag_tsc();
            int32_t nxt[Q_SPEC_MAX + 1];
            for (int i = 0; i < m - 1; i++) nxt[i] = e->hist[P - m + 1 + i];
            nxt[m - 1] = tok;
            const int lr = m - 1, r0 = 0;
            q_mtp_forward(e, e->X, Q_HCW, nxt, m, P - m, &lr, k > 0, qlog);
            e->mtp_valid = P;
            for (int i = 0; i < k; i++) {
                if (i > 0) q_mtp_forward(e, e->X2, Q_HCW, &vt[i], 1, P - 1 + i, &r0, 1, qlog + (size_t)i * Q_VOCAB);
                memcpy(e->hist + P, vt, sizeof(int32_t) * (size_t)(i + 1));  // penalties see the drafted prefix
                const int hn = P + i + 1 > 256 ? 256 : P + i + 1;
                const int *ix;
                const float *pr;
                const int c = sampler_dist(&smp, qlog + (size_t)i * Q_VOCAB, e->hist + P + i + 1 - hn, hn, &ix, &pr);
                float *q = qd + (size_t)i * Q_VOCAB;
                for (int j = 0; j < c; j++) q[ix[j]] = pr[j];
                const double r = sampler_rand_f(&smp);
                double acc = 0;
                int d = ix[c - 1];
                for (int j = 0; j < c; j++) { acc += pr[j]; if (r < acc) { d = ix[j]; break; } }
                vt[i + 1] = d;
            }
            diag_add(DG_DRAFT, -1, tdr, 0, 0);
        } else {
            if (use_mtp(e) && e->mtp_valid >= 0) {
                LOGI("MTP cache out of step (%d filled, %d expected): speculative decoding off", e->mtp_valid, P - m);
                e->mtp_valid = -1;
            }
            k = 0;
        }
        // verify [tok, drafts] in one main pass; the recurrent state is snapshotted after each draft position
        diag_set_phase(k ? PH_VERIFY : PH_DECODE);
        e->snap_n = k;
        q_forward(e, vt, k + 1, rows, k + 1, logits);
        e->snap_n = 0;
        g_diag.phase_tokens[k ? PH_VERIFY : PH_DECODE] += (uint64_t)(k + 1);
        // acceptance, position by position; `next` is the token after the accepted ones
        int a = 0, next = -1;
        for (;; a++) {
            const int hn = P + a + 1 > 256 ? 256 : P + a + 1;
            const int *ix;
            const float *pr;
            ts = diag_tsc();
            const int c = sampler_dist(&smp, logits + (size_t)a * Q_VOCAB, e->hist + P + a + 1 - hn, hn, &ix, &pr);
            if (a == k) {                       // nothing (left) to check: sample from the target distribution
                const double r = sampler_rand_f(&smp);
                double acc = 0;
                next = ix[c - 1];
                for (int j = 0; j < c; j++) { acc += pr[j]; if (r < acc) { next = ix[j]; break; } }
                diag_add(DG_SAMPLE, -1, ts, 0, 0);
                break;
            }
            for (int j = 0; j < c; j++) pd[ix[j]] = pr[j];
            const bool ok = spec_accept(&smp, pd, ix, c, qd + (size_t)a * Q_VOCAB, vt[a + 1], &next);
            for (int j = 0; j < c; j++) pd[ix[j]] = 0;
            diag_add(DG_VERIFY_EXTRA, -1, ts, 0, 0);
            if (!ok) break;
        }
        for (int i = 0; i < k; i++) memset(qd + (size_t)i * Q_VOCAB, 0, sizeof(float) * Q_VOCAB);
        if (k) {
            g_diag.tokens_drafted += (uint64_t)k;
            g_diag.tokens_accepted += (uint64_t)a;
            g_diag.verify_steps++;
            res->drafted += k;
            res->accepted += a;
        }
        // emit the accepted drafts; `keep` = index of the last token of vt the state must end with
        int keep_i = a;
        bool stop = false;
        for (int i = 1; i <= a; i++) {
            if (is_stop(g, vt[i])) { res->finish_reason = "stop"; keep_i = i - 1; stop = true; break; }
            res->gen_tokens++;
            diag_token_emitted();
            if (!cb(ud, vt[i])) { res->finish_reason = "stop"; keep_i = i; stop = true; break; }
            if (res->gen_tokens >= g->max_tokens) { res->finish_reason = "length"; keep_i = i; stop = true; break; }
        }
        if (keep_i < k) {                       // rewind the recurrent state to just after vt[keep_i]
            rec_copy(&e->st, &e->snap[keep_i]);
            e->n_past = P + keep_i + 1;
            memcpy(e->x_last, e->X + (size_t)keep_i * Q_HCW, sizeof(float) * Q_HCW);
        }
        m = keep_i + 1;
        if (stop) break;
        tok = next;
    }
    // leave the MTP cache one slot behind the main one (the next request fills the rest)
    if (use_mtp(e) && e->mtp_valid >= 0 && m > 1 && e->mtp_valid == e->n_past - m) {
        int32_t nxt[Q_SPEC_MAX + 1];
        for (int i = 0; i < m - 1; i++) nxt[i] = e->hist[e->n_past - m + 1 + i];
        q_mtp_forward(e, e->X, Q_HCW, nxt, m - 1, e->n_past - m, NULL, 0, NULL);
        e->mtp_valid = e->n_past - 1;
    }
    res->t_decode = inz_now() - td;
    g_diag.decode_tps_last = res->t_decode > 0 ? res->gen_tokens / res->t_decode : 0;
    diag_set_phase(PH_DECODE);
    sampler_free(&smp);
    free(logits);
    free(qlog);
    free(pd);
    free(qd);
    e->state = "idle";
    return 0;
}

float *engine_eval_logits_ex(inz_engine *e, const int32_t *tokens, int n, int last, bool stepwise) {
    if (last <= 0 || last > n) last = n;
    if (n > e->ctx) inz_die("eval: %d tokens exceed the context (%d)", n, e->ctx);
    const int from = n - last;
    float *out = inz_malloc(sizeof(float) * (size_t)last * Q_VOCAB);
    q_state_reset(e);
    e->ck_pos = -1;
    e->mtp_valid = -1;  // evaluation does not maintain the MTP cache
    int rows[Q_MAX_T];
    for (int done = 0; done < n;) {
        const bool step = stepwise && done >= from;
        int k = step ? 1 : (n - done < Q_MAX_T ? n - done : Q_MAX_T);
        if (!step && done < from && done + k > from) k = from - done;  // stop the prefill at the first logits row
        diag_set_phase(step ? PH_DECODE : PH_PREFILL);
        int nr = 0;
        for (int i = 0; i < k; i++)
            if (done + i >= from) rows[nr++] = i;
        q_forward(e, tokens + done, k, rows, nr, nr ? out + (size_t)(done + rows[0] - from) * Q_VOCAB : NULL);
        g_diag.phase_tokens[step ? PH_DECODE : PH_PREFILL] += (uint64_t)k;
        if (step) diag_token_emitted();
        done += k;
    }
    return out;
}

// Validation of the speculative rewind: positions [n - last, n) are decoded in verify passes of three tokens
// (two real ones and a junk draft) with snapshots, then rewound as a rejected draft would be, alternately to
// after the first and after the second token. The logits must equal a plain evaluation's.
float *engine_eval_snapshots(inz_engine *e, const int32_t *tokens, int n, int last) {
    if (last <= 0 || last > n) last = n;
    const int from = n - last;
    float *out = inz_malloc(sizeof(float) * (size_t)last * Q_VOCAB);
    float *lg = inz_malloc(sizeof(float) * 3 * Q_VOCAB);
    q_state_reset(e);
    e->ck_pos = -1;
    e->mtp_valid = -1;
    prefill(e, tokens, 0, from);
    const int rows[3] = {0, 1, 2};
    for (int p = from, step = 0; p < n; step++) {
        const bool two = (step & 1) && p + 1 < n;  // keep two real tokens (snapshot 1) or one (snapshot 0)
        const int32_t vt[3] = {tokens[p], p + 1 < n ? tokens[p + 1] : 0, 1234};
        e->snap_n = 2;
        q_forward(e, vt, 3, rows, 3, lg);
        e->snap_n = 0;
        const int keep = two ? 1 : 0;
        for (int i = 0; i <= keep; i++) memcpy(out + (size_t)(p - from + i) * Q_VOCAB, lg + (size_t)i * Q_VOCAB, sizeof(float) * Q_VOCAB);
        rec_copy(&e->st, &e->snap[keep]);
        e->n_past = p + keep + 1;
        memcpy(e->x_last, e->X + (size_t)keep * Q_HCW, sizeof(float) * Q_HCW);
        p += keep + 1;
    }
    free(lg);
    return out;
}

float *engine_eval_logits(inz_engine *e, const int32_t *tokens, int n) { return engine_eval_logits_ex(e, tokens, n, 0, false); }

/* ------------------------------------------------------------------ */
/* Diagnostics snapshot                                                */
/* ------------------------------------------------------------------ */

void engine_diag_json(inz_engine *e, jw *w) {
    jw_obj(w);
    jw_kstr(w, "state", e->state);
    jw_kstr(w, "model", e->model_name);
    jw_kstr(w, "recipe", gguf_get_str(e->g, "inz.recipe", ""));
    jw_kstr(w, "isa", inz_kernel_isa());
    jw_kint(w, "threads", tp_size(e->tp));
    jw_kint(w, "n_past", e->n_past);
    jw_kint(w, "ctx", e->ctx);
    jw_kint(w, "checkpoint", e->ck_pos);
    jw_key(w, "diag");
    diag_write_json(w);
    if (e->ec) {
        const ec_stats *st = ec_get_stats(e->ec);
        static uint64_t prev_lookups, prev_hits;
        static double prev_t;
        const double now = inz_now();
        const double recent = st->lookups > prev_lookups ? (double)(st->hits - prev_hits) / (double)(st->lookups - prev_lookups)
                                                         : (st->lookups ? (double)st->hits / st->lookups : 0);
        if (now - prev_t > 2.0) { prev_lookups = st->lookups; prev_hits = st->hits; prev_t = now; }
        jw_key(w, "cache");
        jw_obj(w);
        jw_kint(w, "slots", (int64_t)st->slots);
        jw_kint(w, "resident", (int64_t)st->resident);
        jw_kint(w, "lookups", (int64_t)st->lookups);
        jw_kint(w, "hits", (int64_t)st->hits);
        jw_kint(w, "misses", (int64_t)st->misses);
        jw_knum(w, "hit_rate", st->lookups ? (double)st->hits / st->lookups : 0);
        jw_knum(w, "hit_rate_recent", recent);
        jw_kint(w, "bytes_read", (int64_t)st->bytes_read);
        jw_kint(w, "reads", (int64_t)st->reads);
        jw_knum(w, "read_lat_avg_us", st->reads ? st->read_lat_sum_s / st->reads * 1e6 : 0);
        jw_knum(w, "stall_s", st->stall_s);
        jw_kint(w, "evictions", (int64_t)st->evictions);
        jw_key(w, "latency_hist_us_log2");
        jw_arr(w);
        for (int i = 0; i < 16; i++) jw_int(w, (int64_t)st->lat_hist[i]);
        jw_arr_end(w);
        jw_key(w, "layer_hit_rate");
        jw_arr(w);
        for (int l = 0; l < e->n_layer; l++) {
            const uint64_t h = st->layer_hits[l], m = st->layer_misses[l];
            jw_num(w, h + m ? (double)h / (h + m) : 0);
        }
        jw_arr_end(w);
        jw_obj_end(w);
    }
    jw_key(w, "head");
    jw_obj(w);
    jw_kbool(w, "two_stage", e->two_stage);
    jw_knum(w, "screen_sigma", e->head_sigma);
    jw_knum(w, "avg_candidates", e->head_rows ? (double)e->head_cands / (double)e->head_rows : 0);
    jw_obj_end(w);
    jw_key(w, "memory");
    jw_obj(w);
    jw_knum(w, "rss_gb", inz_rss() / 1e9);
    jw_knum(w, "available_gb", inz_mem_available() / 1e9);
    jw_knum(w, "dense_gb", e->dense_bytes / 1e9);
    jw_knum(w, "cache_gb", e->ec ? ec_slot_count(e->ec) * (double)e->blob_bytes / 1e9 : 0);
    jw_knum(w, "recurrent_state_gb", (double)(S_FLOATS + CONV_FLOATS + PLE_FLOATS) * 4 / 1e9);
    jw_knum(w, "kv_gb", (double)e->n_past * Q_N_ATTN * (2.0 * Q_KV * 2 + Q_IDX_HD * 4 + Q_IDX_HD) / 1e9);
    jw_obj_end(w);
    const tp_stats *ts = tp_get_stats(e->tp);
    jw_key(w, "thread_busy_s");
    jw_arr(w);
    for (int i = 0; i < tp_size(e->tp); i++) jw_num(w, ts->busy_s[i]);
    jw_arr_end(w);
    jw_obj_end(w);
}

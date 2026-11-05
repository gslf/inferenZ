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
    o->model_path = "models/laguna-s-2.1.inz.gguf";  // the official NVFP4 experts, 8-bit rest (README)
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
// weights that every token needs. Routed experts and the embedding table stay in the file.
typedef struct { const gguf_tensor *t; size_t off; } dense_item;
typedef struct { inz_engine *e; dense_item *items; int n; } dense_set;

static bool resident(const gguf_tensor *t) { return t->type != INZ_T_EXPERTS && strcmp(t->name, "token_embd.weight"); }

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

// one decoder layer "blk.N": attention, then the dense MLP (layer 0) or the MoE (the experts are bound by setup_experts)
static void bind_layer(const dense_set *d, q_layer *L, int l) {
    char f[64];
#define N(suffix) (snprintf(f, sizeof(f), "blk.%d." suffix, l), f)
    L->nh = q_nh(l);
    L->attn_norm = dvec(d, N("attn_norm"), 0, Q_H);
    L->mlp_norm = dvec(d, N("mlp_norm"), 0, Q_H);
    dlin(d, &L->at_q, N("at.q"), 0, L->nh * Q_HD, Q_H);
    dlin(d, &L->at_k, N("at.k"), 0, Q_KV, Q_H);
    dlin(d, &L->at_v, N("at.v"), 0, Q_KV, Q_H);
    dlin(d, &L->at_o, N("at.o"), 0, Q_H, L->nh * Q_HD);
    dlin(d, &L->at_g, N("at.g"), 0, L->nh, Q_H);
    L->q_norm = dvec(d, N("at.q_norm"), 0, Q_HD);
    L->k_norm = dvec(d, N("at.k_norm"), 0, Q_HD);
    if (l == 0) {
        dlin(d, &L->d_gu, N("mlp.gate_up"), 0, 2 * Q_DENSE_FF, Q_H);
        dlin(d, &L->d_down, N("mlp.down"), 0, Q_H, Q_DENSE_FF);
    } else {
        dlin(d, &L->router, N("router"), 0, Q_N_EXPERT, Q_H);
        L->router_bias = dvec(d, N("router_bias"), 0, Q_N_EXPERT);
        dlin(d, &L->sh_gu, N("sh.gate_up"), 0, 2 * Q_SHARED_FF, Q_H);
        dlin(d, &L->sh_down, N("sh.down"), 0, Q_H, Q_SHARED_FF);
    }
#undef N
}

// Recipe studies: INZ_OVERRIDE="other.gguf:pattern1,pattern2" takes the dense tensors whose name contains a
// pattern from another conversion of the same model (e.g. `inz convert --dense-only --attn F16 ...` for exact
// attention in the 8-bit recipe).
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
        if (!resident(t)) continue;
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

    for (int l = 0; l < e->n_layer; l++) bind_layer(&d, &e->L[l], l);
    e->n_moe = e->n_layer - 1;
    e->norm_f = dvec(&d, "output_norm", 0, Q_H);
    dlin(&d, &e->head, "output.weight", 0, Q_VOCAB, Q_H);
    e->mtp.present = false;
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
    if (!inz_qmat_view(&e->embed, e->embed_hdr) || e->embed.rows != Q_VOCAB || e->embed.cols != Q_H || e->embed.rot_block ||
        e->embed.type == INZ_T_Q4N || e->embed.type == INZ_T_F8T)
        inz_die("model: token_embd.weight must be F16 or an unrotated %dx%d matrix with one scale per group", Q_VOCAB, Q_H);
}

// The MoE layers form blob classes by expert storage type (the official NVFP4 layers as Q4N; the layers the checkpoint
// keeps in BF16 as an 8-bit format). Each class has its own cache of fixed-size slots; the memory budget is split in
// proportion to the classes' sizes, so both keep the same fraction of their experts.
static void setup_experts(inz_engine *e, uint64_t other_bytes) {
    gguf_file *g = e->g;
    if (e->n_moe <= 0) return;  // a truncated test model without MoE layers
    const gguf_kv *kt = gguf_find_kv(g, "inz.expert_types");
    if (!kt || kt->arr_n < (uint64_t)e->n_layer) inz_die("model: inz.expert_types missing");
    const int32_t *types = (const int32_t *)kt->data;
    int nl[2] = {0, 0};
    for (int l = 1; l < e->n_layer; l++) {
        int c = 0;
        while (c < e->n_xc && e->xc[c].type != types[l]) c++;
        if (c == e->n_xc) {
            if (e->n_xc == 2) inz_die("model: more than two expert types");
            e->xc[c].type = types[l];
            e->n_xc++;
        }
        char name[64];
        snprintf(name, sizeof(name), "blk.%d.experts", l);
        const gguf_tensor *t = gguf_find_tensor(g, name);
        if (!t) inz_die("model: missing %s", name);
        q_xclass *x = &e->xc[c];
        if (!x->blob_bytes) {
            x->blob_bytes = t->ne[0];
            x->gu_bytes = inz_mat_bytes(x->type, 2 * Q_FF, Q_H);
            if (x->gu_bytes + inz_mat_bytes(x->type, Q_H, Q_FF) > x->blob_bytes || x->blob_bytes % 4096)
                inz_die("model: expert blob size %llu does not fit type %s", (unsigned long long)x->blob_bytes, inz_qtype_name(x->type));
        } else if (t->ne[0] != x->blob_bytes) inz_die("model: %s has another blob size", name);
        e->L[l].xc = x;
        e->L[l].xl = nl[c]++;
    }
    uint64_t total[2] = {0, 0}, all_total = 0;
    for (int c = 0; c < e->n_xc; c++) all_total += total[c] = (uint64_t)nl[c] * Q_N_EXPERT * e->xc[c].blob_bytes;
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
    }
    for (int c = 0; c < e->n_xc; c++) {
        q_xclass *x = &e->xc[c];
        uint64_t *off = inz_malloc(sizeof(uint64_t) * (size_t)nl[c] * Q_N_EXPERT);
        for (int l = 1; l < e->n_layer; l++) {
            if (e->L[l].xc != x) continue;
            char name[64];
            snprintf(name, sizeof(name), "blk.%d.experts", l);
            const gguf_tensor *t = gguf_find_tensor(g, name);
            for (int k = 0; k < Q_N_EXPERT; k++)
                off[(size_t)e->L[l].xl * Q_N_EXPERT + k] = g->data_offset + t->offset + (uint64_t)k * x->blob_bytes;
        }
        const uint64_t share = budget >= all_total ? total[c] : (uint64_t)((double)budget * total[c] / all_total);
        x->all_resident = share >= total[c];
        ec_config cfg = {.path = g->path, .n_layer = nl[c], .n_expert = Q_N_EXPERT, .offset = off, .blob_bytes = x->blob_bytes,
                         .cache_bytes = share, .queue_depth = 64, .sqpoll_cpu = c ? -1 : e->opt.sqpoll_cpu,
                         .decay = e->opt.cache_decay, .all_resident = x->all_resident};
        char err[256];
        x->ec = ec_open(&cfg, err, sizeof(err));
        free(off);
        if (!x->ec) inz_die("expert cache: %s", err);
        LOGI("experts %d: %s, %.2f MiB each, %d layers x %d = %.1f GiB on SSD, cache %.1f GiB (%.0f%%)", c, inz_qtype_name(x->type),
             x->blob_bytes / 1048576.0, nl[c], Q_N_EXPERT, total[c] / 1073741824.0,
             ec_slot_count(x->ec) * (double)x->blob_bytes / 1073741824.0, 100.0 * ec_slot_count(x->ec) / (double)(nl[c] * Q_N_EXPERT));
        if (!x->all_resident) {
            if (c == 0) {
                g_diag.peak_ssd_gbs = ec_probe_bandwidth(x->ec, 24);
                LOGI("SSD expert-read bandwidth probe: %.2f GB/s", g_diag.peak_ssd_gbs);
            }
            if (e->opt.hotlist) {
                char hp[1100];
                snprintf(hp, sizeof(hp), "%s.%d", e->opt.hotlist, c);
                const double t0 = inz_now();
                const int n = ec_load_and_preload(x->ec, hp);
                if (n) LOGI("warm start: preloaded %d experts from %s in %.1f s", n, hp, inz_now() - t0);
            }
        } else {
            int32_t *pairs = inz_malloc(sizeof(int32_t) * 2 * (size_t)nl[c] * Q_N_EXPERT);
            int k = 0;
            for (int l = 0; l < nl[c]; l++)
                for (int xx = 0; xx < Q_N_EXPERT; xx++) { pairs[2 * k] = l; pairs[2 * k + 1] = xx; k++; }
            const double t0 = inz_now();
            ec_preload(x->ec, pairs, k);
            LOGI("all experts %d resident (%.1f s)", c, inz_now() - t0);
            free(pairs);
        }
    }
}

/* ------------------------------------------------------------------ */
/* Sequence state                                                      */
/* ------------------------------------------------------------------ */

#define N_SWA (Q_N_LAYER - Q_N_FULL)

static void state_alloc(inz_engine *e) {
    q_state *st = &e->st;
    const size_t ctx = (size_t)e->ctx;
    // pages are touched lazily: an unused context costs no memory
    st->kf = inz_mmap_anon((size_t)Q_N_FULL * ctx * Q_KV * 2, true);
    st->vf = inz_mmap_anon((size_t)Q_N_FULL * ctx * Q_KV * 2, true);
    st->ks = inz_mmap_anon((size_t)N_SWA * Q_RING * Q_KV * 2, true);
    st->vs = inz_mmap_anon((size_t)N_SWA * Q_RING * Q_KV * 2, true);
    e->ck_x = inz_aligned(64, sizeof(float) * Q_H);
    e->x_last = inz_aligned(64, sizeof(float) * Q_H);
    e->Xm = inz_aligned(64, sizeof(float) * Q_H);
    e->ck_pos = -1;
    e->hist = inz_malloc(sizeof(int32_t) * ctx);
    e->state_bytes = ((size_t)Q_N_FULL * ctx + (size_t)N_SWA * Q_RING) * Q_KV * 2 * 2;
}

static void state_free(inz_engine *e) {
    q_state *st = &e->st;
    const size_t ctx = (size_t)e->ctx;
    inz_munmap(st->kf, (size_t)Q_N_FULL * ctx * Q_KV * 2);
    inz_munmap(st->vf, (size_t)Q_N_FULL * ctx * Q_KV * 2);
    inz_munmap(st->ks, (size_t)N_SWA * Q_RING * Q_KV * 2);
    inz_munmap(st->vs, (size_t)N_SWA * Q_RING * Q_KV * 2);
    free(e->ck_x);
    free(e->x_last);
    free(e->Xm);
    free(e->hist);
}

// Every cache is positional, so the sequence can be rewound to any position p whose sliding window the rings still
// hold: the positions written after p must not have wrapped onto (p - window, p].
static bool can_rewind(const inz_engine *e, int p) { return p >= 0 && p <= e->n_past && e->n_past - p <= Q_RING - Q_WINDOW; }

static void checkpoint_save(inz_engine *e) { e->ck_pos = e->n_past; }

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
    snprintf(e->model_name, sizeof(e->model_name), "laguna-s-2.1");
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
    diag_init(Q_N_LAYER, Q_N_MOE, Q_N_EXPERT, Q_TOPK);  // routing per MoE layer
    if (o->route_trace) g_diag.route_trace = fopen(o->route_trace, "w");
    LOGI("RAM bandwidth probe: %.1f GB/s", g_diag.peak_ram_gbs);
    load_dense(e);
    setup_embed(e);
    e->ctx = o->ctx > 0 ? o->ctx : 32768;
    state_alloc(e);
    q_scratch_alloc(e);
    q_rope_init(e);
    q_state_reset(e);
    if (e->imx_path) {
        const size_t nl = (size_t)(e->n_moe > 0 ? e->n_moe : 0);
        e->imx_exp = inz_calloc(nl * Q_N_EXPERT * (Q_H + Q_FF), sizeof(double));
        e->imx_exp_n = inz_calloc(nl * Q_N_EXPERT, sizeof(uint64_t));
        LOGI("imatrix: collecting input statistics into %s", e->imx_path);
    }
    // state pages are touched lazily; budget the full context so the cache never starves long chats
    setup_experts(e, e->state_bytes + (2ull << 30) + (e->imx_path ? (uint64_t)Q_N_MOE * Q_N_EXPERT * (Q_H + Q_FF) * 8 : 0));
    e->t_loaded = inz_now();
    e->state = "idle";
    return e;
}

void engine_close(inz_engine *e) {
    if (!e) return;
    if (e->imx_path) q_imx_write(e);
    for (int c = 0; c < e->n_xc; c++) {
        char hp[1100];
        if (e->opt.hotlist) snprintf(hp, sizeof(hp), "%s.%d", e->opt.hotlist, c);
        if (e->xc[c].ec && e->opt.hotlist && ec_save_scores(e->xc[c].ec, hp)) LOGI("expert scores saved to %s", hp);
        ec_close(e->xc[c].ec);
    }
    q_scratch_free(e);
    state_free(e);
    inz_munmap(e->dense_mem, e->dense_bytes);
    if (e->embed_fd > 0) close(e->embed_fd);
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
    if (p0 > 0) memcpy(X + (size_t)m++ * Q_H, x_prev, sizeof(float) * Q_H);
    memcpy(X + (size_t)m * Q_H, e->X, sizeof(float) * Q_H * (size_t)(n - 1));
    m += n - 1;
    q_mtp_forward(e, X, Q_H, true, e->hist + first + 1, m, first, NULL, 0, NULL);
    e->mtp_valid = last;
}

// one main forward of tokens at e->n_past with MTP bookkeeping
static void forward_tracked(inz_engine *e, const int32_t *tok, int n, const int *rows, int nr, float *logits) {
    float xp[Q_H];
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
    // to the end of the common part (the caches are positional), else start over
    int keep = 0;
    while (keep < e->n_past && keep < n_prompt && e->hist[keep] == prompt[keep]) keep++;
    if (!(keep == e->n_past && keep < n_prompt)) {
        const int target = keep < n_prompt ? keep : n_prompt - 1;
        if (can_rewind(e, target)) e->n_past = target;
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
            int32_t nxt[Q_SPEC_MAX + 1];
            for (int i = 0; i < m - 1; i++) nxt[i] = e->hist[P - m + 1 + i];
            nxt[m - 1] = tok;
            const int lr = m - 1, r0 = 0;
            q_mtp_forward(e, e->X, Q_H, true, nxt, m, P - m, &lr, k > 0, qlog);
            e->mtp_valid = P;
            for (int i = 0; i < k; i++) {
                if (i > 0) q_mtp_forward(e, e->X2 + (size_t)(i == 1 ? m - 1 : 0) * Q_H, Q_H, false, &vt[i], 1, P - 1 + i, &r0, 1,
                                         qlog + (size_t)i * Q_VOCAB);
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
        if (keep_i < k) {                       // rewind to just after vt[keep_i] (positional caches)
            e->n_past = P + keep_i + 1;
            memcpy(e->x_last, e->X + (size_t)keep_i * Q_H, sizeof(float) * Q_H);
        }
        m = keep_i + 1;
        if (stop) break;
        tok = next;
    }
    // leave the MTP cache one slot behind the main one (the next request fills the rest)
    if (use_mtp(e) && e->mtp_valid >= 0 && m > 1 && e->mtp_valid == e->n_past - m) {
        int32_t nxt[Q_SPEC_MAX + 1];
        for (int i = 0; i < m - 1; i++) nxt[i] = e->hist[e->n_past - m + 1 + i];
        q_mtp_forward(e, e->X, Q_H, true, nxt, m - 1, e->n_past - m, NULL, 0, NULL);
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

// Validation of rewinds: positions [n - last, n) are decoded in passes of three tokens (two real ones and a junk
// draft), then rewound as a rejected draft would be, alternately to after the first and after the second token. The
// logits must equal a plain evaluation's.
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
        q_forward(e, vt, 3, rows, 3, lg);
        const int keep = two ? 1 : 0;
        for (int i = 0; i <= keep; i++) memcpy(out + (size_t)(p - from + i) * Q_VOCAB, lg + (size_t)i * Q_VOCAB, sizeof(float) * Q_VOCAB);
        e->n_past = p + keep + 1;
        memcpy(e->x_last, e->X + (size_t)keep * Q_H, sizeof(float) * Q_H);
        p += keep + 1;
    }
    free(lg);
    return out;
}

float *engine_eval_logits(inz_engine *e, const int32_t *tokens, int n) { return engine_eval_logits_ex(e, tokens, n, 0, false); }

// Validation of the MTP head: the main model over tokens [0, n), then the MTP slots 0 .. n-2, slot t with the token at
// t + 1. Returns malloc'ed [n - 1][vocab] draft logits (NULL without the MTP head).
float *engine_eval_mtp_logits(inz_engine *e, const int32_t *tokens, int n) {
    if (!e->mtp.present || n < 2 || n > Q_MAX_T) return NULL;
    float *out = inz_malloc(sizeof(float) * (size_t)(n - 1) * Q_VOCAB);
    q_state_reset(e);
    e->ck_pos = -1;
    e->mtp_valid = -1;
    q_forward(e, tokens, n, NULL, 0, NULL);
    int rows[Q_MAX_T];
    for (int i = 0; i < n - 1; i++) rows[i] = i;
    for (int i = 0; i < n - 1; i += 64) {  // the head takes up to 64 rows per call
        const int k = n - 1 - i < 64 ? n - 1 - i : 64;
        if (i == 0) q_mtp_forward(e, e->X, Q_H, true, tokens + 1, n - 1, 0, rows, k, out);
        else q_head(e, e->X2, n - 1, rows + i, k, out + (size_t)i * Q_VOCAB);
    }
    return out;
}

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
    if (e->n_xc) {
        // both expert classes together
        uint64_t slots = 0, resident = 0, lookups = 0, hits = 0, misses = 0, bytes = 0, reads = 0, evictions = 0, hist[16] = {0};
        double lat = 0, stall = 0;
        for (int c = 0; c < e->n_xc; c++) {
            const ec_stats *st = ec_get_stats(e->xc[c].ec);
            slots += st->slots; resident += st->resident; lookups += st->lookups; hits += st->hits; misses += st->misses;
            bytes += st->bytes_read; reads += st->reads; evictions += st->evictions; lat += st->read_lat_sum_s; stall += st->stall_s;
            for (int i = 0; i < 16; i++) hist[i] += st->lat_hist[i];
        }
        static uint64_t prev_lookups, prev_hits;
        static double prev_t;
        const double now = inz_now();
        const double recent = lookups > prev_lookups ? (double)(hits - prev_hits) / (double)(lookups - prev_lookups)
                                                     : (lookups ? (double)hits / lookups : 0);
        if (now - prev_t > 2.0) { prev_lookups = lookups; prev_hits = hits; prev_t = now; }
        jw_key(w, "cache");
        jw_obj(w);
        jw_kint(w, "slots", (int64_t)slots);
        jw_kint(w, "resident", (int64_t)resident);
        jw_kint(w, "lookups", (int64_t)lookups);
        jw_kint(w, "hits", (int64_t)hits);
        jw_kint(w, "misses", (int64_t)misses);
        jw_knum(w, "hit_rate", lookups ? (double)hits / lookups : 0);
        jw_knum(w, "hit_rate_recent", recent);
        jw_kint(w, "bytes_read", (int64_t)bytes);
        jw_kint(w, "reads", (int64_t)reads);
        jw_knum(w, "read_lat_avg_us", reads ? lat / reads * 1e6 : 0);
        jw_knum(w, "stall_s", stall);
        jw_kint(w, "evictions", (int64_t)evictions);
        jw_key(w, "latency_hist_us_log2");
        jw_arr(w);
        for (int i = 0; i < 16; i++) jw_int(w, (int64_t)hist[i]);
        jw_arr_end(w);
        jw_key(w, "layer_hit_rate");
        jw_arr(w);
        for (int l = 1; l < e->n_layer; l++) {
            const ec_stats *st = ec_get_stats(e->L[l].xc->ec);
            const uint64_t h = st->layer_hits[e->L[l].xl], m = st->layer_misses[e->L[l].xl];
            jw_num(w, h + m ? (double)h / (h + m) : 0);
        }
        jw_arr_end(w);
        jw_obj_end(w);
    }
    jw_key(w, "memory");
    jw_obj(w);
    jw_knum(w, "rss_gb", inz_rss() / 1e9);
    jw_knum(w, "available_gb", inz_mem_available() / 1e9);
    jw_knum(w, "dense_gb", e->dense_bytes / 1e9);
    double cache_gb = 0;
    for (int c = 0; c < e->n_xc; c++) cache_gb += ec_slot_count(e->xc[c].ec) * (double)e->xc[c].blob_bytes / 1e9;
    jw_knum(w, "cache_gb", cache_gb);
    jw_knum(w, "recurrent_state_gb", 0);
    const int swa = e->n_past < Q_RING ? e->n_past : Q_RING;
    jw_knum(w, "kv_gb", ((double)e->n_past * Q_N_FULL + (double)swa * N_SWA) * 2.0 * Q_KV * 2 / 1e9);
    jw_obj_end(w);
    const tp_stats *ts = tp_get_stats(e->tp);
    jw_key(w, "thread_busy_s");
    jw_arr(w);
    for (int i = 0; i < tp_size(e->tp); i++) jw_num(w, ts->busy_s[i]);
    jw_arr_end(w);
    jw_obj_end(w);
}

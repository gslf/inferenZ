#define _GNU_SOURCE
#include "tokenizer.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "json.h"
#include "util.h"
#include "unicode_tables.inc"

/* ------------------------------------------------------------------ */
/* Unicode helpers                                                     */
/* ------------------------------------------------------------------ */

static uint16_t uc_flags(uint32_t cp) {
    int lo = 0, hi = (int)(sizeof(uc_ranges) / sizeof(uc_ranges[0])) - 1;
    while (lo <= hi) {
        int mid = (lo + hi) >> 1;
        if (cp < uc_ranges[mid].lo) hi = mid - 1;
        else if (cp > uc_ranges[mid].hi) lo = mid + 1;
        else return uc_ranges[mid].f;
    }
    return 0;
}

static uint8_t uc_ccc_of(uint32_t cp) {
    if (cp < 0x300) return 0;
    int lo = 0, hi = (int)(sizeof(uc_ccc) / sizeof(uc_ccc[0])) - 1;
    while (lo <= hi) {
        int mid = (lo + hi) >> 1;
        if (cp < uc_ccc[mid].lo) hi = mid - 1;
        else if (cp > uc_ccc[mid].hi) lo = mid + 1;
        else return uc_ccc[mid].c;
    }
    return 0;
}

// decode one UTF-8 sequence; returns length (>=1). invalid bytes yield cp = 0x110000 + byte
static int utf8_dec(const unsigned char *s, size_t n, uint32_t *cp) {
    unsigned char c = s[0];
    if (c < 0x80) { *cp = c; return 1; }
    int len = (c & 0xE0) == 0xC0 ? 2 : (c & 0xF0) == 0xE0 ? 3 : (c & 0xF8) == 0xF0 ? 4 : 0;
    if (len == 0 || (size_t)len > n) { *cp = 0x110000u + c; return 1; }
    uint32_t v = len == 2 ? (c & 0x1Fu) : len == 3 ? (c & 0x0Fu) : (c & 0x07u);
    for (int i = 1; i < len; i++) {
        if ((s[i] & 0xC0) != 0x80) { *cp = 0x110000u + c; return 1; }
        v = (v << 6) | (s[i] & 0x3Fu);
    }
    if ((len == 2 && v < 0x80) || (len == 3 && v < 0x800) || (len == 4 && (v < 0x10000 || v > 0x10FFFF)) ||
        (v >= 0xD800 && v <= 0xDFFF)) { *cp = 0x110000u + c; return 1; }
    *cp = v;
    return len;
}

static int utf8_enc(uint32_t cp, char *o) {
    if (cp >= 0x110000u) { o[0] = (char)(cp - 0x110000u); return 1; }
    if (cp < 0x80) { o[0] = (char)cp; return 1; }
    if (cp < 0x800) { o[0] = (char)(0xC0 | (cp >> 6)); o[1] = (char)(0x80 | (cp & 0x3F)); return 2; }
    if (cp < 0x10000) {
        o[0] = (char)(0xE0 | (cp >> 12)); o[1] = (char)(0x80 | ((cp >> 6) & 0x3F)); o[2] = (char)(0x80 | (cp & 0x3F));
        return 3;
    }
    o[0] = (char)(0xF0 | (cp >> 18)); o[1] = (char)(0x80 | ((cp >> 12) & 0x3F));
    o[2] = (char)(0x80 | ((cp >> 6) & 0x3F)); o[3] = (char)(0x80 | (cp & 0x3F));
    return 4;
}

enum { SBase = 0xAC00, LBase = 0x1100, VBase = 0x1161, TBase = 0x11A7, LCount = 19, VCount = 21, TCount = 28,
       NCount = VCount * TCount, SCount = LCount * NCount };

static int find_decomp(uint32_t cp) {
    int lo = 0, hi = (int)(sizeof(uc_decomp) / sizeof(uc_decomp[0])) - 1;
    while (lo <= hi) {
        int mid = (lo + hi) >> 1;
        if (cp < uc_decomp[mid].cp) hi = mid - 1;
        else if (cp > uc_decomp[mid].cp) lo = mid + 1;
        else return mid;
    }
    return -1;
}

static uint32_t compose_pair(uint32_t a, uint32_t b) {
    if (a >= LBase && a < LBase + LCount && b >= VBase && b < VBase + VCount)
        return SBase + ((a - LBase) * VCount + (b - VBase)) * TCount;
    if (a >= SBase && a < SBase + SCount && (a - SBase) % TCount == 0 && b > TBase && b < TBase + TCount)
        return a + (b - TBase);
    int lo = 0, hi = (int)(sizeof(uc_comp) / sizeof(uc_comp[0])) - 1;
    while (lo <= hi) {
        int mid = (lo + hi) >> 1;
        if (a < uc_comp[mid].a || (a == uc_comp[mid].a && b < uc_comp[mid].b)) hi = mid - 1;
        else if (a > uc_comp[mid].a || (a == uc_comp[mid].a && b > uc_comp[mid].b)) lo = mid + 1;
        else return uc_comp[mid].c;
    }
    return 0;
}

size_t inz_nfc(const char *in, size_t len, char **out) {
    bool ascii = true;
    for (size_t i = 0; i < len; i++) if ((unsigned char)in[i] >= 0x80) { ascii = false; break; }
    if (ascii) {
        *out = inz_malloc(len + 1);
        memcpy(*out, in, len);
        (*out)[len] = 0;
        return len;
    }
    // decompose
    uint32_t *d = inz_malloc(sizeof(uint32_t) * (len * 4 + 4));
    size_t nd = 0;
    for (size_t i = 0; i < len;) {
        uint32_t cp;
        i += (size_t)utf8_dec((const unsigned char *)in + i, len - i, &cp);
        if (cp >= SBase && cp < SBase + SCount) {
            uint32_t s = cp - SBase;
            d[nd++] = LBase + s / NCount;
            d[nd++] = VBase + (s % NCount) / TCount;
            if (s % TCount) d[nd++] = TBase + s % TCount;
            continue;
        }
        int k = cp < 0xC0 ? -1 : find_decomp(cp);
        if (k >= 0) for (int j = 0; j < uc_decomp[k].len; j++) d[nd++] = uc_dpool[uc_decomp[k].off + j];
        else d[nd++] = cp;
    }
    // canonical ordering (insertion sort inside runs of non-starters)
    for (size_t i = 1; i < nd; i++) {
        uint8_t c = uc_ccc_of(d[i]);
        if (!c) continue;
        size_t j = i;
        while (j > 0 && uc_ccc_of(d[j - 1]) > c) { uint32_t t = d[j]; d[j] = d[j - 1]; d[j - 1] = t; j--; }
    }
    // compose (canonical composition algorithm, UAX #15)
    size_t w = 0, starter = (size_t)-1;
    int last_cc = 0;
    for (size_t i = 0; i < nd; i++) {
        uint32_t c = d[i];
        int cc = uc_ccc_of(c);
        if (starter != (size_t)-1) {
            bool adjacent = w == starter + 1;
            bool blocked = !adjacent && (last_cc == 0 || last_cc >= cc);
            if (!blocked) {
                uint32_t comp = compose_pair(d[starter], c);
                if (comp) { d[starter] = comp; continue; }
            }
        }
        if (cc == 0) starter = w;
        d[w++] = c;
        last_cc = cc;
    }
    char *o = inz_malloc(w * 4 + 1);
    size_t ol = 0;
    for (size_t i = 0; i < w; i++) ol += (size_t)utf8_enc(d[i], o + ol);
    o[ol] = 0;
    free(d);
    *out = o;
    return ol;
}

/* ------------------------------------------------------------------ */
/* Hash tables                                                         */
/* ------------------------------------------------------------------ */

static uint64_t hbytes(const void *p, size_t n) {
    const unsigned char *s = p;
    uint64_t h = 1469598103934665603ull;
    for (size_t i = 0; i < n; i++) { h ^= s[i]; h *= 1099511628211ull; }
    return h ^ (h >> 29);
}

static uint64_t hpair(uint32_t a, uint32_t b) {
    uint64_t x = ((uint64_t)a << 32) | b;
    x ^= x >> 33; x *= 0xff51afd7ed558ccdull; x ^= x >> 33; x *= 0xc4ceb9fe1a85ec53ull; x ^= x >> 33;
    return x;
}

typedef struct { int32_t a, b, rank, res; } merge_ent;

typedef struct {
    char *key;         // cached pre-token bytes
    uint32_t klen;
    int32_t *ids;
    uint32_t nids;
} cache_ent;

struct tokenizer {
    int n_vocab;
    int split;            // pre-tokenizer: TOK_SPLIT_O200K or TOK_SPLIT_QWEN
    char **piece;
    uint32_t *plen;
    bool *special;
    // piece lookup
    int32_t *vtab;
    uint32_t vmask;
    // merges
    merge_ent *mtab;
    uint32_t mmask;
    int n_merges;
    int32_t *m_a, *m_b, *m_res;
    // added tokens
    int n_added;
    int32_t *added_id;
    char **added_str;
    uint32_t *added_len;
    uint8_t *added_special;
    int32_t byte_tok[256];
    // encode cache
    cache_ent *cache;
    uint32_t cmask, ccount;
};

static void vtab_build(tokenizer *t) {
    uint32_t sz = 1;
    while (sz < (uint32_t)t->n_vocab * 2) sz <<= 1;
    t->vmask = sz - 1;
    t->vtab = inz_malloc(sizeof(int32_t) * sz);
    for (uint32_t i = 0; i < sz; i++) t->vtab[i] = -1;
    for (int id = 0; id < t->n_vocab; id++) {
        if (!t->piece[id]) continue;
        uint32_t h = (uint32_t)hbytes(t->piece[id], t->plen[id]) & t->vmask;
        bool dup = false;
        while (t->vtab[h] >= 0) {
            int o = t->vtab[h];
            if (t->plen[o] == t->plen[id] && !memcmp(t->piece[o], t->piece[id], t->plen[id])) { dup = true; break; }
            h = (h + 1) & t->vmask;
        }
        if (!dup) t->vtab[h] = id;
    }
}

static int vfind(const tokenizer *t, const char *s, size_t n) {
    uint32_t h = (uint32_t)hbytes(s, n) & t->vmask;
    while (t->vtab[h] >= 0) {
        int id = t->vtab[h];
        if (t->plen[id] == n && !memcmp(t->piece[id], s, n)) return id;
        h = (h + 1) & t->vmask;
    }
    return -1;
}

static void mtab_build(tokenizer *t) {
    uint32_t sz = 1;
    while (sz < (uint32_t)t->n_merges * 2 + 2) sz <<= 1;
    t->mmask = sz - 1;
    t->mtab = inz_malloc(sizeof(merge_ent) * sz);
    for (uint32_t i = 0; i < sz; i++) t->mtab[i].a = -1;
    for (int r = 0; r < t->n_merges; r++) {
        uint32_t h = (uint32_t)hpair((uint32_t)t->m_a[r], (uint32_t)t->m_b[r]) & t->mmask;
        bool dup = false;
        while (t->mtab[h].a >= 0) {
            if (t->mtab[h].a == t->m_a[r] && t->mtab[h].b == t->m_b[r]) { dup = true; break; }
            h = (h + 1) & t->mmask;
        }
        if (!dup) t->mtab[h] = (merge_ent){t->m_a[r], t->m_b[r], r, t->m_res[r]};
    }
}

static const merge_ent *mfind(const tokenizer *t, int32_t a, int32_t b) {
    uint32_t h = (uint32_t)hpair((uint32_t)a, (uint32_t)b) & t->mmask;
    while (t->mtab[h].a >= 0) {
        if (t->mtab[h].a == a && t->mtab[h].b == b) return &t->mtab[h];
        h = (h + 1) & t->mmask;
    }
    return NULL;
}

static void finish_init(tokenizer *t) {
    vtab_build(t);
    mtab_build(t);
    for (int b = 0; b < 256; b++) {
        char c = (char)b;
        t->byte_tok[b] = vfind(t, &c, 1);
    }
    t->special = inz_calloc((size_t)t->n_vocab, 1);
    for (int i = 0; i < t->n_added; i++)
        if (t->added_id[i] >= 0 && t->added_id[i] < t->n_vocab) t->special[t->added_id[i]] = t->added_special[i];
    t->cmask = (1u << 16) - 1;
    t->cache = inz_calloc(t->cmask + 1, sizeof(cache_ent));
}

/* ------------------------------------------------------------------ */
/* Loading                                                             */
/* ------------------------------------------------------------------ */

// GPT-2 bytes_to_unicode inverse: unicode code point -> byte
static int bl_inv[512];
static void bl_init(void) {
    static bool done = false;
    if (done) return;
    for (int i = 0; i < 512; i++) bl_inv[i] = -1;
    int n = 0;
    for (int b = 0; b < 256; b++) {
        bool direct = (b >= 33 && b <= 126) || (b >= 161 && b <= 172) || (b >= 174 && b <= 255);
        if (direct) bl_inv[b] = b;
        else bl_inv[256 + n++] = b;
    }
    done = true;
}

// byte-level encoded string -> raw bytes. returns length or -1 if not byte-level.
static int bl_decode(const char *s, size_t n, char *out) {
    size_t o = 0;
    for (size_t i = 0; i < n;) {
        uint32_t cp;
        i += (size_t)utf8_dec((const unsigned char *)s + i, n - i, &cp);
        if (cp >= 512 || bl_inv[cp] < 0) return -1;
        out[o++] = (char)bl_inv[cp];
    }
    return (int)o;
}

tokenizer *tok_load_hf_json(const char *path, char *err, size_t errlen) {
    bl_init();
    size_t len;
    char *txt = inz_read_file(path, &len);
    if (!txt) { snprintf(err, errlen, "cannot read %s", path); return NULL; }
    jval *root = json_parse(txt, len, err, errlen);
    free(txt);
    if (!root) return NULL;
    jval *model = json_get(root, "model");
    jval *vocab = json_get(model, "vocab");
    jval *merges = json_get(model, "merges");
    jval *added = json_get(root, "added_tokens");
    // pre-tokenizer flavour from the split regex (Qwen isolates single digits and joins letters with marks)
    int split = TOK_SPLIT_O200K;
    {
        jval *pre = json_get(root, "pre_tokenizer");
        jval *list = pre ? json_get(pre, "pretokenizers") : NULL;
        for (jval *x = list ? list->child : (pre ? pre : NULL); x; x = list ? x->next : NULL) {
            const char *rx = json_str(json_path(x, "pattern.Regex"), "");
            if (strstr(rx, "[\\p{L}\\p{M}]+|\\p{N}|")) split = TOK_SPLIT_QWEN;
            if (!list) break;
        }
    }
    if (!vocab || !merges) { snprintf(err, errlen, "tokenizer.json: missing model.vocab/merges"); json_free(root); return NULL; }
    tokenizer *t = inz_calloc(1, sizeof(*t));
    int maxid = 0;
    for (jval *v = vocab->child; v; v = v->next) if ((int)v->num > maxid) maxid = (int)v->num;
    for (jval *a = added ? added->child : NULL; a; a = a->next) {
        int id = (int)json_int(json_get(a, "id"), 0);
        if (id > maxid) maxid = id;
    }
    t->n_vocab = maxid + 1;
    t->piece = inz_calloc((size_t)t->n_vocab, sizeof(char *));
    t->plen = inz_calloc((size_t)t->n_vocab, sizeof(uint32_t));
    char *buf = inz_malloc(4096);
    for (jval *v = vocab->child; v; v = v->next) {
        int id = (int)v->num;
        size_t kl = strlen(v->key);
        if (kl * 1 + 1 > 4096) continue;
        int bl = bl_decode(v->key, kl, buf);
        if (bl < 0) { // not byte-level: store verbatim
            memcpy(buf, v->key, kl);
            bl = (int)kl;
        }
        t->piece[id] = inz_malloc((size_t)bl + 1);
        memcpy(t->piece[id], buf, (size_t)bl);
        t->piece[id][bl] = 0;
        t->plen[id] = (uint32_t)bl;
    }
    // added tokens
    t->split = split;
    t->n_added = added ? (int)added->len : 0;
    t->added_id = inz_calloc((size_t)t->n_added + 1, sizeof(int32_t));
    t->added_str = inz_calloc((size_t)t->n_added + 1, sizeof(char *));
    t->added_len = inz_calloc((size_t)t->n_added + 1, sizeof(uint32_t));
    t->added_special = inz_calloc((size_t)t->n_added + 1, 1);
    int ai = 0;
    for (jval *a = added ? added->child : NULL; a; a = a->next, ai++) {
        int id = (int)json_int(json_get(a, "id"), 0);
        const char *content = json_str(json_get(a, "content"), "");
        t->added_id[ai] = id;
        t->added_str[ai] = inz_strdup(content);
        t->added_len[ai] = (uint32_t)strlen(content);
        t->added_special[ai] = json_bool(json_get(a, "special"), false);
        if (!t->piece[id]) {
            t->piece[id] = inz_strdup(content);
            t->plen[id] = (uint32_t)strlen(content);
        }
    }
    // merges: "a b" strings or ["a","b"] pairs
    t->n_merges = 0;
    t->m_a = inz_malloc(sizeof(int32_t) * merges->len);
    t->m_b = inz_malloc(sizeof(int32_t) * merges->len);
    t->m_res = inz_malloc(sizeof(int32_t) * merges->len);
    char *b2 = inz_malloc(4096);
    // temporary piece lookup for merge resolution
    vtab_build(t);
    for (jval *m = merges->child; m; m = m->next) {
        const char *sa, *sb;
        size_t la, lb;
        char tmp[2048];
        if (m->type == JSTR) {
            const char *sp = strchr(m->str + 1, ' ');  // first char may itself be a space-like symbol
            if (!sp) continue;
            la = (size_t)(sp - m->str);
            if (la >= sizeof(tmp)) continue;
            memcpy(tmp, m->str, la);
            tmp[la] = 0;
            sa = tmp;
            sb = sp + 1;
            lb = strlen(sb);
        } else if (m->type == JARR && m->len == 2) {
            sa = json_str(json_at(m, 0), "");
            sb = json_str(json_at(m, 1), "");
            la = strlen(sa);
            lb = strlen(sb);
        } else continue;
        int na = bl_decode(sa, la, buf), nb = bl_decode(sb, lb, b2);
        if (na < 0 || nb < 0 || na + nb >= 4096) continue;
        int ia = vfind(t, buf, (size_t)na), ib = vfind(t, b2, (size_t)nb);
        memcpy(buf + na, b2, (size_t)nb);
        int ir = vfind(t, buf, (size_t)(na + nb));
        if (ia < 0 || ib < 0 || ir < 0) continue;
        t->m_a[t->n_merges] = ia;
        t->m_b[t->n_merges] = ib;
        t->m_res[t->n_merges] = ir;
        t->n_merges++;
    }
    free(t->vtab);
    t->vtab = NULL;
    free(buf);
    free(b2);
    json_free(root);
    finish_init(t);
    return t;
}

tokenizer *tok_create(int n_vocab, const char *const *pieces, const uint32_t *piece_len, int n_merges,
                      const int32_t *merge_a, const int32_t *merge_b, const int32_t *merge_res,
                      int n_added, const int32_t *added_ids, const char *const *added_str, const uint8_t *added_special) {
    tokenizer *t = inz_calloc(1, sizeof(*t));
    t->n_vocab = n_vocab;
    t->piece = inz_calloc((size_t)n_vocab, sizeof(char *));
    t->plen = inz_calloc((size_t)n_vocab, sizeof(uint32_t));
    for (int i = 0; i < n_vocab; i++) {
        if (!pieces[i] || (piece_len[i] == 0 && !pieces[i][0])) continue;
        t->piece[i] = inz_malloc(piece_len[i] + 1);
        memcpy(t->piece[i], pieces[i], piece_len[i]);
        t->piece[i][piece_len[i]] = 0;
        t->plen[i] = piece_len[i];
    }
    t->n_merges = n_merges;
    t->m_a = inz_malloc(sizeof(int32_t) * (n_merges + 1));
    t->m_b = inz_malloc(sizeof(int32_t) * (n_merges + 1));
    t->m_res = inz_malloc(sizeof(int32_t) * (n_merges + 1));
    memcpy(t->m_a, merge_a, sizeof(int32_t) * n_merges);
    memcpy(t->m_b, merge_b, sizeof(int32_t) * n_merges);
    memcpy(t->m_res, merge_res, sizeof(int32_t) * n_merges);
    t->n_added = n_added;
    t->added_id = inz_calloc((size_t)n_added + 1, sizeof(int32_t));
    t->added_str = inz_calloc((size_t)n_added + 1, sizeof(char *));
    t->added_len = inz_calloc((size_t)n_added + 1, sizeof(uint32_t));
    t->added_special = inz_calloc((size_t)n_added + 1, 1);
    for (int i = 0; i < n_added; i++) {
        t->added_id[i] = added_ids[i];
        t->added_str[i] = inz_strdup(added_str[i]);
        t->added_len[i] = (uint32_t)strlen(added_str[i]);
        t->added_special[i] = added_special[i];
    }
    finish_init(t);
    return t;
}

void tok_free(tokenizer *t) {
    if (!t) return;
    for (int i = 0; i < t->n_vocab; i++) free(t->piece[i]);
    for (int i = 0; i < t->n_added; i++) free(t->added_str[i]);
    for (uint32_t i = 0; i <= t->cmask; i++) { free(t->cache[i].key); free(t->cache[i].ids); }
    free(t->piece); free(t->plen); free(t->special); free(t->vtab); free(t->mtab);
    free(t->m_a); free(t->m_b); free(t->m_res);
    free(t->added_id); free(t->added_str); free(t->added_len); free(t->added_special);
    free(t->cache);
    free(t);
}

void tok_get_export(const tokenizer *t, tok_export *e) {
    e->n_vocab = t->n_vocab;
    e->pieces = (const char **)t->piece;
    e->piece_len = t->plen;
    e->n_merges = t->n_merges;
    e->merge_a = t->m_a;
    e->merge_b = t->m_b;
    e->merge_res = t->m_res;
    e->n_added = t->n_added;
    e->added_ids = t->added_id;
    e->added_str = (const char **)t->added_str;
    e->added_special = t->added_special;
}

int tok_vocab(const tokenizer *t) { return t->n_vocab; }
bool tok_is_special(const tokenizer *t, int id) { return id >= 0 && id < t->n_vocab && t->special[id]; }

const char *tok_piece(const tokenizer *t, int id, size_t *len) {
    if (id < 0 || id >= t->n_vocab || !t->piece[id]) { *len = 0; return ""; }
    *len = t->plen[id];
    return t->piece[id];
}

int tok_find(const tokenizer *t, const char *str) {
    for (int i = 0; i < t->n_added; i++) if (!strcmp(t->added_str[i], str)) return t->added_id[i];
    return vfind(t, str, strlen(str));
}

/* ------------------------------------------------------------------ */
/* Pre-tokenizer (o200k-style split, as in the MiniMax tokenizer.json) */
/* ------------------------------------------------------------------ */

typedef struct { uint32_t cp; uint16_t f; uint32_t off; } ucp;

#define IS_A(f) ((f) & (UC_LU | UC_LT | UC_LM | UC_LO | UC_M))
#define IS_B(f) ((f) & (UC_LL | UC_LM | UC_LO | UC_M))
#define IS_WS(f) ((f) & UC_WS)

static inline bool is_nl(uint32_t c) { return c == '\r' || c == '\n'; }
static inline bool p1(const ucp *u) { return !is_nl(u->cp) && !(u->f & (UC_L | UC_N)); }
static inline bool c4(const ucp *u) { return !(u->f & (UC_WS | UC_L | UC_N)); }

static int contraction(const ucp *u, int n, int e) {
    if (e >= n || u[e].cp != '\'') return e;
    if (e + 1 >= n) return e;
    uint32_t c1 = u[e + 1].cp | 0x20;
    if (c1 == 's' || c1 == 't' || c1 == 'm' || c1 == 'd') return e + 2;
    if (e + 2 < n) {
        uint32_t c2 = u[e + 2].cp | 0x20;
        if ((c1 == 'r' && c2 == 'e') || (c1 == 'v' && c2 == 'e') || (c1 == 'l' && c2 == 'l')) return e + 3;
    }
    return e;
}

// Qwen (Qwen3.x / Qwen3.8) split:
//   (?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+|\p{N}| ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*
//   |\s*[\r\n]+|\s+(?!\S)|\s+
// Alternatives are tried in order (leftmost-first), so a contraction wins over a longer letter run.
#define IS_LM(f) ((f) & (UC_L | UC_M))
static inline bool q4(const ucp *u) { return !(u->f & (UC_WS | UC_L | UC_M | UC_N)); }
static inline uint32_t lower_ascii(uint32_t c) { return c >= 'A' && c <= 'Z' ? c | 0x20 : c; }

static int match_at_qwen(const ucp *u, int n, int i) {
    if (u[i].cp == '\'' && i + 1 < n) {
        const uint32_t c1 = lower_ascii(u[i + 1].cp);
        if (c1 == 's' || c1 == 't' || c1 == 'm' || c1 == 'd') return i + 2;
        if (i + 2 < n) {
            const uint32_t c2 = lower_ascii(u[i + 2].cp);
            if ((c1 == 'r' && c2 == 'e') || (c1 == 'v' && c2 == 'e') || (c1 == 'l' && c2 == 'l')) return i + 3;
        }
    }
    {
        int j = i;
        if (!is_nl(u[j].cp) && !(u[j].f & (UC_L | UC_N)) && j + 1 < n && IS_LM(u[j + 1].f)) j++;
        if (IS_LM(u[j].f)) {
            int e = j;
            while (e < n && IS_LM(u[e].f)) e++;
            return e;
        }
    }
    if (u[i].f & UC_N) return i + 1;
    {
        int j = i;
        if (u[j].cp == ' ' && j + 1 < n && q4(&u[j + 1])) j++;
        if (q4(&u[j])) {
            int e = j;
            while (e < n && q4(&u[e])) e++;
            while (e < n && is_nl(u[e].cp)) e++;
            return e;
        }
    }
    if (IS_WS(u[i].f)) {
        int e = i;
        while (e < n && IS_WS(u[e].f)) e++;
        for (int k = e - 1; k >= i; k--) if (is_nl(u[k].cp)) return k + 1;  // \s*[\r\n]+
        if (e == n) return e;                                              // \s+(?!\S)
        if (e - i >= 2) return e - 1;
        return e;                                                          // \s+
    }
    return i + 1;
}

static int match_at(const ucp *u, int n, int i) {
    // alternatives 1 and 2 (letters with optional leading symbol)
    for (int alt = 1; alt <= 2; alt++) {
        for (int pre = 1; pre >= 0; pre--) {
            int j = i;
            if (pre) { if (j < n && p1(&u[j])) j++; else continue; }
            if (alt == 1) {
                int ae = j;
                while (ae < n && IS_A(u[ae].f)) ae++;
                for (int k = ae; k >= j; k--) {
                    if (k < n && IS_B(u[k].f)) {
                        int e = k;
                        while (e < n && IS_B(u[e].f)) e++;
                        return contraction(u, n, e);
                    }
                }
            } else {
                if (j < n && IS_A(u[j].f)) {
                    int e = j;
                    while (e < n && IS_A(u[e].f)) e++;
                    while (e < n && IS_B(u[e].f)) e++;
                    return contraction(u, n, e);
                }
            }
        }
    }
    // \p{N}{1,3}
    if (u[i].f & UC_N) {
        int e = i;
        while (e < n && e < i + 3 && (u[e].f & UC_N)) e++;
        return e;
    }
    // ' ?[^\s\p{L}\p{N}]+[\r\n/]*'
    for (int sp = 1; sp >= 0; sp--) {
        int j = i;
        if (sp) { if (u[j].cp == ' ') j++; else continue; }
        if (j < n && c4(&u[j])) {
            int e = j;
            while (e < n && c4(&u[e])) e++;
            while (e < n && (is_nl(u[e].cp) || u[e].cp == '/')) e++;
            return e;
        }
    }
    if (IS_WS(u[i].f)) {
        int e = i;
        while (e < n && IS_WS(u[e].f)) e++;
        // \s*[\r\n]+
        for (int k = e - 1; k >= i; k--) if (is_nl(u[k].cp)) return k + 1;
        // \s+(?!\S)
        if (e == n) return e;
        if (e - i >= 2) return e - 1;
        // \s+
        return e;
    }
    return i + 1;
}

/* ------------------------------------------------------------------ */
/* BPE                                                                 */
/* ------------------------------------------------------------------ */

typedef struct { int32_t rank; int32_t pos; int32_t a, b; } heap_ent;

static void heap_push(heap_ent *h, int *n, heap_ent e) {
    int i = (*n)++;
    h[i] = e;
    while (i > 0) {
        int p = (i - 1) / 2;
        if (h[p].rank < h[i].rank || (h[p].rank == h[i].rank && h[p].pos <= h[i].pos)) break;
        heap_ent t = h[p]; h[p] = h[i]; h[i] = t;
        i = p;
    }
}

static heap_ent heap_pop(heap_ent *h, int *n) {
    heap_ent top = h[0];
    h[0] = h[--(*n)];
    int i = 0;
    for (;;) {
        int l = 2 * i + 1, r = l + 1, m = i;
        if (l < *n && (h[l].rank < h[m].rank || (h[l].rank == h[m].rank && h[l].pos < h[m].pos))) m = l;
        if (r < *n && (h[r].rank < h[m].rank || (h[r].rank == h[m].rank && h[r].pos < h[m].pos))) m = r;
        if (m == i) break;
        heap_ent t = h[m]; h[m] = h[i]; h[i] = t;
        i = m;
    }
    return top;
}

static int bpe_word(tokenizer *t, const char *s, int n, int32_t *out) {
    if (n == 1) { out[0] = t->byte_tok[(unsigned char)s[0]]; return 1; }
    int32_t *id = inz_malloc(sizeof(int32_t) * n), *prev = inz_malloc(sizeof(int32_t) * n), *next = inz_malloc(sizeof(int32_t) * n);
    heap_ent *heap = inz_malloc(sizeof(heap_ent) * (size_t)(n * 2 + 2));
    int hn = 0;
    for (int i = 0; i < n; i++) {
        id[i] = t->byte_tok[(unsigned char)s[i]];
        prev[i] = i - 1;
        next[i] = i + 1 < n ? i + 1 : -1;
    }
    for (int i = 0; i + 1 < n; i++) {
        const merge_ent *m = mfind(t, id[i], id[i + 1]);
        if (m) heap_push(heap, &hn, (heap_ent){m->rank, i, id[i], id[i + 1]});
    }
    while (hn) {
        heap_ent e = heap_pop(heap, &hn);
        int i = e.pos;
        if (id[i] != e.a || next[i] < 0 || id[next[i]] != e.b) continue;  // stale
        int j = next[i];
        const merge_ent *m = mfind(t, e.a, e.b);
        id[i] = m->res;
        id[j] = -1;
        next[i] = next[j];
        if (next[j] >= 0) prev[next[j]] = i;
        if (prev[i] >= 0) {
            const merge_ent *pm = mfind(t, id[prev[i]], id[i]);
            if (pm) heap_push(heap, &hn, (heap_ent){pm->rank, prev[i], id[prev[i]], id[i]});
        }
        if (next[i] >= 0) {
            const merge_ent *nm = mfind(t, id[i], id[next[i]]);
            if (nm) heap_push(heap, &hn, (heap_ent){nm->rank, i, id[i], id[next[i]]});
        }
    }
    int k = 0;
    for (int i = 0; i >= 0; i = next[i]) out[k++] = id[i];
    free(id); free(prev); free(next); free(heap);
    return k;
}

static int encode_chunk(tokenizer *t, const char *s, size_t len, int32_t *out, int cap, int count) {
    // NFC normalize, split, BPE
    char *norm;
    size_t nl = inz_nfc(s, len, &norm);
    ucp *u = inz_malloc(sizeof(ucp) * (nl + 1));
    int n = 0;
    for (size_t i = 0; i < nl;) {
        uint32_t cp;
        int l = utf8_dec((const unsigned char *)norm + i, nl - i, &cp);
        u[n].cp = cp;
        u[n].f = cp >= 0x110000u ? 0 : uc_flags(cp);
        u[n].off = (uint32_t)i;
        n++;
        i += (size_t)l;
    }
    u[n].off = (uint32_t)nl;
    int32_t tmp_small[256];
    for (int i = 0; i < n;) {
        int e = t->split == TOK_SPLIT_QWEN ? match_at_qwen(u, n, i) : match_at(u, n, i);
        if (e <= i) e = i + 1;
        const char *ws = norm + u[i].off;
        int wl = (int)(u[e].off - u[i].off);
        // cache lookup
        uint32_t h = (uint32_t)hbytes(ws, (size_t)wl) & t->cmask;
        cache_ent *ce = &t->cache[h];
        if (ce->key && ce->klen == (uint32_t)wl && !memcmp(ce->key, ws, (size_t)wl)) {
            for (uint32_t k = 0; k < ce->nids; k++) { if (count < cap) out[count] = ce->ids[k]; count++; }
        } else {
            int32_t *ids = wl <= 256 ? tmp_small : inz_malloc(sizeof(int32_t) * wl);
            int nt = bpe_word(t, ws, wl, ids);
            for (int k = 0; k < nt; k++) { if (count < cap) out[count] = ids[k]; count++; }
            if (wl <= 64) {
                free(ce->key); free(ce->ids);
                ce->key = inz_malloc((size_t)wl);
                memcpy(ce->key, ws, (size_t)wl);
                ce->klen = (uint32_t)wl;
                ce->ids = inz_malloc(sizeof(int32_t) * nt);
                memcpy(ce->ids, ids, sizeof(int32_t) * nt);
                ce->nids = (uint32_t)nt;
            }
            if (ids != tmp_small) free(ids);
        }
        i = e;
    }
    free(u);
    free(norm);
    return count;
}

int tok_encode(tokenizer *t, const char *text, size_t len, bool parse_special, int32_t *out, int cap) {
    int count = 0;
    size_t seg = 0;
    if (parse_special) {
        for (size_t i = 0; i < len;) {
            int best = -1;
            uint32_t bl = 0;
            for (int a = 0; a < t->n_added; a++) {
                uint32_t l = t->added_len[a];
                if (l > bl && l <= len - i && t->added_str[a][0] == text[i] && !memcmp(text + i, t->added_str[a], l)) {
                    best = a;
                    bl = l;
                }
            }
            if (best >= 0) {
                if (i > seg) count = encode_chunk(t, text + seg, i - seg, out, cap, count);
                if (count < cap) out[count] = t->added_id[best];
                count++;
                i += bl;
                seg = i;
            } else i++;
        }
    }
    if (seg < len) count = encode_chunk(t, text + seg, len - seg, out, cap, count);
    return count;
}

size_t tok_stream_push(const tokenizer *t, tok_stream *s, int id, char *out, size_t cap) {
    size_t pl;
    const char *p = tok_piece(t, id, &pl);
    size_t o = 0;
    for (size_t i = 0; i < pl; i++) {
        if (s->npend < (int)sizeof(s->pend)) s->pend[s->npend++] = p[i];
        // flush every complete sequence at the head of pend
        for (;;) {
            if (s->npend == 0) break;
            unsigned char c = (unsigned char)s->pend[0];
            int need = c < 0x80 ? 1 : (c & 0xE0) == 0xC0 ? 2 : (c & 0xF0) == 0xE0 ? 3 : (c & 0xF8) == 0xF0 ? 4 : 1;
            if (s->npend < need) break;
            for (int k = 0; k < need && o < cap; k++) out[o++] = s->pend[k];
            memmove(s->pend, s->pend + need, (size_t)(s->npend - need));
            s->npend -= need;
        }
    }
    if (o < cap) out[o] = 0;
    return o;
}

void tok_set_split(tokenizer *t, int split) { t->split = split; }
int tok_get_split(const tokenizer *t) { return t->split; }

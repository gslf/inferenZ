#include "json.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    const char *p, *end;
    char *err;
    size_t errlen;
    int depth;
} jparser;

static void jerr(jparser *ps, const char *msg) {
    if (ps->err && ps->err[0] == 0) snprintf(ps->err, ps->errlen, "%s", msg);
}

static void skip_ws(jparser *ps) {
    while (ps->p < ps->end && (*ps->p == ' ' || *ps->p == '\n' || *ps->p == '\r' || *ps->p == '\t')) ps->p++;
}

static jval *jnew(jtype t) {
    jval *v = calloc(1, sizeof(jval));
    v->type = t;
    return v;
}

static int hexv(char c) {
    if (c >= '0' && c <= '9') return c - '0';
    if (c >= 'a' && c <= 'f') return c - 'a' + 10;
    if (c >= 'A' && c <= 'F') return c - 'A' + 10;
    return -1;
}

static size_t utf8_put(char *o, uint32_t cp) {
    if (cp < 0x80) { o[0] = (char)cp; return 1; }
    if (cp < 0x800) { o[0] = (char)(0xC0 | (cp >> 6)); o[1] = (char)(0x80 | (cp & 0x3F)); return 2; }
    if (cp < 0x10000) {
        o[0] = (char)(0xE0 | (cp >> 12)); o[1] = (char)(0x80 | ((cp >> 6) & 0x3F));
        o[2] = (char)(0x80 | (cp & 0x3F)); return 3;
    }
    o[0] = (char)(0xF0 | (cp >> 18)); o[1] = (char)(0x80 | ((cp >> 12) & 0x3F));
    o[2] = (char)(0x80 | ((cp >> 6) & 0x3F)); o[3] = (char)(0x80 | (cp & 0x3F)); return 4;
}

static char *parse_string_raw(jparser *ps, size_t *outlen) {
    if (ps->p >= ps->end || *ps->p != '"') { jerr(ps, "expected string"); return NULL; }
    ps->p++;
    const char *s = ps->p;
    size_t cap = 16;
    // first pass to find end and size upper bound
    while (ps->p < ps->end && *ps->p != '"') {
        if (*ps->p == '\\') ps->p++;
        ps->p++;
    }
    if (ps->p >= ps->end) { jerr(ps, "unterminated string"); return NULL; }
    cap = (size_t)(ps->p - s) + 1;
    char *out = malloc(cap + 4);
    size_t o = 0;
    for (const char *q = s; q < ps->p;) {
        if (*q != '\\') { out[o++] = *q++; continue; }
        q++;
        char c = *q++;
        switch (c) {
        case '"': out[o++] = '"'; break;
        case '\\': out[o++] = '\\'; break;
        case '/': out[o++] = '/'; break;
        case 'b': out[o++] = '\b'; break;
        case 'f': out[o++] = '\f'; break;
        case 'n': out[o++] = '\n'; break;
        case 'r': out[o++] = '\r'; break;
        case 't': out[o++] = '\t'; break;
        case 'u': {
            if (q + 4 > ps->p) { jerr(ps, "bad \\u escape"); free(out); return NULL; }
            uint32_t cp = 0;
            for (int i = 0; i < 4; i++) {
                int h = hexv(q[i]);
                if (h < 0) { jerr(ps, "bad hex"); free(out); return NULL; }
                cp = cp * 16 + (uint32_t)h;
            }
            q += 4;
            if (cp >= 0xD800 && cp <= 0xDBFF && q + 6 <= ps->p && q[0] == '\\' && q[1] == 'u') {
                uint32_t lo = 0;
                bool ok = true;
                for (int i = 0; i < 4; i++) {
                    int h = hexv(q[2 + i]);
                    if (h < 0) { ok = false; break; }
                    lo = lo * 16 + (uint32_t)h;
                }
                if (ok && lo >= 0xDC00 && lo <= 0xDFFF) {
                    cp = 0x10000 + ((cp - 0xD800) << 10) + (lo - 0xDC00);
                    q += 6;
                }
            }
            o += utf8_put(out + o, cp);
            break;
        }
        default: out[o++] = c; break;
        }
    }
    out[o] = 0;
    ps->p++;  // closing quote
    if (outlen) *outlen = o;
    return out;
}

static jval *parse_value(jparser *ps);

static jval *parse_value(jparser *ps) {
    skip_ws(ps);
    if (ps->p >= ps->end) { jerr(ps, "unexpected end"); return NULL; }
    if (++ps->depth > 512) { jerr(ps, "nesting too deep"); return NULL; }
    jval *v = NULL;
    char c = *ps->p;
    if (c == '{') {
        v = jnew(JOBJ);
        ps->p++;
        skip_ws(ps);
        jval **tail = &v->child;
        if (ps->p < ps->end && *ps->p == '}') { ps->p++; goto done; }
        for (;;) {
            skip_ws(ps);
            size_t kl;
            char *k = parse_string_raw(ps, &kl);
            if (!k) { json_free(v); v = NULL; goto done; }
            skip_ws(ps);
            if (ps->p >= ps->end || *ps->p != ':') { free(k); jerr(ps, "expected ':'"); json_free(v); v = NULL; goto done; }
            ps->p++;
            jval *m = parse_value(ps);
            if (!m) { free(k); json_free(v); v = NULL; goto done; }
            m->key = k;
            *tail = m;
            tail = &m->next;
            v->len++;
            skip_ws(ps);
            if (ps->p < ps->end && *ps->p == ',') { ps->p++; continue; }
            if (ps->p < ps->end && *ps->p == '}') { ps->p++; break; }
            jerr(ps, "expected ',' or '}'");
            json_free(v); v = NULL; goto done;
        }
    } else if (c == '[') {
        v = jnew(JARR);
        ps->p++;
        skip_ws(ps);
        jval **tail = &v->child;
        if (ps->p < ps->end && *ps->p == ']') { ps->p++; goto done; }
        for (;;) {
            jval *m = parse_value(ps);
            if (!m) { json_free(v); v = NULL; goto done; }
            *tail = m;
            tail = &m->next;
            v->len++;
            skip_ws(ps);
            if (ps->p < ps->end && *ps->p == ',') { ps->p++; continue; }
            if (ps->p < ps->end && *ps->p == ']') { ps->p++; break; }
            jerr(ps, "expected ',' or ']'");
            json_free(v); v = NULL; goto done;
        }
    } else if (c == '"') {
        v = jnew(JSTR);
        v->str = parse_string_raw(ps, &v->len);
        if (!v->str) { free(v); v = NULL; }
    } else if (c == 't' && ps->end - ps->p >= 4 && !memcmp(ps->p, "true", 4)) {
        v = jnew(JBOOL); v->b = true; ps->p += 4;
    } else if (c == 'f' && ps->end - ps->p >= 5 && !memcmp(ps->p, "false", 5)) {
        v = jnew(JBOOL); v->b = false; ps->p += 5;
    } else if (c == 'n' && ps->end - ps->p >= 4 && !memcmp(ps->p, "null", 4)) {
        v = jnew(JNULL); ps->p += 4;
    } else if (c == 'N' && ps->end - ps->p >= 3 && !memcmp(ps->p, "NaN", 3)) {
        v = jnew(JNUM); v->num = NAN; ps->p += 3;
    } else if (c == 'I' && ps->end - ps->p >= 8 && !memcmp(ps->p, "Infinity", 8)) {
        v = jnew(JNUM); v->num = INFINITY; ps->p += 8;
    } else if (c == '-' || (c >= '0' && c <= '9')) {
        char tmp[64];
        size_t n = 0;
        while (ps->p < ps->end && n < sizeof(tmp) - 1 &&
               (strchr("+-.eE", *ps->p) || (*ps->p >= '0' && *ps->p <= '9')))
            tmp[n++] = *ps->p++;
        tmp[n] = 0;
        if (!strcmp(tmp, "-") && ps->end - ps->p >= 8 && !memcmp(ps->p, "Infinity", 8)) {
            ps->p += 8;
            v = jnew(JNUM); v->num = -INFINITY;
        } else {
            v = jnew(JNUM);
            v->num = strtod(tmp, NULL);
        }
    } else {
        jerr(ps, "unexpected character");
    }
done:
    ps->depth--;
    return v;
}

jval *json_parse(const char *text, size_t len, char *err, size_t errlen) {
    if (err && errlen) err[0] = 0;
    jparser ps = {.p = text, .end = text + len, .err = err, .errlen = errlen};
    jval *v = parse_value(&ps);
    if (v) {
        skip_ws(&ps);
        if (ps.p != ps.end && *ps.p != 0) {
            jerr(&ps, "trailing characters");
            json_free(v);
            return NULL;
        }
    }
    return v;
}

void json_free(jval *v) {
    while (v) {
        jval *n = v->next;
        if (v->child) json_free(v->child);
        free(v->str);
        free(v->key);
        free(v);
        v = n;
    }
}

jval *json_get(const jval *obj, const char *key) {
    if (!obj || obj->type != JOBJ) return NULL;
    for (jval *c = obj->child; c; c = c->next)
        if (c->key && !strcmp(c->key, key)) return c;
    return NULL;
}

jval *json_at(const jval *arr, size_t idx) {
    if (!arr || (arr->type != JARR && arr->type != JOBJ)) return NULL;
    jval *c = arr->child;
    while (c && idx--) c = c->next;
    return c;
}

const char *json_str(const jval *v, const char *def) { return v && v->type == JSTR ? v->str : def; }
double json_num(const jval *v, double def) {
    if (!v) return def;
    if (v->type == JNUM) return v->num;
    if (v->type == JBOOL) return v->b;
    return def;
}
int64_t json_int(const jval *v, int64_t def) { return v && v->type == JNUM ? (int64_t)v->num : def; }
bool json_bool(const jval *v, bool def) {
    if (!v) return def;
    if (v->type == JBOOL) return v->b;
    if (v->type == JNUM) return v->num != 0;
    return def;
}

jval *json_path(const jval *root, const char *dotted) {
    char key[256];
    const jval *cur = root;
    while (cur && *dotted) {
        const char *dot = strchr(dotted, '.');
        size_t n = dot ? (size_t)(dot - dotted) : strlen(dotted);
        if (n >= sizeof(key)) return NULL;
        memcpy(key, dotted, n);
        key[n] = 0;
        cur = json_get(cur, key);
        dotted += n + (dot ? 1 : 0);
        if (!dot) break;
    }
    return (jval *)cur;
}

/* ---------------- writer ---------------- */

static void wgrow(jw *w, size_t add) {
    if (w->len + add + 1 > w->cap) {
        size_t nc = w->cap ? w->cap * 2 : 256;
        while (nc < w->len + add + 1) nc *= 2;
        w->buf = realloc(w->buf, nc);
        w->cap = nc;
    }
}

static void wput(jw *w, const char *s, size_t n) {
    wgrow(w, n);
    memcpy(w->buf + w->len, s, n);
    w->len += n;
    w->buf[w->len] = 0;
}

void json_escape_append(char **buf, size_t *len, size_t *cap, const char *s, size_t n) {
    static const char hex[] = "0123456789abcdef";
    for (size_t i = 0; i < n; i++) {
        unsigned char c = (unsigned char)s[i];
        char tmp[8];
        size_t tl = 0;
        switch (c) {
        case '"': tmp[0] = '\\'; tmp[1] = '"'; tl = 2; break;
        case '\\': tmp[0] = '\\'; tmp[1] = '\\'; tl = 2; break;
        case '\n': tmp[0] = '\\'; tmp[1] = 'n'; tl = 2; break;
        case '\r': tmp[0] = '\\'; tmp[1] = 'r'; tl = 2; break;
        case '\t': tmp[0] = '\\'; tmp[1] = 't'; tl = 2; break;
        default:
            if (c < 0x20) {
                tmp[0] = '\\'; tmp[1] = 'u'; tmp[2] = '0'; tmp[3] = '0';
                tmp[4] = hex[c >> 4]; tmp[5] = hex[c & 15]; tl = 6;
            } else { tmp[0] = (char)c; tl = 1; }
        }
        if (*len + tl + 1 > *cap) {
            size_t nc = *cap ? *cap * 2 : 256;
            while (nc < *len + tl + 1) nc *= 2;
            *buf = realloc(*buf, nc);
            *cap = nc;
        }
        memcpy(*buf + *len, tmp, tl);
        *len += tl;
        (*buf)[*len] = 0;
    }
}

void jw_init(jw *w) { memset(w, 0, sizeof(*w)); wgrow(w, 64); w->buf[0] = 0; }
void jw_free(jw *w) { free(w->buf); memset(w, 0, sizeof(*w)); }

static void sep(jw *w) {
    if (w->depth > 0 && w->need_comma[w->depth]) wput(w, ",", 1);
    if (w->depth > 0) w->need_comma[w->depth] = true;
}

void jw_obj(jw *w) { sep(w); wput(w, "{", 1); w->depth++; w->need_comma[w->depth] = false; }
void jw_obj_end(jw *w) { w->depth--; wput(w, "}", 1); }
void jw_arr(jw *w) { sep(w); wput(w, "[", 1); w->depth++; w->need_comma[w->depth] = false; }
void jw_arr_end(jw *w) { w->depth--; wput(w, "]", 1); }
void jw_key(jw *w, const char *k) {
    sep(w);
    wput(w, "\"", 1);
    json_escape_append(&w->buf, &w->len, &w->cap, k, strlen(k));
    wput(w, "\":", 2);
    w->need_comma[w->depth] = false;  // value follows without comma
    // mark that after the value a comma is needed
    w->need_comma[w->depth] = false;
    // trick: the value's sep() will set need_comma back to true
}
void jw_strn(jw *w, const char *s, size_t n) {
    sep(w);
    wput(w, "\"", 1);
    json_escape_append(&w->buf, &w->len, &w->cap, s ? s : "", s ? n : 0);
    wput(w, "\"", 1);
}
void jw_str(jw *w, const char *s) { jw_strn(w, s, s ? strlen(s) : 0); }
void jw_num(jw *w, double v) {
    char tmp[64];
    if (!isfinite(v)) snprintf(tmp, sizeof(tmp), "null");
    else if (v == (double)(int64_t)v && fabs(v) < 1e15) snprintf(tmp, sizeof(tmp), "%lld", (long long)v);
    else snprintf(tmp, sizeof(tmp), "%.6g", v);
    sep(w);
    wput(w, tmp, strlen(tmp));
}
void jw_int(jw *w, int64_t v) {
    char tmp[32];
    snprintf(tmp, sizeof(tmp), "%lld", (long long)v);
    sep(w);
    wput(w, tmp, strlen(tmp));
}
void jw_bool(jw *w, bool v) { sep(w); wput(w, v ? "true" : "false", v ? 4 : 5); }
void jw_null(jw *w) { sep(w); wput(w, "null", 4); }
void jw_raw(jw *w, const char *raw) { sep(w); wput(w, raw, strlen(raw)); }
void jw_kstr(jw *w, const char *k, const char *v) { jw_key(w, k); jw_str(w, v); }
void jw_knum(jw *w, const char *k, double v) { jw_key(w, k); jw_num(w, v); }
void jw_kint(jw *w, const char *k, int64_t v) { jw_key(w, k); jw_int(w, v); }
void jw_kbool(jw *w, const char *k, bool v) { jw_key(w, k); jw_bool(w, v); }

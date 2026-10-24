# Reference solutions for tests/code_eval.py: `code_eval.py --selftest` checks that every hidden test is
# satisfiable and that the harness (sandbox, compiler flags) works.
REF = {}

REF["c_reverse_words"] = r'''
#include <stdlib.h>
#include <string.h>
char *reverse_words(const char *s) {
    if (!s) return NULL;
    size_t n = strlen(s);
    char *out = malloc(n + 1), *o = out;
    if (!out) return NULL;
    size_t end = n;
    while (end > 0) {
        while (end > 0 && s[end - 1] == ' ') end--;
        size_t start = end;
        while (start > 0 && s[start - 1] != ' ') start--;
        if (start < end) {
            if (o != out) *o++ = ' ';
            memcpy(o, s + start, end - start);
            o += end - start;
        }
        end = start;
    }
    *o = 0;
    return out;
}
'''

REF["c_parse_long"] = r'''
#include <limits.h>
int parse_long(const char *s, long *out) {
    int neg = 0;
    if (*s == '+' || *s == '-') neg = *s++ == '-';
    if (!*s) return -1;
    long v = 0;
    for (; *s; s++) {
        if (*s < '0' || *s > '9') return -1;
        int d = *s - '0';
        if (neg) { if (v < (LONG_MIN + d) / 10) return -1; v = v * 10 - d; }
        else { if (v > (LONG_MAX - d) / 10) return -1; v = v * 10 + d; }
    }
    *out = v;
    return 0;
}
'''

REF["c_intvec"] = r'''
#include <stdlib.h>
#include <string.h>
typedef struct { int *data; size_t len, cap; } intvec;
void iv_init(intvec *v) { v->data = NULL; v->len = v->cap = 0; }
static int grow(intvec *v) {
    size_t nc = v->cap ? v->cap * 2 : 8;
    int *d = realloc(v->data, nc * sizeof *d);
    if (!d) return -1;
    v->data = d; v->cap = nc;
    return 0;
}
int iv_push(intvec *v, int x) { if (v->len == v->cap && grow(v)) return -1; v->data[v->len++] = x; return 0; }
int iv_pop(intvec *v, int *out) { if (!v->len) return -1; *out = v->data[--v->len]; return 0; }
int iv_insert(intvec *v, size_t i, int x) {
    if (i > v->len) return -1;
    if (v->len == v->cap && grow(v)) return -1;
    memmove(v->data + i + 1, v->data + i, (v->len - i) * sizeof *v->data);
    v->data[i] = x; v->len++;
    return 0;
}
void iv_free(intvec *v) { free(v->data); iv_init(v); }
'''

REF["c_utf8_len"] = r'''
#include <stddef.h>
long utf8_count(const char *s, size_t n) {
    const unsigned char *p = (const unsigned char *)s;
    long count = 0;
    size_t i = 0;
    while (i < n) {
        unsigned c = p[i];
        size_t len; unsigned cp, min;
        if (c < 0x80) { i++; count++; continue; }
        else if ((c & 0xE0) == 0xC0) { len = 2; cp = c & 0x1F; min = 0x80; }
        else if ((c & 0xF0) == 0xE0) { len = 3; cp = c & 0x0F; min = 0x800; }
        else if ((c & 0xF8) == 0xF0) { len = 4; cp = c & 0x07; min = 0x10000; }
        else return -1;
        if (n - i < len) return -1;
        for (size_t k = 1; k < len; k++) {
            if ((p[i + k] & 0xC0) != 0x80) return -1;
            cp = (cp << 6) | (p[i + k] & 0x3F);
        }
        if (cp < min || cp > 0x10FFFF || (cp >= 0xD800 && cp <= 0xDFFF)) return -1;
        i += len; count++;
    }
    return count;
}
'''

REF["c_balanced"] = r'''
#include <stdlib.h>
int balanced(const char *s) {
    size_t cap = 64, n = 0;
    char *st = malloc(cap);
    if (!st) return 0;
    int in_str = 0, ok = 1;
    for (; *s && ok; s++) {
        char c = *s;
        if (in_str) {
            if (c == '\\' && s[1]) s++;
            else if (c == '"') in_str = 0;
            continue;
        }
        if (c == '"') in_str = 1;
        else if (c == '(' || c == '[' || c == '{') {
            if (n == cap) { char *t = realloc(st, cap * 2); if (!t) { ok = 0; break; } st = t; cap *= 2; }
            st[n++] = c == '(' ? ')' : c == '[' ? ']' : '}';
        } else if (c == ')' || c == ']' || c == '}') {
            if (!n || st[--n] != c) ok = 0;
        }
    }
    ok = ok && !in_str && n == 0;
    free(st);
    return ok;
}
'''

REF["c_base64"] = r'''
#include <string.h>
static const char T[] = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
size_t b64_encode(const unsigned char *in, size_t n, char *out) {
    size_t o = 0;
    for (size_t i = 0; i < n; i += 3) {
        unsigned v = (unsigned)in[i] << 16 | (i + 1 < n ? (unsigned)in[i + 1] << 8 : 0) | (i + 2 < n ? in[i + 2] : 0);
        out[o++] = T[v >> 18 & 63]; out[o++] = T[v >> 12 & 63];
        out[o++] = i + 1 < n ? T[v >> 6 & 63] : '=';
        out[o++] = i + 2 < n ? T[v & 63] : '=';
    }
    out[o] = 0;
    return o;
}
static int val(char c) { const char *p = c ? strchr(T, c) : NULL; return p ? (int)(p - T) : -1; }
long b64_decode(const char *in, unsigned char *out) {
    size_t n = strlen(in);
    if (n % 4) return -1;
    long o = 0;
    for (size_t i = 0; i < n; i += 4) {
        int last = i + 4 == n, pad = 0;
        if (in[i + 3] == '=') pad = in[i + 2] == '=' ? 2 : 1;
        if (pad && !last) return -1;
        int a = val(in[i]), b = val(in[i + 1]), c = pad == 2 ? 0 : val(in[i + 2]), d = pad ? 0 : val(in[i + 3]);
        if (a < 0 || b < 0 || c < 0 || d < 0) return -1;
        unsigned v = (unsigned)a << 18 | (unsigned)b << 12 | (unsigned)c << 6 | (unsigned)d;
        out[o++] = (unsigned char)(v >> 16);
        if (pad < 2) out[o++] = (unsigned char)(v >> 8);
        if (pad < 1) out[o++] = (unsigned char)v;
    }
    return o;
}
'''

REF["c_hashmap"] = r'''
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
typedef struct { char *key; int value; int state; } slot;  /* state: 0 empty, 1 full, 2 deleted */
typedef struct strmap { slot *s; size_t cap, len, used; } strmap;
static uint64_t h(const char *k) { uint64_t x = 1469598103934665603ull; for (; *k; k++) x = (x ^ (unsigned char)*k) * 1099511628211ull; return x; }
strmap *sm_new(void) {
    strmap *m = calloc(1, sizeof *m);
    if (!m) return NULL;
    m->cap = 16; m->s = calloc(m->cap, sizeof *m->s);
    if (!m->s) { free(m); return NULL; }
    return m;
}
void sm_free(strmap *m) { if (!m) return; for (size_t i = 0; i < m->cap; i++) if (m->s[i].state == 1) free(m->s[i].key); free(m->s); free(m); }
static slot *find(const strmap *m, const char *k) {
    for (size_t i = h(k) & (m->cap - 1);; i = (i + 1) & (m->cap - 1)) {
        slot *s = &m->s[i];
        if (s->state == 0) return NULL;
        if (s->state == 1 && !strcmp(s->key, k)) return s;
    }
}
static int rehash(strmap *m, size_t nc) {
    slot *ns = calloc(nc, sizeof *ns);
    if (!ns) return -1;
    for (size_t i = 0; i < m->cap; i++) if (m->s[i].state == 1) {
        size_t j = h(m->s[i].key) & (nc - 1);
        while (ns[j].state) j = (j + 1) & (nc - 1);
        ns[j] = m->s[i];
    }
    free(m->s); m->s = ns; m->cap = nc; m->used = m->len;
    return 0;
}
int sm_set(strmap *m, const char *key, int value) {
    slot *s = find(m, key);
    if (s) { s->value = value; return 0; }
    if ((m->used + 1) * 4 > m->cap * 3 && rehash(m, m->len * 4 > m->cap ? m->cap * 2 : m->cap)) return -1;
    char *k = strdup(key);
    if (!k) return -1;
    size_t i = h(key) & (m->cap - 1);
    while (m->s[i].state == 1) i = (i + 1) & (m->cap - 1);
    if (m->s[i].state == 0) m->used++;
    m->s[i] = (slot){k, value, 1};
    m->len++;
    return 0;
}
int sm_get(const strmap *m, const char *key, int *value) { slot *s = find(m, key); if (!s) return 0; *value = s->value; return 1; }
int sm_del(strmap *m, const char *key) { slot *s = find(m, key); if (!s) return 0; free(s->key); s->key = NULL; s->state = 2; m->len--; return 1; }
size_t sm_len(const strmap *m) { return m->len; }
'''

REF["c_ring"] = r'''
#include <stdlib.h>
#include <string.h>
typedef struct { unsigned char *buf; size_t cap, head, len; } ring;
int ring_init(ring *r, size_t cap) {
    if (!cap) return -1;
    r->buf = malloc(cap);
    if (!r->buf) return -1;
    r->cap = cap; r->head = r->len = 0;
    return 0;
}
size_t ring_write(ring *r, const void *src, size_t n) {
    size_t room = r->cap - r->len;
    if (n > room) n = room;
    size_t tail = (r->head + r->len) % r->cap, first = r->cap - tail < n ? r->cap - tail : n;
    memcpy(r->buf + tail, src, first);
    memcpy(r->buf, (const unsigned char *)src + first, n - first);
    r->len += n;
    return n;
}
size_t ring_read(ring *r, void *dst, size_t n) {
    if (n > r->len) n = r->len;
    size_t first = r->cap - r->head < n ? r->cap - r->head : n;
    memcpy(dst, r->buf + r->head, first);
    memcpy((unsigned char *)dst + first, r->buf, n - first);
    r->head = (r->head + n) % r->cap; r->len -= n;
    return n;
}
void ring_free(ring *r) { free(r->buf); r->buf = NULL; r->cap = r->head = r->len = 0; }
'''

REF["c_merge_lists"] = r'''
#include <stddef.h>
struct node { int val; struct node *next; };
struct node *merge_sorted(struct node *a, struct node *b) {
    struct node head = {0, NULL}, *t = &head;
    while (a && b) {
        if (b->val < a->val) { t->next = b; b = b->next; }
        else { t->next = a; a = a->next; }
        t = t->next;
    }
    t->next = a ? a : b;
    return head.next;
}
'''

REF["c_csv_field"] = r'''
#include <stdlib.h>
#include <string.h>
void csv_free(char **fields, size_t n) { for (size_t i = 0; i < n; i++) free(fields[i]); free(fields); }
int csv_split(const char *line, char ***fields, size_t *n) {
    size_t cap = 4, cnt = 0, len = strlen(line);
    char **f = malloc(cap * sizeof *f);
    if (!f) return -1;
    const char *p = line;
    for (;;) {
        char *buf = malloc(len + 1), *o = buf;
        if (!buf) { csv_free(f, cnt); return -1; }
        if (*p == '"') {
            p++;
            for (;;) {
                if (!*p) { free(buf); csv_free(f, cnt); return -1; }
                if (*p == '"') { if (p[1] == '"') { *o++ = '"'; p += 2; continue; } p++; break; }
                *o++ = *p++;
            }
            if (*p && *p != ',') { free(buf); csv_free(f, cnt); return -1; }
        } else {
            while (*p && *p != ',') *o++ = *p++;
        }
        *o = 0;
        if (cnt == cap) { char **t = realloc(f, cap * 2 * sizeof *f); if (!t) { free(buf); csv_free(f, cnt); return -1; } f = t; cap *= 2; }
        f[cnt++] = buf;
        if (*p == ',') { p++; continue; }
        break;
    }
    *fields = f; *n = cnt;
    return 0;
}
'''

REF["js_lru"] = r'''
class LRUCache {
  constructor(capacity) { this.cap = capacity; this.m = new Map(); }
  get(key) { if (!this.m.has(key)) return undefined; const v = this.m.get(key); this.m.delete(key); this.m.set(key, v); return v; }
  set(key, value) {
    if (this.m.has(key)) this.m.delete(key);
    this.m.set(key, value);
    if (this.m.size > this.cap) this.m.delete(this.m.keys().next().value);
  }
  get size() { return this.m.size; }
}
'''

REF["js_csv"] = r'''
function parseCSV(text) {
  const rows = []; let row = [], field = "", i = 0, quoted = false, any = false;
  while (i < text.length) {
    const c = text[i];
    if (quoted) {
      if (c === '"') { if (text[i + 1] === '"') { field += '"'; i += 2; continue; } quoted = false; i++; continue; }
      field += c; i++; continue;
    }
    if (c === '"' && field === "") { quoted = true; any = true; i++; continue; }
    if (c === ",") { row.push(field); field = ""; any = true; i++; continue; }
    if (c === "\n" || (c === "\r" && text[i + 1] === "\n")) {
      row.push(field); rows.push(row); row = []; field = ""; any = false; i += c === "\r" ? 2 : 1; continue;
    }
    field += c; any = true; i++;
  }
  if (quoted) throw new SyntaxError("unterminated quote");
  if (any || field !== "" || row.length) { row.push(field); rows.push(row); }
  return rows;
}
'''

REF["js_deep_equal"] = r'''
function deepEqual(a, b, seen = new Map()) {
  if (Object.is(a, b) || (a === 0 && b === 0)) return true;
  if (typeof a !== "object" || typeof b !== "object" || a === null || b === null) return false;
  if (Object.getPrototypeOf(a) !== Object.getPrototypeOf(b)) return false;
  if (seen.get(a) === b) return true;
  seen.set(a, b);
  if (a instanceof Date) return a.getTime() === b.getTime();
  if (a instanceof RegExp) return a.source === b.source && a.flags === b.flags;
  if (a instanceof Set) { if (a.size !== b.size) return false; for (const x of a) if (!b.has(x)) return false; return true; }
  if (a instanceof Map) {
    if (a.size !== b.size) return false;
    for (const [k, v] of a) if (!b.has(k) || !deepEqual(v, b.get(k), seen)) return false;
    return true;
  }
  if (Array.isArray(a) !== Array.isArray(b)) return false;
  const ka = Object.keys(a), kb = Object.keys(b);
  if (ka.length !== kb.length) return false;
  for (const k of ka) if (!Object.prototype.hasOwnProperty.call(b, k) || !deepEqual(a[k], b[k], seen)) return false;
  return true;
}
'''

REF["js_pool"] = r'''
async function runPool(tasks, limit) {
  const out = new Array(tasks.length); let next = 0, failed = false;
  async function worker() {
    while (!failed && next < tasks.length) {
      const i = next++;
      try { out[i] = await tasks[i](); } catch (e) { failed = true; throw e; }
    }
  }
  await Promise.all(Array.from({length: Math.min(limit, tasks.length)}, worker));
  return out;
}
'''

REF["js_roman"] = r'''
const ROM = [[1000, "M"], [900, "CM"], [500, "D"], [400, "CD"], [100, "C"], [90, "XC"], [50, "L"], [40, "XL"], [10, "X"], [9, "IX"], [5, "V"], [4, "IV"], [1, "I"]];
function toRoman(n) {
  if (!Number.isInteger(n) || n < 1 || n > 3999) throw new RangeError("out of range");
  let s = "";
  for (const [v, r] of ROM) while (n >= v) { s += r; n -= v; }
  return s;
}
function fromRoman(s) {
  if (typeof s !== "string" || !/^M{0,3}(CM|CD|D?C{0,3})(XC|XL|L?X{0,3})(IX|IV|V?I{0,3})$/.test(s) || s === "") throw new SyntaxError("bad numeral");
  let n = 0, i = 0;
  for (const [v, r] of ROM) while (s.startsWith(r, i)) { n += v; i += r.length; }
  return n;
}
'''

REF["js_emitter"] = r'''
class Emitter {
  constructor() { this.l = new Map(); }
  on(ev, fn) { if (!this.l.has(ev)) this.l.set(ev, []); const rec = {fn}; this.l.get(ev).push(rec); return () => this._remove(ev, rec); }
  once(ev, fn) { const rec = {fn, once: true}; if (!this.l.has(ev)) this.l.set(ev, []); this.l.get(ev).push(rec); return () => this._remove(ev, rec); }
  _remove(ev, rec) { const a = this.l.get(ev); if (!a) return; const i = a.indexOf(rec); if (i >= 0) a.splice(i, 1); }
  off(ev, fn) { const a = this.l.get(ev); if (!a) return; for (let i = a.length - 1; i >= 0; i--) if (a[i].fn === fn) { a.splice(i, 1); return; } }
  emit(ev, ...args) {
    const a = (this.l.get(ev) || []).slice(); let err = null, n = 0;
    for (const rec of a) {
      if (rec.once) this._remove(ev, rec);
      n++;
      try { rec.fn(...args); } catch (e) { if (!err) err = e; }
    }
    if (err) throw err;
    return n;
  }
}
'''

REF["js_flatten"] = r'''
const isPlain = (v) => v !== null && typeof v === "object" && (Array.isArray(v) || Object.getPrototypeOf(v) === Object.prototype);
function flatten(obj, prefix = "", out = {}) {
  for (const [k, v] of Object.entries(obj)) {
    const key = prefix ? prefix + "." + k : k;
    if (isPlain(v) && Object.keys(v).length) flatten(v, key, out); else out[key] = v;
  }
  return out;
}
function unflatten(flat) {
  const root = {};
  for (const [path, v] of Object.entries(flat)) {
    const parts = path.split("."); let cur = root;
    for (let i = 0; i < parts.length - 1; i++) {
      const p = parts[i];
      if (cur[p] === undefined) cur[p] = /^\d+$/.test(parts[i + 1]) ? [] : {};
      cur = cur[p];
    }
    cur[parts[parts.length - 1]] = v;
  }
  return root;
}
'''

REF["js_retry"] = r'''
async function retry(fn, {retries = 3, baseMs = 100, factor = 2, shouldRetry = () => true, sleep = (ms) => new Promise((r) => setTimeout(r, ms))} = {}) {
  for (let attempt = 0; ; attempt++) {
    try { return await fn(attempt); } catch (e) {
      if (attempt >= retries || !shouldRetry(e)) throw e;
      await sleep(baseMs * Math.pow(factor, attempt));
    }
  }
}
'''

REF["js_template"] = r'''
function render(template, data) {
  const get = (p) => p.split(".").reduce((o, k) => (o == null ? undefined : o[k]), data);
  const esc = (s) => s.replace(/[&<>"']/g, (c) => ({"&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;"})[c]);
  const str = (v) => (v === undefined || v === null ? "" : String(v));
  return template.replace(/\\\{\{|\{\{\{\s*([\w.]+)\s*\}\}\}|\{\{\s*([\w.]+)\s*\}\}/g, (m, raw, path) => {
    if (m === "\\{{") return "{{";
    return raw !== undefined ? str(get(raw)) : esc(str(get(path)));
  });
}
'''

REF["js_interval_merge"] = r'''
function mergeIntervals(list) {
  for (const p of list) if (!Array.isArray(p) || p.length !== 2 || typeof p[0] !== "number" || typeof p[1] !== "number" || p[0] > p[1] || Number.isNaN(p[0]) || Number.isNaN(p[1])) throw new TypeError("bad interval");
  const s = list.map((p) => [p[0], p[1]]).sort((a, b) => a[0] - b[0]);
  const out = [];
  for (const p of s) { if (out.length && p[0] <= out[out.length - 1][1]) out[out.length - 1][1] = Math.max(out[out.length - 1][1], p[1]); else out.push(p); }
  return out;
}
'''

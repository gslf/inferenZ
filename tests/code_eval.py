#!/usr/bin/env python3
"""Coding quality harness: C and JavaScript tasks with hidden tests, run against an inferenZ server.

Each task asks the model for one function (or class) with an exact signature. The answer's last fenced code
block is extracted and checked by hidden tests that probe the edge cases a careful programmer handles:
  C:  gcc -std=gnu23 -O1 -g -Wall -Wextra -Werror -fsanitize=address,undefined  (warnings, leaks, UB all fail)
  JS: node, the solution followed by assert-based tests (uncaught errors and rejected promises fail)
Programs run inside bubblewrap (no network, read-only system, private /tmp) with a time limit.

    python3 tests/code_eval.py [--url http://127.0.0.1:8080] [--only c|js] [--task ID[,ID...]] [--effort high]
                               [--samples N] [--save DIR]

Prints one line per task and a pass rate; --save keeps prompts, answers and test logs for inspection.
"""
import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.request

SYSTEM_C = ("Answer with a short explanation and then exactly one ```c code block containing only the requested "
            "function(s), the #include lines they need and any static helpers. Do not write main().")
SYSTEM_JS = ("Answer with a short explanation and then exactly one ```javascript code block containing only the "
             "requested function(s) or class. Do not export anything and do not write tests.")

C_TASKS = [
    {
        "id": "c_reverse_words",
        "prompt": "Scrivi in C la funzione `char *reverse_words(const char *s);` che restituisce una nuova stringa "
                  "(allocata con malloc, la libera il chiamante) con le parole di `s` in ordine inverso. Le parole "
                  "sono separate da uno o più spazi; nel risultato sono separate da un solo spazio, senza spazi "
                  "iniziali o finali. Se `s` è NULL restituisci NULL.",
        "test": r'''
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
char *reverse_words(const char *s);
static int fails;
static void t(const char *in, const char *want) {
    char *got = reverse_words(in);
    if (!got || strcmp(got, want)) { printf("reverse_words(\"%s\") = \"%s\", want \"%s\"\n", in, got ? got : "(null)", want); fails++; }
    free(got);
}
int main(void) {
    t("hello world", "world hello");
    t("  uno   due tre  ", "tre due uno");
    t("solo", "solo");
    t("", "");
    t("     ", "");
    t("a b c d e f", "f e d c b a");
    if (reverse_words(NULL) != NULL) { puts("NULL input must give NULL"); fails++; }
    return fails != 0;
}''',
    },
    {
        "id": "c_parse_long",
        "prompt": "Write the C function `int parse_long(const char *s, long *out);`. It parses a base-10 integer "
                  "with an optional leading '+' or '-'. The whole string must be consumed: no leading or trailing "
                  "spaces, no empty string, no digits missing after the sign. It must detect overflow without "
                  "relying on undefined behavior. On success it stores the value in *out and returns 0; on any "
                  "error it returns -1 and leaves *out unchanged. Do not use strtol.",
        "test": r'''
#include <limits.h>
#include <stdio.h>
int parse_long(const char *s, long *out);
static int fails;
static void ok(const char *s, long want) {
    long v = 12345;
    if (parse_long(s, &v) != 0 || v != want) { printf("parse_long(\"%s\") -> %ld, want %ld\n", s, v, want); fails++; }
}
static void bad(const char *s) {
    long v = 777;
    if (parse_long(s, &v) != -1 || v != 777) { printf("parse_long(\"%s\") must fail and keep *out\n", s); fails++; }
}
int main(void) {
    char buf[64];
    ok("0", 0); ok("42", 42); ok("-42", -42); ok("+7", 7); ok("007", 7);
    snprintf(buf, sizeof buf, "%ld", LONG_MAX); ok(buf, LONG_MAX);
    snprintf(buf, sizeof buf, "%ld", LONG_MIN); ok(buf, LONG_MIN);
    bad(""); bad("-"); bad("+"); bad(" 1"); bad("1 "); bad("12a"); bad("--1"); bad("0x10");
    bad("9223372036854775808"); bad("-9223372036854775809"); bad("99999999999999999999999");
    return fails != 0;
}''',
    },
    {
        "id": "c_intvec",
        "prompt": "Implementa in C un vettore dinamico di interi con questa interfaccia:\n"
                  "```c\ntypedef struct { int *data; size_t len, cap; } intvec;\n"
                  "void iv_init(intvec *v);            // vettore vuoto, nessuna allocazione\n"
                  "int iv_push(intvec *v, int x);      // 0 se ok, -1 se manca memoria (il vettore resta valido)\n"
                  "int iv_pop(intvec *v, int *out);    // 0 e l'ultimo elemento in *out, -1 se vuoto\n"
                  "int iv_insert(intvec *v, size_t i, int x); // inserisce in posizione i (0..len), -1 se i > len\n"
                  "void iv_free(intvec *v);            // libera e torna vuoto\n```\n"
                  "La capacità deve crescere in modo geometrico. Includi la definizione del tipo nel codice.",
        "test": r'''
#include <stdio.h>
#include <stdlib.h>
typedef struct { int *data; size_t len, cap; } intvec;
void iv_init(intvec *v); int iv_push(intvec *v, int x); int iv_pop(intvec *v, int *out);
int iv_insert(intvec *v, size_t i, int x); void iv_free(intvec *v);
int main(void) {
    intvec v; iv_init(&v);
    int x, fails = 0;
    if (v.len != 0 || iv_pop(&v, &x) != -1) { puts("empty pop must fail"); fails++; }
    for (int i = 0; i < 100000; i++) if (iv_push(&v, i)) { puts("push failed"); return 1; }
    if (v.len != 100000 || v.cap < v.len) { puts("bad len/cap"); fails++; }
    if (iv_insert(&v, 0, -1) || iv_insert(&v, v.len, -2) || iv_insert(&v, 50000, -3)) { puts("insert failed"); fails++; }
    if (iv_insert(&v, v.len + 1, 9) != -1) { puts("insert past end must fail"); fails++; }
    if (v.data[0] != -1 || v.data[50000] != -3 || v.data[50001] != 49999 || v.data[v.len - 1] != -2) { puts("insert order wrong"); fails++; }
    if (iv_pop(&v, &x) || x != -2) { puts("pop wrong"); fails++; }
    iv_free(&v);
    if (v.len != 0 || v.data != NULL) { puts("free must reset"); fails++; }
    if (iv_push(&v, 5) || v.data[0] != 5) { puts("reuse after free failed"); fails++; }
    iv_free(&v);
    return fails != 0;
}''',
    },
    {
        "id": "c_utf8_len",
        "prompt": "Write `long utf8_count(const char *s, size_t n);` in C: it returns the number of Unicode code "
                  "points in the first n bytes of s, or -1 if those bytes are not valid UTF-8. Reject overlong "
                  "encodings, surrogates (U+D800..U+DFFF), code points above U+10FFFF and truncated sequences. "
                  "s may contain NUL bytes, which count as code points.",
        "test": r'''
#include <stdio.h>
#include <string.h>
long utf8_count(const char *s, size_t n);
static int fails;
static void t(const char *s, size_t n, long want) {
    long got = utf8_count(s, n);
    if (got != want) { printf("case len %zu: got %ld want %ld\n", n, got, want); fails++; }
}
int main(void) {
    t("", 0, 0); t("abc", 3, 3); t("caf\xc3\xa9", 5, 4); t("\xe2\x82\xac", 3, 1); t("\xf0\x9f\x98\x80!", 5, 2);
    t("a\0b", 3, 3);
    t("\xc3", 1, -1);              /* truncated */
    t("\xc0\xaf", 2, -1);          /* overlong '/' */
    t("\xe0\x80\xaf", 3, -1);      /* overlong */
    t("\xed\xa0\x80", 3, -1);      /* surrogate */
    t("\xf4\x90\x80\x80", 4, -1);  /* > U+10FFFF */
    t("\xf8\x88\x80\x80\x80", 5, -1);
    t("\x80", 1, -1);              /* lone continuation */
    t("\xef\xbf\xbf", 3, 1);       /* U+FFFF is valid */
    t("\xf4\x8f\xbf\xbf", 4, 1);   /* U+10FFFF */
    return fails != 0;
}''',
    },
    {
        "id": "c_balanced",
        "prompt": "Scrivi in C `int balanced(const char *s);` che restituisce 1 se le parentesi (), [] e {} in s "
                  "sono bilanciate e correttamente annidate, 0 altrimenti. Gli altri caratteri vanno ignorati, tranne "
                  "che all'interno di stringhe tra doppi apici, dove le parentesi non contano; dentro una stringa \\\" "
                  "è un apice escaped. Una stringa non chiusa rende il testo non bilanciato. La profondità di "
                  "annidamento può essere arbitraria (non usare uno stack di dimensione fissa).",
        "test": r'''
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
int balanced(const char *s);
static int fails;
static void t(const char *s, int want) { if (balanced(s) != want) { printf("balanced(%s) != %d\n", s, want); fails++; } }
int main(void) {
    t("", 1); t("()", 1); t("([]{})", 1); t("f(a[1], {b: 2})", 1); t("(", 0); t(")(", 0); t("([)]", 0);
    t("\"(\"", 1); t("(\"a)\")", 1); t("\"\\\"(\"", 1); t("\"abc", 0); t("{\"}\"}", 1); t("}", 0);
    size_t n = 200000; char *deep = malloc(2 * n + 1);
    for (size_t i = 0; i < n; i++) { deep[i] = '['; deep[2 * n - 1 - i] = ']'; }
    deep[2 * n] = 0; t(deep, 1); deep[2 * n - 1] = ')'; t(deep, 0); free(deep);
    return fails != 0;
}''',
    },
    {
        "id": "c_base64",
        "prompt": "Write two C functions: `size_t b64_encode(const unsigned char *in, size_t n, char *out);` writes "
                  "the standard base64 (with '=' padding) of in[0..n) into out followed by a NUL and returns the "
                  "length without the NUL; `long b64_decode(const char *in, unsigned char *out);` decodes a "
                  "NUL-terminated base64 string and returns the number of bytes written, or -1 if the input is "
                  "invalid (bad characters, wrong length, misplaced padding). The caller guarantees out is large "
                  "enough.",
        "test": r'''
#include <stdio.h>
#include <string.h>
size_t b64_encode(const unsigned char *in, size_t n, char *out);
long b64_decode(const char *in, unsigned char *out);
static int fails;
static void rt(const char *plain, const char *enc) {
    char e[256]; unsigned char d[256];
    size_t n = strlen(plain), m = b64_encode((const unsigned char *)plain, n, e);
    if (m != strlen(enc) || strcmp(e, enc)) { printf("encode(%s) = %s want %s\n", plain, e, enc); fails++; }
    long k = b64_decode(enc, d);
    if (k != (long)n || memcmp(d, plain, n)) { printf("decode(%s) wrong (%ld)\n", enc, k); fails++; }
}
int main(void) {
    rt("", ""); rt("f", "Zg=="); rt("fo", "Zm8="); rt("foo", "Zm9v"); rt("foob", "Zm9vYg=="); rt("fooba", "Zm9vYmE=");
    rt("foobar", "Zm9vYmFy");
    unsigned char all[256], d[256]; char e[400];
    for (int i = 0; i < 256; i++) all[i] = (unsigned char)i;
    b64_encode(all, 256, e);
    if (b64_decode(e, d) != 256 || memcmp(all, d, 256)) { puts("binary round trip failed"); fails++; }
    const char *bad[] = {"Zg=", "Z===", "Zm9v!", "Zg==Zg==", "=Zg=", "Zm9", "Zm=v"};
    for (int i = 0; i < 7; i++) if (b64_decode(bad[i], d) != -1) { printf("decode(%s) must fail\n", bad[i]); fails++; }
    return fails != 0;
}''',
    },
    {
        "id": "c_hashmap",
        "prompt": "Implementa in C una hash map da stringhe a int:\n```c\ntypedef struct strmap strmap;\n"
                  "strmap *sm_new(void);\nvoid sm_free(strmap *m);\n"
                  "int sm_set(strmap *m, const char *key, int value); // copia la chiave; 0 ok, -1 memoria\n"
                  "int sm_get(const strmap *m, const char *key, int *value); // 1 trovata, 0 no\n"
                  "int sm_del(strmap *m, const char *key); // 1 rimossa, 0 assente\n"
                  "size_t sm_len(const strmap *m);\n```\n"
                  "Deve ridimensionarsi quando si riempie e restare corretta dopo molte cancellazioni e reinserimenti.",
        "test": r'''
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
typedef struct strmap strmap;
strmap *sm_new(void); void sm_free(strmap *m); int sm_set(strmap *m, const char *key, int value);
int sm_get(const strmap *m, const char *key, int *value); int sm_del(strmap *m, const char *key); size_t sm_len(const strmap *m);
int main(void) {
    strmap *m = sm_new(); char k[32]; int v, fails = 0;
    for (int i = 0; i < 50000; i++) { snprintf(k, sizeof k, "key%d", i); if (sm_set(m, k, i)) return 1; }
    if (sm_len(m) != 50000) { puts("len"); fails++; }
    for (int i = 0; i < 50000; i += 2) { snprintf(k, sizeof k, "key%d", i); if (sm_del(m, k) != 1) { fails++; break; } }
    snprintf(k, sizeof k, "key%d", 0); if (sm_del(m, k) != 0 || sm_get(m, k, &v)) { puts("deleted key still there"); fails++; }
    for (int i = 1; i < 50000; i += 2) { snprintf(k, sizeof k, "key%d", i); if (!sm_get(m, k, &v) || v != i) { printf("lost %s\n", k); fails++; break; } }
    for (int r = 0; r < 20; r++) for (int i = 0; i < 1000; i++) { snprintf(k, sizeof k, "tmp%d", i); sm_set(m, k, r); sm_del(m, k); }
    if (sm_len(m) != 25000) { printf("len after churn %zu\n", sm_len(m)); fails++; }
    char buf[8] = "dup"; sm_set(m, buf, 1); buf[0] = 'X'; if (!sm_get(m, "dup", &v) || v != 1) { puts("key not copied"); fails++; }
    sm_set(m, "dup", 2); if (!sm_get(m, "dup", &v) || v != 2 || sm_len(m) != 25001) { puts("overwrite"); fails++; }
    sm_set(m, "", 9); if (!sm_get(m, "", &v) || v != 9) { puts("empty key"); fails++; }
    sm_free(m);
    return fails != 0;
}''',
    },
    {
        "id": "c_ring",
        "prompt": "Write a fixed-capacity ring buffer of bytes in C:\n```c\ntypedef struct { unsigned char *buf; size_t cap, head, len; } ring;\n"
                  "int ring_init(ring *r, size_t cap);  // 0 ok, -1 if cap == 0 or no memory\n"
                  "size_t ring_write(ring *r, const void *src, size_t n); // writes as much as fits, returns bytes written\n"
                  "size_t ring_read(ring *r, void *dst, size_t n);        // reads up to n bytes, returns bytes read\n"
                  "void ring_free(ring *r);\n```\nInclude the struct definition. Use memcpy for the (at most two) contiguous pieces.",
        "test": r'''
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
typedef struct { unsigned char *buf; size_t cap, head, len; } ring;
int ring_init(ring *r, size_t cap); size_t ring_write(ring *r, const void *src, size_t n);
size_t ring_read(ring *r, void *dst, size_t n); void ring_free(ring *r);
int main(void) {
    ring r; int fails = 0;
    if (ring_init(&r, 0) != -1) { puts("cap 0 must fail"); fails++; }
    if (ring_init(&r, 7)) return 1;
    unsigned char out[64], in[64]; unsigned seq = 0, rseq = 0;
    srand(1);
    for (int it = 0; it < 100000; it++) {
        size_t n = (size_t)(rand() % 10);
        for (size_t i = 0; i < n; i++) in[i] = (unsigned char)(seq + i);
        size_t w = ring_write(&r, in, n);
        size_t room = 7 - (size_t)(seq - rseq);
        if (w != (n < room ? n : room)) { printf("write %zu got %zu\n", n, w); fails++; break; }
        seq += (unsigned)w;
        size_t m = (size_t)(rand() % 10), got = ring_read(&r, out, m);
        size_t have = (size_t)(seq - rseq);
        if (got != (m < have ? m : have)) { printf("read %zu got %zu\n", m, got); fails++; break; }
        for (size_t i = 0; i < got; i++) if (out[i] != (unsigned char)(rseq + i)) { puts("data mismatch"); fails++; it = 1 << 30; break; }
        rseq += (unsigned)got;
    }
    ring_free(&r);
    return fails != 0;
}''',
    },
    {
        "id": "c_merge_lists",
        "prompt": "In C, given `struct node { int val; struct node *next; };`, write "
                  "`struct node *merge_sorted(struct node *a, struct node *b);` that merges two ascending lists into "
                  "one ascending list by relinking the existing nodes (no allocation). It must be stable: on equal "
                  "values, nodes of `a` come first. It must work for lists of a million nodes (no recursion). "
                  "Include the struct definition.",
        "test": r'''
#include <stdio.h>
#include <stdlib.h>
struct node { int val; struct node *next; };
struct node *merge_sorted(struct node *a, struct node *b);
static struct node *mk(int n, int start, int step, struct node *pool) {
    for (int i = 0; i < n; i++) { pool[i].val = start + i * step; pool[i].next = i + 1 < n ? &pool[i + 1] : NULL; }
    return n ? pool : NULL;
}
int main(void) {
    int fails = 0, N = 1000000;
    struct node *pa = malloc(sizeof *pa * N), *pb = malloc(sizeof *pb * N);
    struct node *m = merge_sorted(mk(N, 0, 2, pa), mk(N, 0, 2, pb));
    long cnt = 0; struct node *prev = NULL;
    for (struct node *p = m; p; prev = p, p = p->next, cnt++) {
        if (prev && prev->val > p->val) { puts("not sorted"); fails++; break; }
        if (prev && prev->val == p->val && !(prev >= pa && prev < pa + N && p >= pb && p < pb + N)) { puts("not stable"); fails++; break; }
    }
    if (cnt != 2L * N) { printf("count %ld\n", cnt); fails++; }
    if (merge_sorted(NULL, NULL) != NULL) { puts("null,null"); fails++; }
    struct node one = {5, NULL};
    if (merge_sorted(&one, NULL) != &one || merge_sorted(NULL, &one) != &one) { puts("single"); fails++; }
    free(pa); free(pb);
    return fails != 0;
}''',
    },
    {
        "id": "c_csv_field",
        "prompt": "Scrivi in C `int csv_split(const char *line, char ***fields, size_t *n);` che divide una riga "
                  "CSV (RFC 4180) in campi: i campi sono separati da virgole, un campo può essere tra doppi apici e "
                  "allora può contenere virgole e apici raddoppiati (\"\" diventa \"). Alloca con malloc un array di "
                  "stringhe (ognuna allocata) e restituisce 0, oppure -1 se la riga è malformata (apice non chiuso o "
                  "caratteri dopo l'apice di chiusura prima della virgola), senza perdite di memoria. Una riga vuota "
                  "ha un campo vuoto. Scrivi anche `void csv_free(char **fields, size_t n);`.",
        "test": r'''
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
int csv_split(const char *line, char ***fields, size_t *n); void csv_free(char **fields, size_t n);
static int fails;
static void t(const char *line, int nwant, const char **want) {
    char **f = NULL; size_t n = 0;
    int rc = csv_split(line, &f, &n);
    if (nwant < 0) { if (rc != -1) { printf("csv_split(%s) must fail\n", line); fails++; if (rc == 0) csv_free(f, n); } return; }
    if (rc != 0 || n != (size_t)nwant) { printf("csv_split(%s): rc %d n %zu\n", line, rc, n); fails++; if (rc == 0) csv_free(f, n); return; }
    for (size_t i = 0; i < n; i++) if (strcmp(f[i], want[i])) { printf("field %zu of %s: [%s] want [%s]\n", i, line, f[i], want[i]); fails++; }
    csv_free(f, n);
}
int main(void) {
    t("a,b,c", 3, (const char *[]){"a", "b", "c"});
    t("", 1, (const char *[]){""});
    t(",", 2, (const char *[]){"", ""});
    t("\"x,y\",z", 2, (const char *[]){"x,y", "z"});
    t("\"he said \"\"hi\"\"\",2", 2, (const char *[]){"he said \"hi\"", "2"});
    t("\"\",\"\"", 2, (const char *[]){"", ""});
    t("a,,b,", 4, (const char *[]){"a", "", "b", ""});
    t("\"unterminated", -1, NULL);
    t("\"ab\"c,d", -1, NULL);
    return fails != 0;
}''',
    },
]

JS_TASKS = [
    {
        "id": "js_lru",
        "prompt": "Write a JavaScript class `LRUCache` with `constructor(capacity)`, `get(key)` (returns the value or "
                  "undefined and marks the key as most recently used), `set(key, value)` (inserts or updates, "
                  "evicting the least recently used entry when over capacity) and a `size` getter. Keys can be any "
                  "value (objects included). All operations must be O(1).",
        "test": r'''
const assert = require("assert");
const c = new LRUCache(2);
c.set("a", 1); c.set("b", 2); assert.strictEqual(c.get("a"), 1);
c.set("c", 3); assert.strictEqual(c.get("b"), undefined); assert.strictEqual(c.get("a"), 1); assert.strictEqual(c.get("c"), 3);
c.set("a", 10); c.set("d", 4); assert.strictEqual(c.get("c"), undefined); assert.strictEqual(c.get("a"), 10); assert.strictEqual(c.size, 2);
const k1 = {}, k2 = {}; const o = new LRUCache(1); o.set(k1, "x"); assert.strictEqual(o.get(k2), undefined); assert.strictEqual(o.get(k1), "x");
o.set(k2, undefined); assert.strictEqual(o.size, 1); assert.strictEqual(o.get(k1), undefined);
const big = new LRUCache(1000); const t0 = Date.now();
for (let i = 0; i < 300000; i++) { big.set(i, i); big.get(i - 500); }
assert.ok(Date.now() - t0 < 3000, "too slow"); assert.strictEqual(big.size, 1000); assert.strictEqual(big.get(299999), 299999);
''',
    },
    {
        "id": "js_csv",
        "prompt": "Scrivi in JavaScript `function parseCSV(text)` che restituisce un array di righe, ognuna un array "
                  "di stringhe, secondo RFC 4180: separatore virgola, campi tra doppi apici che possono contenere "
                  "virgole, a capo e apici raddoppiati; le righe terminano con \\n o \\r\\n; un a capo finale non "
                  "crea una riga vuota. Lancia un `SyntaxError` se un apice non è chiuso.",
        "test": r'''
const assert = require("assert");
assert.deepStrictEqual(parseCSV("a,b\nc,d"), [["a", "b"], ["c", "d"]]);
assert.deepStrictEqual(parseCSV("a,b\r\nc,d\r\n"), [["a", "b"], ["c", "d"]]);
assert.deepStrictEqual(parseCSV('"x,y","line1\nline2"\n'), [["x,y", "line1\nline2"]]);
assert.deepStrictEqual(parseCSV('"say ""hi"""'), [['say "hi"']]);
assert.deepStrictEqual(parseCSV("a,,\n,"), [["a", "", ""], ["", ""]]);
assert.deepStrictEqual(parseCSV(""), []);
assert.throws(() => parseCSV('"open'), SyntaxError);
''',
    },
    {
        "id": "js_deep_equal",
        "prompt": "Write `function deepEqual(a, b)` in JavaScript: structural equality for primitives, arrays, plain "
                  "objects, Map, Set (of primitives), Date and RegExp. NaN equals NaN, +0 and -0 are equal, objects "
                  "with different prototypes are not equal, arrays are not equal to objects with the same keys, and "
                  "cyclic structures must not overflow the stack.",
        "test": r'''
const assert = require("assert");
assert.ok(deepEqual(1, 1)); assert.ok(!deepEqual(1, "1")); assert.ok(deepEqual(NaN, NaN)); assert.ok(deepEqual(0, -0));
assert.ok(deepEqual({a: [1, {b: 2}]}, {a: [1, {b: 2}]})); assert.ok(!deepEqual({a: 1}, {a: 1, b: undefined}));
assert.ok(!deepEqual([1, 2], {0: 1, 1: 2})); assert.ok(!deepEqual([], {}));
assert.ok(deepEqual(new Map([[1, {x: 1}]]), new Map([[1, {x: 1}]]))); assert.ok(!deepEqual(new Map([[1, 2]]), new Map([[1, 3]])));
assert.ok(deepEqual(new Set([1, 2]), new Set([2, 1]))); assert.ok(!deepEqual(new Set([1]), new Set([2])));
assert.ok(deepEqual(new Date(5), new Date(5))); assert.ok(!deepEqual(new Date(5), new Date(6)));
assert.ok(deepEqual(/a/g, /a/g)); assert.ok(!deepEqual(/a/g, /a/i));
class P { constructor() { this.x = 1; } } assert.ok(!deepEqual(new P(), {x: 1}));
const c1 = {n: 1}; c1.self = c1; const c2 = {n: 1}; c2.self = c2; assert.ok(deepEqual(c1, c2));
assert.ok(!deepEqual(null, {})); assert.ok(deepEqual(null, null)); assert.ok(!deepEqual(undefined, null));
''',
    },
    {
        "id": "js_pool",
        "prompt": "Scrivi in JavaScript `async function runPool(tasks, limit)`: `tasks` è un array di funzioni che "
                  "restituiscono promise; esegue al massimo `limit` task contemporaneamente e restituisce un array "
                  "con i risultati nello stesso ordine di `tasks`. Se un task fallisce, `runPool` rifiuta con quel "
                  "primo errore e non avvia altri task. Con un array vuoto restituisce [].",
        "test": r'''
const assert = require("assert");
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
(async () => {
  let active = 0, peak = 0;
  const mk = (i, ms) => async () => { active++; peak = Math.max(peak, active); await sleep(ms); active--; return i * 10; };
  const res = await runPool([mk(0, 30), mk(1, 5), mk(2, 20), mk(3, 1), mk(4, 10), mk(5, 2)], 2);
  assert.deepStrictEqual(res, [0, 10, 20, 30, 40, 50]); assert.strictEqual(peak, 2);
  assert.deepStrictEqual(await runPool([], 3), []);
  let started = 0;
  const failing = [async () => { started++; await sleep(5); throw new Error("boom"); },
                   async () => { started++; await sleep(50); return 1; },
                   async () => { started++; return 2; }, async () => { started++; return 3; }];
  await assert.rejects(runPool(failing, 2), /boom/);
  await sleep(80);
  assert.ok(started <= 3, "started tasks after the failure: " + started);
  const r3 = await runPool([() => Promise.resolve("a")], 10); assert.deepStrictEqual(r3, ["a"]);
  console.log("ok");
})().catch((e) => { console.error(e); process.exit(1); });
''',
    },
    {
        "id": "js_roman",
        "prompt": "Write two JavaScript functions: `toRoman(n)` for integers 1..3999 (throw a RangeError otherwise, "
                  "including for non-integers) and `fromRoman(s)` that accepts only canonical numerals (e.g. 'IIII', "
                  "'IC', 'MMMM' and '' are invalid and must throw a SyntaxError).",
        "test": r'''
const assert = require("assert");
const cases = [[1, "I"], [4, "IV"], [9, "IX"], [14, "XIV"], [40, "XL"], [90, "XC"], [400, "CD"], [1994, "MCMXCIV"], [3999, "MMMCMXCIX"], [2024, "MMXXIV"]];
for (const [n, s] of cases) { assert.strictEqual(toRoman(n), s); assert.strictEqual(fromRoman(s), n); }
for (let n = 1; n <= 3999; n++) assert.strictEqual(fromRoman(toRoman(n)), n);
for (const bad of [0, 4000, -1, 1.5, NaN]) assert.throws(() => toRoman(bad), RangeError);
for (const bad of ["IIII", "IC", "MMMM", "", "VX", "IIV", "abc", "XM", "CCCC"]) assert.throws(() => fromRoman(bad), SyntaxError);
''',
    },
    {
        "id": "js_emitter",
        "prompt": "Implementa in JavaScript una classe `Emitter` con `on(evento, fn)` che restituisce una funzione per "
                  "rimuovere il listener, `once(evento, fn)`, `off(evento, fn)` (rimuove una sola registrazione di fn, la più recente) ed `emit(evento, ...args)` che "
                  "restituisce il numero di listener chiamati. Lo stesso fn registrato due volte viene chiamato due "
                  "volte. Rimuovere o aggiungere listener durante un emit non deve influenzare quell'emit. Un errore "
                  "lanciato da un listener non deve impedire la chiamata degli altri: alla fine emit rilancia il "
                  "primo errore.",
        "test": r'''
const assert = require("assert");
const e = new Emitter(); const log = [];
const f = (x) => log.push("f" + x); e.on("a", f); e.on("a", f); assert.strictEqual(e.emit("a", 1), 2); assert.deepStrictEqual(log, ["f1", "f1"]);
e.off("a", f); assert.strictEqual(e.emit("a", 2), 1);
const un = e.on("b", () => log.push("b")); un(); assert.strictEqual(e.emit("b"), 0);
e.once("c", (x, y) => log.push(x + y)); assert.strictEqual(e.emit("c", 2, 3), 1); assert.strictEqual(e.emit("c", 2, 3), 0); assert.strictEqual(log.at(-1), 5);
const g = new Emitter(); const seen = [];
const h2 = () => seen.push(2); g.on("x", () => { seen.push(1); g.off("x", h2); g.on("x", () => seen.push(3)); }); g.on("x", h2);
assert.strictEqual(g.emit("x"), 2); assert.deepStrictEqual(seen, [1, 2]);
const k = new Emitter(); let after = false; k.on("y", () => { throw new Error("first"); }); k.on("y", () => { after = true; });
assert.throws(() => k.emit("y"), /first/); assert.ok(after);
assert.strictEqual(new Emitter().emit("none"), 0);
''',
    },
    {
        "id": "js_flatten",
        "prompt": "Write `function flatten(obj)` in JavaScript that turns nested plain objects and arrays into a flat "
                  "object with dotted keys: {a: {b: 1}, c: [2, {d: 3}]} becomes {'a.b': 1, 'c.0': 2, 'c.1.d': 3}. "
                  "Empty objects and empty arrays are kept as values. Dates, null and other non-plain values are leaf "
                  "values. Also write `function unflatten(flat)` that inverts it, rebuilding arrays for numeric "
                  "segments.",
        "test": r'''
const assert = require("assert");
const o = {a: {b: 1}, c: [2, {d: 3}], e: {}, f: [], g: null, h: new Date(0), i: "s"};
const f = flatten(o);
assert.deepStrictEqual(Object.keys(f).sort(), ["a.b", "c.0", "c.1.d", "e", "f", "g", "h", "i"]);
assert.strictEqual(f["c.1.d"], 3); assert.deepStrictEqual(f.e, {}); assert.deepStrictEqual(f.f, []); assert.strictEqual(f.g, null); assert.ok(f.h instanceof Date);
assert.deepStrictEqual(unflatten(f), o);
assert.deepStrictEqual(flatten({}), {}); assert.deepStrictEqual(unflatten({}), {});
assert.ok(Array.isArray(unflatten({"x.0": 1, "x.1": 2}).x));
''',
    },
    {
        "id": "js_retry",
        "prompt": "Scrivi in JavaScript `async function retry(fn, {retries = 3, baseMs = 100, factor = 2, "
                  "shouldRetry = () => true, sleep = (ms) => new Promise(r => setTimeout(r, ms))} = {})`: chiama "
                  "`fn(tentativo)` (tentativo parte da 0) e se rifiuta riprova fino a `retries` volte, attendendo "
                  "baseMs * factor^k prima del k-esimo nuovo tentativo (k da 0), usando la funzione `sleep` passata. "
                  "Se `shouldRetry(err)` restituisce false rilancia subito. Dopo l'ultimo fallimento rilancia l'ultimo "
                  "errore.",
        "test": r'''
const assert = require("assert");
(async () => {
  const waits = []; const sleep = async (ms) => { waits.push(ms); };
  let n = 0; const r = await retry(async (a) => { assert.strictEqual(a, n); n++; if (n < 3) throw new Error("e" + n); return "done"; }, {sleep});
  assert.strictEqual(r, "done"); assert.deepStrictEqual(waits, [100, 200]);
  waits.length = 0; let calls = 0;
  await assert.rejects(retry(async () => { calls++; throw new Error("x" + calls); }, {retries: 2, baseMs: 10, factor: 3, sleep}), /x3/);
  assert.strictEqual(calls, 3); assert.deepStrictEqual(waits, [10, 30]);
  waits.length = 0; calls = 0;
  await assert.rejects(retry(async () => { calls++; throw new TypeError("fatal"); }, {sleep, shouldRetry: (e) => !(e instanceof TypeError)}), TypeError);
  assert.strictEqual(calls, 1); assert.deepStrictEqual(waits, []);
  assert.strictEqual(await retry(async () => 7, {retries: 0, sleep}), 7);
  console.log("ok");
})().catch((e) => { console.error(e); process.exit(1); });
''',
    },
    {
        "id": "js_template",
        "prompt": "Write `function render(template, data)` in JavaScript: it replaces `{{ path }}` placeholders (spaces "
                  "optional, path with dots like user.name or items.0) with the value from data, HTML-escaping "
                  "& < > \" ' as &amp; &lt; &gt; &quot; &#39;. `{{{ path }}}` inserts the value without escaping. Missing values render as an "
                  "empty string, but 0 and false render as '0' and 'false'. `\\{{` is a literal '{{'.",
        "test": r'''
const assert = require("assert");
assert.strictEqual(render("Hi {{name}}!", {name: "Ann"}), "Hi Ann!");
assert.strictEqual(render("{{ user.name }} / {{items.1}}", {user: {name: "Bo"}, items: ["a", "b"]}), "Bo / b");
assert.strictEqual(render("{{x}}", {x: "<a href=\"q\">&'"}), "&lt;a href=&quot;q&quot;&gt;&amp;&#39;");
assert.strictEqual(render("{{{x}}}", {x: "<b>"}), "<b>");
assert.strictEqual(render("[{{missing}}][{{a.b.c}}]", {}), "[][]");
assert.strictEqual(render("{{z}} {{f}}", {z: 0, f: false}), "0 false");
assert.strictEqual(render("\\{{name}}", {name: "x"}), "{{name}}");
assert.strictEqual(render("no tags", {}), "no tags");
''',
    },
    {
        "id": "js_interval_merge",
        "prompt": "Scrivi in JavaScript `function mergeIntervals(list)` che riceve un array di coppie [inizio, fine] "
                  "(numeri, inizio <= fine, in ordine qualsiasi) e restituisce un nuovo array di intervalli disgiunti "
                  "ordinati, unendo quelli che si sovrappongono o si toccano ([1,3] e [3,5] diventano [1,5]). Non "
                  "deve modificare l'array in ingresso. Lancia un TypeError se una coppia non è valida.",
        "test": r'''
const assert = require("assert");
assert.deepStrictEqual(mergeIntervals([[1, 3], [2, 6], [8, 10], [15, 18]]), [[1, 6], [8, 10], [15, 18]]);
assert.deepStrictEqual(mergeIntervals([[3, 5], [1, 3]]), [[1, 5]]);
assert.deepStrictEqual(mergeIntervals([]), []);
assert.deepStrictEqual(mergeIntervals([[5, 5]]), [[5, 5]]);
assert.deepStrictEqual(mergeIntervals([[1, 10], [2, 3], [4, 5]]), [[1, 10]]);
assert.deepStrictEqual(mergeIntervals([[-5, -1], [-3, 2]]), [[-5, 2]]);
const input = [[4, 6], [1, 2]]; const copy = JSON.parse(JSON.stringify(input)); mergeIntervals(input); assert.deepStrictEqual(input, copy);
const out = mergeIntervals(input); out[0][0] = 99; assert.deepStrictEqual(input, copy);
assert.throws(() => mergeIntervals([[3, 1]]), TypeError); assert.throws(() => mergeIntervals([[1]]), TypeError); assert.throws(() => mergeIntervals([["a", 2]]), TypeError);
''',
    },
]


def chat(url, messages, effort, max_tokens, temperature):
    body = json.dumps({"model": "inferenz", "messages": messages, "reasoning_effort": effort, "max_tokens": max_tokens,
                       "temperature": temperature, "stream": False}).encode()
    req = urllib.request.Request(url + "/v1/chat/completions", data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=24 * 3600) as r:
        j = json.load(r)
    m = j["choices"][0]["message"]
    return m.get("content") or "", m.get("reasoning_content") or "", j.get("usage", {})


def last_block(text, lang):
    names = ("c",) if lang == "c" else ("javascript", "js")
    blocks = re.findall(r"```([a-zA-Z0-9_+-]*)[^\n]*\n(.*?)```", text, re.S)
    good = [b for tag, b in blocks if tag.lower() in names]
    return good[-1] if good else (blocks[-1][1] if blocks else None)


def sandbox(cmd, cwd, timeout):
    """Run cmd with bubblewrap: read-only system, the work dir writable, no network, private /tmp."""
    bw = ["bwrap", "--ro-bind", "/usr", "/usr", "--symlink", "usr/lib", "/lib", "--symlink", "usr/lib64", "/lib64",
          "--symlink", "usr/bin", "/bin", "--proc", "/proc", "--dev", "/dev", "--tmpfs", "/tmp",
          "--bind", cwd, "/work", "--chdir", "/work", "--unshare-all", "--die-with-parent", "--new-session"]
    node = shutil.which("node")
    if node and not node.startswith("/usr/"):
        root = os.path.dirname(os.path.dirname(os.path.realpath(node)))
        bw += ["--ro-bind", root, root]
    try:
        p = subprocess.run(bw + cmd, capture_output=True, text=True, timeout=timeout)
        return p.returncode, (p.stdout + p.stderr)[-4000:]
    except subprocess.TimeoutExpired:
        return -9, "timeout"


def check(task, lang, code):
    d = tempfile.mkdtemp(prefix="inz-eval-")
    try:
        if lang == "c":
            open(os.path.join(d, "sol.c"), "w").write(code)
            open(os.path.join(d, "test.c"), "w").write(task["test"])
            rc, log = sandbox(["gcc", "-std=gnu23", "-O1", "-g", "-Wall", "-Wextra", "-Werror", "-fsanitize=address,undefined",
                               "-fno-sanitize-recover=all", "-o", "t", "sol.c", "test.c"], d, 60)
            if rc != 0:
                return False, "compile: " + log
            rc, log = sandbox(["./t"], d, 30)
            return rc == 0, log
        open(os.path.join(d, "t.js"), "w").write('"use strict";\n' + code + "\n;\n" + task["test"])
        rc, log = sandbox([shutil.which("node") or "node", "t.js"], d, 30)
        return rc == 0, log
    finally:
        shutil.rmtree(d, ignore_errors=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8080")
    ap.add_argument("--only", choices=["c", "js"])
    ap.add_argument("--task")
    ap.add_argument("--effort", default="high")
    ap.add_argument("--samples", type=int, default=1)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--max-tokens", type=int, default=24000)
    ap.add_argument("--save")
    ap.add_argument("--selftest", action="store_true", help="check the harness itself with reference solutions")
    a = ap.parse_args()
    tasks = [("c", t) for t in C_TASKS] + [("js", t) for t in JS_TASKS]
    if a.selftest:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from code_eval_ref import REF
        bad = 0
        for lang, t in tasks:
            ok, log = check(t, lang, REF[t["id"]])
            bad += not ok
            print(f"{'ok  ' if ok else 'FAIL'} {t['id']}" + ("" if ok else "\n" + log))
        sys.exit(bad != 0)
    if a.only:
        tasks = [x for x in tasks if x[0] == a.only]
    if a.task:  # one id or several, comma-separated
        want = a.task.split(",")
        tasks = [x for x in tasks if x[1]["id"] in want]
    if a.save:
        os.makedirs(a.save, exist_ok=True)
    passed = total = 0
    for lang, t in tasks:
        for s in range(a.samples):
            t0 = time.time()
            msgs = [{"role": "system", "content": SYSTEM_C if lang == "c" else SYSTEM_JS}, {"role": "user", "content": t["prompt"]}]
            content, reasoning, usage = chat(a.url, msgs, a.effort, a.max_tokens, a.temperature)
            code = last_block(content, lang)
            ok, log = (False, "no code block") if code is None else check(t, lang, code)
            passed += ok
            total += 1
            print(f"{'PASS' if ok else 'FAIL'} {t['id']:<18} sample {s}  {time.time() - t0:6.0f}s  "
                  f"{usage.get('completion_tokens', 0)} tokens  {'' if ok else log.strip().splitlines()[0][:100] if log.strip() else ''}",
                  flush=True)
            if a.save:
                with open(os.path.join(a.save, f"{t['id']}.{s}.md"), "w") as f:
                    f.write(f"# {t['id']} ({'PASS' if ok else 'FAIL'})\n\n## Prompt\n\n{t['prompt']}\n\n## Reasoning\n\n{reasoning}\n\n"
                            f"## Answer\n\n{content}\n\n## Test log\n\n```\n{log}\n```\n")
    print(f"\npass rate {passed}/{total} = {100.0 * passed / max(total, 1):.0f}%")


if __name__ == "__main__":
    main()

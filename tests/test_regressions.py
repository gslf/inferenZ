#!/usr/bin/env python3
"""Model-free checks of sampling, head selection and streamed stop strings."""
from pathlib import Path
import os
import shlex
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[1]

SAMPLER = r'''
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include "sampler.h"
static float values[257];
static int compare(const void *a, const void *b) {
    int i = *(const int *)a, j = *(const int *)b;
    return values[i] < values[j] ? 1 : values[i] > values[j] ? -1 : (i > j) - (i < j);
}
int main(void) {
    for (int trial = 0; trial < 64; trial++) {
        int want[257], got[257];
        for (int i = 0; i < 257; i++) { values[i] = rand() % 17 - 8; want[i] = i; }
        qsort(want, 257, sizeof(int), compare);
        for (int k = 0; k <= 260; k++) {
            int count = topk_f32(values, 257, k, got);
            assert(count == (k <= 0 || k > 257 ? 257 : k));
            for (int i = 0; i < count; i++) assert(got[i] == want[i]);
        }
    }
    sampler_params p; sampler_default(&p);
    p.temperature = 1; p.top_p = 1; p.top_k = 0;
    sampler s; sampler_init(&s, &p, 10000);
    float *l = calloc(10000, sizeof(float));
    const int *ix; const float *pr;
    int n = sampler_dist(&s, l, NULL, 0, &ix, &pr);
    printf("support %d\n", n);
    s.p.top_k = 1; l[9999] = 0.001f;
    sampler_dist(&s, l, NULL, 0, &ix, &pr);
    printf("winner %d\n", ix[0]);
    s.p.temperature = 0; s.p.repeat_penalty = 2; l[0] = 4;
    int32_t h[] = {0, 0};
    sampler_dist(&s, l, h, 2, &ix, &pr);
    printf("repeat_logit %g\n", l[0]);
    sampler_free(&s); free(l);
    return 0;
}
'''

HEAD = r'''
#include <fcntl.h>
#include <assert.h>
#include "model_fwd.c"
int main(void) {
    inz_engine *e = calloc(1, sizeof(*e));
    float *l = calloc(Q_VOCAB, sizeof(float));
    float x[Q_H] = {0};
    e->tp = tp_create(1, NULL);
    e->head.type = INZ_T_F16;
    e->head_fd = open("/dev/zero", O_RDONLY);
    if (e->head_fd < 0) return 2;
    for (int i = 1; i < Q_VOCAB; i++) l[i] = -2;
    head_rescore(e, x, l);
    for (int i = 0; i < Q_VOCAB; i++) l[i] = 0;
    l[Q_VOCAB - 1] = 0.001f;
    head_rescore(e, x, l);
    assert(l[Q_VOCAB - 1] == 0);
    assert(e->head_cands <= 2 * HEAD_MAX_CAND);
    puts("returned");
    close(e->head_fd); tp_destroy(e->tp); free(l); free(e);
    return 0;
}
'''

STOP = r'''
#include "server.c"
void engine_diag_json(inz_engine *e, jw *w) { (void)e; (void)w; }
char *diag_prometheus(void) { return NULL; }
static char wire[128];
static size_t wire_n;
bool http_sse_data(http_conn *c, const char *data) {
    (void)c;
    char err[128];
    jval *v = json_parse(data, strlen(data), err, sizeof(err));
    if (!v) exit(2);
    const jval *choice = json_at(json_get(v, "choices"), 0);
    const jval *t = json_get(choice, "text");
    if (!t) t = json_get(json_get(choice, "delta"), "content");
    if (t && t->type == JSTR) {
        if (wire_n + t->len >= sizeof(wire)) exit(3);
        memcpy(wire + wire_n, t->str, t->len);
        wire_n += t->len; wire[wire_n] = 0;
    }
    json_free(v);
    return true;
}
bool http_conn_alive(http_conn *c) { (void)c; return true; }
int main(void) {
    const char *pieces[] = {"hello ST", "OP tail", "uff", "x END STOP tail"};
    uint32_t lens[] = {8, 7, 3, 15};
    tokenizer *t = tok_create(4, pieces, lens, 0, NULL, NULL, NULL,
                              0, NULL, NULL, NULL);
    for (int test = 0; test < 3; test++) {
        gen_ctx g = {0};
        g.tok = t; g.chat = test > 0; g.stream = test < 2;
        g.n_stops = 1; g.stops[0] = "STOP";
        wire_n = 0; wire[0] = 0;
        chat_parser_init(&g.cp, false, NULL);
        on_token(&g, 0); on_token(&g, 1);
        printf("case%d [%s]\n", test, g.stream ? wire : g.cp.content);
        chat_parser_free(&g.cp); free(g.text);
    }
    for (int mode = 0; mode < 4; mode++) {
        for (int test = 0; test < 3; test++) {
            gen_ctx g = {0};
            g.tok = t; g.chat = mode / 2; g.stream = mode % 2;
            g.n_stops = 2; g.stops[0] = "STOP"; g.stops[1] = "END";
            wire_n = 0; wire[0] = 0;
            chat_parser_init(&g.cp, false, NULL);
            if (test == 2) on_token(&g, 3);
            else {
                on_token(&g, 0);
                if (test == 1) on_token(&g, 2);
            }
            const char *dc; size_t nc;
            content_delta(&g, true, &dc, &nc);
            if (g.stream && nc) sse_chunk(&g, NULL, 0, dc, nc, NULL, NULL);
            const char *result = g.stream ? wire : g.chat ? g.cp.content : g.text;
            const char *want = test == 2 ? "x " : test == 1 ? "hello STuff" : "hello ST";
            if (strcmp(result, want)) return 4;
            chat_parser_free(&g.cp); free(g.text);
        }
    }
    tok_free(t);
    return 0;
}
'''

CACHE = r'''
#include <assert.h>
#include <string.h>
#include "expert_cache.h"
int main(void) {
    uint64_t offsets[] = {0, 4096};
    ec_config cfg = {.path = "/dev/zero", .n_layer = 1, .n_expert = 2,
                     .offset = offsets, .blob_bytes = 4096, .cache_bytes = 4096};
    char err[256];
    ec_cache *c = ec_open(&cfg, err, sizeof(err));
    assert(!c && strstr(err, "cache needs at least"));
}
'''


def main():
    failed = 0

    def check(name, got, want):
        nonlocal failed
        ok = got == want
        failed += not ok
        print(f"{'PASS' if ok else 'FAIL'} {name}: {got!r}; expected {want!r}")

    with tempfile.TemporaryDirectory(prefix="inferenz-tests-") as td:
        d = Path(td)

        def build(name, source, dependencies):
            c, exe = d / f"{name}.c", d / name
            c.write_text(source)
            cmd = shlex.split(os.environ.get("CC", "cc")) + [
                "-O2", "-march=native", "-std=gnu23", "-fopenmp",
                "-ffunction-sections", "-fdata-sections", "-I", str(ROOT / "src"),
                str(c), *[str(ROOT / "src" / f) for f in dependencies],
                "-Wl,--gc-sections", "-lm", "-lpthread", "-luring", "-o", str(exe),
            ]
            subprocess.run(cmd, check=True)
            return exe

        s = build("sampler", SAMPLER, ["sampler.c"])
        out = subprocess.run([s], check=True, capture_output=True, text=True, timeout=10)
        values = dict(line.split() for line in out.stdout.splitlines())
        check("top_k=0 preserves full support", int(values["support"]), 10000)
        check("top_k=1 keeps global maximum", int(values["winner"]), 9999)
        check("repeat penalty applies once per token ID", float(values["repeat_logit"]), 2.0)

        h = build("head", HEAD, ["quant.c", "threadpool.c", "util.c", "sampler.c"])
        try:
            subprocess.run([h], check=True, capture_output=True, text=True, timeout=3)
        except subprocess.TimeoutExpired:
            check("head candidate selection terminates", "timeout", "returned")
        else:
            check("head candidate selection terminates", "returned", "returned")

        s = build("stop", STOP, ["tokenizer.c", "chat.c", "json.c", "util.c"])
        out = subprocess.run([s], check=True, capture_output=True, text=True, timeout=10)
        values = dict(line.split(" ", 1) for line in out.stdout.splitlines())
        for i, name in enumerate(["completion stream", "chat stream", "chat non-stream"]):
            check(f"split stop string: {name}", values[f"case{i}"], "[hello ]")

        c = build("cache", CACHE, ["expert_cache.c", "util.c"])
        subprocess.run([c], check=True, capture_output=True, text=True, timeout=10)
        check("cache respects insufficient budget", True, True)

    print(f"\n{failed}/8 regressions detected")
    return int(failed > 0)


if __name__ == "__main__":
    raise SystemExit(main())

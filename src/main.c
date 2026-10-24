// inz: command line entry point.
//   inz serve   -m model.gguf [--host 127.0.0.1] [--port 8080]   web UI + OpenAI API + diagnostics
//   inz run     -m model.gguf [-p prompt]                         interactive chat in the terminal
//   inz bench   -m model.gguf [--tokens N]                        decode/prefill benchmark with breakdown
//   inz convert --src DIR --out FILE [--experts q3l] ...           official checkpoint -> inferenZ GGUF
//   inz probe   [--ssd FILE]                                      measure RAM / SSD / VNNI limits
//   inz info    -m model.gguf                                     print GGUF metadata and tensor sizes
#define _GNU_SOURCE
#include <getopt.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>

#include "chat.h"
#include "convert.h"
#include "diag.h"
#include "engine.h"
#include "gguf.h"
#include "quant.h"
#include "server.h"
#include "util.h"

static uint64_t parse_size(const char *s) {
    char *e;
    double v = strtod(s, &e);
    if (*e == 'G' || *e == 'g') v *= 1024.0 * 1024 * 1024;
    else if (*e == 'M' || *e == 'm') v *= 1024.0 * 1024;
    return (uint64_t)v;
}

static void usage(void) {
    fprintf(stderr,
            "inferenZ %s - Qwen3.8-Flash-Next on CPU with SSD expert streaming\n\n"
            "usage: inz <command> [options]\n\n"
            "commands:\n"
            "  serve    web UI, OpenAI-compatible API and live diagnostics\n"
            "  run      interactive chat in the terminal\n"
            "  bench    speed benchmark with per-stage breakdown\n"
            "  convert  official safetensors checkpoint -> inferenZ GGUF\n"
            "  dump     teacher-forced logits for validation\n"
            "  info     show GGUF metadata and tensors\n\n"
            "common options:\n"
            "  -m, --model FILE        converted model (default models/qwen3.8-flash-next-q4q8.inz.gguf)\n"
            "  -t, --threads N         compute threads (default: P+E cores)\n"
            "  -c, --ctx N             context length (default 32768)\n"
            "      --cache SIZE        expert cache budget, e.g. 40G (default: free RAM minus the reserve)\n"
            "      --ram-reserve GB    RAM left to the rest of the system when sizing the cache (default: max(10, 30%%))\n"
            "      --effort E          reasoning effort: none, low, medium, high (default high)\n"
            "      --temp T            sampling temperature for run (default 1.0, the model's own)\n"
            "      --no-spec           disable MTP speculative decoding\n"
            "      --spec-k N          tokens drafted per step by the MTP layer (1-3, default 1)\n"
            "      --exact-head        full exact LM head in RAM instead of screen + exact candidates (validation)\n"
            "      --lookahead N       decode: prefetch the experts the next layer's router predicts (1 on, 0 off)\n"
            "      --sqpoll CPU        io_uring submission thread on CPU (default: a low-power core; -1 off)\n"
            "      --cache-decay D     expert cache LRFU decay per token (default 0.01; 0 = LFU, 0.2 ~ LRU)\n"
            "      --hotlist F         expert warm-start scores: file, auto or off (default auto)\n"
            "      --route-trace FILE  record routing decisions for offline cache simulation\n"
            "  -v                      verbose\n",
            INZ_VERSION);
}

static const struct option common_opts[] = {
    {"model", required_argument, 0, 'm'}, {"threads", required_argument, 0, 't'}, {"ctx", required_argument, 0, 'c'},
    {"cache", required_argument, 0, 1001}, {"no-spec", no_argument, 0, 1002}, {"effort", required_argument, 0, 1003},
    {"route-trace", required_argument, 0, 1006}, {"host", required_argument, 0, 1007}, {"port", required_argument, 0, 1008},
    {"prompt", required_argument, 0, 'p'}, {"tokens", required_argument, 0, 'n'}, {"api-key", required_argument, 0, 1009},
    {"hotlist", required_argument, 0, 1010}, {"mlock", no_argument, 0, 1011}, {"temp", required_argument, 0, 1013},
    {"cache-decay", required_argument, 0, 1015}, {"sqpoll", required_argument, 0, 1018}, {"spec-k", required_argument, 0, 1021}, {"exact-head", no_argument, 0, 1022}, {"lookahead", required_argument, 0, 1023}, {"ram-reserve", required_argument, 0, 1024},
    {0, 0, 0, 0}};

typedef struct {
    inz_opts eo;
    const char *host, *prompt, *api_key, *effort;
    int port, ntok;
    float temp;
} cli;

static void parse_common(int argc, char **argv, cli *c) {
    inz_opts_default(&c->eo);
    c->host = "127.0.0.1";
    c->port = 8080;
    c->ntok = 128;
    c->temp = 1.0f;   // the model's generation_config.json: temperature 1.0, top_p 0.95, top_k 20
    int ch;
    optind = 1;
    while ((ch = getopt_long(argc, argv, "m:t:c:p:n:v", common_opts, NULL)) != -1) {
        switch (ch) {
        case 'm': c->eo.model_path = optarg; break;
        case 't': c->eo.threads = atoi(optarg); break;
        case 'c': c->eo.ctx = atoi(optarg); break;
        case 'p': c->prompt = optarg; break;
        case 'n': c->ntok = atoi(optarg); break;
        case 'v': c->eo.verbose++; inz_log_level++; break;
        case 1001: c->eo.cache_bytes = parse_size(optarg); break;
        case 1002: c->eo.spec = false; break;
        case 1003: c->effort = optarg; break;
        case 1006: c->eo.route_trace = optarg; break;
        case 1007: c->host = optarg; break;
        case 1008: c->port = atoi(optarg); break;
        case 1009: c->api_key = optarg; break;
        case 1010: c->eo.hotlist = optarg; break;  // file, "auto" or "off"
        case 1011: c->eo.mlock = true; break;
        case 1013: c->temp = (float)atof(optarg); break;
        case 1015: c->eo.cache_decay = (float)atof(optarg); break;
        case 1018: c->eo.sqpoll_cpu = atoi(optarg); break;
        case 1021: c->eo.spec_k = atoi(optarg); break;
        case 1022: c->eo.exact_head = true; break;
        case 1023: c->eo.lookahead = atoi(optarg); break;
        case 1024: c->eo.ram_reserve_gb = atof(optarg); break;
        default: usage(); exit(1);
        }
    }
}

typedef struct { tokenizer *tok; tok_stream ts; chat_parser cp; bool in_think; } run_ctx;

static bool run_cb(void *ud, int token) {
    run_ctx *r = ud;
    char buf[1024];
    size_t n = tok_stream_push(r->tok, &r->ts, token, buf, sizeof(buf));
    if (!n) return true;
    const char *dr, *dc;
    size_t nr, nc;
    chat_parser_feed(&r->cp, buf, n, &dr, &nr, &dc, &nc);
    if (nr) { if (!r->in_think) { fputs("\x1b[2m", stdout); r->in_think = true; } fwrite(dr, 1, nr, stdout); }
    if (nc) { if (r->in_think) { fputs("\x1b[0m\n", stdout); r->in_think = false; } fwrite(dc, 1, nc, stdout); }
    fflush(stdout);
    return true;
}

static int cmd_run(int argc, char **argv, bool bench) {
    cli c;
    memset(&c, 0, sizeof(c));
    parse_common(argc, argv, &c);
    char err[512];
    inz_engine *e = engine_open(&c.eo, err, sizeof(err));
    if (!e) { fprintf(stderr, "inz: %s\n", err); return 1; }
    tokenizer *tok = engine_tokenizer(e);
    char err2[64];
    jval *msgs = json_parse("[]", 2, err2, sizeof(err2));
    char line[65536];
    int turn = 0;
    for (;;) {
        const char *user = NULL;
        if (c.prompt && turn == 0) user = c.prompt;
        else if (bench && turn > 0) break;
        else {
            printf("\n\x1b[1m> \x1b[0m");
            fflush(stdout);
            if (!fgets(line, sizeof(line), stdin)) break;
            line[strcspn(line, "\n")] = 0;
            if (!strcmp(line, "/quit") || !strcmp(line, "/exit")) break;
            if (!*line) continue;
            user = line;
        }
        if (!user) user = "Write a short poem about the sea.";
        // append user message
        jw w;
        jw_init(&w);
        jw_obj(&w);
        jw_kstr(&w, "role", "user");
        jw_kstr(&w, "content", user);
        jw_obj_end(&w);
        jval *m = json_parse(w.buf, w.len, err2, sizeof(err2));
        jw_free(&w);
        jval **tail = &msgs->child;
        while (*tail) tail = &(*tail)->next;
        *tail = m;
        msgs->len++;
        char *prompt = chat_render(msgs, NULL, c.effort, true);
        size_t pl = strlen(prompt);
        int32_t *ids = inz_malloc(sizeof(int32_t) * (pl + 16));
        int n = tok_encode(tok, prompt, pl, true, ids, (int)pl + 16);
        free(prompt);
        gen_opts go;
        memset(&go, 0, sizeof(go));
        sampler_default(&go.sp);
        go.sp.temperature = bench ? 0.0f : c.temp;
        go.sp.top_p = 0.95f;
        go.sp.top_k = 20;
        go.max_tokens = bench ? c.ntok : 32768;
        const int32_t stops[2] = {engine_eos(e), engine_eot(e)};
        go.stop_ids = stops;
        go.n_stop = 2;
        go.spec = c.eo.spec;
        run_ctx rc;
        memset(&rc, 0, sizeof(rc));
        rc.tok = tok;
        chat_parser_init(&rc.cp, chat_thinking_enabled(c.effort), NULL);
        gen_result res;
        engine_generate(e, ids, n, &go, run_cb, &rc, &res);
        free(ids);
        if (rc.in_think) fputs("\x1b[0m", stdout);
        printf("\n\x1b[2m[%d prompt (%d cached) @ %.1f tok/s, %d generated @ %.2f tok/s, ttft %.2fs",
               res.prompt_tokens, res.cached_tokens,
               res.t_prefill > 0 ? (res.prompt_tokens - res.cached_tokens) / res.t_prefill : 0, res.gen_tokens,
               res.t_decode > 0 ? res.gen_tokens / res.t_decode : 0, res.ttft);
        if (res.drafted) printf(", spec %d/%d accepted", res.accepted, res.drafted);
        printf("]\x1b[0m\n");
        // append assistant reply to history
        jw_init(&w);
        jw_obj(&w);
        jw_kstr(&w, "role", "assistant");
        jw_key(&w, "content");
        jw_strn(&w, rc.cp.content ? rc.cp.content : "", rc.cp.content_len);
        jw_obj_end(&w);
        jval *a = json_parse(w.buf, w.len, err2, sizeof(err2));
        jw_free(&w);
        tail = &msgs->child;
        while (*tail) tail = &(*tail)->next;
        *tail = a;
        msgs->len++;
        chat_parser_free(&rc.cp);
        turn++;
    }
    if (bench) {
        jw w;
        jw_init(&w);
        engine_diag_json(e, &w);
        printf("%s\n", w.buf);
        jw_free(&w);
    }
    json_free(msgs);
    engine_close(e);
    return 0;
}

static int cmd_serve(int argc, char **argv) {
    cli c;
    memset(&c, 0, sizeof(c));
    parse_common(argc, argv, &c);
    char err[512];
    inz_engine *e = engine_open(&c.eo, err, sizeof(err));
    if (!e) { fprintf(stderr, "inz: %s\n", err); return 1; }
    int rc = server_run(e, c.host, c.port, c.api_key);
    engine_close(e);
    return rc;
}

static int qtype_arg(const char *s) {
    int t = inz_qtype_from_name(s);
    if (t < 0) inz_die("unknown storage type %s (use F16, F32, Q8G, Q4L, Q3L, Q2L)", s);
    return t;
}

static int cmd_convert(int argc, char **argv) {
    convert_opts o;
    convert_opts_default(&o);
    static const struct option lo[] = {
        {"src", required_argument, 0, 's'}, {"out", required_argument, 0, 'o'}, {"experts", required_argument, 0, 1},
        {"dense", required_argument, 0, 2}, {"hc", required_argument, 0, 3}, {"shared", required_argument, 0, 4},
        {"router", required_argument, 0, 5}, {"ple", required_argument, 0, 6}, {"head", required_argument, 0, 7},
        {"embed", required_argument, 0, 12}, {"layers", required_argument, 0, 8}, {"rot8", no_argument, 0, 9}, {"no-rot8", no_argument, 0, 10},
        {"no-ngram", no_argument, 0, 11}, {"screen", required_argument, 0, 13}, {"imatrix", required_argument, 0, 14}, {"threads", required_argument, 0, 't'}, {"verbose", no_argument, 0, 'v'}, {0, 0, 0, 0}};
    int ch;
    optind = 1;
    while ((ch = getopt_long(argc, argv, "s:o:t:v", lo, NULL)) != -1) {
        switch (ch) {
        case 's': o.src_dir = optarg; break;
        case 'o': o.out_path = optarg; break;
        case 't': o.threads = atoi(optarg); break;
        case 'v': inz_log_level++; break;
        case 1: o.t_experts = qtype_arg(optarg); break;
        case 2: o.t_dense = qtype_arg(optarg); break;
        case 3: o.t_hc = qtype_arg(optarg); break;
        case 4: o.t_shared = qtype_arg(optarg); break;
        case 5: o.t_router = qtype_arg(optarg); break;
        case 6: o.t_ple = qtype_arg(optarg); break;
        case 7: o.t_head = qtype_arg(optarg); break;
        case 8: o.max_layers = atoi(optarg); break;
        case 9: o.rot8 = true; break;
        case 10: o.rot8 = false; break;
        case 11: o.no_ngram = true; break;
        case 12: o.t_embed = qtype_arg(optarg); break;
        case 13: o.t_head_screen = strcasecmp(optarg, "none") ? qtype_arg(optarg) : 0; break;
        case 14: o.imatrix = optarg; break;
        default:
            fprintf(stderr, "usage: inz convert --src DIR --out FILE [--experts T] [--dense T] [--hc T] [--shared T]\n"
                            "                   [--router T] [--ple T] [--head T] [--screen T|none] [--embed T] [--layers N]\n"
                            "                   [--no-rot8] [--no-ngram] [--imatrix FILE]\n"
                            "types: F16 (exact for BF16 sources), F32, Q8G, Q4L, Q3L, Q2L\n");
            return 1;
        }
    }
    if (!o.src_dir || !o.out_path) { fprintf(stderr, "inz convert: --src and --out are required\n"); return 1; }
    return convert_run(&o);
}

static int cmd_dump(int argc, char **argv) {
    // inz dump -m model --tokens 1,2,3 --out logits.f32 : raw float32 logits [n][vocab] for validation
    const char *toks = NULL, *out = NULL;
    int last = 0;
    bool stepwise = false, snaptest = false;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--stepwise")) stepwise = true;
        else if (!strcmp(argv[i], "--snaptest")) snaptest = true;
        else if (!strcmp(argv[i], "--tokens") && i + 1 < argc) toks = argv[++i];
        else if (!strcmp(argv[i], "--out") && i + 1 < argc) out = argv[++i];
        else if (!strcmp(argv[i], "--last") && i + 1 < argc) last = atoi(argv[++i]);
    }
    char *tokbuf = NULL;
    if (toks && toks[0] == '@') { tokbuf = inz_read_file(toks + 1, NULL); toks = tokbuf; }
    cli c;
    memset(&c, 0, sizeof(c));
    int na = 0;
    char *av[64];
    for (int i = 0; i < argc && na < 63; i++) {
        if ((!strcmp(argv[i], "--tokens") || !strcmp(argv[i], "--out") || !strcmp(argv[i], "--last")) && i + 1 < argc) { i++; continue; }
        if (!strcmp(argv[i], "--stepwise") || !strcmp(argv[i], "--snaptest")) continue;
        av[na++] = argv[i];
    }
    parse_common(na, av, &c);
    if (!toks || !out) { fprintf(stderr, "usage: inz dump -m MODEL --tokens a,b,c --out FILE\n"); return 1; }
    static int32_t ids[65536];
    int n = 0;
    for (char *p = (char *)toks; *p && n < 65536;) { ids[n++] = (int32_t)strtol(p, &p, 10); if (*p == ',') p++; else if (*p) p++; }
    char err[512];
    inz_engine *e = engine_open(&c.eo, err, sizeof(err));
    if (!e) { fprintf(stderr, "inz: %s\n", err); return 1; }
    double t0 = inz_now();
    float *lg = snaptest ? engine_eval_snapshots(e, ids, n, last) : engine_eval_logits_ex(e, ids, n, last, stepwise);
    fprintf(stderr, "inz dump: %d tokens in %.2f s\n", n, inz_now() - t0);
    int from = last > 0 && last < n ? n - last : 0;
    FILE *f = fopen(out, "wb");
    fwrite(lg, sizeof(float), (size_t)(n - from) * engine_vocab(), f);
    fclose(f);
    for (int t = from; t < n; t++) {
        int best = 0;
        const float *r = lg + (size_t)(t - from) * engine_vocab();
        for (int v = 1; v < engine_vocab(); v++) if (r[v] > r[best]) best = v;
        printf("%d%s", best, t + 1 < n ? "," : "\n");
    }
    free(lg);
    if (c.eo.verbose) {
        jw w;
        jw_init(&w);
        engine_diag_json(e, &w);
        fprintf(stderr, "%s\n", w.buf);
        jw_free(&w);
    }
    engine_close(e);
    return 0;
}

static int cmd_info(int argc, char **argv) {
    cli c;
    memset(&c, 0, sizeof(c));
    parse_common(argc, argv, &c);
    char err[256];
    gguf_file *g = gguf_open(c.eo.model_path, err, sizeof(err));
    if (!g) { fprintf(stderr, "inz: %s\n", err); return 1; }
    printf("GGUF v%u, %llu tensors, %llu kv, alignment %llu, %.2f GiB\n", g->version, (unsigned long long)g->n_tensors,
           (unsigned long long)g->n_kv, (unsigned long long)g->alignment, g->size / 1073741824.0);
    for (uint64_t i = 0; i < g->n_kv; i++) {
        const gguf_kv *kv = &g->kv[i];
        if (kv->type == GGUF_T_STR) {
            size_t l = strlen(kv->s);
            printf("  %-32s %.*s%s\n", kv->key, (int)(l > 90 ? 90 : l), kv->s, l > 90 ? "..." : "");
        } else if (kv->type == GGUF_T_ARR) printf("  %-32s [array of %llu]\n", kv->key, (unsigned long long)kv->arr_n);
        else printf("  %-32s %lld\n", kv->key, (long long)gguf_get_int(g, kv->key, 0));
    }
    double tot[256] = {0};
    for (uint64_t i = 0; i < g->n_tensors; i++) tot[g->tensors[i].type & 255] += (double)g->tensors[i].nbytes;
    for (int t = 0; t < 256; t++) if (tot[t] > 0) printf("  type %-8s %8.2f GiB\n", inz_qtype_name(t), tot[t] / 1073741824.0);
    gguf_close(g);
    return 0;
}

int main(int argc, char **argv) {
    if (argc < 2) { usage(); return 1; }
    const char *cmd = argv[1];
    argc--;
    argv++;
    if (!strcmp(cmd, "serve")) return cmd_serve(argc, argv);
    if (!strcmp(cmd, "run")) return cmd_run(argc, argv, false);
    if (!strcmp(cmd, "bench")) return cmd_run(argc, argv, true);
    if (!strcmp(cmd, "convert")) return cmd_convert(argc, argv);
    if (!strcmp(cmd, "info")) return cmd_info(argc, argv);
    if (!strcmp(cmd, "dump")) return cmd_dump(argc, argv);
    usage();
    return 1;
}

// HTTP front-end: OpenAI-compatible API, web UI and diagnostics endpoints.
#define _GNU_SOURCE
#include "server.h"

#include <pthread.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>
#include <time.h>

#include "chat.h"
#include "diag.h"
#include "http.h"
#include "util.h"

static const char web_index[] = {
#embed "../web/index.html"
    , 0};

typedef struct {
    inz_engine *eng;
    pthread_mutex_t mu;
    pthread_mutex_t diag_mu;
    char *diag_json, *metrics;
    double diag_time;
    const char *api_key;
    char model_id[64];
    volatile int stop;
} srv;

static srv g_srv;

static void publish_diag(bool force) {
    if (!g_srv.eng) return;
    double now = inz_now();
    if (!force && now - g_srv.diag_time < 0.5) return;
    jw w;
    jw_init(&w);
    engine_diag_json(g_srv.eng, &w);
    char *metrics = diag_prometheus();
    pthread_mutex_lock(&g_srv.diag_mu);
    free(g_srv.diag_json);
    free(g_srv.metrics);
    g_srv.diag_json = w.buf;
    g_srv.metrics = metrics;
    pthread_mutex_unlock(&g_srv.diag_mu);
    g_srv.diag_time = now;
}

static void send_error(http_conn *c, int status, const char *msg) {
    jw w;
    jw_init(&w);
    jw_obj(&w);
    jw_key(&w, "error");
    jw_obj(&w);
    jw_kstr(&w, "message", msg);
    jw_kstr(&w, "type", status == 400 ? "invalid_request_error" : "server_error");
    jw_obj_end(&w);
    jw_obj_end(&w);
    http_send_json(c, status, w.buf);
    jw_free(&w);
}

static bool check_auth(http_req *r, http_conn *c) {
    if (!g_srv.api_key || !*g_srv.api_key) return true;
    char tmp[512];
    const char *h = http_header(r, "Authorization", tmp, sizeof(tmp));
    if (h && !strncasecmp(h, "Bearer ", 7) && !strcmp(h + 7, g_srv.api_key)) return true;
    send_error(c, 401, "invalid api key");
    return false;
}

/* ---------------- generation plumbing ---------------- */

typedef struct {
    http_conn *c;
    tokenizer *tok;
    tok_stream ts;
    chat_parser cp;
    bool stream, chat, include_usage;
    char id[48];
    long created;
    // non-streaming accumulation
    char *text; size_t text_len, text_cap;
    // stop strings
    const char *stops[8];
    int n_stops;
    bool stopped_by_string;
    bool client_gone;
    size_t emitted_content;   // bytes of content already sent (for stop-string holdback)
    int ntok;
} gen_ctx;

static void append(char **b, size_t *n, size_t *cap, const char *p, size_t k) {
    if (*n + k + 1 > *cap) {
        size_t nc = *cap ? *cap * 2 : 1024;
        while (nc < *n + k + 1) nc *= 2;
        *b = realloc(*b, nc);
        *cap = nc;
    }
    memcpy(*b + *n, p, k);
    *n += k;
    (*b)[*n] = 0;
}

static void sse_chunk(gen_ctx *g, const char *reasoning, size_t nr, const char *content, size_t nc, const char *tool_calls,
                      const char *finish) {
    jw w;
    jw_init(&w);
    jw_obj(&w);
    jw_kstr(&w, "id", g->id);
    jw_kstr(&w, "object", g->chat ? "chat.completion.chunk" : "text_completion");
    jw_kint(&w, "created", g->created);
    jw_kstr(&w, "model", g_srv.model_id);
    jw_key(&w, "choices");
    jw_arr(&w);
    jw_obj(&w);
    jw_kint(&w, "index", 0);
    if (g->chat) {
        jw_key(&w, "delta");
        jw_obj(&w);
        if (nr) { jw_key(&w, "reasoning_content"); jw_strn(&w, reasoning, nr); }
        if (nc) { jw_key(&w, "content"); jw_strn(&w, content, nc); }
        if (tool_calls) { jw_key(&w, "tool_calls"); jw_raw(&w, tool_calls); }
        jw_obj_end(&w);
    } else {
        jw_key(&w, "text");
        jw_strn(&w, content, nc);
    }
    jw_key(&w, "finish_reason");
    if (finish) jw_str(&w, finish); else jw_null(&w);
    jw_obj_end(&w);
    jw_arr_end(&w);
    jw_obj_end(&w);
    if (!http_sse_data(g->c, w.buf)) g->client_gone = true;
    jw_free(&w);
}

static void content_delta(gen_ctx *g, bool final, const char **delta, size_t *n) {
    char *text = g->chat ? g->cp.content : g->text;
    size_t *len = g->chat ? &g->cp.content_len : &g->text_len;
    size_t end = *len;
    for (int i = 0; i < g->n_stops && text; i++) {
        const char *stop = g->stops[i];
        if (!*stop) continue;
        const char *hit = strstr(text + g->emitted_content, stop);
        if (hit && (size_t)(hit - text) < end) end = (size_t)(hit - text);
    }
    if (end < *len) {
        *len = end;
        text[end] = 0;
        g->stopped_by_string = true;
    } else if (!final) {
        for (int i = 0; i < g->n_stops; i++) {
            size_t k = strlen(g->stops[i]);
            if (!k) continue;
            k--;
            if (k > *len - g->emitted_content) k = *len - g->emitted_content;
            for (; k; k--) if (!memcmp(text + *len - k, g->stops[i], k)) {
                if (*len - k < end) end = *len - k;
                break;
            }
        }
    }
    *delta = text ? text + g->emitted_content : "";
    *n = end - g->emitted_content;
    g->emitted_content = end;
}

static bool on_token(void *ud, int token) {
    publish_diag(false);
    gen_ctx *g = ud;
    g->ntok++;
    char buf[1024];
    size_t n = tok_stream_push(g->tok, &g->ts, token, buf, sizeof(buf));
    if (!n) return !g->client_gone;
    if (g->chat) {
        const char *dr, *dc;
        size_t nr, nc;
        chat_parser_feed(&g->cp, buf, n, &dr, &nr, &dc, &nc);
        content_delta(g, false, &dc, &nc);
        if (g->stream && (nr || nc)) sse_chunk(g, dr, nr, dc, nc, NULL, NULL);
    } else {
        append(&g->text, &g->text_len, &g->text_cap, buf, n);
        const char *dc;
        size_t nc;
        content_delta(g, false, &dc, &nc);
        if (g->stream && nc) sse_chunk(g, NULL, 0, dc, nc, NULL, NULL);
    }
    if (g->stream && !http_conn_alive(g->c)) g->client_gone = true;
    return !g->client_gone && !g->stopped_by_string;
}

static void write_usage(jw *w, const gen_result *r) {
    jw_key(w, "usage");
    jw_obj(w);
    jw_kint(w, "prompt_tokens", r->prompt_tokens);
    jw_kint(w, "completion_tokens", r->gen_tokens);
    jw_kint(w, "total_tokens", r->prompt_tokens + r->gen_tokens);
    jw_key(w, "prompt_tokens_details");
    jw_obj(w);
    jw_kint(w, "cached_tokens", r->cached_tokens);
    jw_obj_end(w);
    jw_obj_end(w);
}

static void write_timings(jw *w, const gen_result *r) {
    jw_key(w, "timings");
    jw_obj(w);
    jw_kint(w, "prompt_n", r->prompt_tokens);
    jw_kint(w, "prompt_cached", r->cached_tokens);
    jw_knum(w, "prompt_ms", r->t_prefill * 1e3);
    int pn = r->prompt_tokens - r->cached_tokens;
    jw_knum(w, "prompt_per_second", r->t_prefill > 0 ? pn / r->t_prefill : 0);
    jw_kint(w, "predicted_n", r->gen_tokens);
    jw_knum(w, "predicted_ms", r->t_decode * 1e3);
    jw_knum(w, "predicted_per_second", r->t_decode > 0 ? r->gen_tokens / r->t_decode : 0);
    jw_knum(w, "ttft_ms", r->ttft * 1e3);
    jw_kint(w, "drafted", r->drafted);
    jw_kint(w, "accepted", r->accepted);
    jw_obj_end(w);
}

static void parse_sampling(const jval *req, gen_opts *go) {
    sampler_default(&go->sp);
    go->sp.temperature = (float)json_num(json_get(req, "temperature"), 1.0);
    go->sp.top_p = (float)json_num(json_get(req, "top_p"), 0.95);
    go->sp.top_k = (int)json_num(json_get(req, "top_k"), 20);  // defaults from the model's generation_config.json
    go->sp.min_p = (float)json_num(json_get(req, "min_p"), 0.0);
    go->sp.presence_penalty = (float)json_num(json_get(req, "presence_penalty"), 0);
    go->sp.frequency_penalty = (float)json_num(json_get(req, "frequency_penalty"), 0);
    go->sp.repeat_penalty = (float)json_num(json_get(req, "repeat_penalty"), 1.0);
    const jval *seed = json_get(req, "seed");
    go->sp.seed = seed ? (uint64_t)json_int(seed, 0) : (uint64_t)inz_now_ns();
    // thinking at high effort can take tens of thousands of tokens before the answer
    int mt = (int)json_num(json_get(req, "max_completion_tokens"), json_num(json_get(req, "max_tokens"), 32768));
    go->max_tokens = mt > 0 ? mt : 32768;
    go->spec = json_bool(json_get(req, "speculative"), true);
}

static void handle_completion(http_req *r, http_conn *c, bool chat) {
    char err[256];
    jval *req = json_parse(r->body, r->body_len, err, sizeof(err));
    if (!req) { send_error(c, 400, err); return; }
    tokenizer *tok = engine_tokenizer(g_srv.eng);
    char *prompt = NULL;
    const char *effort = NULL;
    const jval *tools = NULL;
    if (chat) {
        const jval *msgs = json_get(req, "messages");
        if (!msgs || msgs->type != JARR) { send_error(c, 400, "messages must be an array"); json_free(req); return; }
        effort = json_str(json_get(req, "reasoning_effort"), NULL);
        if (!effort) effort = json_str(json_path(req, "reasoning.effort"), NULL);
        tools = json_get(req, "tools");
        prompt = chat_render(msgs, tools, effort, true);
    } else {
        const jval *p = json_get(req, "prompt");
        if (p && p->type == JARR) p = json_at(p, 0);
        prompt = inz_strdup(json_str(p, ""));
    }
    size_t plen = strlen(prompt);
    int cap = (int)plen + 16;
    int32_t *ids = inz_malloc(sizeof(int32_t) * cap);
    int n = tok_encode(tok, prompt, plen, true, ids, cap);
    free(prompt);

    gen_opts go;
    memset(&go, 0, sizeof(go));
    parse_sampling(req, &go);
    const int32_t stop_ids[2] = {engine_eos(g_srv.eng), engine_eot(g_srv.eng)};
    go.stop_ids = stop_ids;
    go.n_stop = 2;

    gen_ctx g;
    memset(&g, 0, sizeof(g));
    g.c = c;
    g.tok = tok;
    g.chat = chat;
    g.stream = json_bool(json_get(req, "stream"), false);
    g.include_usage = json_bool(json_path(req, "stream_options.include_usage"), false);
    snprintf(g.id, sizeof(g.id), "%s-%llx", chat ? "chatcmpl" : "cmpl", (unsigned long long)inz_now_ns());
    g.created = (long)time(NULL);
    chat_parser_init(&g.cp, chat && chat_thinking_enabled(effort), chat ? tools : NULL);
    const jval *stop = json_get(req, "stop");
    if (stop && stop->type == JSTR && stop->len) g.stops[g.n_stops++] = stop->str;
    else if (stop && stop->type == JARR)
        for (jval *s = stop->child; s && g.n_stops < 8; s = s->next)
            if (s->type == JSTR && s->len) g.stops[g.n_stops++] = s->str;

    if (g.stream) {
        http_sse_begin(c);
        if (chat) {  // role chunk
            jw w;
            jw_init(&w);
            jw_obj(&w);
            jw_kstr(&w, "id", g.id);
            jw_kstr(&w, "object", "chat.completion.chunk");
            jw_kint(&w, "created", g.created);
            jw_kstr(&w, "model", g_srv.model_id);
            jw_key(&w, "choices");
            jw_arr(&w);
            jw_obj(&w);
            jw_kint(&w, "index", 0);
            jw_key(&w, "delta");
            jw_obj(&w);
            jw_kstr(&w, "role", "assistant");
            jw_kstr(&w, "content", "");
            jw_obj_end(&w);
            jw_key(&w, "finish_reason");
            jw_null(&w);
            jw_obj_end(&w);
            jw_arr_end(&w);
            jw_obj_end(&w);
            http_sse_data(c, w.buf);
            jw_free(&w);
        }
    }

    gen_result res;
    memset(&res, 0, sizeof(res));
    pthread_mutex_lock(&g_srv.mu);
    int rc = engine_generate(g_srv.eng, ids, n, &go, on_token, &g, &res);
    publish_diag(true);
    pthread_mutex_unlock(&g_srv.mu);
    free(ids);

    char *tool_calls = NULL;
    const char *dr = "", *dc = "";
    size_t nr = 0, nc = 0;
    if (chat && !g.stopped_by_string) {
        tool_calls = chat_parser_finish(&g.cp, &dr, &nr, &dc, &nc);
    }
    content_delta(&g, true, &dc, &nc);
    if (g.stopped_by_string) { free(tool_calls); tool_calls = NULL; }
    const char *finish = g.stopped_by_string ? "stop" : tool_calls ? "tool_calls" : res.finish_reason ? res.finish_reason : "stop";
    if (rc != 0 && !g.client_gone) {
        if (g.stream) http_sse_data(c, "{\"error\":{\"message\":\"generation failed\"}}");
        else send_error(c, 500, "generation failed");
    } else if (g.stream) {
        if (nr || nc || tool_calls) sse_chunk(&g, dr, nr, dc, nc, tool_calls, NULL);
        // final chunk with finish reason, usage and timings
        jw w;
        jw_init(&w);
        jw_obj(&w);
        jw_kstr(&w, "id", g.id);
        jw_kstr(&w, "object", chat ? "chat.completion.chunk" : "text_completion");
        jw_kint(&w, "created", g.created);
        jw_kstr(&w, "model", g_srv.model_id);
        jw_key(&w, "choices");
        jw_arr(&w);
        jw_obj(&w);
        jw_kint(&w, "index", 0);
        if (chat) { jw_key(&w, "delta"); jw_obj(&w); jw_obj_end(&w); }
        else jw_kstr(&w, "text", "");
        jw_kstr(&w, "finish_reason", finish);
        jw_obj_end(&w);
        jw_arr_end(&w);
        if (g.include_usage) write_usage(&w, &res);
        write_timings(&w, &res);
        jw_obj_end(&w);
        http_sse_data(c, w.buf);
        http_sse_data(c, "[DONE]");
        jw_free(&w);
    } else {
        jw w;
        jw_init(&w);
        jw_obj(&w);
        jw_kstr(&w, "id", g.id);
        jw_kstr(&w, "object", chat ? "chat.completion" : "text_completion");
        jw_kint(&w, "created", g.created);
        jw_kstr(&w, "model", g_srv.model_id);
        jw_key(&w, "choices");
        jw_arr(&w);
        jw_obj(&w);
        jw_kint(&w, "index", 0);
        if (chat) {
            jw_key(&w, "message");
            jw_obj(&w);
            jw_kstr(&w, "role", "assistant");
            jw_key(&w, "content");
            jw_strn(&w, g.cp.content ? g.cp.content : "", g.cp.content_len);
            if (g.cp.reasoning_len) { jw_key(&w, "reasoning_content"); jw_strn(&w, g.cp.reasoning, g.cp.reasoning_len); }
            if (tool_calls) { jw_key(&w, "tool_calls"); jw_raw(&w, tool_calls); }
            jw_obj_end(&w);
        } else {
            jw_key(&w, "text");
            jw_strn(&w, g.text ? g.text : "", g.text_len);
        }
        jw_kstr(&w, "finish_reason", finish);
        jw_obj_end(&w);
        jw_arr_end(&w);
        write_usage(&w, &res);
        write_timings(&w, &res);
        jw_obj_end(&w);
        http_send_json(c, 200, w.buf);
        jw_free(&w);
    }
    free(tool_calls);
    free(g.text);
    chat_parser_free(&g.cp);
    json_free(req);
}

static void handle(http_req *r, http_conn *c, void *ud) {
    (void)ud;
    const bool get = !strcmp(r->method, "GET"), post = !strcmp(r->method, "POST");
    if (get && (!strcmp(r->path, "/") || !strcmp(r->path, "/index.html"))) {
        http_send(c, 200, "text/html; charset=utf-8", web_index, sizeof(web_index) - 1);
        return;
    }
    if (get && !strcmp(r->path, "/health")) { http_send_json(c, 200, "{\"status\":\"ok\"}"); return; }
    if (get && !strcmp(r->path, "/v1/models")) {
        jw w;
        jw_init(&w);
        jw_obj(&w);
        jw_kstr(&w, "object", "list");
        jw_key(&w, "data");
        jw_arr(&w);
        jw_obj(&w);
        jw_kstr(&w, "id", g_srv.model_id);
        jw_kstr(&w, "object", "model");
        jw_kint(&w, "created", 1761177600);
        jw_kstr(&w, "owned_by", "inferenz");
        jw_obj_end(&w);
        jw_arr_end(&w);
        jw_obj_end(&w);
        http_send_json(c, 200, w.buf);
        jw_free(&w);
        return;
    }
    if (get && !strcmp(r->path, "/v1/diag")) {
        pthread_mutex_lock(&g_srv.diag_mu);
        char *p = inz_strdup(g_srv.diag_json);
        pthread_mutex_unlock(&g_srv.diag_mu);
        http_send_json(c, 200, p);
        free(p);
        return;
    }
    if (get && !strcmp(r->path, "/metrics")) {
        pthread_mutex_lock(&g_srv.diag_mu);
        char *p = inz_strdup(g_srv.metrics);
        pthread_mutex_unlock(&g_srv.diag_mu);
        http_send(c, 200, "text/plain; version=0.0.4", p, strlen(p));
        free(p);
        return;
    }
    if (!strcmp(r->path, "/v1/diag/trace")) {
        if (post) {
            int n = 16;
            const char *q = strstr(r->query, "tokens=");
            if (q) n = atoi(q + 7);
            trace_arm(n);
            http_send_json(c, 200, "{\"armed\":true}");
        } else {
            size_t len = 0;
            char *t = trace_json(&len);
            if (!t) send_error(c, 404, "no trace captured yet: POST /v1/diag/trace?tokens=16 first");
            else { http_send(c, 200, "application/json", t, len); free(t); }
        }
        return;
    }
    if (post && !strcmp(r->path, "/v1/chat/completions")) { if (check_auth(r, c)) handle_completion(r, c, true); return; }
    if (post && !strcmp(r->path, "/v1/completions")) { if (check_auth(r, c)) handle_completion(r, c, false); return; }
    if (post && !strcmp(r->path, "/tokenize")) {
        char err[128];
        jval *req = json_parse(r->body, r->body_len, err, sizeof(err));
        const char *text = json_str(json_get(req, "content"), "");
        size_t tl = strlen(text);
        int32_t *ids = inz_malloc(sizeof(int32_t) * (tl + 16));
        int n = tok_encode(engine_tokenizer(g_srv.eng), text, tl, true, ids, (int)tl + 16);
        jw w;
        jw_init(&w);
        jw_obj(&w);
        jw_key(&w, "tokens");
        jw_arr(&w);
        for (int i = 0; i < n; i++) jw_int(&w, ids[i]);
        jw_arr_end(&w);
        jw_obj_end(&w);
        http_send_json(c, 200, w.buf);
        jw_free(&w);
        free(ids);
        json_free(req);
        return;
    }
    send_error(c, 404, "not found");
}

static void on_sigint(int s) { (void)s; g_srv.stop = 1; }

int server_run(inz_engine *e, const char *host, int port, const char *api_key) {
    memset(&g_srv, 0, sizeof(g_srv));
    g_srv.eng = e;
    g_srv.api_key = api_key;
    pthread_mutex_init(&g_srv.mu, NULL);
    pthread_mutex_init(&g_srv.diag_mu, NULL);
    publish_diag(true);
    snprintf(g_srv.model_id, sizeof(g_srv.model_id), "%s", engine_model_name(e));
    signal(SIGINT, on_sigint);
    signal(SIGTERM, on_sigint);
    fprintf(stderr, "inz: serving on http://%s:%d  (web UI, /v1/chat/completions, /v1/diag, /metrics)\n", host, port);
    return http_serve(host, port, handle, NULL, &g_srv.stop);
}

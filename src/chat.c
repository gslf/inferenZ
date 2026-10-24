#include "chat.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "util.h"

#define IM_START "<|im_start|>"
#define IM_END "<|im_end|>"
#define THINK_BEGIN "<think>"
#define THINK_END "</think>"
#define TOOLCALL_BEGIN "<tool_call>"
#define TOOLCALL_END "</tool_call>"

typedef struct { char *b; size_t n, cap; } sbuf;

static void sb_put(sbuf *s, const char *p, size_t n) {
    if (n > SIZE_MAX / 4) inz_die("chat: string too long");
    if (s->n + n + 1 > s->cap) {
        size_t nc = s->cap ? s->cap * 2 : 1024;
        while (nc < s->n + n + 1) nc *= 2;
        s->b = realloc(s->b, nc);
        s->cap = nc;
    }
    memcpy(s->b + s->n, p, n);
    s->n += n;
    s->b[s->n] = 0;
}
static void sb_puts(sbuf *s, const char *p) { sb_put(s, p, strlen(p)); }

/* Python json.dumps(x, ensure_ascii=False) with default separators (", ", ": "): the `tojson` filter of
 * transformers' chat templates. */
static void py_json_str(sbuf *s, const char *str, size_t n) {
    sb_puts(s, "\"");
    for (size_t i = 0; i < n; i++) {
        unsigned char c = (unsigned char)str[i];
        char tmp[8];
        switch (c) {
        case '"': sb_puts(s, "\\\""); break;
        case '\\': sb_puts(s, "\\\\"); break;
        case '\n': sb_puts(s, "\\n"); break;
        case '\r': sb_puts(s, "\\r"); break;
        case '\t': sb_puts(s, "\\t"); break;
        case '\b': sb_puts(s, "\\b"); break;
        case '\f': sb_puts(s, "\\f"); break;
        default:
            if (c < 0x20) { snprintf(tmp, sizeof(tmp), "\\u%04x", c); sb_puts(s, tmp); }
            else sb_put(s, (const char *)&str[i], 1);
        }
    }
    sb_puts(s, "\"");
}

static void py_num(sbuf *s, double v) {
    char tmp[64];
    if (v == floor(v) && fabs(v) < 1e16) snprintf(tmp, sizeof(tmp), "%lld", (long long)v);
    else {
        for (int prec = 1; prec <= 17; prec++) {  // shortest round-trip form, like Python repr
            snprintf(tmp, sizeof(tmp), "%.*g", prec, v);
            if (strtod(tmp, NULL) == v) break;
        }
    }
    sb_puts(s, tmp);
}

static void py_json(sbuf *s, const jval *v) {
    if (!v) { sb_puts(s, "null"); return; }
    switch (v->type) {
    case JNULL: sb_puts(s, "null"); break;
    case JBOOL: sb_puts(s, v->b ? "true" : "false"); break;
    case JNUM: py_num(s, v->num); break;
    case JSTR: py_json_str(s, v->str, v->len); break;
    case JARR:
        sb_puts(s, "[");
        for (jval *c = v->child; c; c = c->next) { if (c != v->child) sb_puts(s, ", "); py_json(s, c); }
        sb_puts(s, "]");
        break;
    case JOBJ:
        sb_puts(s, "{");
        for (jval *c = v->child; c; c = c->next) {
            if (c != v->child) sb_puts(s, ", ");
            py_json_str(s, c->key, strlen(c->key));
            sb_puts(s, ": ");
            py_json(s, c);
        }
        sb_puts(s, "}");
        break;
    }
}

// render_content(): strings as is, content parts concatenated (text parts only: no vision in this engine)
static void render_content(sbuf *s, const jval *content) {
    if (!content || content->type == JNULL) return;
    if (content->type == JSTR) { sb_put(s, content->str, content->len); return; }
    if (content->type == JARR) {
        for (jval *it = content->child; it; it = it->next) {
            if (it->type == JSTR) { sb_put(s, it->str, it->len); continue; }
            const jval *t = json_get(it, "text");
            if (t) sb_puts(s, json_str(t, ""));
        }
    }
}

static bool is_ws(char c) { return c == ' ' || c == '\t' || c == '\n' || c == '\r' || c == '\v' || c == '\f'; }

// Jinja's `trim` (Python str.strip())
static char *trimmed(const jval *content, size_t *len) {
    sbuf s = {0};
    render_content(&s, content);
    const char *p = s.b ? s.b : "";
    size_t n = s.n;
    size_t a = 0;
    while (a < n && is_ws(p[a])) a++;
    while (n > a && is_ws(p[n - 1])) n--;
    char *r = inz_malloc(n - a + 1);
    memcpy(r, p + a, n - a);
    r[n - a] = 0;
    *len = n - a;
    free(s.b);
    return r;
}

static const char *resolve_effort(const char *effort, bool *thinking) {
    *thinking = true;
    if (!effort || !*effort) return "xhigh";
    if (!strcmp(effort, "none") || !strcmp(effort, "minimal") || !strcmp(effort, "off")) { *thinking = false; return NULL; }
    if (!strcmp(effort, "low")) return "low";
    if (!strcmp(effort, "medium")) return "medium";
    return "xhigh";  // "high", "xhigh", "max" and anything else
}

bool chat_thinking_enabled(const char *effort) {
    bool t;
    resolve_effort(effort, &t);
    return t;
}

static const char *reasoning_text(const char *eff) {
    if (!eff) return "";
    if (!strcmp(eff, "xhigh"))
        return "Reasoning effort is set to xhigh. Please think carefully through the task, validate key assumptions, "
               "consider plausible alternatives, and prioritize correctness, consistency, and clarity in the final answer.";
    if (!strcmp(eff, "low"))
        return "Reasoning effort is set to low. Keep your thinking brief and focused, moving directly to the conclusion "
               "without unnecessary elaboration.";
    return "";  // medium: no instructions
}

static const char *role_of(const jval *m) { return json_str(json_get(m, "role"), ""); }

// tool call arguments as an object: OpenAI clients send a JSON string, the template iterates over items
static jval *args_object(const jval *args, jval **owned) {
    *owned = NULL;
    if (!args) return NULL;
    if (args->type == JOBJ) return (jval *)args;
    if (args->type == JSTR) {
        char err[64];
        jval *v = json_parse(args->str, args->len, err, sizeof(err));
        if (v && v->type == JOBJ) { *owned = v; return v; }
        json_free(v);
    }
    return NULL;
}

char *chat_render(const jval *messages, const jval *tools, const char *effort, bool add_generation_prompt) {
    sbuf s = {0};
    bool thinking;
    const char *eff = resolve_effort(effort, &thinking);
    const char *ri = thinking ? reasoning_text(eff) : "";
    const size_t nmsg = messages && messages->type == JARR ? messages->len : 0;
    const jval *first = json_at(messages, 0);
    const bool first_system = first && !strcmp(role_of(first), "system");

    if (tools && tools->type == JARR && tools->len) {
        sb_puts(&s, IM_START "system\n");
        if (*ri) { sb_puts(&s, ri); sb_puts(&s, "\n\n"); }
        sb_puts(&s, "# Tools\n\nYou have access to the following functions:\n\n<tools>");
        for (jval *t = tools->child; t; t = t->next) { sb_puts(&s, "\n"); py_json(&s, t); }
        sb_puts(&s, "\n</tools>");
        sb_puts(&s, "\n\nIf you choose to call a function ONLY reply in the following format with NO suffix:\n\n<tool_call>\n"
                    "<function=example_function_name>\n<parameter=example_parameter_1>\nvalue_1\n</parameter>\n"
                    "<parameter=example_parameter_2>\nThis is the value for the second parameter\nthat can span\nmultiple "
                    "lines\n</parameter>\n</function>\n</tool_call>\n\n<IMPORTANT>\nReminder:\n- Function calls MUST follow "
                    "the specified format: an inner <function=...></function> block must be nested within "
                    "<tool_call></tool_call> XML tags\n- Required parameters MUST be specified\n- You may provide optional "
                    "reasoning for your function call in natural language BEFORE the function call, but NOT after\n- If "
                    "there is no function call available, answer the question like normal with your current knowledge and "
                    "do not tell the user about function calls\n</IMPORTANT>");
        if (first_system) {
            size_t n;
            char *c = trimmed(json_get(first, "content"), &n);
            if (n) { sb_puts(&s, "\n\n"); sb_put(&s, c, n); }
            free(c);
        }
        sb_puts(&s, IM_END "\n");
    } else if (first_system) {
        size_t n;
        char *c = trimmed(json_get(first, "content"), &n);
        if (n) {
            sb_puts(&s, IM_START "system\n");
            if (*ri) { sb_puts(&s, ri); sb_puts(&s, "\n\n"); }
            sb_put(&s, c, n);
            sb_puts(&s, IM_END "\n");
        } else if (*ri) {
            sb_puts(&s, IM_START "system\n"); sb_puts(&s, ri); sb_puts(&s, IM_END "\n");
        }
        free(c);
    } else if (*ri) {
        sb_puts(&s, IM_START "system\n"); sb_puts(&s, ri); sb_puts(&s, IM_END "\n");
    }

    // last_query_index: the last user message that is not a pure tool_response wrapper
    long last_query = -1;
    for (long i = (long)nmsg - 1; i >= 0; i--) {
        const jval *m = json_at(messages, (size_t)i);
        if (strcmp(role_of(m), "user")) continue;
        size_t n;
        char *c = trimmed(json_get(m, "content"), &n);
        const bool wrapped = n >= 15 && !strncmp(c, "<tool_response>", 15) && n >= 16 && !strcmp(c + n - 16, "</tool_response>");
        free(c);
        if (!wrapped) { last_query = i; break; }
    }

    for (size_t i = 0; i < nmsg; i++) {
        const jval *m = json_at(messages, i);
        const char *role = role_of(m);
        size_t cn;
        char *content = trimmed(json_get(m, "content"), &cn);
        if (!strcmp(role, "system")) {
            // only valid first; rendered above
        } else if (!strcmp(role, "user")) {
            sb_puts(&s, IM_START "user\n"); sb_put(&s, content, cn); sb_puts(&s, IM_END "\n");
        } else if (!strcmp(role, "assistant")) {
            const jval *rc = json_get(m, "reasoning_content");
            size_t rn = 0;
            char *reasoning = rc && rc->type == JSTR ? trimmed(rc, &rn) : inz_strdup("");
            // preserve_thinking is undefined (true) here: every assistant turn keeps its <think> block
            sb_puts(&s, IM_START "assistant\n" THINK_BEGIN "\n");
            sb_put(&s, reasoning, rn);
            sb_puts(&s, "\n" THINK_END "\n\n");
            sb_put(&s, content, cn);
            free(reasoning);
            const jval *tc = json_get(m, "tool_calls");
            if (tc && tc->type == JARR) {
                size_t k = 0;
                for (jval *call = tc->child; call; call = call->next, k++) {
                    const jval *fn = json_get(call, "function") ? json_get(call, "function") : call;
                    const char *name = json_str(json_get(fn, "name"), "");
                    if (k == 0) sb_puts(&s, cn ? "\n\n" TOOLCALL_BEGIN "\n<function=" : TOOLCALL_BEGIN "\n<function=");
                    else sb_puts(&s, "\n" TOOLCALL_BEGIN "\n<function=");
                    sb_puts(&s, name);
                    sb_puts(&s, ">\n");
                    jval *owned;
                    const jval *args = args_object(json_get(fn, "arguments"), &owned);
                    for (const jval *a = args ? args->child : NULL; a; a = a->next) {
                        sb_puts(&s, "<parameter="); sb_puts(&s, a->key); sb_puts(&s, ">\n");
                        if (a->type == JSTR) sb_put(&s, a->str, a->len);
                        else py_json(&s, a);
                        sb_puts(&s, "\n</parameter>\n");
                    }
                    json_free(owned);
                    sb_puts(&s, "</function>\n" TOOLCALL_END);
                }
            }
            sb_puts(&s, IM_END "\n");
        } else if (!strcmp(role, "tool")) {
            const jval *prev = i ? json_at(messages, i - 1) : NULL;
            const jval *next = i + 1 < nmsg ? json_at(messages, i + 1) : NULL;
            if (prev && strcmp(role_of(prev), "tool")) sb_puts(&s, IM_START "user");
            sb_puts(&s, "\n<tool_response>\n");
            sb_put(&s, content, cn);
            sb_puts(&s, "\n</tool_response>");
            if (!next || strcmp(role_of(next), "tool")) sb_puts(&s, IM_END "\n");
        }
        free(content);
    }
    (void)last_query;  // only affects rendering when preserve_thinking is false
    if (add_generation_prompt) {
        sb_puts(&s, IM_START "assistant\n");
        sb_puts(&s, thinking ? THINK_BEGIN "\n" : THINK_BEGIN "\n\n" THINK_END "\n\n");
    }
    if (!s.b) sb_puts(&s, "");
    return s.b;
}

/* ------------------------------------------------------------------ */
/* Output parser                                                       */
/* ------------------------------------------------------------------ */

static const char *memstr(const char *h, size_t hn, const char *n) {
    size_t nn = strlen(n);
    if (nn > hn) return NULL;
    for (size_t i = 0; i + nn <= hn; i++) if (!memcmp(h + i, n, nn)) return h + i;
    return NULL;
}

static void buf_put(char **b, size_t *n, size_t *cap, const char *p, size_t k) {
    if (*n + k + 1 > *cap) {
        size_t nc = *cap ? *cap * 2 : 256;
        while (nc < *n + k + 1) nc *= 2;
        *b = realloc(*b, nc);
        *cap = nc;
    }
    memcpy(*b + *n, p, k);
    *n += k;
    (*b)[*n] = 0;
}

void chat_parser_init(chat_parser *p, bool thinking, const jval *tools) {
    memset(p, 0, sizeof(*p));
    p->phase = thinking ? CP_THINK : CP_CONTENT;
    p->tools = tools;
}

void chat_parser_free(chat_parser *p) {
    free(p->raw); free(p->reasoning); free(p->content); free(p->tool_raw);
    memset(p, 0, sizeof(*p));
}

// length of the longest suffix of s[0..n) that is a proper prefix of tag
static size_t partial_suffix(const char *s, size_t n, const char *tag) {
    size_t tl = strlen(tag);
    for (size_t k = tl - 1; k > 0; k--) if (k <= n && !memcmp(s + n - k, tag, k)) return k;
    return 0;
}

static void advance(chat_parser *p, bool final, size_t *r0, size_t *rn, size_t *c0, size_t *cn) {
    *r0 = p->reasoning_len; *c0 = p->content_len;
    for (;;) {
        const char *s = p->raw + p->scanned;
        size_t n = p->raw_len - p->scanned;
        if (n == 0) break;
        if (p->phase == CP_START) p->phase = CP_CONTENT;
        if (p->phase == CP_THINK) {
            const char *e = memstr(s, n, THINK_END);
            if (e) {
                size_t k = (size_t)(e - s);
                while (k && s[k - 1] == '\n') k--;  // the template writes "\n</think>"
                buf_put(&p->reasoning, &p->reasoning_len, &p->reasoning_cap, s, k);
                p->scanned += (size_t)(e - s) + strlen(THINK_END);
                p->phase = CP_CONTENT;
                p->skip_nl = true;
                continue;
            }
            // hold back a partial "</think>" and trailing newlines that may precede it
            size_t hold = final ? 0 : partial_suffix(s, n, THINK_END);
            size_t k = n - hold;
            if (!final) while (k && s[k - 1] == '\n') k--;
            buf_put(&p->reasoning, &p->reasoning_len, &p->reasoning_cap, s, k);
            p->scanned += k;
            break;
        }
        if (p->phase == CP_CONTENT) {
            if (p->skip_nl) {
                size_t k = 0;
                while (k < n && s[k] == '\n') k++;
                p->scanned += k;
                if (k == n) break;
                p->skip_nl = false;
                continue;
            }
            const char *e = memstr(s, n, TOOLCALL_BEGIN);
            if (e) {
                size_t k = (size_t)(e - s);
                while (k && s[k - 1] == '\n') k--;  // "\n\n<tool_call>" after content
                buf_put(&p->content, &p->content_len, &p->content_cap, s, k);
                p->scanned += (size_t)(e - s);
                p->phase = CP_TOOL;
                continue;
            }
            size_t hold = final ? 0 : partial_suffix(s, n, TOOLCALL_BEGIN);
            size_t k = n - hold;
            if (!final) while (k && s[k - 1] == '\n') k--;  // could be the newlines before a tool call
            buf_put(&p->content, &p->content_len, &p->content_cap, s, k);
            p->scanned += k;
            break;
        }
        // CP_TOOL: keep everything (more tool calls may follow)
        buf_put(&p->tool_raw, &p->tool_len, &p->tool_cap, s, n);
        p->scanned += n;
        break;
    }
    *rn = p->reasoning_len - *r0;
    *cn = p->content_len - *c0;
}

void chat_parser_feed(chat_parser *p, const char *text, size_t n, const char **dr, size_t *nr, const char **dc, size_t *nc) {
    buf_put(&p->raw, &p->raw_len, &p->raw_cap, text, n);
    size_t r0, c0;
    advance(p, false, &r0, nr, &c0, nc);
    *dr = p->reasoning ? p->reasoning + r0 : "";
    *dc = p->content ? p->content + c0 : "";
}

// declared JSON-schema type of a tool parameter ("string", "integer", ...), or NULL
static const char *param_type(const jval *tools, const char *fn, size_t fnl, const char *param, size_t pl) {
    for (const jval *t = tools && tools->type == JARR ? tools->child : NULL; t; t = t->next) {
        const jval *f = json_get(t, "function") ? json_get(t, "function") : t;
        const char *name = json_str(json_get(f, "name"), "");
        if (strlen(name) != fnl || memcmp(name, fn, fnl)) continue;
        for (const jval *pr = json_path(f, "parameters.properties") ? json_path(f, "parameters.properties")->child : NULL; pr; pr = pr->next)
            if (strlen(pr->key) == pl && !memcmp(pr->key, param, pl)) return json_str(json_get(pr, "type"), NULL);
    }
    return NULL;
}

// A parameter value is raw text between "<parameter=x>\n" and "\n</parameter>". Strings stay strings; other
// declared types (or, without a schema, text that parses as JSON) are passed through as JSON values.
static void put_value(sbuf *out, const char *v, size_t n, const char *type) {
    if (n && v[0] == '\n') { v++; n--; }
    if (n && v[n - 1] == '\n') n--;
    if (type && !strcmp(type, "string")) { py_json_str(out, v, n); return; }
    char err[32];
    jval *j = n ? json_parse(v, n, err, sizeof(err)) : NULL;
    if (j && (type || j->type != JSTR)) { py_json(out, j); json_free(j); return; }
    json_free(j);
    py_json_str(out, v, n);
}

char *chat_parser_finish(chat_parser *p, const char **dr, size_t *nr, const char **dc, size_t *nc) {
    size_t r0, c0;
    advance(p, true, &r0, nr, &c0, nc);
    *dr = p->reasoning ? p->reasoning + r0 : "";
    *dc = p->content ? p->content + c0 : "";
    if (!p->tool_len) return NULL;
    // tool_raw: <tool_call>\n<function=NAME>\n<parameter=P>\nVALUE\n</parameter>\n</function>\n</tool_call> ...
    sbuf out = {0};
    sb_puts(&out, "[");
    const char *s = p->tool_raw, *end = p->tool_raw + p->tool_len;
    int idx = 0;
    const char *fnp;
    while ((fnp = memstr(s, (size_t)(end - s), "<function="))) {
        const char *name = fnp + strlen("<function=");
        const char *gt = memstr(name, (size_t)(end - name), ">");
        if (!gt) break;
        const size_t nl = (size_t)(gt - name);
        const char *body = gt + 1;
        const char *fend = memstr(body, (size_t)(end - body), "</function>");
        if (!fend) fend = end;
        sbuf args = {0};
        sb_puts(&args, "{");
        bool first = true;
        const char *q = body;
        const char *pp;
        while ((pp = memstr(q, (size_t)(fend - q), "<parameter="))) {
            const char *pn = pp + strlen("<parameter=");
            const char *pg = memstr(pn, (size_t)(fend - pn), ">");
            if (!pg) break;
            const char *v = pg + 1;
            const char *ve = memstr(v, (size_t)(fend - v), "</parameter>");
            if (!ve) ve = fend;
            if (!first) sb_puts(&args, ", ");
            first = false;
            py_json_str(&args, pn, (size_t)(pg - pn));
            sb_puts(&args, ": ");
            put_value(&args, v, (size_t)(ve - v), param_type(p->tools, name, nl, pn, (size_t)(pg - pn)));
            q = ve < fend ? ve + strlen("</parameter>") : fend;
        }
        sb_puts(&args, "}");
        char head[160];
        snprintf(head, sizeof(head), "%s{\"index\": %d, \"id\": \"call_%d_%08x\", \"type\": \"function\", \"function\": {\"name\": ",
                 idx ? ", " : "", idx, idx, (unsigned)(inz_now_ns() & 0xffffffff));
        sb_puts(&out, head);
        py_json_str(&out, name, nl);
        sb_puts(&out, ", \"arguments\": ");
        py_json_str(&out, args.b, args.n);
        sb_puts(&out, "}}");
        free(args.b);
        idx++;
        s = fend < end ? fend + strlen("</function>") : end;
    }
    sb_puts(&out, "]");
    if (idx == 0) { free(out.b); return NULL; }
    return out.b;
}

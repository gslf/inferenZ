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

// Jinja's `trim` (Python str.strip()) of a string
static void trim_span(const char *p, size_t n, const char **a, size_t *len) {
    size_t i = 0;
    while (i < n && is_ws(p[i])) i++;
    while (n > i && is_ws(p[n - 1])) n--;
    *a = p + i;
    *len = n - i;
}

// Jinja's `| string`: Python str() of a JSON value (True/False/None for the literals, json for containers)
static void py_str(sbuf *s, const jval *v) {
    if (!v || v->type == JNULL) { sb_puts(s, "None"); return; }
    if (v->type == JSTR) { sb_put(s, v->str, v->len); return; }
    if (v->type == JBOOL) { sb_puts(s, v->b ? "True" : "False"); return; }
    if (v->type == JNUM) {
        // the parsed JSON keeps no int/float distinction: integral values print as Python ints
        if (v->num == floor(v->num) && fabs(v->num) < 1e16) { py_num(s, v->num); return; }
        char tmp[64];
        for (int prec = 1; prec <= 17; prec++) {  // Python repr of a float: shortest round trip, ".0" when integral
            snprintf(tmp, sizeof(tmp), "%.*g", prec, v->num);
            if (strtod(tmp, NULL) == v->num) break;
        }
        if (!strpbrk(tmp, ".eEn")) strcat(tmp, ".0");
        sb_puts(s, tmp);
        return;
    }
    py_json(s, v);  // Python would print a dict / list repr; templates only stringify scalars here
}

// Effort: "none" | "minimal" | "off" -> thinking disabled; anything else (default) -> thinking on (Laguna has no
// intermediate level).
static void resolve_effort(const char *effort, bool *thinking) {
    *thinking = !(effort && (!strcmp(effort, "none") || !strcmp(effort, "minimal") || !strcmp(effort, "off")));
}

bool chat_thinking_enabled(const char *effort) {
    bool t;
    resolve_effort(effort, &t);
    return t;
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

#define SYS_DEFAULT "You are a helpful, conversationally-fluent assistant made by Poolside. You are here to be helpful to users " \
                    "through natural language conversations."

// C port of chat_template.jinja (enable_thinking from the effort, preserve_thinking false). The template renders
// content-part lists as empty strings; here their text parts are joined, which is the only deliberate difference.
char *chat_render(const jval *messages, const jval *tools, const char *effort, bool add_generation_prompt) {
    sbuf s = {0};
    bool thinking;
    resolve_effort(effort, &thinking);
    const size_t nmsg = messages && messages->type == JARR ? messages->len : 0;
    const jval *first = json_at(messages, 0);
    const bool first_system = first && !strcmp(role_of(first), "system");
    const bool have_tools = tools && tools->type == JARR && tools->len;
    sbuf sys = {0};
    if (first_system) render_content(&sys, json_get(first, "content"));
    else sb_puts(&sys, SYS_DEFAULT);
    const char *sp = sys.b ? sys.b : "";
    const char *ta;
    size_t tl;
    trim_span(sp, sys.n, &ta, &tl);
    sb_puts(&s, "〈|EOS|〉");
    if (tl || have_tools || thinking) {
        sb_puts(&s, "<system>");
        if (tl) {
            size_t n = sys.n;
            while (n && is_ws(sp[n - 1])) n--;  // rstrip()
            sb_put(&s, sp, n);
            if (have_tools) sb_puts(&s, "\n\n");
        }
        if (have_tools) {
            sb_puts(&s, "### Tools\n\nYou may call functions to assist with the user query.\nAll available function signatures are "
                        "listed below:\n<available_tools>\n");
            for (const jval *t = tools->child; t; t = t->next) { py_json(&s, t); sb_puts(&s, "\n"); }
            sb_puts(&s, "</available_tools>");
        }
        sb_puts(&s, "</system>\n");
    }
    free(sys.b);
    for (size_t i = first_system ? 1 : 0; i < nmsg; i++) {
        const jval *m = json_at(messages, i);
        const char *role = role_of(m);
        sbuf cb = {0};
        render_content(&cb, json_get(m, "content"));
        if (!cb.b) sb_puts(&cb, "");
        if (!strcmp(role, "user")) {
            sb_puts(&s, "<user>"); sb_put(&s, cb.b, cb.n); sb_puts(&s, "</user>\n");
        } else if (!strcmp(role, "assistant")) {
            sb_puts(&s, "<assistant>");
            const jval *r = json_get(m, "reasoning");
            if (!r || r->type != JSTR) r = json_get(m, "reasoning_content");
            if (thinking) {
                sb_puts(&s, THINK_BEGIN);
                if (r && r->type == JSTR) sb_put(&s, r->str, r->len);
                sb_puts(&s, THINK_END);
            } else sb_puts(&s, THINK_END);
            sb_put(&s, cb.b, cb.n);
            const jval *tc = json_get(m, "tool_calls");
            for (const jval *call = tc && tc->type == JARR ? tc->child : NULL; call; call = call->next) {
                const jval *fn = json_get(call, "function") ? json_get(call, "function") : call;
                sb_puts(&s, TOOLCALL_BEGIN);
                py_str(&s, json_get(fn, "name"));
                jval *owned;
                const jval *args = args_object(json_get(fn, "arguments"), &owned);
                for (const jval *a = args ? args->child : NULL; a; a = a->next) {
                    sb_puts(&s, "<arg_key>"); sb_puts(&s, a->key); sb_puts(&s, "</arg_key><arg_value>");
                    if (a->type == JSTR) sb_put(&s, a->str, a->len);
                    else py_json(&s, a);
                    sb_puts(&s, "</arg_value>");
                }
                json_free(owned);
                sb_puts(&s, TOOLCALL_END);
            }
            sb_puts(&s, "</assistant>\n");
        } else if (!strcmp(role, "tool")) {
            sb_puts(&s, "<tool_response>"); sb_put(&s, cb.b, cb.n); sb_puts(&s, "</tool_response>\n");
        } else if (!strcmp(role, "system")) {
            sb_puts(&s, "<system>"); sb_put(&s, cb.b, cb.n); sb_puts(&s, "</system>\n");
        }
        free(cb.b);
    }
    if (add_generation_prompt) sb_puts(&s, thinking ? "<assistant>" THINK_BEGIN : "<assistant>" THINK_END);
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
    // tool_raw: <tool_call>NAME<arg_key>K</arg_key><arg_value>V</arg_value>...</tool_call> ...
    sbuf out = {0};
    sb_puts(&out, "[");
    const char *s = p->tool_raw, *end = p->tool_raw + p->tool_len;
    int idx = 0;
    const char *tp;
    while ((tp = memstr(s, (size_t)(end - s), TOOLCALL_BEGIN))) {
        const char *name = tp + strlen(TOOLCALL_BEGIN);
        const char *tend = memstr(name, (size_t)(end - name), TOOLCALL_END);
        if (!tend) tend = end;
        const char *ak = memstr(name, (size_t)(tend - name), "<arg_key>");
        const char *nend = ak ? ak : tend;
        const char *na;
        size_t nl;
        trim_span(name, (size_t)(nend - name), &na, &nl);
        sbuf args = {0};
        sb_puts(&args, "{");
        bool first = true;
        for (const char *q = ak; q;) {
            const char *k = q + strlen("<arg_key>");
            const char *ke = memstr(k, (size_t)(tend - k), "</arg_key>");
            if (!ke) break;
            const char *v = memstr(ke, (size_t)(tend - ke), "<arg_value>");
            if (!v) break;
            v += strlen("<arg_value>");
            const char *ve = memstr(v, (size_t)(tend - v), "</arg_value>");
            if (!ve) ve = tend;
            if (!first) sb_puts(&args, ", ");
            first = false;
            py_json_str(&args, k, (size_t)(ke - k));
            sb_puts(&args, ": ");
            put_value(&args, v, (size_t)(ve - v), param_type(p->tools, na, nl, k, (size_t)(ke - k)));
            q = ve < tend ? memstr(ve, (size_t)(tend - ve), "<arg_key>") : NULL;
        }
        sb_puts(&args, "}");
        char head[160];
        snprintf(head, sizeof(head), "%s{\"index\": %d, \"id\": \"call_%d_%08x\", \"type\": \"function\", \"function\": {\"name\": ",
                 idx ? ", " : "", idx, idx, (unsigned)(inz_now_ns() & 0xffffffff));
        sb_puts(&out, head);
        py_json_str(&out, na, nl);
        sb_puts(&out, ", \"arguments\": ");
        py_json_str(&out, args.b, args.n);
        sb_puts(&out, "}}");
        free(args.b);
        idx++;
        s = tend < end ? tend + strlen(TOOLCALL_END) : end;
    }
    sb_puts(&out, "]");
    if (idx == 0) { free(out.b); return NULL; }
    return out.b;
}

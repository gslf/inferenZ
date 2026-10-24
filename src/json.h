// Minimal JSON DOM: parse into a tree of nodes, query, and a streaming writer.
// Used for safetensors headers, HF config/tokenizer files and the HTTP API.
#ifndef INZ_JSON_H
#define INZ_JSON_H

#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>

typedef enum { JNULL, JBOOL, JNUM, JSTR, JARR, JOBJ } jtype;

typedef struct jval {
    jtype type;
    bool b;
    double num;
    char *str;           // JSTR: decoded UTF-8 (NUL terminated); also key storage for members
    size_t len;          // JSTR: byte length; JARR/JOBJ: number of children
    struct jval *child;  // first child (arrays/objects)
    struct jval *next;   // next sibling
    char *key;           // member key when parent is an object
} jval;

jval *json_parse(const char *text, size_t len, char *err, size_t errlen);
void json_free(jval *v);

jval *json_get(const jval *obj, const char *key);          // object member or NULL
jval *json_at(const jval *arr, size_t idx);                // array element or NULL
const char *json_str(const jval *v, const char *def);
double json_num(const jval *v, double def);
int64_t json_int(const jval *v, int64_t def);
bool json_bool(const jval *v, bool def);
jval *json_path(const jval *root, const char *dotted);      // "a.b.c"

// Streaming writer into a growable buffer.
typedef struct {
    char *buf;
    size_t len, cap;
    int depth;
    bool need_comma[64];
} jw;

void jw_init(jw *w);
void jw_free(jw *w);
void jw_obj(jw *w);
void jw_obj_end(jw *w);
void jw_arr(jw *w);
void jw_arr_end(jw *w);
void jw_key(jw *w, const char *k);
void jw_str(jw *w, const char *s);
void jw_strn(jw *w, const char *s, size_t n);
void jw_num(jw *w, double v);
void jw_int(jw *w, int64_t v);
void jw_bool(jw *w, bool v);
void jw_null(jw *w);
void jw_raw(jw *w, const char *raw);  // pre-serialized JSON value
// convenience: key + value
void jw_kstr(jw *w, const char *k, const char *v);
void jw_knum(jw *w, const char *k, double v);
void jw_kint(jw *w, const char *k, int64_t v);
void jw_kbool(jw *w, const char *k, bool v);

void json_escape_append(char **buf, size_t *len, size_t *cap, const char *s, size_t n);

#endif

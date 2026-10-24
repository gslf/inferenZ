// Small HTTP/1.1 server: one thread per connection, keep-alive, SSE streaming.
#ifndef INZ_HTTP_H
#define INZ_HTTP_H

#include <stdbool.h>
#include <stddef.h>

typedef struct {
    char method[16];
    char path[1024];
    char query[1024];
    char *headers;      // raw header block (NUL terminated)
    char *body;
    size_t body_len;
    char remote[64];
} http_req;

typedef struct http_conn http_conn;
typedef void (*http_handler)(http_req *req, http_conn *c, void *ud);

const char *http_header(const http_req *r, const char *name, char *buf, size_t buflen);
bool http_send(http_conn *c, int status, const char *ctype, const void *body, size_t len);
bool http_send_json(http_conn *c, int status, const char *json);
bool http_sse_begin(http_conn *c);
bool http_sse_data(http_conn *c, const char *data);        // "data: <data>\n\n"
bool http_sse_event(http_conn *c, const char *event, const char *data);
bool http_write_raw(http_conn *c, const void *p, size_t n);
bool http_conn_alive(http_conn *c);                         // false once client disconnected
void http_conn_close_after(http_conn *c);

int http_serve(const char *host, int port, http_handler h, void *ud, volatile int *stop);

#endif

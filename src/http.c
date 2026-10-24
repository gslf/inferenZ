#define _GNU_SOURCE
#include "http.h"

#include <arpa/inet.h>
#include <errno.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <poll.h>
#include <pthread.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>
#include <sys/socket.h>
#include <unistd.h>

struct http_conn {
    int fd;
    bool close_after;
    bool dead;
    http_handler h;
    void *ud;
    char remote[64];
};

static const char *status_text(int s) {
    switch (s) {
    case 200: return "OK";
    case 204: return "No Content";
    case 400: return "Bad Request";
    case 401: return "Unauthorized";
    case 404: return "Not Found";
    case 405: return "Method Not Allowed";
    case 413: return "Payload Too Large";
    case 429: return "Too Many Requests";
    case 500: return "Internal Server Error";
    case 503: return "Service Unavailable";
    default: return "OK";
    }
}

bool http_write_raw(http_conn *c, const void *p, size_t n) {
    const char *b = p;
    while (n && !c->dead) {
        ssize_t w = send(c->fd, b, n, MSG_NOSIGNAL);
        if (w < 0) {
            if (errno == EINTR) continue;
            c->dead = true;
            return false;
        }
        b += w;
        n -= (size_t)w;
    }
    return !c->dead;
}

bool http_conn_alive(http_conn *c) {
    if (c->dead) return false;
    struct pollfd pf = {.fd = c->fd, .events = POLLIN};
    if (poll(&pf, 1, 0) > 0) {
        if (pf.revents & (POLLHUP | POLLERR)) { c->dead = true; return false; }
        char b;
        ssize_t r = recv(c->fd, &b, 1, MSG_PEEK | MSG_DONTWAIT);
        if (r == 0) { c->dead = true; return false; }
    }
    return true;
}

void http_conn_close_after(http_conn *c) { c->close_after = true; }

static const char *CORS =
    "Access-Control-Allow-Origin: *\r\n"
    "Access-Control-Allow-Headers: *\r\n"
    "Access-Control-Allow-Methods: GET, POST, OPTIONS\r\n";

bool http_send(http_conn *c, int status, const char *ctype, const void *body, size_t len) {
    char hdr[1024];
    int n = snprintf(hdr, sizeof(hdr),
                     "HTTP/1.1 %d %s\r\nContent-Type: %s\r\nContent-Length: %zu\r\n%sConnection: %s\r\n\r\n",
                     status, status_text(status), ctype, len, CORS, c->close_after ? "close" : "keep-alive");
    return http_write_raw(c, hdr, (size_t)n) && (len == 0 || http_write_raw(c, body, len));
}

bool http_send_json(http_conn *c, int status, const char *json) {
    return http_send(c, status, "application/json", json, strlen(json));
}

bool http_sse_begin(http_conn *c) {
    c->close_after = true;
    char hdr[512];
    int n = snprintf(hdr, sizeof(hdr),
                     "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nCache-Control: no-cache\r\n"
                     "X-Accel-Buffering: no\r\n%sConnection: close\r\n\r\n", CORS);
    return http_write_raw(c, hdr, (size_t)n);
}

bool http_sse_data(http_conn *c, const char *data) {
    return http_write_raw(c, "data: ", 6) && http_write_raw(c, data, strlen(data)) && http_write_raw(c, "\n\n", 2);
}

bool http_sse_event(http_conn *c, const char *event, const char *data) {
    return http_write_raw(c, "event: ", 7) && http_write_raw(c, event, strlen(event)) &&
           http_write_raw(c, "\n", 1) && http_sse_data(c, data);
}

const char *http_header(const http_req *r, const char *name, char *buf, size_t buflen) {
    size_t nl = strlen(name);
    const char *p = r->headers;
    while (p && *p) {
        const char *eol = strstr(p, "\r\n");
        if (!eol) eol = p + strlen(p);
        if ((size_t)(eol - p) > nl && !strncasecmp(p, name, nl) && p[nl] == ':') {
            const char *v = p + nl + 1;
            while (*v == ' ' || *v == '\t') v++;
            size_t vl = (size_t)(eol - v);
            if (vl >= buflen) vl = buflen - 1;
            memcpy(buf, v, vl);
            buf[vl] = 0;
            return buf;
        }
        p = *eol ? eol + 2 : eol;
    }
    return NULL;
}

#define MAX_HDR (64 * 1024)
#define MAX_BODY (64 * 1024 * 1024)

static void *conn_thread(void *arg) {
    http_conn *c = arg;
    char *buf = malloc(MAX_HDR + 1);
    size_t have = 0;
    for (;;) {
        // read until end of headers
        char *hend = NULL;
        while (!(hend = memmem(buf, have, "\r\n\r\n", 4))) {
            if (have >= MAX_HDR) goto out;
            struct pollfd pf = {.fd = c->fd, .events = POLLIN};
            int pr = poll(&pf, 1, 120000);  // keep-alive idle timeout
            if (pr <= 0) goto out;
            ssize_t r = recv(c->fd, buf + have, MAX_HDR - have, 0);
            if (r <= 0) goto out;
            have += (size_t)r;
        }
        size_t hlen = (size_t)(hend - buf) + 4;
        buf[hlen - 2] = 0;
        http_req req;
        memset(&req, 0, sizeof(req));
        snprintf(req.remote, sizeof(req.remote), "%s", c->remote);
        char target[2048] = {0};
        if (sscanf(buf, "%15s %2047s", req.method, target) != 2) goto out;
        char *q = strchr(target, '?');
        if (q) { *q = 0; snprintf(req.query, sizeof(req.query), "%s", q + 1); }
        snprintf(req.path, sizeof(req.path), "%s", target);
        char *line_end = strstr(buf, "\r\n");
        req.headers = line_end ? line_end + 2 : buf + hlen - 2;
        char tmp[64];
        size_t clen = 0;
        if (http_header(&req, "Content-Length", tmp, sizeof(tmp))) clen = strtoull(tmp, NULL, 10);
        if (clen > MAX_BODY) { http_conn_close_after(c); http_send_json(c, 413, "{\"error\":\"body too large\"}"); goto out; }
        c->close_after = false;
        if (http_header(&req, "Connection", tmp, sizeof(tmp)) && !strcasecmp(tmp, "close")) c->close_after = true;
        // copy headers since buf may be reused
        char *hcopy = strdup(req.headers);
        req.headers = hcopy;
        req.body = malloc(clen + 1);
        size_t extra = have - hlen;
        size_t take = extra < clen ? extra : clen;
        memcpy(req.body, buf + hlen, take);
        size_t got = take;
        while (got < clen) {
            ssize_t r = recv(c->fd, req.body + got, clen - got, 0);
            if (r <= 0) { free(req.body); free(hcopy); goto out; }
            got += (size_t)r;
        }
        req.body[clen] = 0;
        req.body_len = clen;
        // keep pipelined leftovers
        size_t consumed = hlen + take;
        memmove(buf, buf + consumed, have - consumed);
        have -= consumed;

        if (!strcmp(req.method, "OPTIONS")) http_send(c, 204, "text/plain", "", 0);
        else c->h(&req, c, c->ud);
        free(req.body);
        free(hcopy);
        if (c->close_after || c->dead) break;
    }
out:
    free(buf);
    close(c->fd);
    free(c);
    return NULL;
}

int http_serve(const char *host, int port, http_handler h, void *ud, volatile int *stop) {
    signal(SIGPIPE, SIG_IGN);
    int s = socket(AF_INET, SOCK_STREAM, 0);
    int one = 1;
    setsockopt(s, SOL_SOCKET, SO_REUSEADDR, &one, sizeof(one));
    struct sockaddr_in a = {.sin_family = AF_INET, .sin_port = htons((uint16_t)port)};
    if (inet_pton(AF_INET, host, &a.sin_addr) != 1) { fprintf(stderr, "bad host %s\n", host); return -1; }
    if (bind(s, (struct sockaddr *)&a, sizeof(a)) < 0) { perror("bind"); return -1; }
    if (listen(s, 64) < 0) { perror("listen"); return -1; }
    while (!stop || !*stop) {
        struct pollfd pf = {.fd = s, .events = POLLIN};
        if (poll(&pf, 1, 500) <= 0) continue;
        struct sockaddr_in ca;
        socklen_t cl = sizeof(ca);
        int fd = accept(s, (struct sockaddr *)&ca, &cl);
        if (fd < 0) continue;
        setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof(one));
        http_conn *c = calloc(1, sizeof(*c));
        c->fd = fd;
        c->h = h;
        c->ud = ud;
        inet_ntop(AF_INET, &ca.sin_addr, c->remote, sizeof(c->remote));
        pthread_t t;
        pthread_attr_t at;
        pthread_attr_init(&at);
        pthread_attr_setdetachstate(&at, PTHREAD_CREATE_DETACHED);
        pthread_attr_setstacksize(&at, 1 << 20);
        if (pthread_create(&t, &at, conn_thread, c) != 0) { close(fd); free(c); }
        pthread_attr_destroy(&at);
    }
    close(s);
    return 0;
}

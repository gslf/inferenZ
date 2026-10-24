// Reads NUL-separated UTF-8 strings from stdin, prints token ids per string (one line each).
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "../src/tokenizer.h"
#include "../src/util.h"

int main(int argc, char **argv) {
    if (argc < 2) { fprintf(stderr, "usage: %s tokenizer.json < strings\n", argv[0]); return 1; }
    char err[256];
    double t0 = inz_now();
    tokenizer *t = tok_load_hf_json(argv[1], err, sizeof(err));
    if (!t) { fprintf(stderr, "load: %s\n", err); return 1; }
    fprintf(stderr, "loaded vocab %d in %.2fs\n", tok_vocab(t), inz_now() - t0);
    size_t cap = 1 << 20, n = 0;
    char *buf = malloc(cap);
    int c;
    int32_t *ids = malloc(sizeof(int32_t) * (1 << 20));
    double tenc = 0; size_t nbytes = 0, ntok = 0;
    while ((c = getchar()) != EOF) {
        if (c == 0) {
            double a = inz_now();
            int k = tok_encode(t, buf, n, argc > 2, ids, 1 << 20);
            tenc += inz_now() - a; nbytes += n; ntok += k;
            for (int i = 0; i < k; i++) printf(i ? " %d" : "%d", ids[i]);
            printf("\n");
            n = 0;
        } else {
            if (n + 1 >= cap) { cap *= 2; buf = realloc(buf, cap); }
            buf[n++] = (char)c;
        }
    }
    fprintf(stderr, "encoded %zu bytes -> %zu tokens in %.3fs (%.1f MB/s)\n", nbytes, ntok, tenc, nbytes / tenc / 1e6);
    tok_free(t);
    return 0;
}

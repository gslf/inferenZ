// cachesim: replay an inferenZ routing trace (--route-trace) against expert cache policies.
//
//   cc -O2 -o tools/cachesim tools/cachesim.c -lm
//   tools/cachesim trace.txt [-e experts_per_layer] [-c coarse_fraction] [capacity_pct ...]
//   (default: 512 experts per layer; capacities 10%..50% of all experts)
//
// Accesses are replayed token-major (sorted by token, then layer), as decode issues them. A global cache shared
// by all layers is simulated, as in the engine. Two tables are printed:
//
//   1. miss rate of single-level caches: LRU, LRFU at several decays (the engine's score), Belady's OPT (the
//      offline optimum, a bound for any online policy);
//   2. SSD traffic of the two-pool planar cache (Q3P): the same memory holds more experts when only a fraction
//      of them keep their fine planes. A full miss reads 1 blob, a hit on a coarse-only resident expert reads
//      the fine planes (1 - c blobs). Traffic is reported in blobs per access, so frac = 1 equals the miss rate.
#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct { uint32_t t; int32_t key; uint32_t ord; float w; } acc;  // w: normalized routing weight

static acc *A;
static size_t na, cap_a;
static int n_layer = 0, n_expert = 512, nk;  // Qwen3.8-Flash-Next: 512 experts per layer (-e N to change)

static void push(uint32_t t, int32_t key) {
    if (na == cap_a) { cap_a = cap_a ? cap_a * 2 : 1 << 20; A = realloc(A, sizeof(acc) * cap_a); }
    A[na] = (acc){t, key, (uint32_t)na, 0.25f};
    na++;
}


static int cmp_acc(const void *a, const void *b) {
    const acc *x = a, *y = b;
    if (x->t != y->t) return x->t < y->t ? -1 : 1;
    int lx = x->key / n_expert, ly = y->key / n_expert;
    if (lx != ly) return lx < ly ? -1 : 1;
    return x->ord < y->ord ? -1 : x->ord > y->ord;
}

/* ---- indexed min-heap over keys ---- */
typedef struct { int n; int32_t *h; int32_t *pos; double *pri; } heap;

static void hp_init(heap *q) {
    q->n = 0;
    q->h = malloc(sizeof(int32_t) * nk);
    q->pos = malloc(sizeof(int32_t) * nk);
    q->pri = calloc(nk, sizeof(double));
    for (int i = 0; i < nk; i++) q->pos[i] = -1;
}
static void hp_free(heap *q) { free(q->h); free(q->pos); free(q->pri); }
static void hp_swap(heap *q, int a, int b) {
    int32_t t = q->h[a]; q->h[a] = q->h[b]; q->h[b] = t;
    q->pos[q->h[a]] = a; q->pos[q->h[b]] = b;
}
static void hp_up(heap *q, int i) { while (i > 0 && q->pri[q->h[(i - 1) / 2]] > q->pri[q->h[i]]) { hp_swap(q, i, (i - 1) / 2); i = (i - 1) / 2; } }
static void hp_down(heap *q, int i) {
    for (;;) {
        int l = 2 * i + 1, r = l + 1, m = i;
        if (l < q->n && q->pri[q->h[l]] < q->pri[q->h[m]]) m = l;
        if (r < q->n && q->pri[q->h[r]] < q->pri[q->h[m]]) m = r;
        if (m == i) return;
        hp_swap(q, i, m);
        i = m;
    }
}
static void hp_set(heap *q, int32_t k, double p) {
    q->pri[k] = p;
    if (q->pos[k] < 0) { q->h[q->n] = k; q->pos[k] = q->n; q->n++; hp_up(q, q->n - 1); }
    else { hp_up(q, q->pos[k]); hp_down(q, q->pos[k]); }
}
static int32_t hp_pop(heap *q) {
    int32_t k = q->h[0];
    hp_swap(q, 0, q->n - 1);
    q->n--;
    q->pos[k] = -1;
    if (q->n) hp_down(q, 0);
    return k;
}
static void hp_del(heap *q, int32_t k) {
    int i = q->pos[k];
    if (i < 0) return;
    hp_swap(q, i, q->n - 1);
    q->n--;
    q->pos[k] = -1;
    if (i < q->n) { hp_up(q, i); hp_down(q, i); }
}

/* ---- scores ----
 * LRFU: crf(t) = crf(tk) * 2^(-lambda (t - tk)). All keys decay at the same rate, so the order of
 * log2(crf(tk)) + lambda * tk never changes between touches: it is the heap priority.
 * mode: 0 = LRU, 1 = LRFU(lambda), 2 = OPT (priority = -next use). */
typedef struct { int mode; double lambda; int admit; } policy;

static double *g_crf; static uint32_t *g_tk; static size_t *g_next;

static double touch(const policy *p, size_t i) {
    int32_t k = A[i].key;
    uint32_t t = A[i].t;
    if (p->mode == 0) return (double)i;
    if (p->mode == 2) return -(double)g_next[i];
    double c = g_crf[k] * exp2(-p->lambda * (double)(t - g_tk[k])) + 1.0;
    g_crf[k] = c;
    g_tk[k] = t;
    return log2(c) + p->lambda * (double)t;
}

static void scores_reset(void) {
    memset(g_crf, 0, sizeof(double) * nk);
    memset(g_tk, 0, sizeof(uint32_t) * nk);
}

// Two-pool cache: `slots` experts resident (coarse part at least), `nfine` of them with fine planes.
// Returns SSD traffic in blobs per access; *miss gets the full-miss rate.
static double sim(const policy *p, int slots, int nfine, double cfrac, double *miss) {
    heap res, fine;
    hp_init(&res); hp_init(&fine);
    char *in = calloc(nk, 1), *fn = calloc(nk, 1);
    scores_reset();
    double bytes = 0;
    uint64_t misses = 0;
    for (size_t i = 0; i < na; i++) {
        int32_t k = A[i].key;
        bool need_fine = false;
        if (!in[k] && p->admit && res.n == slots && p->mode == 1) {
            // compare decayed scores: newcomer after this touch vs the current minimum
            double pr = touch(p, i);
            int32_t v = res.h[0];
            misses++;
            bytes += 1.0;
            if (pr <= res.pri[v]) continue;  // bypass
            hp_pop(&res);
            in[v] = 0;
            if (fn[v]) { fn[v] = 0; hp_del(&fine, v); }
            in[k] = 1;
            hp_set(&res, k, pr);
            if (nfine > 0) { if (fine.n == nfine) { int32_t f2 = hp_pop(&fine); fn[f2] = 0; } fn[k] = 1; hp_set(&fine, k, pr); }
            continue;
        }
        if (!in[k]) {
            misses++;
            bytes += 1.0;
            if (res.n == slots) {
                int32_t v = hp_pop(&res);
                in[v] = 0;
                if (fn[v]) { fn[v] = 0; hp_del(&fine, v); }
            }
            in[k] = 1;
            need_fine = true;
        } else if (!fn[k]) {
            bytes += 1.0 - cfrac;
            need_fine = true;
        }
        double pr = touch(p, i);
        hp_set(&res, k, pr);
        if (need_fine && nfine > 0) {
            if (fine.n == nfine) { int32_t v = hp_pop(&fine); fn[v] = 0; }
            fn[k] = 1;
        }
        if (fn[k]) hp_set(&fine, k, pr);
    }
    hp_free(&res); hp_free(&fine);
    free(in); free(fn);
    if (miss) *miss = (double)misses / na;
    return bytes / na;
}

// HOBBIT-style coarse misses on a single-level cache of `slots` full-size slots: a miss whose normalized
// weight is below tau reads only the coarse part (cfrac) and is computed at 2-bit precision; a later use with
// weight >= tau upgrades it (1 - cfrac). Returns blobs per access; *cshare = share of routed weight computed coarse.
static double sim_coarse(double lambda, int slots, double tau, double cfrac, double *cshare) {
    heap res;
    hp_init(&res);
    char *lvl = calloc(nk, 1);  // 0 absent, 1 coarse, 2 full
    scores_reset();
    policy P = {1, lambda, 0};
    double bytes = 0, wc = 0, wt = 0;
    for (size_t i = 0; i < na; i++) {
        int32_t k = A[i].key;
        const bool low = A[i].w < tau;
        wt += A[i].w;
        if (!lvl[k]) {
            if (res.n == slots) { int32_t v = hp_pop(&res); lvl[v] = 0; }
            lvl[k] = low ? 1 : 2;
            bytes += low ? cfrac : 1.0;
        } else if (lvl[k] == 1 && !low) {
            lvl[k] = 2;
            bytes += 1.0 - cfrac;
        }
        if (lvl[k] == 1) wc += A[i].w;
        hp_set(&res, k, touch(&P, i));
    }
    hp_free(&res);
    free(lvl);
    *cshare = wt > 0 ? wc / wt : 0;
    return bytes / na;
}

int main(int argc, char **argv) {
    if (argc < 2) { fprintf(stderr, "usage: %s trace.txt [-e experts_per_layer] [-c coarse_fraction] [capacity_pct...]\n", argv[0]); return 1; }
    double cfrac = 0.69;
    int ai = 2;
    for (;;) {
        if (argc > ai + 1 && !strcmp(argv[ai], "-c")) { cfrac = atof(argv[ai + 1]); ai += 2; continue; }
        if (argc > ai + 1 && !strcmp(argv[ai], "-e")) { n_expert = atoi(argv[ai + 1]); ai += 2; continue; }
        break;
    }
    FILE *f = fopen(argv[1], "r");
    if (!f) { perror(argv[1]); return 1; }
    char line[1024];
    int min_layer = 1 << 30, max_layer = 0;
    while (fgets(line, sizeof(line), f)) {
        char *p = line;
        unsigned long t = strtoul(p, &p, 10);
        int layer = (int)strtol(p, &p, 10);
        if (layer < min_layer) min_layer = layer;
        if (layer > max_layer) max_layer = layer;
        size_t first = na;
        for (;;) {
            char *q;
            long e = strtol(p, &q, 10);
            if (q == p) break;
            p = q;
            push((uint32_t)t, layer * 1000 + (int)e);  // remapped below
        }
        char *bar = strchr(p, '|');
        if (bar) {
            float wsum = 0;
            p = bar + 1;
            for (size_t j = first; j < na; j++) { A[j].w = strtof(p, &p); wsum += A[j].w; }
            if (wsum > 0) for (size_t j = first; j < na; j++) A[j].w /= wsum;
        }
    }
    fclose(f);
    if (!na) { fprintf(stderr, "empty trace\n"); return 1; }
    n_layer = max_layer - min_layer + 1;
    nk = n_layer * n_expert;
    for (size_t i = 0; i < na; i++) {
        int layer = A[i].key / 1000, e = A[i].key % 1000;
        A[i].key = (layer - min_layer) * n_expert + e;
    }
    qsort(A, na, sizeof(acc), cmp_acc);
    g_crf = calloc(nk, sizeof(double));
    g_tk = calloc(nk, sizeof(uint32_t));
    g_next = malloc(sizeof(size_t) * na);
    {
        size_t *np = malloc(sizeof(size_t) * nk);
        for (int x = 0; x < nk; x++) np[x] = SIZE_MAX / 2;
        for (size_t i = na; i-- > 0;) { g_next[i] = np[A[i].key]; np[A[i].key] = i; }
        free(np);
    }
    printf("trace: %zu accesses, %u tokens, %d layers x %d experts = %d experts, coarse part %.0f%%\n", na,
           A[na - 1].t - A[0].t + 1, n_layer, n_expert, nk, 100 * cfrac);
    int pcts[32], nc = 0;
    for (int i = ai; i < argc && nc < 32; i++) pcts[nc++] = atoi(argv[i]);
    if (!nc) for (int pct = 10; pct <= 50; pct += 10) pcts[nc++] = pct;

    const double lambdas[] = {0.0, 0.002, 0.01, 0.05, 0.2};
    printf("\nmiss rate, single-level cache\n%6s %7s", "cap", "LRU");
    for (int j = 0; j < 5; j++) printf("  LRFU%-5g", lambdas[j]);
    printf(" %7s %9s %9s\n", "OPT", "adm.01", "adm.05");
    for (int i = 0; i < nc; i++) {
        int slots = nk * pcts[i] / 100;
        double m;
        printf("%5d%% ", pcts[i]);
        sim(&(policy){0, 0}, slots, slots, cfrac, &m);
        printf("%6.1f%%", 100 * m);
        for (int j = 0; j < 5; j++) { sim(&(policy){1, lambdas[j]}, slots, slots, cfrac, &m); printf("  %8.1f%%", 100 * m); }
        sim(&(policy){2, 0}, slots, slots, cfrac, &m);
        printf(" %6.1f%%", 100 * m);
        sim(&(policy){1, 0.01, 1}, slots, slots, cfrac, &m);
        printf(" %8.1f%%", 100 * m);
        sim(&(policy){1, 0.05, 1}, slots, slots, cfrac, &m);
        printf(" %8.1f%%\n", 100 * m);
    }

    // Speculative verification of w tokens reads the union of their experts once: unique experts per
    // layer per window, and the miss traffic per token of an LRFU cache that sees one access per unique expert.
    printf("\nverify windows: unique experts per layer per step, misses per token (LRFU 0.01, step-level accesses)\n");
    printf("%6s %8s", "window", "unique");
    for (int i = 0; i < nc; i++) printf("  miss/tok@%d%%", pcts[i]);
    printf("\n");
    for (int w = 1; w <= 8; w *= 2) {
        // build a deduplicated access stream: per window, per layer, each expert once (time = window index)
        acc *B = malloc(sizeof(acc) * na);
        size_t nb = 0;
        char *seen = calloc(nk, 1);
        int32_t *list = malloc(sizeof(int32_t) * nk);
        size_t i = 0;
        uint32_t t0 = A[0].t;
        uint64_t windows = 0;
        while (i < na) {
            uint32_t wi = (A[i].t - t0) / w;
            int nl = 0;
            size_t j = i;
            while (j < na && (A[j].t - t0) / w == wi) {
                if (!seen[A[j].key]) { seen[A[j].key] = 1; list[nl++] = A[j].key; B[nb++] = (acc){wi, A[j].key, (uint32_t)nb}; }
                j++;
            }
            for (int q = 0; q < nl; q++) seen[list[q]] = 0;
            windows++;
            i = j;
        }
        acc *saveA = A; size_t savena = na;
        size_t *savenext = g_next;
        A = B; na = nb;
        g_next = malloc(sizeof(size_t) * na);
        { size_t *np = malloc(sizeof(size_t) * nk); for (int x = 0; x < nk; x++) np[x] = SIZE_MAX / 2;
          for (size_t q = na; q-- > 0;) { g_next[q] = np[A[q].key]; np[A[q].key] = q; } free(np); }
        uint32_t ntok = saveA[savena - 1].t - t0 + 1;
        printf("%6d %8.2f", w, (double)nb / windows / n_layer);
        for (int c = 0; c < nc; c++) {
            int slots = nk * pcts[c] / 100;
            double m;
            sim(&(policy){1, 0.01 * w}, slots, slots, cfrac, &m);
            printf("  %11.2f", m * nb / ntok / n_layer);
        }
        printf("   (per layer)\n");
        free(g_next); g_next = savenext;
        A = saveA; na = savena;
        free(B); free(seen); free(list);
    }

    const double taus[] = {0.0, 0.15, 0.2, 0.25, 0.3, 1.1};
    printf("\ncoarse misses (LRFU 0.01): SSD blobs per access / share of routed weight computed at 2-bit\n%6s", "cap");
    for (int j = 0; j < 6; j++) printf("     tau=%-4.2f   ", taus[j]);
    printf("\n");
    for (int i = 0; i < nc; i++) {
        int slots = nk * pcts[i] / 100;
        printf("%5d%% ", pcts[i]);
        for (int j = 0; j < 6; j++) {
            double cs, b = sim_coarse(0.01, slots, taus[j], cfrac, &cs);
            printf("  %6.3f / %4.1f%%", b, 100 * cs);
        }
        printf("\n");
    }

    const double fracs[] = {1.0, 0.8, 0.6, 0.4, 0.2, 0.0};
    for (int pol = 0; pol < 2; pol++) {
        policy P = pol == 0 ? (policy){0, 0} : (policy){1, 0.05};
        printf("\nSSD blobs per access, two-pool cache (%s); memory as a %% of all experts at full size\n%6s",
               pol == 0 ? "LRU" : "LRFU 0.05", "mem");
        for (int j = 0; j < 6; j++) printf("   fine=%.1f", fracs[j]);
        printf("\n");
        for (int i = 0; i < nc; i++) {
            double budget = nk * pcts[i] / 100.0;  // in full blobs
            printf("%5d%% ", pcts[i]);
            for (int j = 0; j < 6; j++) {
                int slots = (int)(budget / (cfrac + fracs[j] * (1 - cfrac)));
                if (slots > nk) slots = nk;
                int nfine = (int)((budget - slots * cfrac) / (1 - cfrac));
                if (nfine > slots) nfine = slots;
                if (nfine < 0) nfine = 0;
                printf("  %8.3f", sim(&P, slots, nfine, cfrac, NULL));
            }
            printf("\n");
        }
    }
    return 0;
}

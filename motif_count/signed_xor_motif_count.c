/*
 * signed_xor_motif_count.c
 *
 * 8-node signed-XOR motif search — COUNT ONLY, no file output.
 * OpenMP parallelised, targets DGX Spark (ARM, Ubuntu 24.04, 20 cores).
 *
 * Motif definition (8 nodes, 12 edges):
 *
 *   Nodes:
 *     E1, E2, E3, E4  — excitatory (label starts with 'e')
 *     INH             — inhibitory, one of 12 target types
 *     XOR             — excitatory (label starts with 'e')
 *     F+              — excitatory (label starts with 'e')
 *     F-              — inhibitory (label starts with 'i')
 *
 *   Edges:
 *     EXC: E1→E2, E1→INH, E3→E4, E3→INH   (inputs to core)
 *     INH: INH→E2, INH→E4                   (lateral inhibition)
 *     EXC: E2→XOR, E4→XOR                   (convergence onto XOR)
 *     EXC: E2→F+,  E4→F-                    (signed feedback)
 *     EXC: XOR→F+, XOR→F-                   (XOR drives both feedback nodes)
 *
 * Search strategy (anchored on INH):
 *   1. For each INH neuron of the 12 target types
 *   2. Pick (E2, E4) pairs from INH->out_inh
 *   3. Find E1 in E2->in_exc ∩ INH->in_exc
 *   4. Find E3 in E4->in_exc ∩ INH->in_exc, E3 ≠ E1
 *   5. Find XOR in E2->out_exc ∩ E4->out_exc
 *   6. Find F+ in XOR->out_exc ∩ E2->out_exc, F+ must be 'e'
 *   7. Find F- in XOR->out_exc ∩ E4->out_exc, F- must be 'i', F- ≠ F+
 *   8. All 8 nodes must be distinct
 *   9. check_subgraph verifies exactly 12 edges, no extras
 *
 * Compile:
 *   gcc -O3 -march=native -fopenmp -o signed_xor_motif_count signed_xor_motif_count.c
 *
 * Run:
 *   OMP_NUM_THREADS=20 ./signed_xor_motif_count [-d] [-n <max_per_type>] <csv_file>
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <math.h>
#include <omp.h>
#include "uthash.h"

/* ── tunables ─────────────────────────────────────────────── */
#define INITIAL_CAPACITY 8

/* ── globals ──────────────────────────────────────────────── */
int    debug        = 0;
long   max_rows     = 0;
int    max_per_type = 0;       /* 0 = all neurons */
double min_weight   = 1e-5;    /* drop edges with syn_weight below this */
const char *weight_stats_prefix = NULL;  /* if set, dump weight analysis */
long   reservoir_size_per_role  = 100000; /* sample size per role */
int    pivot_id = -1;          /* if >=0, restrict to this single neuron */

/* ── node structure ───────────────────────────────────────── */
typedef struct node {
    int    id;
    char   label[64];
    int    has_label;

    int    *out_exc;   double *out_exc_w;  size_t out_exc_count;  size_t out_exc_alloc;
    int    *out_inh;   double *out_inh_w;  size_t out_inh_count;  size_t out_inh_alloc;
    int    *in_exc;    /* in_exc does not need weights — we reach back via out_exc */
                       size_t in_exc_count;   size_t in_exc_alloc;

    UT_hash_handle hh;
} node_t;

node_t  *nodes      = NULL;
node_t **node_array = NULL;
int      node_count = 0;

/* ── helpers ──────────────────────────────────────────────── */
static node_t *get_node(int id) {
    node_t *n;
    HASH_FIND_INT(nodes, &id, n);
    if (!n) {
        n = calloc(1, sizeof(node_t));
        if (!n) { perror("calloc"); exit(1); }
        n->id = id;
        HASH_ADD_INT(nodes, id, n);
    }
    return n;
}

static void array_push(int **arr, size_t *cnt, size_t *alloc, int val) {
    if (*cnt >= *alloc) {
        *alloc = (*alloc == 0) ? INITIAL_CAPACITY : (*alloc * 2);
        *arr   = realloc(*arr, *alloc * sizeof(int));
        if (!*arr) { perror("realloc"); exit(1); }
    }
    (*arr)[(*cnt)++] = val;
}

static void array_push_w(int **arr, double **warr, size_t *cnt, size_t *alloc, int val, double w) {
    if (*cnt >= *alloc) {
        *alloc = (*alloc == 0) ? INITIAL_CAPACITY : (*alloc * 2);
        *arr   = realloc(*arr,  *alloc * sizeof(int));
        *warr  = realloc(*warr, *alloc * sizeof(double));
        if (!*arr || !*warr) { perror("realloc"); exit(1); }
    }
    (*arr)[*cnt]  = val;
    (*warr)[*cnt] = w;
    (*cnt)++;
}

/* Find the weight of edge src→tgt in the given adjacency list pair.
 * Returns the weight if found, or 0.0 if not (caller should know it exists). */
static inline double find_weight(const int *ids, const double *ws, size_t n, int tgt) {
    for (size_t i = 0; i < n; i++)
        if (ids[i] == tgt) return ws[i];
    return 0.0;
}

static inline int in_array(const int *arr, size_t n, int val) {
    for (size_t i = 0; i < n; i++)
        if (arr[i] == val) return 1;
    return 0;
}

/* ── CSV reader ───────────────────────────────────────────── */
static void read_csv(const char *path) {
    FILE *fp = fopen(path, "r");
    if (!fp) { perror("fopen"); exit(1); }

    char line[512];
    long row = 0;
    long kept = 0;
    long dropped_weight = 0;

    /* skip header */
    if (fgets(line, sizeof(line), fp) == NULL) { fclose(fp); return; }

    while (fgets(line, sizeof(line), fp)) {
        row++;
        if (max_rows > 0 && row > max_rows) break;
        line[strcspn(line, "\r\n")] = 0;
        if (!line[0]) continue;

        char *tok = strtok(line, ",");  if (!tok) continue;
        int src_id = atoi(tok);

        tok = strtok(NULL, ",");        if (!tok) continue;
        int tgt_id = atoi(tok);

        tok = strtok(NULL, ",");        if (!tok) continue;
        char label[64];
        strncpy(label, tok, 63); label[63] = 0;

        tok = strtok(NULL, ",");        if (!tok) continue;
        char etype[8];
        strncpy(etype, tok, 7); etype[7] = 0;

        /* 5th column: syn_weight — filter spurious synapses.
         * Allen V1 stores INH weights as negative values (the sign encodes
         * EXC vs INH at the level of the weight itself, in addition to the
         * connection_type column). We filter by the magnitude. */
        tok = strtok(NULL, ",");        if (!tok) continue;
        double weight = atof(tok);
        double abs_weight = weight < 0 ? -weight : weight;
        if (abs_weight < min_weight) { dropped_weight++; continue; }

        node_t *src = get_node(src_id);
        node_t *tgt = get_node(tgt_id);

        if (!src->has_label) {
            strncpy(src->label, label, 63);
            src->has_label = 1;
        }
        if (strcmp(etype, "EXC") == 0) {
            array_push_w(&src->out_exc, &src->out_exc_w, &src->out_exc_count, &src->out_exc_alloc, tgt_id, weight);
            array_push  (&tgt->in_exc,                    &tgt->in_exc_count,  &tgt->in_exc_alloc,  src_id);
            kept++;
        } else if (strcmp(etype, "INH") == 0) {
            array_push_w(&src->out_inh, &src->out_inh_w, &src->out_inh_count, &src->out_inh_alloc, tgt_id, weight);
            kept++;
        }
        if (debug && row % 5000000 == 0)
            printf("  read %ld M rows...\n", row / 1000000);
    }
    printf("CSV loaded. Rows scanned: %ld  kept: %ld  dropped (weight<%.0e): %ld  Nodes: %d\n",
           row, kept, min_weight, dropped_weight, (int)HASH_COUNT(nodes));
    fclose(fp);
}

/* ── flat node array ──────────────────────────────────────── */
static void build_node_array(void) {
    node_count = (int)HASH_COUNT(nodes);
    node_array = malloc(node_count * sizeof(node_t *));
    if (!node_array) { perror("malloc"); exit(1); }
    node_t *n, *tmp;
    int i = 0;
    HASH_ITER(hh, nodes, n, tmp) node_array[i++] = n;
}

/* ── subgraph check ───────────────────────────────────────────
 * Verifies exactly 12 required edges exist and no extra edges
 * between any of the 8 nodes.
 *
 * roles: [0]=E1 [1]=E2 [2]=E3 [3]=E4 [4]=INH [5]=XOR [6]=F+ [7]=F-
 *
 * Required edges:
 *   EXC: E1→E2, E1→INH, E3→E4, E3→INH   (0→1, 0→4, 2→3, 2→4)
 *   INH: INH→E2, INH→E4                   (4→1, 4→3)
 *   EXC: E2→XOR, E4→XOR                   (1→5, 3→5)
 *   EXC: E2→F+,  E4→F-                    (1→6, 3→7)
 *   EXC: XOR→F+, XOR→F-                   (5→6, 5→7)
 */
static int check_subgraph(node_t *m[8]) {
    static const int allowed[12][3] = {
        {0, 1, 1},  /* E1  → E2  EXC */
        {0, 4, 1},  /* E1  → INH EXC */
        {2, 3, 1},  /* E3  → E4  EXC */
        {2, 4, 1},  /* E3  → INH EXC */
        {4, 1, 0},  /* INH → E2  INH */
        {4, 3, 0},  /* INH → E4  INH */
        {1, 5, 1},  /* E2  → XOR EXC */
        {3, 5, 1},  /* E4  → XOR EXC */
        {1, 6, 1},  /* E2  → F+  EXC */
        {3, 7, 1},  /* E4  → F-  EXC */
        {5, 6, 1},  /* XOR → F+  EXC */
        {5, 7, 1},  /* XOR → F-  EXC */
    };

    for (int i = 0; i < 8; i++) {
        for (int j = 0; j < 8; j++) {
            if (i == j) continue;
            int required = 0, req_exc = 0;
            for (int k = 0; k < 12; k++) {
                if (allowed[k][0] == i && allowed[k][1] == j) {
                    required = 1;
                    req_exc  = allowed[k][2];
                    break;
                }
            }
            int has_exc = in_array(m[i]->out_exc, m[i]->out_exc_count, m[j]->id);
            int has_inh = in_array(m[i]->out_inh, m[i]->out_inh_count, m[j]->id);
            if (required) {
                if (req_exc  && (!has_exc || has_inh)) return 0;
                if (!req_exc && (!has_inh || has_exc)) return 0;
            } else {
                if (has_exc || has_inh) return 0;
            }
        }
    }
    return 1;
}

/* ── weight statistics by role ─────────────────────────────
 * 12 edge roles, indexed 0..11 in this order:
 *   0: E1 → E2   EXC
 *   1: E1 → INH  EXC
 *   2: E3 → E4   EXC
 *   3: E3 → INH  EXC
 *   4: INH → E2  INH
 *   5: INH → E4  INH
 *   6: E2 → XOR  EXC
 *   7: E4 → XOR  EXC
 *   8: E2 → F+   EXC
 *   9: E4 → F-   EXC
 *  10: XOR → F+  EXC
 *  11: XOR → F-  EXC
 */
#define N_EDGE_ROLES 12

static const char *edge_role_names[N_EDGE_ROLES] = {
    "E1_E2_EXC", "E1_INH_EXC", "E3_E4_EXC", "E3_INH_EXC",
    "INH_E2_INH", "INH_E4_INH",
    "E2_XOR_EXC", "E4_XOR_EXC",
    "E2_Fpos_EXC", "E4_Fneg_EXC",
    "XOR_Fpos_EXC", "XOR_Fneg_EXC"
};

/* Histogram bins: log-spaced on |w|. We use 60 bins covering 1e-10 to 1e2,
 * which gives 5 bins per decade. Plus one underflow and one overflow bin. */
#define HIST_N_BINS 62
#define HIST_LOG_MIN -10.0
#define HIST_LOG_MAX   2.0
#define HIST_BINS_INNER 60

static inline int log_bin(double abs_w) {
    if (abs_w <= 0.0) return 0;  /* underflow */
    double logw = log10(abs_w);
    if (logw < HIST_LOG_MIN) return 0;
    if (logw >= HIST_LOG_MAX) return HIST_N_BINS - 1;
    double frac = (logw - HIST_LOG_MIN) / (HIST_LOG_MAX - HIST_LOG_MIN);
    int idx = 1 + (int)(frac * HIST_BINS_INNER);
    if (idx < 1) idx = 1;
    if (idx > HIST_BINS_INNER) idx = HIST_BINS_INNER;
    return idx;
}

typedef struct {
    /* exact moments (computed online via Welford for variance) */
    long   n;
    double sum_w;     /* sum of raw weights (signed) */
    double sum_aw;    /* sum of absolute weights */
    double sum_w2;    /* sum of w^2 (for variance) */
    double min_w, max_w;
    double min_aw, max_aw;

    /* log10 histogram of |w| */
    long   hist[HIST_N_BINS];

    /* reservoir sample of raw signed weights */
    double *sample;
    long    sample_seen;   /* total observations so far (for reservoir bookkeeping) */
    long    sample_n;      /* current count in reservoir */
    unsigned int rng_state;
} role_stats_t;

static role_stats_t role_stats[N_EDGE_ROLES];

static void init_role_stats(void) {
    for (int r = 0; r < N_EDGE_ROLES; r++) {
        role_stats[r].n       = 0;
        role_stats[r].sum_w   = 0.0;
        role_stats[r].sum_aw  = 0.0;
        role_stats[r].sum_w2  = 0.0;
        role_stats[r].min_w   =  1e300;
        role_stats[r].max_w   = -1e300;
        role_stats[r].min_aw  =  1e300;
        role_stats[r].max_aw  = -1e300;
        for (int b = 0; b < HIST_N_BINS; b++) role_stats[r].hist[b] = 0;
        role_stats[r].sample = malloc(reservoir_size_per_role * sizeof(double));
        if (!role_stats[r].sample) { perror("malloc sample"); exit(1); }
        role_stats[r].sample_seen = 0;
        role_stats[r].sample_n    = 0;
        role_stats[r].rng_state   = 12345u + r;  /* per-role seed */
    }
}

static void free_role_stats(void) {
    for (int r = 0; r < N_EDGE_ROLES; r++) free(role_stats[r].sample);
}

/* Record one weight observation in role `r`. NOT thread-safe — caller must
 * hold a per-role lock if called concurrently. */
static void record_weight(int r, double w) {
    role_stats_t *s = &role_stats[r];
    double aw = w < 0 ? -w : w;
    s->n++;
    s->sum_w  += w;
    s->sum_aw += aw;
    s->sum_w2 += w * w;
    if (w  < s->min_w)  s->min_w  = w;
    if (w  > s->max_w)  s->max_w  = w;
    if (aw < s->min_aw) s->min_aw = aw;
    if (aw > s->max_aw) s->max_aw = aw;
    s->hist[log_bin(aw)]++;
    s->sample_seen++;
    if (s->sample_n < reservoir_size_per_role) {
        s->sample[s->sample_n++] = w;
    } else {
        /* reservoir replacement: random index in [0, sample_seen) */
        unsigned int r1 = rand_r(&s->rng_state);
        unsigned int r2 = rand_r(&s->rng_state);
        unsigned long roll = ((unsigned long)r1 << 16) ^ (unsigned long)r2;
        unsigned long idx = roll % (unsigned long)s->sample_seen;
        if (idx < (unsigned long)reservoir_size_per_role) {
            s->sample[idx] = w;
        }
    }
}

/* Locks: one per role to minimise contention */
static omp_lock_t role_locks[N_EDGE_ROLES];
static void init_role_locks(void) { for (int r=0;r<N_EDGE_ROLES;r++) omp_init_lock(&role_locks[r]); }
static void destroy_role_locks(void) { for (int r=0;r<N_EDGE_ROLES;r++) omp_destroy_lock(&role_locks[r]); }

/* Record the 12 weights of a counted motif. Called only when --weight-stats
 * is active. The motif nodes are in roles m[0..7] = E1,E2,E3,E4,INH,XOR,F+,F-. */
static void record_motif_weights(node_t *m[8]) {
    /* Look up each of the 12 weights from the source-node's adjacency lists */
    double w[N_EDGE_ROLES] = {
        find_weight(m[0]->out_exc, m[0]->out_exc_w, m[0]->out_exc_count, m[1]->id), /* E1→E2 EXC */
        find_weight(m[0]->out_exc, m[0]->out_exc_w, m[0]->out_exc_count, m[4]->id), /* E1→INH EXC */
        find_weight(m[2]->out_exc, m[2]->out_exc_w, m[2]->out_exc_count, m[3]->id), /* E3→E4 EXC */
        find_weight(m[2]->out_exc, m[2]->out_exc_w, m[2]->out_exc_count, m[4]->id), /* E3→INH EXC */
        find_weight(m[4]->out_inh, m[4]->out_inh_w, m[4]->out_inh_count, m[1]->id), /* INH→E2 INH */
        find_weight(m[4]->out_inh, m[4]->out_inh_w, m[4]->out_inh_count, m[3]->id), /* INH→E4 INH */
        find_weight(m[1]->out_exc, m[1]->out_exc_w, m[1]->out_exc_count, m[5]->id), /* E2→XOR EXC */
        find_weight(m[3]->out_exc, m[3]->out_exc_w, m[3]->out_exc_count, m[5]->id), /* E4→XOR EXC */
        find_weight(m[1]->out_exc, m[1]->out_exc_w, m[1]->out_exc_count, m[6]->id), /* E2→F+ EXC */
        find_weight(m[3]->out_exc, m[3]->out_exc_w, m[3]->out_exc_count, m[7]->id), /* E4→F- EXC */
        find_weight(m[5]->out_exc, m[5]->out_exc_w, m[5]->out_exc_count, m[6]->id), /* XOR→F+ EXC */
        find_weight(m[5]->out_exc, m[5]->out_exc_w, m[5]->out_exc_count, m[7]->id), /* XOR→F- EXC */
    };
    for (int r = 0; r < N_EDGE_ROLES; r++) {
        omp_set_lock(&role_locks[r]);
        record_weight(r, w[r]);
        omp_unset_lock(&role_locks[r]);
    }
}

/* Estimate a quantile of |w| from the log-spaced histogram by inverting the
 * cumulative distribution. Returns the geometric-mean midpoint of the bin
 * containing the target rank (linear in log-space, which is appropriate
 * for log-spaced bins). For very sparse histograms (few observations) this
 * is approximate; for >1000 observations it is accurate to within one bin
 * width, i.e. ~10% relative on |w|. */
static double quantile_from_hist(const role_stats_t *s, double q) {
    if (s->n == 0) return 0.0;
    long target = (long)(q * s->n);
    if (target < 1)         target = 1;
    if (target > s->n)      target = s->n;
    long cum = 0;
    for (int b = 0; b < HIST_N_BINS; b++) {
        cum += s->hist[b];
        if (cum >= target) {
            double lo, hi;
            if (b == 0) { lo = 1e-12; hi = 1e-10; }
            else if (b == HIST_N_BINS - 1) { lo = 1e2; hi = 1e3; }
            else {
                double frac_lo = (double)(b - 1) / HIST_BINS_INNER;
                double frac_hi = (double)(b)     / HIST_BINS_INNER;
                lo = __builtin_pow(10.0, HIST_LOG_MIN + frac_lo * (HIST_LOG_MAX - HIST_LOG_MIN));
                hi = __builtin_pow(10.0, HIST_LOG_MIN + frac_hi * (HIST_LOG_MAX - HIST_LOG_MIN));
            }
            /* geometric-mean midpoint of the bin */
            return __builtin_sqrt(lo * hi);
        }
    }
    return s->max_aw;
}

/* Write the three weight-stats output files. Called after the parallel
 * count phase. */
static void write_weight_stats(const char *prefix) {
    char path[1024];

    /* 1) <prefix>_stats.csv — one row per role, with exact moments + quantiles */
    snprintf(path, sizeof(path), "%s_stats.csv", prefix);
    FILE *fp = fopen(path, "w");
    if (!fp) { perror(path); return; }
    fprintf(fp, "role_index,role_name,n_observations,"
                "mean_w,sd_w,min_w,max_w,"
                "mean_abs_w,min_abs_w,max_abs_w,"
                "p10_abs_w,p25_abs_w,p50_abs_w,p75_abs_w,p90_abs_w\n");
    for (int r = 0; r < N_EDGE_ROLES; r++) {
        role_stats_t *s = &role_stats[r];
        double mean = s->n ? s->sum_w / s->n : 0.0;
        double var  = s->n > 1 ? (s->sum_w2 - s->n * mean * mean) / (s->n - 1) : 0.0;
        if (var < 0.0) var = 0.0;
        double sd   = (var > 0.0) ? __builtin_sqrt(var) : 0.0;
        double meanabs = s->n ? s->sum_aw / s->n : 0.0;
        double p10 = quantile_from_hist(s, 0.10);
        double p25 = quantile_from_hist(s, 0.25);
        double p50 = quantile_from_hist(s, 0.50);
        double p75 = quantile_from_hist(s, 0.75);
        double p90 = quantile_from_hist(s, 0.90);
        fprintf(fp, "%d,%s,%ld,%.6g,%.6g,%.6g,%.6g,%.6g,%.6g,%.6g,%.6g,%.6g,%.6g,%.6g,%.6g\n",
                r, edge_role_names[r], s->n,
                mean, sd,
                s->n ? s->min_w  : 0.0, s->n ? s->max_w  : 0.0,
                meanabs,
                s->n ? s->min_aw : 0.0, s->n ? s->max_aw : 0.0,
                p10, p25, p50, p75, p90);
    }
    fclose(fp);
    printf("Wrote %s\n", path);

    /* 2) <prefix>_histogram.csv — log-magnitude histogram, one column per role */
    snprintf(path, sizeof(path), "%s_histogram.csv", prefix);
    fp = fopen(path, "w");
    if (!fp) { perror(path); return; }
    fprintf(fp, "bin_index,bin_lo,bin_hi");
    for (int r = 0; r < N_EDGE_ROLES; r++) fprintf(fp, ",%s", edge_role_names[r]);
    fprintf(fp, "\n");
    /* Bin 0 = underflow (|w| <= 0 or |w| < 1e-10); bins 1..60 = log-spaced;
     * bin 61 = overflow (|w| >= 1e2). */
    for (int b = 0; b < HIST_N_BINS; b++) {
        double lo, hi;
        if (b == 0) { lo = 0.0; hi = 1e-10; }
        else if (b == HIST_N_BINS - 1) { lo = 1e2; hi = 1.0/0.0; }
        else {
            double frac_lo = (double)(b - 1) / HIST_BINS_INNER;
            double frac_hi = (double)(b)     / HIST_BINS_INNER;
            lo = __builtin_pow(10.0, HIST_LOG_MIN + frac_lo * (HIST_LOG_MAX - HIST_LOG_MIN));
            hi = __builtin_pow(10.0, HIST_LOG_MIN + frac_hi * (HIST_LOG_MAX - HIST_LOG_MIN));
        }
        fprintf(fp, "%d,%.6g,%.6g", b, lo, hi);
        for (int r = 0; r < N_EDGE_ROLES; r++)
            fprintf(fp, ",%ld", role_stats[r].hist[b]);
        fprintf(fp, "\n");
    }
    fclose(fp);
    printf("Wrote %s\n", path);

    /* 3) <prefix>_sample.csv — reservoir sample (long-format, ~12 * sample_n rows) */
    snprintf(path, sizeof(path), "%s_sample.csv", prefix);
    fp = fopen(path, "w");
    if (!fp) { perror(path); return; }
    fprintf(fp, "role_index,role_name,weight\n");
    long total_written = 0;
    for (int r = 0; r < N_EDGE_ROLES; r++) {
        for (long i = 0; i < role_stats[r].sample_n; i++) {
            fprintf(fp, "%d,%s,%.6g\n", r, edge_role_names[r], role_stats[r].sample[i]);
            total_written++;
        }
    }
    fclose(fp);
    printf("Wrote %s (%ld rows)\n", path, total_written);
}

/* ── count signed-XOR motifs for one INH node ────────────── */
static long count_for_inh(node_t *inh) {
    long count = 0;
    if (inh->out_inh_count < 2) return 0;

    /* iterate (E2, E4) pairs from INH->out_inh */
    for (size_t i = 0; i < inh->out_inh_count; i++) {
        node_t *e2 = get_node(inh->out_inh[i]);
        if (!e2->has_label || e2->label[0] != 'e') continue;

        for (size_t j = i + 1; j < inh->out_inh_count; j++) {
            node_t *e4 = get_node(inh->out_inh[j]);
            if (!e4->has_label || e4->label[0] != 'e') continue;

            /* E1: in E2->in_exc ∩ INH->in_exc */
            for (size_t a = 0; a < e2->in_exc_count; a++) {
                node_t *e1 = get_node(e2->in_exc[a]);
                if (!e1->has_label || e1->label[0] != 'e') continue;
                if (!in_array(inh->in_exc, inh->in_exc_count, e1->id)) continue;

                /* E3: in E4->in_exc ∩ INH->in_exc, E3 ≠ E1 */
                for (size_t b = 0; b < e4->in_exc_count; b++) {
                    node_t *e3 = get_node(e4->in_exc[b]);
                    if (e3->id == e1->id) continue;
                    if (!e3->has_label || e3->label[0] != 'e') continue;
                    if (!in_array(inh->in_exc, inh->in_exc_count, e3->id)) continue;

                    /* XOR: in E2->out_exc ∩ E4->out_exc */
                    for (size_t c = 0; c < e2->out_exc_count; c++) {
                        int xor_id = e2->out_exc[c];
                        if (!in_array(e4->out_exc, e4->out_exc_count, xor_id)) continue;

                        node_t *xor = get_node(xor_id);
                        if (!xor->has_label || xor->label[0] != 'e') continue;

                        /* XOR must be distinct from E1,E2,E3,E4,INH */
                        if (xor->id == e1->id || xor->id == e2->id ||
                            xor->id == e3->id || xor->id == e4->id ||
                            xor->id == inh->id) continue;

                        /* F+: in XOR->out_exc ∩ E2->out_exc, must be 'e' */
                        for (size_t d = 0; d < xor->out_exc_count; d++) {
                            int fp_id = xor->out_exc[d];
                            if (!in_array(e2->out_exc, e2->out_exc_count, fp_id)) continue;

                            node_t *fp = get_node(fp_id);
                            if (!fp->has_label || fp->label[0] != 'e') continue;

                            /* F+ distinct from all previous */
                            if (fp->id == e1->id || fp->id == e2->id ||
                                fp->id == e3->id || fp->id == e4->id ||
                                fp->id == inh->id || fp->id == xor->id) continue;

                            /* F-: in XOR->out_exc ∩ E4->out_exc, must be 'i' */
                            for (size_t e = 0; e < xor->out_exc_count; e++) {
                                int fm_id = xor->out_exc[e];
                                if (fm_id == fp_id) continue;
                                if (!in_array(e4->out_exc, e4->out_exc_count, fm_id)) continue;

                                node_t *fm = get_node(fm_id);
                                if (!fm->has_label || fm->label[0] != 'i') continue;

                                /* F- distinct from all previous */
                                if (fm->id == e1->id || fm->id == e2->id ||
                                    fm->id == e3->id || fm->id == e4->id ||
                                    fm->id == inh->id || fm->id == xor->id ||
                                    fm->id == fp->id) continue;

                                node_t *m[8] = {e1, e2, e3, e4, inh, xor, fp, fm};
                                if (!check_subgraph(m)) continue;
                                count++;
                                if (weight_stats_prefix) record_motif_weights(m);
                            }
                        }
                    }
                }
            }
        }
    }
    return count;
}

/* ── free graph ───────────────────────────────────────────── */
static void free_graph(void) {
    node_t *n, *tmp;
    HASH_ITER(hh, nodes, n, tmp) {
        HASH_DEL(nodes, n);
        free(n->out_exc);  free(n->out_exc_w);
        free(n->out_inh);  free(n->out_inh_w);
        free(n->in_exc);
        free(n);
    }
    free(node_array);
}

/* ── target types ─────────────────────────────────────────── */
static const char *TARGET_TYPES[12] = {
    "i23Pvalb", "i23Sst", "i23Htr3a",
    "i4Pvalb",  "i4Sst",  "i4Htr3a",
    "i5Pvalb",  "i5Sst",  "i5Htr3a",
    "i6Pvalb",  "i6Sst",  "i6Htr3a"
};

static int type_index(const char *label) {
    for (int i = 0; i < 12; i++)
        if (strcmp(label, TARGET_TYPES[i]) == 0) return i;
    return -1;
}

/* ── usage ────────────────────────────────────────────────── */
static void usage(const char *prog) {
    fprintf(stderr,
        "Usage: %s [-d] [-t <max_rows>] [-n <max_per_type>] [--min-weight <w>]\n"
        "           [--pivot <node_id>] [--weight-stats <prefix>] [--sample-size <N>] <csv_file>\n"
        "  -d                      debug: per-neuron counts\n"
        "  -t <max_rows>           limit CSV rows (for testing)\n"
        "  -n <max_per_type>       max neurons per type (0 = all, default)\n"
        "  --min-weight <w>        drop synapses with |syn_weight| below w (default 1e-5)\n"
        "  --pivot <node_id>       restrict to a single inhibitory neuron as role-6\n"
        "                          (its node_id must match the source_node_id column)\n"
        "  --weight-stats <prefix> write per-edge-role weight analysis to:\n"
        "                            <prefix>_stats.csv      (exact moments per role)\n"
        "                            <prefix>_histogram.csv  (log10|w| histogram per role)\n"
        "                            <prefix>_sample.csv     (reservoir-sampled weights)\n"
        "  --sample-size <N>       reservoir size per role (default 100000)\n"
        "Counts signed-XOR motifs (8 nodes, 12 edges).\n",
        prog);
}

/* ════════════════════════════════════════════════════════════ */
int main(int argc, char *argv[]) {
    const char *csv_path = NULL;

    for (int i = 1; i < argc; i++) {
        if      (!strcmp(argv[i], "-d")) { debug = 1; }
        else if (!strcmp(argv[i], "-t") && i+1 < argc) { max_rows     = atol(argv[++i]); }
        else if (!strcmp(argv[i], "-n") && i+1 < argc) { max_per_type = atoi(argv[++i]); }
        else if (!strcmp(argv[i], "--min-weight") && i+1 < argc) { min_weight = atof(argv[++i]); }
        else if (!strcmp(argv[i], "--pivot") && i+1 < argc) { pivot_id = atoi(argv[++i]); }
        else if (!strcmp(argv[i], "--weight-stats") && i+1 < argc) { weight_stats_prefix = argv[++i]; }
        else if (!strcmp(argv[i], "--sample-size") && i+1 < argc) { reservoir_size_per_role = atol(argv[++i]); }
        else if (argv[i][0] != '-') { csv_path = argv[i]; }
        else { fprintf(stderr, "Unknown option: %s\n", argv[i]); usage(argv[0]); return 1; }
    }
    if (!csv_path) { usage(argv[0]); return 1; }

    int nthreads = omp_get_max_threads();
    printf("Threads: %d\n", nthreads);
    printf("Min weight: %g (synapses below this are dropped)\n", min_weight);
    if (pivot_id >= 0) printf("Pivot mode: restricted to node_id %d\n", pivot_id);
    if (weight_stats_prefix) {
        printf("Weight stats: writing to %s_{stats,histogram,sample}.csv (reservoir size %ld per role)\n",
               weight_stats_prefix, reservoir_size_per_role);
        init_role_stats();
        init_role_locks();
    }

    /* 1. Load graph */
    printf("Reading CSV: %s\n", csv_path);
    read_csv(csv_path);

    /* 2. Build flat array */
    build_node_array();

    /* 3. Collect INH candidates */
    typedef struct { node_t *n; int tidx; } cand_t;
    cand_t *candidates = malloc(node_count * sizeof(cand_t));
    int     ncands     = 0;
    int     per_type[12] = {0};

    for (int i = 0; i < node_count; i++) {
        node_t *n = node_array[i];
        if (!n->has_label || n->label[0] != 'i') continue;
        int ti = type_index(n->label);
        if (ti < 0) continue;
        if (pivot_id >= 0 && n->id != pivot_id) continue;
        if (max_per_type > 0 && per_type[ti] >= max_per_type) continue;
        candidates[ncands].n    = n;
        candidates[ncands].tidx = ti;
        ncands++;
        per_type[ti]++;
    }
    if (pivot_id >= 0) {
        if (ncands == 0) {
            fprintf(stderr, "ERROR: --pivot %d not found among inhibitory candidates\n"
                    "  (must be a node with label 'i*' and present in CSV as source)\n",
                    pivot_id);
            exit(2);
        }
        printf("INH candidates: %d (restricted to pivot %d, type %s)\n\n",
               ncands, pivot_id, candidates[0].n->label);
    }
    else if (max_per_type > 0)
        printf("INH candidates: %d (max %d per type)\n\n", ncands, max_per_type);
    else
        printf("INH candidates: %d (all neurons)\n\n", ncands);

    /* 4. Parallel count */
    long type_counts[12] = {0};
    long total     = 0;
    long processed = 0;
    double t0 = omp_get_wtime();

    #pragma omp parallel reduction(+:total)
    {
        #pragma omp for schedule(dynamic, 1)
        for (int c = 0; c < ncands; c++) {
            node_t *inh  = candidates[c].n;
            int     tidx = candidates[c].tidx;

            long found = count_for_inh(inh);

            #pragma omp atomic
            type_counts[tidx] += found;

            total += found;

            long done;
            #pragma omp atomic capture
            done = ++processed;

            if (done % 100 == 0 || debug) {
                double elapsed = omp_get_wtime() - t0;
                #pragma omp critical(print)
                {
                    printf("  [%ld/%d | %.1f min] %s neuron %d: %ld motifs\n",
                           done, ncands, elapsed/60.0,
                           inh->label, inh->id, found);
                    fflush(stdout);
                }
            }
        }
    }

    double elapsed = omp_get_wtime() - t0;

    /* 5. Report */
    printf("\n═══════════════════════════════════════════════════════\n");
    if (max_per_type > 0)
        printf("  Signed-XOR motif count — %d neurons sampled per type\n", max_per_type);
    else
        printf("  Signed-XOR motif count — all neurons, all types\n");
    printf("═══════════════════════════════════════════════════════\n");
    printf("%-14s  %12s  %8s\n", "Type", "Motifs", "Neurons");
    printf("───────────────────────────────────────────────────────\n");
    long grand_total = 0;
    for (int i = 0; i < 12; i++) {
        printf("%-14s  %12ld  %8d\n",
               TARGET_TYPES[i], type_counts[i], per_type[i]);
        grand_total += type_counts[i];
    }
    printf("───────────────────────────────────────────────────────\n");
    printf("%-14s  %12ld\n", "TOTAL", grand_total);
    printf("\nTime: %.1f s  (%.1f min)\n", elapsed, elapsed/60.0);
    printf("Threads: %d\n", nthreads);

    /* Write weight stats if requested */
    if (weight_stats_prefix) {
        printf("\nWriting weight stats...\n");
        write_weight_stats(weight_stats_prefix);
        destroy_role_locks();
        free_role_stats();
    }

    free(candidates);
    free_graph();
    return 0;
}

/*
 * drosophila_signed_xor_motif.c
 *
 * Counts the SIGNED-XOR motif: the 6-node XOR motif (8 edges) extended with
 * a signed feedback module of 2 additional neurons (F+ and F-) and 4 new
 * edges, for a total of 8 nodes and 12 edges.
 *
 * BASE XOR (8 edges, unchanged from drosophila_xor_motif.c):
 *      EXC: (1,2) (1,6) (3,4) (3,6) (2,5) (4,5)
 *      INH: (6,2) (6,4)
 *
 * SIGNED FEEDBACK extension (4 new edges, 2 new nodes F+ and F-):
 *      (5, F+)  with sign matching node 5's nt_type
 *      (5, F-)  with sign matching node 5's nt_type
 *      AND either Variant A OR Variant B (not necessarily exclusive):
 *          Variant A: (1, F+) EXC  AND  (3, F-) EXC
 *          Variant B: (2, F+) EXC  AND  (4, F-) EXC
 *      A motif is counted if A OR B holds (we count once even if both hold).
 *
 * NEURON IDENTITY (Dale's principle, encoded in CSV):
 *      F+ must be excitatory (its nt_type is ACH).
 *      F- must be inhibitory (its nt_type is GABA or GLUT).
 *      Node 5's nt_type determines the sign of the (5,F+) and (5,F-) edges.
 *      Identity is determined from outgoing edge types: a neuron with at
 *      least one ACH outgoing edge is EXC, with GABA/GLUT is INH. Mixed
 *      neurons (both EXC and INH outputs) are skipped — Dale violation,
 *      ambiguous identity.
 *
 * DISTINCTNESS:
 *      All 8 neurons must be pairwise distinct.
 *
 * SYMMETRY:
 *      The base XOR has 2-fold automorphism (1<->3, 2<->4). Under that swap,
 *      Variant A maps to Variant B. So the signed motif as a whole has the
 *      same 2-fold symmetry: each biological signed-XOR instance is found
 *      twice in the search. DISTINCT count = RAW count / 2.
 *
 * Compile:
 *      gcc -O3 -march=native -fopenmp -o drosophila_signed_xor_motif \
 *          drosophila_signed_xor_motif.c
 *
 * Usage:
 *      ./drosophila_signed_xor_motif input.csv [--out motifs.csv] [--induced] [-q]
 *
 * Output: same STATS / MOTIF_COUNT format as the XOR counter. CSV (if --out)
 * has 9 columns: n1,n2,n3,n4,n5,n6,fp,fm,variants  (variants in {A,B,AB}).
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <stdint.h>
#include <stdbool.h>
#include <time.h>
#ifdef _OPENMP
#include <omp.h>
#endif
#include "uthash.h"

/* ===================================================================
 *                        STRING -> ID INTERNING
 * =================================================================== */
typedef struct id_entry {
    char *name;
    int   id;
    UT_hash_handle hh;
} id_entry_t;
static id_entry_t *id_table = NULL;
static int next_id = 0;
static char **id_to_name = NULL;
static int    id_to_name_cap = 0;

static int intern_id(const char *s) {
    id_entry_t *e;
    HASH_FIND_STR(id_table, s, e);
    if (e) return e->id;
    e = (id_entry_t*)malloc(sizeof(id_entry_t));
    e->name = strdup(s);
    e->id = next_id++;
    HASH_ADD_KEYPTR(hh, id_table, e->name, strlen(e->name), e);
    if (e->id >= id_to_name_cap) {
        int nc = id_to_name_cap ? id_to_name_cap*2 : 1024;
        while (nc <= e->id) nc *= 2;
        id_to_name = (char**)realloc(id_to_name, nc * sizeof(char*));
        for (int i = id_to_name_cap; i < nc; i++) id_to_name[i] = NULL;
        id_to_name_cap = nc;
    }
    id_to_name[e->id] = e->name;
    return e->id;
}

/* ===================================================================
 *                        NODE ADJACENCY DATA
 * =================================================================== */
typedef struct {
    int   *out_exc; size_t out_exc_n, out_exc_cap;
    int   *out_inh; size_t out_inh_n, out_inh_cap;
    int   *in_exc;  size_t in_exc_n,  in_exc_cap;
    int   *in_inh;  size_t in_inh_n,  in_inh_cap;
    int   *out_other; size_t out_other_n, out_other_cap;
    char is_exc;     /* has at least one outgoing ACH edge */
    char is_inh;     /* has at least one outgoing GABA/GLUT edge */
} node_t;

static node_t *graph = NULL;
static int     N_nodes = 0;
static int     N_cap = 0;

static void ensure_node_capacity(int upto) {
    if (upto < N_cap) return;
    int nc = N_cap ? N_cap*2 : 1024;
    while (upto >= nc) nc *= 2;
    graph = (node_t*)realloc(graph, nc * sizeof(node_t));
    for (int i = N_cap; i < nc; i++) memset(&graph[i], 0, sizeof(node_t));
    N_cap = nc;
}

static void push_int(int **arr, size_t *n, size_t *cap, int v) {
    if (*cap == 0) { *cap = 4; *arr = (int*)malloc(*cap * sizeof(int)); }
    else if (*n >= *cap) { *cap *= 2; *arr = (int*)realloc(*arr, *cap * sizeof(int)); }
    (*arr)[(*n)++] = v;
}

static int int_cmp(const void *a, const void *b) {
    int x = *(const int*)a, y = *(const int*)b;
    return (x>y) - (x<y);
}

static int bsearch_int(const int *arr, size_t n, int v) {
    size_t lo = 0, hi = n;
    while (lo < hi) {
        size_t mid = lo + (hi-lo)/2;
        if (arr[mid] == v) return 1;
        if (arr[mid] < v) lo = mid+1; else hi = mid;
    }
    return 0;
}

/* ===================================================================
 *                    NEUROTRANSMITTER -> CLASS
 * =================================================================== */
typedef enum { EDGE_EXC, EDGE_INH, EDGE_OTHER } edgeclass_t;

static edgeclass_t classify_nt(const char *nt) {
    if (!nt) return EDGE_OTHER;
    while (*nt && (isspace((unsigned char)*nt) || *nt == '"')) nt++;
    if (!strncmp(nt, "ACH",  3)) return EDGE_EXC;
    if (!strncmp(nt, "GABA", 4)) return EDGE_INH;
    if (!strncmp(nt, "GLUT", 4)) return EDGE_INH;
    return EDGE_OTHER;
}

/* ===================================================================
 *                          CSV READER
 * =================================================================== */
typedef struct {
    long n_rows;
    long n_exc, n_inh, n_other;
    long n_neur_exc, n_neur_inh, n_neur_other;
} csv_stats_t;

static int split_csv(char *line, char *out[], int max_fields) {
    int k = 0; char *p = line;
    while (k < max_fields) {
        out[k++] = p;
        char *c = strchr(p, ','); if (!c) break;
        *c = '\0'; p = c + 1;
    }
    return k;
}

static void rstrip(char *s) {
    size_t n = strlen(s);
    while (n && (s[n-1] == '\r' || s[n-1] == '\n' ||
                 s[n-1] == ' '  || s[n-1] == '\t')) s[--n] = '\0';
}

static csv_stats_t read_csv(const char *path) {
    csv_stats_t st = {0};
    FILE *fp = fopen(path, "r");
    if (!fp) { perror(path); exit(1); }
    static char io_buf[8 * 1024 * 1024];
    setvbuf(fp, io_buf, _IOFBF, sizeof(io_buf));

    char line[1024];
    long pos0 = ftell(fp);
    if (fgets(line, sizeof(line), fp)) {
        rstrip(line);
        char *first_comma = strchr(line, ',');
        char first_field[64] = "";
        if (first_comma) {
            size_t k = first_comma - line;
            if (k >= sizeof(first_field)) k = sizeof(first_field)-1;
            memcpy(first_field, line, k); first_field[k] = '\0';
        }
        int looks_numeric = 1;
        for (char *q = first_field; *q; q++) {
            if (!isdigit((unsigned char)*q) && *q != '-' && *q != '+') {
                looks_numeric = 0; break;
            }
        }
        if (!looks_numeric && first_field[0]) {
            /* skip header */
        } else {
            fseek(fp, pos0, SEEK_SET);
        }
    }

    char *seen_exc = NULL, *seen_inh = NULL, *seen_other = NULL;
    int   seen_cap = 0;

    while (fgets(line, sizeof(line), fp)) {
        rstrip(line);
        if (!line[0]) continue;
        char *fields[8];
        int nf = split_csv(line, fields, 8);
        if (nf < 5) continue;

        const char *pre  = fields[0];
        const char *post = fields[1];
        const char *nt   = fields[4];

        int sid = intern_id(pre);
        int tid = intern_id(post);
        edgeclass_t cls = classify_nt(nt);

        ensure_node_capacity(sid > tid ? sid : tid);
        if (sid+1 > N_nodes) N_nodes = sid+1;
        if (tid+1 > N_nodes) N_nodes = tid+1;

        st.n_rows++;
        if (cls == EDGE_EXC) {
            push_int(&graph[sid].out_exc, &graph[sid].out_exc_n, &graph[sid].out_exc_cap, tid);
            push_int(&graph[tid].in_exc,  &graph[tid].in_exc_n,  &graph[tid].in_exc_cap,  sid);
            graph[sid].is_exc = 1;
            st.n_exc++;
        } else if (cls == EDGE_INH) {
            push_int(&graph[sid].out_inh, &graph[sid].out_inh_n, &graph[sid].out_inh_cap, tid);
            push_int(&graph[tid].in_inh,  &graph[tid].in_inh_n,  &graph[tid].in_inh_cap,  sid);
            graph[sid].is_inh = 1;
            st.n_inh++;
        } else {
            push_int(&graph[sid].out_other, &graph[sid].out_other_n, &graph[sid].out_other_cap, tid);
            st.n_other++;
        }

        if (sid >= seen_cap) {
            int nc = seen_cap ? seen_cap*2 : 1024;
            while (sid >= nc) nc *= 2;
            seen_exc   = (char*)realloc(seen_exc,   nc);
            seen_inh   = (char*)realloc(seen_inh,   nc);
            seen_other = (char*)realloc(seen_other, nc);
            for (int i = seen_cap; i < nc; i++) {
                seen_exc[i] = 0; seen_inh[i] = 0; seen_other[i] = 0;
            }
            seen_cap = nc;
        }
        if (cls == EDGE_EXC && !seen_exc[sid])     { seen_exc[sid]=1;   st.n_neur_exc++; }
        if (cls == EDGE_INH && !seen_inh[sid])     { seen_inh[sid]=1;   st.n_neur_inh++; }
        if (cls == EDGE_OTHER && !seen_other[sid]) { seen_other[sid]=1; st.n_neur_other++; }
    }
    fclose(fp);
    free(seen_exc); free(seen_inh); free(seen_other);
    return st;
}

static void sort_adjacencies(void) {
    for (int i = 0; i < N_nodes; i++) {
        if (graph[i].out_exc_n   > 1) qsort(graph[i].out_exc,   graph[i].out_exc_n,   sizeof(int), int_cmp);
        if (graph[i].out_inh_n   > 1) qsort(graph[i].out_inh,   graph[i].out_inh_n,   sizeof(int), int_cmp);
        if (graph[i].in_exc_n    > 1) qsort(graph[i].in_exc,    graph[i].in_exc_n,    sizeof(int), int_cmp);
        if (graph[i].in_inh_n    > 1) qsort(graph[i].in_inh,    graph[i].in_inh_n,    sizeof(int), int_cmp);
        if (graph[i].out_other_n > 1) qsort(graph[i].out_other, graph[i].out_other_n, sizeof(int), int_cmp);
    }
}

static inline int has_exc(int s, int t) { return bsearch_int(graph[s].out_exc, graph[s].out_exc_n, t); }
static inline int has_inh(int s, int t) { return bsearch_int(graph[s].out_inh, graph[s].out_inh_n, t); }
static inline int has_other(int s, int t) { return bsearch_int(graph[s].out_other, graph[s].out_other_n, t); }

/* ===================================================================
 *                INDUCED CHECK ON 8-NODE EXPECTED-EDGE TABLE
 * =================================================================== */
static int check_induced(int N[8], const int8_t E[8][8]) {
    for (int i = 0; i < 8; i++) {
        for (int j = 0; j < 8; j++) {
            if (i == j) continue;
            int want = E[i][j];
            int e = has_exc(N[i], N[j]);
            int h = has_inh(N[i], N[j]);
            int o = has_other(N[i], N[j]);
            if (want == 1)      { if (!e || h || o) return 0; }
            else if (want == 2) { if (!h || e || o) return 0; }
            else                { if (e || h || o)  return 0; }
        }
    }
    return 1;
}

/* Build expected-edge matrix. sign5: 1=EXC, 2=INH. variantA/B: 0/1 each. */
static void build_expected_edges(int8_t E[8][8],
                                 int sign5, int variantA, int variantB) {
    memset(E, 0, 64);
    /* base XOR */
    E[0][1] = 1;  /* 1->2 EXC */
    E[0][5] = 1;  /* 1->6 EXC */
    E[1][4] = 1;  /* 2->5 EXC */
    E[2][3] = 1;  /* 3->4 EXC */
    E[2][5] = 1;  /* 3->6 EXC */
    E[3][4] = 1;  /* 4->5 EXC */
    E[5][1] = 2;  /* 6->2 INH */
    E[5][3] = 2;  /* 6->4 INH */
    /* (5,F+) and (5,F-) match node 5's identity sign */
    E[4][6] = (int8_t)sign5;  /* 5 -> F+ */
    E[4][7] = (int8_t)sign5;  /* 5 -> F- */
    if (variantA) {
        E[0][6] = 1;  /* 1 -> F+ EXC */
        E[2][7] = 1;  /* 3 -> F- EXC */
    }
    if (variantB) {
        E[1][6] = 1;  /* 2 -> F+ EXC */
        E[3][7] = 1;  /* 4 -> F- EXC */
    }
}

/* ===================================================================
 *                              MAIN
 * =================================================================== */
int main(int argc, char **argv) {
    const char *csv = NULL, *out_csv = NULL;
    int induced_mode = 0;
    int quiet = 0;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--out") && i+1 < argc) out_csv = argv[++i];
        else if (!strcmp(argv[i], "--induced")) induced_mode = 1;
        else if (!strcmp(argv[i], "-q")) quiet = 1;
        else if (argv[i][0] != '-') csv = argv[i];
    }
    if (!csv) {
        fprintf(stderr, "Usage: %s input.csv [--out motifs.csv] [--induced] [-q]\n", argv[0]);
        return 1;
    }

    time_t t0 = time(NULL);
    if (!quiet) fprintf(stderr, "Reading %s ...\n", csv);
    csv_stats_t st = read_csv(csv);
    if (!quiet) {
        fprintf(stderr,
            "Total rows: %ld   Nodes: %d\n"
            "  EXC: %ld   INH: %ld   other (skipped): %ld\n"
            "  Source-neurons emitting EXC: %ld   INH: %ld   other: %ld\n",
            st.n_rows, N_nodes,
            st.n_exc, st.n_inh, st.n_other,
            st.n_neur_exc, st.n_neur_inh, st.n_neur_other);
        fprintf(stderr, "Read in %.0fs\n", difftime(time(NULL), t0));
    }
    sort_adjacencies();

    int *r6 = NULL; size_t r6_n = 0, r6_cap = 0;
    for (int i = 0; i < N_nodes; i++) {
        if (graph[i].out_inh_n >= 2) push_int(&r6, &r6_n, &r6_cap, i);
    }
    if (!quiet) fprintf(stderr, "role-6 candidates: %zu\n", r6_n);

    FILE *out_fp = NULL;
    if (out_csv) {
        out_fp = fopen(out_csv, "w");
        if (out_fp) fprintf(out_fp, "n1,n2,n3,n4,n5,n6,fp,fm,variants\n");
    }

    long long raw = 0;
    time_t ts = time(NULL);

    #pragma omp parallel reduction(+:raw)
    {
        long long local = 0;
        #pragma omp for schedule(dynamic, 64)
        for (size_t r = 0; r < r6_n; r++) {
            int n6 = r6[r];
            node_t *N6 = &graph[n6];
            for (size_t i = 0; i < N6->out_inh_n; i++) {
                int n2 = N6->out_inh[i];
                node_t *N2 = &graph[n2];
                for (size_t j = 0; j < N6->out_inh_n; j++) {
                    if (i == j) continue;
                    int n4 = N6->out_inh[j];
                    if (n4 == n2) continue;
                    node_t *N4 = &graph[n4];
                    for (size_t a = 0; a < N2->in_exc_n; a++) {
                        int n1 = N2->in_exc[a];
                        if (n1 == n4 || n1 == n6) continue;
                        if (!has_exc(n1, n6)) continue;
                        for (size_t b = 0; b < N4->in_exc_n; b++) {
                            int n3 = N4->in_exc[b];
                            if (n3 == n1 || n3 == n2 || n3 == n6) continue;
                            /* No canonicalization needed: the (1<->3, 2<->4)
                               base-XOR symmetry would require F+<->F- to also
                               swap, but F+/F- have asymmetric identity
                               constraints (F+ must be ACH, F- must be
                               GABA/GLUT), breaking the symmetry. So each
                               biological signed motif is found exactly once
                               by the search. raw == distinct. */
                            if (!has_exc(n3, n6)) continue;
                            int *A; size_t An;
                            int *B; size_t Bn;
                            if (N2->out_exc_n <= N4->out_exc_n) {
                                A = N2->out_exc; An = N2->out_exc_n;
                                B = N4->out_exc; Bn = N4->out_exc_n;
                            } else {
                                A = N4->out_exc; An = N4->out_exc_n;
                                B = N2->out_exc; Bn = N2->out_exc_n;
                            }
                            for (size_t c = 0; c < An; c++) {
                                int n5 = A[c];
                                if (!bsearch_int(B, Bn, n5)) continue;
                                if (n5 == n1 || n5 == n2 || n5 == n3 ||
                                    n5 == n4 || n5 == n6) continue;

                                /* === XOR core. Now look for signed feedback. === */
                                node_t *N5 = &graph[n5];
                                int sign5;
                                if (N5->is_exc && !N5->is_inh)      sign5 = 1;
                                else if (N5->is_inh && !N5->is_exc) sign5 = 2;
                                else continue;  /* mixed / sink-only: skip */

                                int *n5_out;
                                size_t n5_out_n;
                                if (sign5 == 1) {
                                    n5_out = N5->out_exc; n5_out_n = N5->out_exc_n;
                                } else {
                                    n5_out = N5->out_inh; n5_out_n = N5->out_inh_n;
                                }

                                /* Iterate F+ candidates: nodes that node 5
                                   sends a sign5 edge to. We then filter by
                                   F+'s identity (must be excitatory neuron). */
                                for (size_t fpi = 0; fpi < n5_out_n; fpi++) {
                                    int fp = n5_out[fpi];
                                    if (fp == n1 || fp == n2 || fp == n3 ||
                                        fp == n4 || fp == n5 || fp == n6) continue;
                                    node_t *FP = &graph[fp];
                                    if (!FP->is_exc || FP->is_inh) continue;

                                    int A_fp = has_exc(n1, fp);
                                    int B_fp = has_exc(n2, fp);
                                    if (!A_fp && !B_fp) continue;

                                    /* Iterate F- candidates: same sign5 from n5,
                                       but F- must be inhibitory neuron. */
                                    for (size_t fmi = 0; fmi < n5_out_n; fmi++) {
                                        int fm = n5_out[fmi];
                                        if (fm == fp) continue;
                                        if (fm == n1 || fm == n2 || fm == n3 ||
                                            fm == n4 || fm == n5 || fm == n6) continue;
                                        node_t *FM = &graph[fm];
                                        if (!FM->is_inh || FM->is_exc) continue;

                                        int A_ok = A_fp && has_exc(n3, fm);
                                        int B_ok = B_fp && has_exc(n4, fm);
                                        if (!A_ok && !B_ok) continue;

                                        if (induced_mode) {
                                            int N8[8] = {n1,n2,n3,n4,n5,n6,fp,fm};
                                            int8_t E[8][8];
                                            build_expected_edges(E, sign5, A_ok, B_ok);
                                            if (!check_induced(N8, E)) continue;
                                        }

                                        local++;
                                        if (out_fp) {
                                            const char *vlabel = (A_ok && B_ok) ? "AB" :
                                                                 A_ok            ? "A"  : "B";
                                            #pragma omp critical
                                            fprintf(out_fp,
                                                "%s,%s,%s,%s,%s,%s,%s,%s,%s\n",
                                                id_to_name[n1], id_to_name[n2],
                                                id_to_name[n3], id_to_name[n4],
                                                id_to_name[n5], id_to_name[n6],
                                                id_to_name[fp], id_to_name[fm],
                                                vlabel);
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        raw += local;
    }

    if (!quiet) fprintf(stderr, "Search time: %.0fs\n", difftime(time(NULL), ts));
    if (out_fp) fclose(out_fp);

    long long distinct = raw;  /* canonicalization (n1 < n3) breaks the
                                  σ automorphism, so each motif is found exactly
                                  once. raw == distinct. */

    printf("STATS raw=%lld distinct=%lld rows=%ld nodes=%d "
           "exc_conn=%ld inh_conn=%ld other_conn=%ld "
           "exc_neur=%ld inh_neur=%ld other_neur=%ld\n",
           raw, distinct, st.n_rows, N_nodes,
           st.n_exc, st.n_inh, st.n_other,
           st.n_neur_exc, st.n_neur_inh, st.n_neur_other);
    printf("MOTIF_COUNT_RAW %lld  MOTIF_COUNT_DISTINCT %lld\n", raw, distinct);

    if (!quiet) {
        fprintf(stderr,
            "Raw mapping count:                    %lld\n"
            "Distinct biological motifs:           %lld\n",
            raw, distinct);
    }
    return 0;
}

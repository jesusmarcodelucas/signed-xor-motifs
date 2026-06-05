/*
 * randomize_drosophila.c
 *
 * Drosophila adaptation of the Roberts & Coolen (Phys. Rev. E 85, 046103, 2012)
 * degree-preserving edge-swap randomizer.
 *
 * Same swap kernel as the mouse randomize_directed_signed.c. The only thing
 * that changes is *what counts as a layer*.
 *
 * Drosophila CSV format (5 columns, header optional):
 *   pre_id, post_id, col2, col3, nt_type
 * where nt_type in {ACH, GABA, GLUT, SER, DA, OCT, ...}
 *
 * Two operating modes:
 *
 *   --by-nt   (default, biologically correct)
 *       Each distinct nt_type is its own layer. ACH edges only swap with
 *       ACH edges, GABA with GABA, etc. Preserves Dale's principle: every
 *       neuron retains its identity as ACH/GABA/GLUT/... source.
 *       Layers tracked: any nt_type that appears in the file.
 *
 *   --lump-inhibitory  (mouse-equivalent)
 *       Two layers: EXC (= ACH) and INH (= GABA + GLUT lumped together).
 *       Same null space as randomize_directed_signed.c on the mouse data.
 *       Note: this allows a GABAergic neuron to "become" glutamatergic in
 *       the null, which is biologically problematic for Drosophila but
 *       sometimes desired for cross-species methodological consistency.
 *
 *   --include-other
 *       By default, edges with nt_type not in {ACH, GABA, GLUT} are kept
 *       as-is and not randomized (consistent with the Python script which
 *       lumps them as 'other' and never matches them in the motif). Pass
 *       this flag to also randomize "other" edges within their own per-NT
 *       layers.
 *
 * Acceptance: accept-all (Roberts/Coolen Sec. VI.C: bias is immaterial for
 * sparse fat-tailed biological networks).
 *
 * Compile:
 *   gcc -O3 -march=native -o randomize_drosophila randomize_drosophila.c
 *
 * Usage:
 *   ./randomize_drosophila in.csv --out out.csv --seed N \
 *       [--mixing 100] [--by-nt | --lump-inhibitory] [--include-other] [-q]
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <stdint.h>
#include <stdbool.h>
#include <time.h>
#include "uthash.h"

/* ---------- xoshiro256** PRNG ---------- */
static uint64_t rng_s[4];
static inline uint64_t rotl(uint64_t x, int k){return (x<<k)|(x>>(64-k));}
static uint64_t rng_next(void){
    const uint64_t r = rotl(rng_s[1]*5,7)*9;
    const uint64_t t = rng_s[1]<<17;
    rng_s[2]^=rng_s[0]; rng_s[3]^=rng_s[1];
    rng_s[1]^=rng_s[2]; rng_s[0]^=rng_s[3];
    rng_s[2]^=t; rng_s[3]=rotl(rng_s[3],45);
    return r;
}
static inline uint64_t rng_below(uint64_t n){
    uint64_t x=rng_next();
    __uint128_t m=(__uint128_t)x*(__uint128_t)n;
    uint64_t l=(uint64_t)m;
    if(l<n){
        uint64_t t=(-n)%n;
        while(l<t){ x=rng_next(); m=(__uint128_t)x*(__uint128_t)n; l=(uint64_t)m; }
    }
    return (uint64_t)(m>>64);
}
static void rng_seed(uint64_t s){
    for(int i=0;i<4;i++){
        s+=0x9E3779B97F4A7C15ULL;
        uint64_t z=s;
        z=(z^(z>>30))*0xBF58476D1CE4E5B9ULL;
        z=(z^(z>>27))*0x94D049BB133111EBULL;
        rng_s[i]=z^(z>>31);
    }
}

/* ---------- string -> dense int id ---------- */
typedef struct id_entry {
    char *name;
    int   id;
    UT_hash_handle hh;
} id_entry_t;
static id_entry_t *id_table = NULL;
static int next_id = 0;
static char **id_to_name = NULL;
static int id_to_name_cap = 0;

static int intern_id(const char *s){
    id_entry_t *e;
    HASH_FIND_STR(id_table,s,e);
    if(e) return e->id;
    e=(id_entry_t*)malloc(sizeof(id_entry_t));
    e->name=strdup(s);
    e->id=next_id++;
    HASH_ADD_KEYPTR(hh,id_table,e->name,strlen(e->name),e);
    if(e->id>=id_to_name_cap){
        int nc = id_to_name_cap?id_to_name_cap*2:1024;
        while(nc<=e->id) nc*=2;
        id_to_name=(char**)realloc(id_to_name,nc*sizeof(char*));
        for(int i=id_to_name_cap;i<nc;i++) id_to_name[i]=NULL;
        id_to_name_cap=nc;
    }
    id_to_name[e->id]=e->name;
    return e->id;
}

/* ---------- per-layer edge list + edge hash set ---------- */
typedef struct {
    int *src, *dst;
    size_t n, cap;
} edgelist_t;

static void el_push(edgelist_t *E, int s, int d){
    if(E->n>=E->cap){
        size_t nc = E->cap?E->cap*2:1024;
        E->src=(int*)realloc(E->src,nc*sizeof(int));
        E->dst=(int*)realloc(E->dst,nc*sizeof(int));
        if(!E->src||!E->dst){fprintf(stderr,"OOM el_push\n");exit(1);}
        E->cap=nc;
    }
    E->src[E->n]=s; E->dst[E->n]=d; E->n++;
}

#define EMPTY_KEY 0ULL
typedef struct {
    uint64_t *slots;
    size_t cap, mask, size;
} edgeset_t;

static inline uint64_t pack_key(int s,int d){
    uint64_t us=(uint32_t)(s+1), ud=(uint32_t)(d+1);
    return (us<<32)|ud;
}
static inline uint64_t mix64(uint64_t x){
    x^=x>>30; x*=0xBF58476D1CE4E5B9ULL;
    x^=x>>27; x*=0x94D049BB133111EBULL;
    x^=x>>31; return x;
}
static size_t next_pow2(size_t x){ size_t p=1; while(p<x) p<<=1; return p; }

static void es_init(edgeset_t *S, size_t expected){
    size_t cap = next_pow2(expected*2+16);
    S->slots = (uint64_t*)calloc(cap,sizeof(uint64_t));
    if(!S->slots){fprintf(stderr,"OOM es_init cap=%zu\n",cap);exit(1);}
    S->cap=cap; S->mask=cap-1; S->size=0;
}
static inline bool es_contains(const edgeset_t *S, int s, int d){
    uint64_t k=pack_key(s,d);
    size_t i=mix64(k)&S->mask;
    while(1){
        uint64_t v=S->slots[i];
        if(v==EMPTY_KEY) return false;
        if(v==k) return true;
        i=(i+1)&S->mask;
    }
}
static inline void es_insert(edgeset_t *S, int s, int d){
    uint64_t k=pack_key(s,d);
    size_t i=mix64(k)&S->mask;
    while(1){
        uint64_t v=S->slots[i];
        if(v==EMPTY_KEY){S->slots[i]=k; S->size++; return;}
        if(v==k) return;
        i=(i+1)&S->mask;
    }
}
static inline void es_remove(edgeset_t *S, int s, int d){
    uint64_t k=pack_key(s,d);
    size_t i=mix64(k)&S->mask;
    while(1){
        uint64_t v=S->slots[i];
        if(v==EMPTY_KEY) return;
        if(v==k) break;
        i=(i+1)&S->mask;
    }
    S->slots[i]=EMPTY_KEY; S->size--;
    /* Robin-Hood backward shift */
    size_t j=(i+1)&S->mask;
    while(S->slots[j]!=EMPTY_KEY){
        uint64_t vj=S->slots[j];
        size_t home=mix64(vj)&S->mask;
        size_t dj=(j-home)&S->mask;
        size_t di=(i-home)&S->mask;
        if(dj>di){ S->slots[i]=vj; S->slots[j]=EMPTY_KEY; i=j; }
        j=(j+1)&S->mask;
    }
}

/* ---------- per-layer container indexed by nt-type string ---------- */
#define MAX_NT_LEN 16
typedef struct layer {
    char nt[MAX_NT_LEN];   /* e.g. "ACH", "GABA", "GLUT", or "EXC", "INH" */
    int  randomize;        /* 1 = swap edges in this layer; 0 = leave alone */
    edgelist_t E;
    edgeset_t  S;
    UT_hash_handle hh;
} layer_t;

static layer_t *layers = NULL;
static int n_layers = 0;

static layer_t *get_layer(const char *nt, int randomize){
    layer_t *L;
    HASH_FIND_STR(layers, nt, L);
    if(L) return L;
    L = (layer_t*)calloc(1,sizeof(layer_t));
    strncpy(L->nt, nt, MAX_NT_LEN-1);
    L->randomize = randomize;
    HASH_ADD_STR(layers, nt, L);
    n_layers++;
    return L;
}

/* ---------- options ---------- */
typedef enum { MODE_BY_NT, MODE_LUMP_INH } rand_mode_t;
static rand_mode_t mode = MODE_BY_NT;
static int    include_other = 0;
static int    quiet = 0;

/* All edges, in original CSV order, so we can write them back faithfully.
   Each entry stores: (src, dst, layer_ptr, edge_index_in_layer). */
typedef struct {
    int sid;
    int tid;
    layer_t *L;          /* NULL if this edge is not in any layer (e.g. SER
                            when --include-other is off) */
    size_t   idx;        /* index in L->E (only valid if L != NULL) */
    char     nt[MAX_NT_LEN];
    char     col2[64];   /* preserve original cols 2/3 for round-trip */
    char     col3[64];
} all_edge_t;

static all_edge_t *all_edges = NULL;
static size_t all_n = 0, all_cap = 0;

static void all_push(int s, int t, layer_t *L, size_t idx,
                     const char *nt, const char *c2, const char *c3){
    if(all_n>=all_cap){
        size_t nc=all_cap?all_cap*2:1024;
        all_edges=(all_edge_t*)realloc(all_edges,nc*sizeof(all_edge_t));
        if(!all_edges){fprintf(stderr,"OOM all_push\n");exit(1);}
        all_cap=nc;
    }
    all_edges[all_n].sid=s;
    all_edges[all_n].tid=t;
    all_edges[all_n].L=L;
    all_edges[all_n].idx=idx;
    strncpy(all_edges[all_n].nt, nt, MAX_NT_LEN-1); all_edges[all_n].nt[MAX_NT_LEN-1]=0;
    strncpy(all_edges[all_n].col2, c2, 63); all_edges[all_n].col2[63]=0;
    strncpy(all_edges[all_n].col3, c3, 63); all_edges[all_n].col3[63]=0;
    all_n++;
}

/* ---------- helpers ---------- */
static void rstrip(char *s){
    size_t n=strlen(s);
    while(n && (s[n-1]=='\r'||s[n-1]=='\n'||s[n-1]==' '||s[n-1]=='\t')) s[--n]=0;
}
static int split_csv(char *line, char *out[], int max_fields){
    int k=0; char *p=line;
    while(k<max_fields){
        out[k++]=p;
        char *c=strchr(p,',');
        if(!c) break;
        *c=0; p=c+1;
    }
    return k;
}
/* Trim trailing whitespace/quotes from a field; returns pointer into
   possibly-shifted string (we don't allocate). */
static char *trim_field(char *s){
    while(*s && (isspace((unsigned char)*s) || *s=='"')) s++;
    size_t n=strlen(s);
    while(n && (isspace((unsigned char)*s) || s[n-1]=='"' || s[n-1]==' '||s[n-1]=='\t')) s[--n]=0;
    return s;
}

/* Decide which layer name an edge of nt_type X belongs to under current mode.
   Returns: layer name string, OR NULL if this edge should not be randomized
   (e.g. nt_type "SER" with --include-other off). */
static const char *layer_for_nt(const char *nt){
    static const char *EXC = "EXC", *INH = "INH";
    if(mode == MODE_LUMP_INH){
        if(!strcmp(nt,"ACH")) return EXC;
        if(!strcmp(nt,"GABA")||!strcmp(nt,"GLUT")) return INH;
        return include_other ? nt : NULL;
    } else {
        /* MODE_BY_NT: each nt is its own layer, but only the canonical
           three (ACH, GABA, GLUT) are randomized by default. */
        if(!strcmp(nt,"ACH")||!strcmp(nt,"GABA")||!strcmp(nt,"GLUT")) return nt;
        return include_other ? nt : NULL;
    }
}

/* ---------- I/O ---------- */
static int header_present(const char *path){
    FILE *fp=fopen(path,"r"); if(!fp) return 0;
    char line[1024];
    int hp=0;
    if(fgets(line,sizeof(line),fp)){
        rstrip(line);
        char *c=strchr(line,',');
        if(c){
            *c=0;
            char *t=trim_field(line);
            int has_nondigit=0;
            for(char *q=t; *q; q++){
                if(!isdigit((unsigned char)*q) && *q!='-' && *q!='+'){
                    has_nondigit=1; break;
                }
            }
            if(has_nondigit && *t) hp=1;
        }
    }
    fclose(fp);
    return hp;
}

static char header_line[1024] = "";
static int  has_header = 0;

static void read_csv(const char *path){
    has_header = header_present(path);
    FILE *fp=fopen(path,"r"); if(!fp){perror(path);exit(1);}
    static char io_buf[8*1024*1024];
    setvbuf(fp,io_buf,_IOFBF,sizeof(io_buf));

    char line[1024];
    if(has_header){
        if(fgets(header_line,sizeof(header_line),fp)){
            rstrip(header_line);
        }
    }

    while(fgets(line,sizeof(line),fp)){
        rstrip(line);
        if(!line[0]) continue;
        char *fields[8];
        int nf=split_csv(line,fields,8);
        if(nf<5) continue;

        char *pre = trim_field(fields[0]);
        char *post= trim_field(fields[1]);
        char *c2  = trim_field(fields[2]);
        char *c3  = trim_field(fields[3]);
        char *nt  = trim_field(fields[4]);

        int sid = intern_id(pre);
        int tid = intern_id(post);

        const char *lname = layer_for_nt(nt);
        if(!lname){
            all_push(sid, tid, NULL, 0, nt, c2, c3);
            continue;
        }
        layer_t *L = get_layer(lname, 1);
        size_t idx = L->E.n;
        el_push(&L->E, sid, tid);
        all_push(sid, tid, L, idx, nt, c2, c3);
    }
    fclose(fp);
}

static void build_edgesets(void){
    layer_t *L, *tmp;
    HASH_ITER(hh, layers, L, tmp){
        es_init(&L->S, L->E.n);
        for(size_t i=0;i<L->E.n;i++) es_insert(&L->S, L->E.src[i], L->E.dst[i]);
        if(!quiet)
            fprintf(stderr, "  layer %-6s: %zu edges (set load %.2f)\n",
                    L->nt, L->E.n, (double)L->S.size/L->S.cap);
    }
}

/* ---------- swap kernel ---------- */
static inline int try_swap(layer_t *L){
    edgelist_t *E = &L->E;
    edgeset_t  *S = &L->S;
    if(E->n<2) return 0;
    size_t i=rng_below(E->n), j=rng_below(E->n);
    if(i==j) return 0;
    int a=E->src[i], b=E->dst[i];
    int c=E->src[j], d=E->dst[j];
    if(a==c||a==d||b==c||b==d) return 0;
    if(es_contains(S,a,d)) return 0;
    if(es_contains(S,c,b)) return 0;
    es_remove(S,a,b); es_remove(S,c,d);
    es_insert(S,a,d); es_insert(S,c,b);
    E->dst[i]=d; E->dst[j]=b;
    return 1;
}

static void randomize_layer(layer_t *L, int mixing){
    long long att = (long long)mixing * (long long)L->E.n;
    long long acc = 0;
    long long step = att/10; if(step<1) step=1;
    time_t t0 = time(NULL);
    for(long long t=0;t<att;t++){
        if(try_swap(L)) acc++;
        if(!quiet && (t+1)%step==0){
            time_t now=time(NULL);
            double sec=difftime(now,t0);
            double rate = sec>0 ? (t+1)/sec : 0;
            fprintf(stderr, "    %s: %lld/%lld (%.0f%%, %.1fM/s)\n",
                    L->nt, t+1, att, 100.0*(t+1)/att, rate/1e6);
        }
    }
    if(!quiet)
        fprintf(stderr, "  %s done: %lld/%lld accepted (%.2f%%)\n",
                L->nt, acc, att, 100.0*acc/att);
}

/* ---------- write output ---------- *
 *
 * Square swap on edges (a->b) and (c->d) becomes (a->d) and (c->b).
 * Source a still has the same number of out-edges (one), it just goes to d
 * now instead of b. Source c still has one out-edge, going to b instead of d.
 * Target b has its in-edge moved from src=a to src=c. Target d has its
 * in-edge moved from src=c to src=a. So out-degree per source is preserved
 * AND in-degree per target is preserved.
 *
 * To write the result, we walk the original all_edges[] and substitute the
 * new dst from the layer's edgelist when L != NULL. This also faithfully
 * preserves the original edge order in the output, which is nice for diff.
 */
static void write_csv(const char *path){
    FILE *fp=fopen(path,"w"); if(!fp){perror(path);exit(1);}
    static char io_buf[8*1024*1024];
    setvbuf(fp,io_buf,_IOFBF,sizeof(io_buf));
    if(has_header && header_line[0]) fprintf(fp,"%s\n",header_line);
    for(size_t i=0;i<all_n;i++){
        all_edge_t *e = &all_edges[i];
        int s = e->sid;
        int d = e->tid;
        if(e->L) d = e->L->E.dst[e->idx];
        fprintf(fp, "%s,%s,%s,%s,%s\n",
                id_to_name[s], id_to_name[d],
                e->col2, e->col3, e->nt);
    }
    fclose(fp);
}

/* ---------- main ---------- */
int main(int argc, char **argv){
    const char *in_csv=NULL, *out_csv=NULL;
    uint64_t seed = (uint64_t)time(NULL);
    int mixing = 100;
    for(int i=1;i<argc;i++){
        if(!strcmp(argv[i],"--out") && i+1<argc) out_csv=argv[++i];
        else if(!strcmp(argv[i],"--seed") && i+1<argc) seed=strtoull(argv[++i],NULL,10);
        else if(!strcmp(argv[i],"--mixing") && i+1<argc) mixing=atoi(argv[++i]);
        else if(!strcmp(argv[i],"--by-nt")) mode = MODE_BY_NT;
        else if(!strcmp(argv[i],"--lump-inhibitory")) mode = MODE_LUMP_INH;
        else if(!strcmp(argv[i],"--include-other")) include_other = 1;
        else if(!strcmp(argv[i],"-q")) quiet = 1;
        else if(argv[i][0] != '-') in_csv = argv[i];
    }
    if(!in_csv || !out_csv){
        fprintf(stderr,
            "Usage: %s in.csv --out out.csv --seed N [--mixing K]\n"
            "       [--by-nt | --lump-inhibitory] [--include-other] [-q]\n",
            argv[0]);
        return 1;
    }
    rng_seed(seed);

    if(!quiet){
        fprintf(stderr, "Mode: %s\n",
                mode==MODE_BY_NT?"by-nt (per-NT-type layers, Dale-preserving)":
                                 "lump-inhibitory (mouse-equivalent EXC/INH layers)");
        fprintf(stderr, "Reading %s ...\n", in_csv);
    }
    time_t t0 = time(NULL);
    read_csv(in_csv);
    if(!quiet)
        fprintf(stderr,"Read %zu edges into %d layer(s) in %.0fs.\n",
                all_n, n_layers, difftime(time(NULL),t0));

    if(!quiet) fprintf(stderr, "Building edge hash sets...\n");
    build_edgesets();

    layer_t *L,*tmp;
    HASH_ITER(hh, layers, L, tmp){
        if(!L->randomize) continue;
        if(!quiet) fprintf(stderr, "Randomizing layer %s (mixing=%d)...\n", L->nt, mixing);
        randomize_layer(L, mixing);
    }

    if(!quiet) fprintf(stderr, "Writing %s ...\n", out_csv);
    write_csv(out_csv);
    if(!quiet) fprintf(stderr, "Done.\n");
    return 0;
}

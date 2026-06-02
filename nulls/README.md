# nulls — degree- and sign-preserving null models

C tools that generate randomized connectomes for testing whether the
signed-XOR motif is enriched beyond chance. Each randomizer rewires the input
edge list while preserving the in/out degree of every node and the sign
(excitatory/inhibitory identity) of every edge, so motif counts on the
randomized graphs form the null distribution against which the observed count
is compared (Z-score, fold-enrichment).

| file                    | input format                          |
|-------------------------|---------------------------------------|
| `randomize_drosophila.c`| FlyWire / FAFB edge list              |
| `randomize_celegans.c`  | C. elegans (Cook et al.) connectome   |

## Build

Requires a C compiler with OpenMP. The bundled `uthash.h` is the only
dependency.

```bash
gcc -O3 -march=native -fopenmp -o randomize_drosophila randomize_drosophila.c -lm
gcc -O3 -march=native -fopenmp -o randomize_celegans   randomize_celegans.c   -lm
```

## Workflow

Generate N randomized graphs, run the matching finder from `../motif_count/`
on each, and compare the observed motif count to the null distribution. See
the header comment in each `.c` file for command-line options (number of
swaps, number of nulls, random seed).

## Data

No connectome data is included. Supply your own edge-list CSV obtained from the
original provider under its terms. See `../data/README.md`.

## Third-party

`uthash.h` © Troy D. Hanson, revised BSD license (notice retained in the file
header). Not covered by this repository's CC BY-NC 4.0 license.

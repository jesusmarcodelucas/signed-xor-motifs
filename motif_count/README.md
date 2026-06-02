# motif_count — signed-XOR motif finders

C tools that enumerate the 8-node, 12-edge signed-XOR motif in a connectome
edge list. One finder per dataset format; the motif definition is identical
across them.

| file                            | input format                                  |
|---------------------------------|-----------------------------------------------|
| `signed_xor_motif_count.c`      | Allen V1 / MICrONS proofread CSV (5 columns)  |
| `drosophila_signed_xor_motif.c` | FlyWire / FAFB edge list                       |
| `celegans_signed_xor_motif.c`   | C. elegans (Cook et al.) connectome            |

## Build

Requires a C compiler with OpenMP. The bundled `uthash.h` is the only
dependency.

```bash
gcc -O3 -march=native -fopenmp -o signed_xor_motif_count signed_xor_motif_count.c -lm
gcc -O3 -march=native -fopenmp -o drosophila_signed_xor_motif drosophila_signed_xor_motif.c -lm
gcc -O3 -march=native -fopenmp -o celegans_signed_xor_motif celegans_signed_xor_motif.c -lm
```

## Run

Each tool takes an edge-list CSV and prints per-type motif counts. See the
header comment in each `.c` file for its command-line options (e.g.
`-n <max_per_type>`, `--weight-stats <prefix>` in `signed_xor_motif_count.c`).

```bash
OMP_NUM_THREADS=$(nproc) ./signed_xor_motif_count <edges.csv>
```

## Data

No connectome data is included. Obtain the datasets from their original
providers (MICrONS, FlyWire/FAFB, C. elegans / Cook et al.) under their
respective terms, and supply your own edge-list CSV. See `../data/README.md`.

## Third-party

`uthash.h` © Troy D. Hanson, distributed under the revised BSD license
(notice retained in the file header). Not covered by this repository's
CC BY-NC 4.0 license.

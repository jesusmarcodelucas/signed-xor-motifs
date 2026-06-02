"""
Experiment 2 -- Fast-spiking (PV) requirement and robustness.

The preprint explicitly flags that parvalbumin interneurons fire 2-3x faster
than pyramidal cells and that this "would substantially affect functional motif
behaviour" -- but does not model it. Here we test it directly.

We sweep the membrane time constant of the two inhibitory neurons (the INH pivot
and the F- neuron) from fast-spiking PV values (~5-6 ms) up to pyramidal values
(~15 ms) and, at each value, measure the fraction of multi-seed trials whose full
signed-XOR truth table is correct.

We also compare two named compositions:
    - PV_PIVOT  (INH = i4Pvalb, fast-spiking): expected to work
    - RANK1     (INH = i4Sst,  not fast-spiking): expected to fail

Outputs:
    exp2_pv_sweep.csv
    exp2_pv_sweep.png
    exp2_summary.json
"""

from __future__ import annotations

import json
import os
import csv

import numpy as np

from signed_xor import run_trial, is_correct, PV_PIVOT, RANK1_COMPOSITION
from signed_xor.readout import DEFAULT_SPIKE_THRESHOLD

RESULTS = os.path.join(os.path.dirname(__file__), "..", "results")
CONDS = [(0, 0), (1, 0), (0, 1), (1, 1)]


def _table_success_rate(inh_tau, n_seeds=15, noise_hz=20.0):
    ok = []
    for seed in range(n_seeds):
        full = all(
            is_correct(s1, s2,
                       run_trial(s1, s2, composition=PV_PIVOT,
                                 inh_tau_override_ms=inh_tau,
                                 noise_rate_hz=noise_hz, seed=2000 + seed)["counts"],
                       DEFAULT_SPIKE_THRESHOLD)
            for s1, s2 in CONDS
        )
        ok.append(full)
    return float(np.mean(ok))


def main():
    os.makedirs(RESULTS, exist_ok=True)
    taus = [4, 5, 6, 7, 8, 9, 10, 12, 15]
    n_seeds = 15

    success = [_table_success_rate(t, n_seeds=n_seeds) for t in taus]
    print("  inhibitory tau_m (ms) -> truth-table success rate (n=%d seeds):" % n_seeds)
    for t, s in zip(taus, success):
        print(f"    {t:>4} ms : {s*100:5.1f}%   "
              + ("fast-spiking PV regime" if t <= 6 else
                 "intermediate" if t <= 8 else "pyramidal-speed (slow)"))

    # largest tau that still gives a perfect table (working ceiling)
    working = [t for t, s in zip(taus, success) if s >= 0.99]
    working_max = max(working) if working else None

    # composition comparison at native tau (no override)
    pv_native = all(
        is_correct(s1, s2, run_trial(s1, s2, composition=PV_PIVOT, seed=0)["counts"],
                   DEFAULT_SPIKE_THRESHOLD) for s1, s2 in CONDS
    )
    rank1_native = all(
        is_correct(s1, s2, run_trial(s1, s2, composition=RANK1_COMPOSITION, seed=0)["counts"],
                   DEFAULT_SPIKE_THRESHOLD) for s1, s2 in CONDS
    )
    print(f"\n  PV-pivot (i4Pvalb) composition correct (native): {pv_native}")
    print(f"  Rank-1 (i4Sst, not fast-spiking) composition correct (native): {rank1_native}")

    # outputs
    with open(os.path.join(RESULTS, "exp2_pv_sweep.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["inh_tau_ms", "success_rate"])
        for t, s in zip(taus, success):
            w.writerow([t, s])

    from signed_xor import plotting
    plotting.plot_pv_sweep(taus, success, n_seeds,
                           os.path.join(RESULTS, "exp2_pv_sweep.png"),
                           working_max_tau=working_max)

    summary = {
        "taus_ms": taus,
        "success_rate": success,
        "working_max_tau_ms": working_max,
        "pv_pivot_correct_native": bool(pv_native),
        "rank1_sst_correct_native": bool(rank1_native),
        "n_seeds": n_seeds,
    }
    with open(os.path.join(RESULTS, "exp2_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    return summary


if __name__ == "__main__":
    main()

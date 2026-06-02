"""
Experiment 1 -- Static truth-table validation.

Drives the four binary input conditions (S1, S2) in {0,1}^2 through the spiking
signed-XOR motif (PV-pivot composition) and checks that the decoded XOR / F+ / F-
output matches the expected signed-error truth table. Also runs a multi-seed
version with background Poisson noise to report a success rate with error bars
(addressing the "single seed, no error bars" caveat in the preprint).

Outputs (written to results/):
    exp1_truth_table.csv        per-condition spike counts and correctness
    exp1_rasters.png            4-panel spike rasters
    exp1_truth_table_heatmap.png
    exp1_summary.json           machine-readable summary for the LaTeX snippet
"""

from __future__ import annotations

import json
import os
import csv

import numpy as np

from signed_xor import run_trial, decode, is_correct, error_label, ROLE_ORDER, PV_PIVOT
from signed_xor.readout import DEFAULT_SPIKE_THRESHOLD
from signed_xor import plotting

RESULTS = os.path.join(os.path.dirname(__file__), "..", "results")
CONDS = [(0, 0), (1, 0), (0, 1), (1, 1)]


def main():
    os.makedirs(RESULTS, exist_ok=True)
    th = DEFAULT_SPIKE_THRESHOLD

    # --- deterministic truth table -------------------------------------
    results_by_cond = {}
    counts_by_cond = {}
    rows = []
    all_correct = True
    for s1, s2 in CONDS:
        res = run_trial(s1, s2, composition=PV_PIVOT, record_traces=True, seed=0)
        results_by_cond[(s1, s2)] = res
        counts = res["counts"]
        counts_by_cond[(s1, s2)] = counts
        xor, fp, fm = decode(counts, th)
        ok = is_correct(s1, s2, counts, th)
        all_correct &= ok
        rows.append({
            "S1": s1, "S2": s2,
            **{role: int(counts[k]) for k, role in enumerate(ROLE_ORDER)},
            "XOR_active": int(xor), "Fplus_active": int(fp), "Fminus_active": int(fm),
            "expected": error_label(s1, s2), "correct": int(ok),
        })
        print(f"  (S1={s1}, S2={s2})  "
              + "  ".join(f"{role}={int(counts[k])}" for k, role in enumerate(ROLE_ORDER))
              + f"  ->  XOR={int(xor)} F+={int(fp)} F-={int(fm)}  "
              + ("[OK]" if ok else "[FAIL]"))

    print(f"\n  Deterministic truth table correct on all 4 conditions: {all_correct}")

    # --- multi-seed noisy version (error bars) -------------------------
    n_seeds = 25
    noisy_correct = []
    for seed in range(n_seeds):
        ok_all = all(
            is_correct(s1, s2,
                       run_trial(s1, s2, composition=PV_PIVOT,
                                 noise_rate_hz=20.0, seed=1000 + seed)["counts"], th)
            for s1, s2 in CONDS
        )
        noisy_correct.append(ok_all)
    noisy_rate = float(np.mean(noisy_correct))
    print(f"  Noisy (20 Hz background, n={n_seeds} seeds) full-table success rate: "
          f"{noisy_rate * 100:.1f}%")

    # --- write outputs -------------------------------------------------
    with open(os.path.join(RESULTS, "exp1_truth_table.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    plotting.plot_condition_rasters(results_by_cond, th, 100.0,
                                    os.path.join(RESULTS, "exp1_rasters.png"))
    plotting.plot_truth_table(counts_by_cond, th,
                              os.path.join(RESULTS, "exp1_truth_table_heatmap.png"))

    summary = {
        "deterministic_all_correct": bool(all_correct),
        "spike_threshold": th,
        "noisy_seeds": n_seeds,
        "noisy_background_hz": 20.0,
        "noisy_success_rate": noisy_rate,
        "counts": {f"{s1}{s2}": [int(x) for x in counts_by_cond[(s1, s2)]] for s1, s2 in CONDS},
        "roles": ROLE_ORDER,
    }
    with open(os.path.join(RESULTS, "exp1_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    return summary


if __name__ == "__main__":
    main()

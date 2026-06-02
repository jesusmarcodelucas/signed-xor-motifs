"""
Experiment 3 -- Graded (rate-coded) signed error.

The static truth table uses binary inputs. The preprint also defines a graded
form (Eqs. 393-394):

    F+ = max(0, x - r)        (under-shoot: target x exceeds prediction r)
    F- = max(0, r - x)        (over-shoot:  prediction r exceeds target x)

Here the target (S1) is held ON at a fixed drive while the prediction (S2) drive
is swept from below to above the target. We measure the F+ and F- spike counts.
The expected signature is: F+ active when prediction < target, a zero-crossing
("homeostasis") where prediction ~ target, and F- active when prediction > target.

Outputs:
    exp3_graded_error.csv
    exp3_graded_error.png
    exp3_summary.json
"""

from __future__ import annotations

import json
import os
import csv

import numpy as np

from signed_xor import run_trial, PV_PIVOT, DEFAULT_DRIVE

RESULTS = os.path.join(os.path.dirname(__file__), "..", "results")


def main():
    os.makedirs(RESULTS, exist_ok=True)
    # target fixed ON (s1 = 1.0). Sweep prediction s2 from 0 to ~1.6x target.
    target = 1.0
    pred_levels = np.linspace(0.0, 1.6, 17)
    fp_counts, fm_counts = [], []
    for s2 in pred_levels:
        res = run_trial(target, float(s2), composition=PV_PIVOT, seed=0)
        c = res["counts"]
        fp_counts.append(int(c[6]))  # Fp
        fm_counts.append(int(c[7]))  # Fm

    fp_counts = np.array(fp_counts)
    fm_counts = np.array(fm_counts)

    # find the approximate zero-crossing / minimum-error prediction level
    total_err = fp_counts + fm_counts
    match_idx = int(np.argmin(total_err))
    match_level = float(pred_levels[match_idx])

    print("  prediction level | F+  F-")
    for lvl, fp, fm in zip(pred_levels, fp_counts, fm_counts):
        tag = "  <- min error (match)" if abs(lvl - match_level) < 1e-9 else ""
        print(f"      {lvl:4.2f}        | {fp:>2}  {fm:>2}{tag}")

    # qualitative checks
    under = pred_levels < match_level - 1e-9
    over = pred_levels > match_level + 1e-9
    fp_dominates_under = bool(np.all(fp_counts[under] >= fm_counts[under])) if under.any() else True
    fm_dominates_over = bool(np.all(fm_counts[over] >= fp_counts[over])) if over.any() else True
    print(f"\n  F+ dominates on under-shoot side: {fp_dominates_under}")
    print(f"  F- dominates on over-shoot side:  {fm_dominates_over}")
    print(f"  minimum-error (match) prediction level ~ {match_level:.2f} x target")

    with open(os.path.join(RESULTS, "exp3_graded_error.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["prediction_level", "Fplus_count", "Fminus_count"])
        for lvl, fp, fm in zip(pred_levels, fp_counts, fm_counts):
            w.writerow([f"{lvl:.4f}", int(fp), int(fm)])

    from signed_xor import plotting
    plotting.plot_graded_error(pred_levels, fp_counts, fm_counts, match_level,
                               os.path.join(RESULTS, "exp3_graded_error.png"))

    summary = {
        "target_level": target,
        "prediction_levels": [float(x) for x in pred_levels],
        "Fplus_counts": [int(x) for x in fp_counts],
        "Fminus_counts": [int(x) for x in fm_counts],
        "match_level": match_level,
        "Fplus_dominates_undershoot": fp_dominates_under,
        "Fminus_dominates_overshoot": fm_dominates_over,
    }
    with open(os.path.join(RESULTS, "exp3_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    return summary


if __name__ == "__main__":
    main()

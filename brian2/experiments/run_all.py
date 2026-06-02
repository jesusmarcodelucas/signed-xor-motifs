"""Run the full functional-validation suite (experiments 1-3) in order."""

import os
import sys

# allow `python experiments/run_all.py` from repo root
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from experiments import exp1_truth_table, exp2_pv_robustness, exp3_graded_error  # noqa: E402


def main():
    print("=" * 70)
    print("Experiment 1: static signed-XOR truth table")
    print("=" * 70)
    exp1_truth_table.main()

    print("\n" + "=" * 70)
    print("Experiment 2: fast-spiking (PV) requirement and robustness")
    print("=" * 70)
    exp2_pv_robustness.main()

    print("\n" + "=" * 70)
    print("Experiment 3: graded (rate-coded) signed error")
    print("=" * 70)
    exp3_graded_error.main()

    print("\nAll experiments complete. See results/ for figures, CSVs, and JSON summaries.")


if __name__ == "__main__":
    main()

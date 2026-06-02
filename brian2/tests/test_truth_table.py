"""
Tests for the signed-XOR functional validation.

These assert the core scientific claims:
  1. The PV-pivot motif computes the full signed-XOR truth table.
  2. F+ fires on under-shoot only; F- fires on over-shoot only.
  3. Homeostasis (matched inputs) produces no error signal.
  4. Fast-spiking inhibition is required: slowing the inhibitory neurons to
     pyramidal speed breaks the table.

Run with:  pytest -q
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pytest

from signed_xor import run_trial, decode, is_correct, PV_PIVOT
from signed_xor.readout import DEFAULT_SPIKE_THRESHOLD, TRUTH_TABLE

CONDS = [(0, 0), (1, 0), (0, 1), (1, 1)]


@pytest.mark.parametrize("s1,s2", CONDS)
def test_truth_table_each_condition(s1, s2):
    counts = run_trial(s1, s2, composition=PV_PIVOT, seed=0)["counts"]
    assert decode(counts, DEFAULT_SPIKE_THRESHOLD) == TRUTH_TABLE[(s1, s2)]


def test_fplus_only_on_undershoot():
    """F+ fires for (1,0) and for no other binary condition."""
    fires = {}
    for s1, s2 in CONDS:
        counts = run_trial(s1, s2, composition=PV_PIVOT, seed=0)["counts"]
        _, fp, _ = decode(counts, DEFAULT_SPIKE_THRESHOLD)
        fires[(s1, s2)] = fp
    assert fires[(1, 0)] is True
    assert fires[(0, 0)] is False
    assert fires[(0, 1)] is False
    assert fires[(1, 1)] is False


def test_fminus_only_on_overshoot():
    """F- fires for (0,1) and for no other binary condition."""
    fires = {}
    for s1, s2 in CONDS:
        counts = run_trial(s1, s2, composition=PV_PIVOT, seed=0)["counts"]
        _, _, fm = decode(counts, DEFAULT_SPIKE_THRESHOLD)
        fires[(s1, s2)] = fm
    assert fires[(0, 1)] is True
    assert fires[(0, 0)] is False
    assert fires[(1, 0)] is False
    assert fires[(1, 1)] is False


def test_homeostasis_no_error():
    """Matched inputs (0,0) and (1,1) produce no XOR / F+ / F- output."""
    for s1, s2 in [(0, 0), (1, 1)]:
        counts = run_trial(s1, s2, composition=PV_PIVOT, seed=0)["counts"]
        xor, fp, fm = decode(counts, DEFAULT_SPIKE_THRESHOLD)
        assert not (xor or fp or fm)


def test_slow_inhibition_breaks_table():
    """Slowing inhibition to pyramidal speed (15 ms) must break the table."""
    all_ok = all(
        is_correct(s1, s2,
                   run_trial(s1, s2, composition=PV_PIVOT,
                             inh_tau_override_ms=15.0, seed=0)["counts"],
                   DEFAULT_SPIKE_THRESHOLD)
        for s1, s2 in CONDS
    )
    assert all_ok is False

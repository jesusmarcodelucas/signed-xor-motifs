"""
Readout: decode the motif's spiking output into the logical signed-XOR result.

A channel (XOR, F+, F-) is declared *active* in a trial if it emits at least
``threshold`` spikes within the integration window. The expected truth table
(Section 2 of the preprint) is:

    (S1, S2)     XOR   F+   F-     meaning
    (0, 0)        0     0    0     homeostasis (no error, no plasticity)
    (1, 0)        1     1    0     under-shoot  -> LTP   (F+)
    (0, 1)        1     0    1     over-shoot   -> LTD   (F-)
    (1, 1)        0     0    0     homeostasis (matched, no error)
"""

from __future__ import annotations

DEFAULT_SPIKE_THRESHOLD = 2

# role index lookup for [E1, E3, E2, E4, INH, XOR, Fp, Fm]
_IDX = {"E1": 0, "E3": 1, "E2": 2, "E4": 3, "INH": 4, "XOR": 5, "Fp": 6, "Fm": 7}

# expected (XOR, Fp, Fm) booleans per binary input condition
TRUTH_TABLE = {
    (0, 0): (False, False, False),
    (1, 0): (True, True, False),
    (0, 1): (True, False, True),
    (1, 1): (False, False, False),
}


def decode(counts, threshold=DEFAULT_SPIKE_THRESHOLD):
    """Return (xor_active, fp_active, fm_active) booleans from a count vector."""
    xor = counts[_IDX["XOR"]] >= threshold
    fp = counts[_IDX["Fp"]] >= threshold
    fm = counts[_IDX["Fm"]] >= threshold
    return bool(xor), bool(fp), bool(fm)


def is_correct(s1, s2, counts, threshold=DEFAULT_SPIKE_THRESHOLD):
    """True if the decoded output matches the expected signed-XOR truth table."""
    return decode(counts, threshold) == TRUTH_TABLE[(int(s1), int(s2))]


def error_label(s1, s2):
    """Human-readable label for an input condition."""
    return {
        (0, 0): "homeostasis (no error)",
        (1, 0): "under-shoot -> LTP (F+)",
        (0, 1): "over-shoot -> LTD (F-)",
        (1, 1): "homeostasis (matched)",
    }[(int(s1), int(s2))]

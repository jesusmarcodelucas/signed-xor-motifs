"""
signed_xor: functional (spiking) validation of the signed-XOR neuronal motif.

A minimal, dependency-light spiking model of the 8-neuron signed-XOR motif built
from mouse-cortex (Allen V1) cell types. It demonstrates that the motif computes
the signed error (F+ / F- / homeostasis) when implemented with biologically
plausible leaky integrate-and-fire neurons, complementing the structural
graph analysis of the preprint.
"""

from .motif import run_trial, DEFAULT_WEIGHTS, DEFAULT_DRIVE
from .readout import decode, is_correct, TRUTH_TABLE, error_label
from .celltypes import PV_PIVOT, RANK1_COMPOSITION, CELL_TYPES, ROLE_ORDER, composition_table

__all__ = [
    "run_trial", "DEFAULT_WEIGHTS", "DEFAULT_DRIVE",
    "decode", "is_correct", "TRUTH_TABLE", "error_label",
    "PV_PIVOT", "RANK1_COMPOSITION", "CELL_TYPES", "ROLE_ORDER", "composition_table",
]

__version__ = "1.0.0"

"""
Cell-type-specific leaky integrate-and-fire (LIF) parameters for the
mouse-cortex neuron classes used in the Allen Institute V1 biophysical model
(Billeh et al., 2020).

The key biological fact encoded here is that parvalbumin-expressing (Pvalb)
interneurons are *fast-spiking*: they fire roughly 2-3x faster than pyramidal
cells (Cardin et al., 2009; Atallah et al., 2012). In a leaky integrate-and-fire
abstraction this is captured by a shorter membrane time constant ``tau_m`` and a
shorter absolute refractory period. Somatostatin (Sst) cells are *not*
fast-spiking; they are low-threshold, adapting interneurons and are modelled
here with pyramidal-like (slow) kinetics. These differences are exactly the
"functional" properties that the structural-only analysis of the preprint does
not model, and that this validation makes explicit.

All values are normalised LIF parameters (membrane potential in mV, with
E_L = -70 mV, V_th = -50 mV, V_reset = -65 mV defined in :mod:`signed_xor.motif`).
They are chosen to be biologically reasonable for mouse cortex rather than fit to
any single dataset; the qualitative conclusions of the validation depend on the
*ordering* of the time constants (PV fast vs pyramidal slow), not on their exact
values, which is precisely the robustness question that ``exp2_pv_robustness``
quantifies.

References
----------
Billeh, Y. N. et al. (2020). Neuron 106(3):388-403.
Cardin, J. A. et al. (2009). Nature 459:663-667.
Atallah, B. V. et al. (2012). Neuron 73(1):159-170.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class CellType:
    """LIF parameters for a single mouse-cortex cell type.

    Parameters
    ----------
    name : str
        Allen V1 cell-type label (e.g. ``"e4other"``, ``"i4Pvalb"``).
    tau_m_ms : float
        Membrane time constant in ms. Smaller = faster integration / higher
        achievable firing rate (the fast-spiking PV signature).
    refractory_ms : float
        Absolute refractory period in ms.
    is_inhibitory : bool
        True for inhibitory cell types (names beginning with ``"i"``), following
        the Dale's-law convention used throughout the preprint.
    layer : str
        Cortical layer, for plotting/annotation only.
    description : str
        Human-readable phenotype.
    """

    name: str
    tau_m_ms: float
    refractory_ms: float
    is_inhibitory: bool
    layer: str
    description: str


# --- Mouse V1 cell-type catalogue ------------------------------------------
# Excitatory (pyramidal-like, regular-spiking, slower membrane):
CELL_TYPES = {
    "e4other": CellType("e4other", 12.0, 3.0, False, "L4", "L4 excitatory (regular-spiking)"),
    "e4Scnn1a": CellType("e4Scnn1a", 12.0, 3.0, False, "L4", "L4 excitatory Scnn1a (regular-spiking)"),
    "e23Cux2": CellType("e23Cux2", 15.0, 3.0, False, "L2/3", "L2/3 excitatory Cux2 (regular-spiking)"),
    "e5Rbp4": CellType("e5Rbp4", 14.0, 3.0, False, "L5", "L5 excitatory Rbp4 (regular-spiking)"),
    # Inhibitory, parvalbumin: FAST-SPIKING (short tau, short refractory) ~2-3x faster
    "i4Pvalb": CellType("i4Pvalb", 6.0, 1.0, True, "L4", "L4 PV fast-spiking interneuron"),
    "i23Pvalb": CellType("i23Pvalb", 6.0, 1.0, True, "L2/3", "L2/3 PV fast-spiking interneuron"),
    "i5Pvalb": CellType("i5Pvalb", 6.0, 1.0, True, "L5", "L5 PV fast-spiking interneuron"),
    # Inhibitory, somatostatin: low-threshold, NOT fast-spiking (slow, adapting)
    "i4Sst": CellType("i4Sst", 12.0, 2.0, True, "L4", "L4 Sst low-threshold interneuron (not fast-spiking)"),
    "i23Sst": CellType("i23Sst", 12.0, 2.0, True, "L2/3", "L2/3 Sst low-threshold interneuron"),
    # Inhibitory, Htr3a/VIP: intermediate
    "i4Htr3a": CellType("i4Htr3a", 10.0, 2.0, True, "L4", "L4 Htr3a/VIP interneuron"),
    "i23Htr3a": CellType("i23Htr3a", 10.0, 2.0, True, "L2/3", "L2/3 Htr3a/VIP interneuron"),
}


# --- Named motif compositions ----------------------------------------------
# Mapping the 8 signed-XOR roles to Allen V1 cell types.
# Role order is fixed: E1=I, E3=P, E2=Li, E4=Lp, INH, XOR, Fp, Fm
ROLE_ORDER = ["I", "P", "Li", "Lp", "INH", "XOR", "Fp", "Fm"]

# Canonical *functional* composition: PV (fast-spiking) inhibitory pivot.
# This is the configuration the preprint's narrative implies (INH described as a
# "low-threshold fast-spiking" cell; i4Pvalb dominates at high weight threshold).
PV_PIVOT = {
    "I": "e4other",     # input / external-signal driver (S1)
    "P": "e4other",     # prediction driver (S2); anatomically symmetric with I 
    "Li": "e23Cux2",     # latent on the I branch
    "Lp": "e23Cux2",     # latent on the P branch
    "INH": "i4Pvalb",    # sparsifier / coincidence pivot (FAST-SPIKING)
    "XOR": "e23Cux2",    # mismatch comparator (excitatory)
    "Fp": "e23Cux2",     # positive error / under-shoot -> LTP  (excitatory)
    "Fm": "i23Pvalb",    # negative error / over-shoot  -> LTD  (inhibitory, fast)
}

# Empirical rank-1 composition reported in the preprint (Sst inhibitory pivot).
# Provided so the validation can test whether a *non*-fast-spiking pivot still
# supports the signed-XOR logic (it does not: see exp2_pv_robustness).
RANK1_COMPOSITION = {
    "I": "e4other",
    "P": "e4other",
    "Li": "e23Cux2",
    "Lp": "e23Cux2",
    "INH": "i4Sst",      # somatostatin pivot (slow / not fast-spiking)
    "XOR": "e23Cux2",
    "Fp": "e23Cux2",
    "Fm": "i23Pvalb",
}


def composition_table(comp):
    """Return a list of (role, cell_type, phenotype) rows for a composition."""
    rows = []
    for role in ROLE_ORDER:
        ct = CELL_TYPES[comp[role]]
        rows.append((role, ct.name, ct.description))
    return rows

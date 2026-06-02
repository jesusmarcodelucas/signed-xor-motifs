"""
Spiking implementation of the 8-neuron signed-XOR motif.

This module builds the motif as a current-based leaky integrate-and-fire (LIF)
network in Brian2 and runs it for a single input condition. It realises, with
biologically plausible mouse-cortex neurons, the circuit described in Section 2
of the preprint:

  Roles (8 neurons):  E1, E3 (inputs S1, S2) -> E2, E4 (latents) ;
                      INH (coincidence pivot) ; XOR (comparator) ;
                      Fp = F+ (positive error, excitatory) ;
                      Fm = F- (negative error, inhibitory).

  Edges (12; the same 12 counted on the Allen V1 graph):
      E1->E2, E1->INH, E3->E4, E3->INH,        (inputs drive latent + pivot)
      INH-|E2, INH-|E4,                        (pivot vetoes both latents)
      E2->XOR, E4->XOR,                        (latents drive comparator)
      E2->Fp, XOR->Fp,                         (F+ = coincidence(E2, XOR))
      E4->Fm, XOR->Fm.                         (F- = coincidence(E4, XOR))

Functional logic
----------------
INH is a *coincidence detector*: a single active input is sub-threshold, only
both inputs together drive it to fire. When INH fires it vetoes the latents.
Hence

    E2 active  <=>  (E1 on  AND  E3 off)        [under-shoot: target on, prediction off]
    E4 active  <=>  (E3 on  AND  E1 off)        [over-shoot:  target off, prediction on]
    XOR active <=>  E1 XOR E3
    Fp fires   <=>  XOR AND E2  =>  under-shoot  ->  LTP  (potentiate)
    Fm fires   <=>  XOR AND E4  =>  over-shoot   ->  LTD  (depress)
    Fp = Fm = 0 <=> E1 == E3   =>  homeostasis  ->  no plasticity

The coincidence/veto logic depends on the *speed* of the inhibitory neurons:
fast-spiking PV cells make it work, slow inhibition breaks it (quantified in
``experiments/exp2_pv_robustness.py``).

The default codegen target is numpy, which needs no C/C++ compiler. This is the
most portable setting and runs out-of-the-box under WSL on Windows. On a machine
with a compiler (e.g. a DGX), set ``brian2.prefs.codegen.target = 'cython'`` for
speed; it is unnecessary at this scale (8 neurons).
"""

from __future__ import annotations

import numpy as np
import brian2 as b2

from .celltypes import CELL_TYPES, ROLE_ORDER, PV_PIVOT

# Portable, compiler-free backend (works under WSL with no extra setup).
b2.prefs.codegen.target = "numpy"
b2.defaultclock.dt = 0.1 * b2.ms

# --- Fixed membrane constants (normalised mV) ------------------------------
E_L = -70.0      # resting potential
V_TH = -50.0     # spike threshold
V_RESET = -65.0  # reset potential

# --- Default synaptic weights (tuned; see README "Parameter tuning") --------
# Units are the same as the membrane potential increments per synaptic event.
DEFAULT_WEIGHTS = {
    "input_to_latent": 84.0,   #85 E1->E2, E3->E4  (one input drives its latent)
    "input_to_inh": 28.0,       #28 E1->INH, E3->INH (sub-threshold alone; coincident -> fire)
    "inh_to_latent": 84.0,     #90 INH-|E2, INH-|E4 (veto strength)
    "latent_to_xor": 84.0,     #85 E2->XOR, E4->XOR
    "xor_to_f": 28.0,           #28 XOR->Fp, XOR->Fm (sub-threshold alone)
    "latent_to_f": 28.0,        #28 E2->Fp, E4->Fm  (sub-threshold alone; coincident -> fire)
}

DEFAULT_DRIVE = 25.0   # external current applied to an input neuron when its bit = 1
DEFAULT_TAU_SYN_MS = 8.0
DEFAULT_DURATION_MS = 100.0


def _edge_list(w):
    """Return (pre_idx, post_idx, weight) tuples for the 12 motif edges.

    Inhibitory edges carry negative weights. Neuron indices follow
    ``[E1, E3, E2, E4, INH, XOR, Fp, Fm] = [0..7]``.
    """
    return [
        (0, 2, +w["input_to_latent"]),  # E1 -> E2
        (0, 4, +w["input_to_inh"]),      # E1 -> INH
        (1, 3, +w["input_to_latent"]),   # E3 -> E4
        (1, 4, +w["input_to_inh"]),      # E3 -> INH
        (4, 2, -w["inh_to_latent"]),     # INH -| E2
        (4, 3, -w["inh_to_latent"]),     # INH -| E4
        (2, 5, +w["latent_to_xor"]),     # E2 -> XOR
        (3, 5, +w["latent_to_xor"]),     # E4 -> XOR
        (2, 6, +w["latent_to_f"]),       # E2 -> Fp
        (5, 6, +w["xor_to_f"]),          # XOR -> Fp
        (3, 7, +w["latent_to_f"]),       # E4 -> Fm
        (5, 7, +w["xor_to_f"]),          # XOR -> Fm
    ]


def run_trial(
    s1,
    s2,
    composition=None,
    weights=None,
    inh_tau_override_ms=None,
    drive=DEFAULT_DRIVE,
    tau_syn_ms=DEFAULT_TAU_SYN_MS,
    duration_ms=DEFAULT_DURATION_MS,
    noise_rate_hz=0.0,
    seed=None,
    record_traces=False,
):
    """Simulate the motif for one input condition and return spike counts.

    Parameters
    ----------
    s1, s2 : float
        Input drive levels for the external signal (S1, role E1) and the internal
        prediction (S2, role E3). Use 0/1 for the binary truth table, or a value
        in [0, 1] (or beyond) for a graded input. The applied current is
        ``drive * s1`` and ``drive * s2``.
    composition : dict or None
        Role -> cell-type mapping (see :mod:`signed_xor.celltypes`). Defaults to
        ``PV_PIVOT`` (fast-spiking PV pivot).
    weights : dict or None
        Synaptic-weight dict (see ``DEFAULT_WEIGHTS``). Defaults to the tuned set.
    inh_tau_override_ms : float or None
        If set, override the membrane time constant of *both* inhibitory neurons
        (INH and Fm). Used by the PV-speed robustness sweep.
    drive : float
        External current magnitude for an "on" input.
    tau_syn_ms : float
        Synaptic (current) decay time constant in ms.
    duration_ms : float
        Simulation duration in ms.
    noise_rate_hz : float
        If > 0, each input neuron additionally receives independent background
        Poisson excitation at this rate (used for multi-seed robustness).
    seed : int or None
        Random seed (affects Poisson noise only).
    record_traces : bool
        If True, also record membrane-potential traces (for raster/trace plots).

    Returns
    -------
    dict
        ``{"counts": np.ndarray(8), "spike_mon": SpikeMonitor,
           "state_mon": StateMonitor or None, "roles": ROLE_ORDER}``
        ``counts`` is the number of spikes per role in ``ROLE_ORDER``.
    """
    composition = composition or PV_PIVOT
    weights = weights or DEFAULT_WEIGHTS
    if seed is not None:
        b2.seed(seed)
        np.random.seed(seed)

    b2.start_scope()
    ms = b2.ms
    tau_syn = tau_syn_ms * ms

    eqs = """
    dv/dt = (E_L - v)/tau + I/tau + Idrive/tau : 1 (unless refractory)
    dI/dt = -I/tau_syn : 1
    tau : second
    reft : second
    Idrive : 1
    """
    G = b2.NeuronGroup(
        8, eqs, threshold="v > V_TH", reset="v = V_RESET",
        refractory="reft", method="euler",
        namespace={"E_L": E_L, "V_TH": V_TH, "V_RESET": V_RESET, "tau_syn": tau_syn},
    )
    G.v = E_L

    # per-neuron membrane tau & refractory from the cell-type catalogue
    taus, refs = [], []
    for role in ROLE_ORDER:
        ct = CELL_TYPES[composition[role]]
        tau_m = ct.tau_m_ms
        if inh_tau_override_ms is not None and ct.is_inhibitory:
            tau_m = inh_tau_override_ms
        taus.append(tau_m)
        refs.append(ct.refractory_ms)
    G.tau = np.array(taus) * ms
    G.reft = np.array(refs) * ms

    # external binary/graded drive onto the two input neurons (E1, E3)
    G.Idrive = [drive * float(s1), drive * float(s2), 0, 0, 0, 0, 0, 0]

    # the 12 motif synapses; inhibitory pivot uses a slightly shorter delay so
    # its (fast) veto can act on the same cycle as the latent excitation.
    edges = _edge_list(weights)
    pre = [e[0] for e in edges]
    post = [e[1] for e in edges]
    wv = [e[2] for e in edges]
    delays = [0.5 if p == 4 else 1.0 for p in pre]  # INH (idx 4) acts faster
    S = b2.Synapses(G, G, "wgt : 1", on_pre="I_post += wgt", method="euler")
    S.connect(i=pre, j=post)
    S.wgt = wv
    S.delay = np.array(delays) * ms

    objs = [G, S]
    if noise_rate_hz > 0:
        P = b2.PoissonGroup(2, rates=noise_rate_hz * b2.Hz)
        SP = b2.Synapses(P, G, on_pre="I_post += 6.0")
        SP.connect(i=[0, 1], j=[0, 1])  # background onto E1, E3
        objs += [P, SP]

    spike_mon = b2.SpikeMonitor(G)
    objs.append(spike_mon)
    state_mon = None
    if record_traces:
        state_mon = b2.StateMonitor(G, "v", record=True)
        objs.append(state_mon)

    net = b2.Network(objs)
    net.run(duration_ms * ms)

    counts = np.array([int(np.sum(spike_mon.i == k)) for k in range(8)])
    return {
        "counts": counts,
        "spike_mon": spike_mon,
        "state_mon": state_mon,
        "roles": ROLE_ORDER,
    }

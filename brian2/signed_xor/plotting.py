"""Plotting helpers for the signed-XOR functional validation figures."""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")  # headless / WSL-safe
import matplotlib.pyplot as plt
import numpy as np

from .celltypes import ROLE_ORDER, CELL_TYPES

# colour code: excitatory = warm, inhibitory = blue, error channels highlighted
_ROLE_COLOR = {
    "I": "#444444", "Li": "#444444",
    "P": "#7D4A00", "Lp": "#7D4A00",
    "INH": "#2E5090", "XOR": "#1A6B3A",
    "Fp": "#C0392B", "Fm": "#2E5090",
}


def plot_condition_rasters(results_by_cond, threshold, duration_ms, path):
    """4-panel raster (one per input condition) of all 8 motif neurons."""
    conds = [(0, 0), (1, 0), (0, 1), (1, 1)]
    titles = {
        (0, 0): "I=0, P=0  -> homeostasis",
        (1, 0): "I=1, P=0  -> under-shoot (F+)",
        (0, 1): "I=0, P=1  -> over-shoot (F-)",
        (1, 1): "I=1, P=1  -> homeostasis",
    }
    fig, axes = plt.subplots(2, 2, figsize=(11, 6.5), sharex=True, sharey=True)
    for ax, cond in zip(axes.ravel(), conds):
        sm = results_by_cond[cond]["spike_mon"]
        t_ms = np.asarray(sm.t) * 1000.0  # Brian2 times are in seconds
        idx = np.asarray(sm.i)
        for k, role in enumerate(ROLE_ORDER):
            sel = idx == k
            ax.scatter(t_ms[sel], np.full(sel.sum(), k), s=28,
                       color=_ROLE_COLOR[role], marker="|", linewidths=1.6)
        ax.set_title(titles[cond], fontsize=10)
        ax.set_yticks(range(8))
        ax.set_yticklabels(ROLE_ORDER, fontsize=8)
        ax.set_xlim(0, duration_ms)
        ax.invert_yaxis()
        ax.grid(alpha=0.15)
    for ax in axes[-1]:
        ax.set_xlabel("time (ms)")
    fig.suptitle("Signed-XOR motif: spike rasters by input condition "
                 "(PV-pivot composition)", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_truth_table(counts_by_cond, threshold, path):
    """Heatmap of XOR/F+/F- spike counts across the four input conditions."""
    conds = [(0, 0), (1, 0), (0, 1), (1, 1)]
    chans = ["XOR", "Fp", "Fm"]
    cidx = [5, 6, 7]
    M = np.array([[counts_by_cond[c][i] for i in cidx] for c in conds], dtype=float)
    fig, ax = plt.subplots(figsize=(5.2, 4.2))
    im = ax.imshow(M, cmap="magma", aspect="auto")
    ax.set_xticks(range(3)); ax.set_xticklabels(["XOR", "F+", "F-"])
    ax.set_yticks(range(4)); ax.set_yticklabels([f"I={a}, P={b}" for a, b in conds])
    for r in range(4):
        for c in range(3):
            v = M[r, c]
            ax.text(c, r, f"{int(v)}", ha="center", va="center",
                    color="white" if v < M.max() * 0.6 else "black", fontsize=11)
    ax.set_title(f"Output spike counts (active if >= {threshold})")
    fig.colorbar(im, ax=ax, label="spikes in window")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_pv_sweep(taus, success, n_seeds, path, working_max_tau=None):
    """Fraction of correct truth tables as a function of inhibitory tau_m."""
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    ax.plot(taus, success, "o-", color="#2E5090", lw=2)
    ax.axhline(1.0, color="grey", ls=":", lw=1)
    ax.set_xlabel("inhibitory membrane time constant  $\\tau_m^{inh}$  (ms)")
    ax.set_ylabel(f"fraction of trials with correct\nsigned-XOR table (n={n_seeds} seeds)")
    ax.set_ylim(-0.05, 1.08)
    ax.axvspan(taus[0], 6.5, color="#1A6B3A", alpha=0.08)
    ax.text(min(taus) + 0.2, 0.15, "fast-spiking PV regime\n(2-3x faster)",
            color="#1A6B3A", fontsize=9)
    ax.set_title("Fast-spiking inhibition is required for signed-XOR computation")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_graded_error(pred_levels, fp_counts, fm_counts, target_level, path):
    """Signed-error curve: F+ and F- vs prediction level at fixed target."""
    fig, ax = plt.subplots(figsize=(6.8, 4.3))
    ax.plot(pred_levels, fp_counts, "o-", color="#C0392B", lw=2, label="F+ (under-shoot -> LTP)")
    ax.plot(pred_levels, fm_counts, "s-", color="#2E5090", lw=2, label="F- (over-shoot -> LTD)")
    ax.axvline(target_level, color="grey", ls="--", lw=1.2, label="target = prediction (match)")
    ax.set_xlabel("prediction drive  (target fixed ON)")
    ax.set_ylabel("error-channel spike count")
    ax.set_title("Graded signed error: F+ on the under side, F- on the over side")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)

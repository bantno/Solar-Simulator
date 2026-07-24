"""Quick-plot a saved value-function table (.npy) from the backward-induction solver.

Table layout (see mdpAnalyticalBackwardSolver / mdp._get_states):
  shape = (num_states, horizon)
  rows [0 : n_soc]          -> mode 0 (floating), SoC 0..100
  rows [n_soc : 2*n_soc]    -> mode 1 (flying),   SoC 0..100
  last row                  -> broken terminal state (always 0)
  columns                   -> time stage 0 .. horizon-1

Usage (pvlib conda env):
    conda run -n pvlib python plot_value_function.py
    conda run -n pvlib python plot_value_function.py "some_table.npy"
"""
import argparse
import glob
import os

import numpy as np
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))


def find_default_npy():
    cands = glob.glob(os.path.join(HERE, "*.npy"))
    if not cands:
        raise SystemExit("No .npy file found in the repo root. Pass a path explicitly.")
    # newest by mtime
    return max(cands, key=os.path.getmtime)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path", nargs="?", default=None, help="value-table .npy (default: newest in root)")
    ap.add_argument("--save", default=None, help="output PNG path (default: alongside the .npy)")
    args = ap.parse_args()

    path = args.path or find_default_npy()
    V = np.load(path)
    if V.ndim != 2:
        raise SystemExit(f"Expected a 2D table, got shape {V.shape}")

    S, T = V.shape
    n_soc = (S - 1) // 2
    soc = np.linspace(0.0, 100.0, n_soc)
    V0 = V[0:n_soc, :]            # floating
    V1 = V[n_soc:2 * n_soc, :]    # flying

    print(f"Loaded {os.path.basename(path)}: shape={V.shape}  n_soc={n_soc}  horizon={T}")
    print(f"value range: [{V.min():.3f}, {V.max():.3f}]")

    # mesh edges for pcolormesh
    soc_edges = np.linspace(0.0, 100.0, n_soc + 1)
    stage_edges = np.arange(T + 1)
    vmin, vmax = float(min(V0.min(), V1.min())), float(max(V0.max(), V1.max()))

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True, constrained_layout=True)
    for ax, data, title in ((axes[0], V0, "Mode 0 — Floating"),
                            (axes[1], V1, "Mode 1 — Flying")):
        pcm = ax.pcolormesh(stage_edges, soc_edges, data, shading="auto",
                            cmap="viridis", vmin=vmin, vmax=vmax)
        ax.set_title(title)
        ax.set_xlabel("Time stage")
    axes[0].set_ylabel("State of charge [%]")
    fig.colorbar(pcm, ax=axes, label="Value V(SoC, stage)")
    fig.suptitle(os.path.basename(path), fontsize=10)

    out = args.save or (os.path.splitext(path)[0] + "_value.png")
    fig.savefig(out, dpi=150)
    print(f"saved: {out}")
    try:
        plt.show()
    except Exception:
        pass


if __name__ == "__main__":
    main()

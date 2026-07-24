#!/usr/bin/env python3
"""Summary figure: weather-only persistence selection reproduces equal-occupancy (panel A);
only decision-aware placement beats it in reward (panel B)."""
import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
METRICS = os.path.join(REPO, "results", "failbin_analysis", "failbin_metrics.csv")
OUT = os.path.join(REPO, "results", "failbin_analysis", "fig8_persistence_summary.png")

SITES = ["florida", "hawaii", "natlantic", "bering"]
LABEL = {"florida": "Florida", "hawaii": "Hawaii", "natlantic": "N. Atlantic", "bering": "Bering"}

# n=3 selection results from persistence_bins.py `select` (all 4 sites, gain +0.0%).
SEL = {
    "florida":   dict(persist=[4.64, 6.87],  eqocc=[4.53, 6.80]),
    "hawaii":    dict(persist=[5.89, 8.79],  eqocc=[6.00, 8.91]),
    "natlantic": dict(persist=[6.80, 10.27], eqocc=[6.89, 10.48]),
    "bering":    dict(persist=[5.91, 9.49],  eqocc=[5.91, 9.49]),
}
C_EQ, C_DEC, C_FAIL = "#DD8452", "#55A868", "#C44E52"


def reward(df, site, arm):
    r = df[(df.location == site) & (df.arm == arm)]
    return float(r.mean_reward.iloc[0]) if len(r) else np.nan


def main():
    df = pd.read_csv(METRICS)
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(13.5, 5.2), gridspec_kw={"width_ratios": [1, 1.15]})

    # ---- Panel A: persistence-selected edges vs equal-occupancy edges (they coincide) ----
    for i, s in enumerate(SITES):
        y = len(SITES) - i
        eq, pe = SEL[s]["eqocc"], SEL[s]["persist"]
        axA.scatter(eq, [y, y], s=190, facecolors="none", edgecolors=C_EQ, linewidths=2,
                    zorder=2, label="equal-occupancy" if i == 0 else None)
        axA.scatter(pe, [y, y], s=32, color="black", zorder=3,
                    label="persistence-optimal" if i == 0 else None)
        dmax = max(abs(a - b) for a, b in zip(eq, pe))
        axA.text(13.6, y, f"max Δ = {dmax:.2f} m/s", va="center", fontsize=8.5, color="0.4")
    axA.set_yticks(range(1, len(SITES) + 1))
    axA.set_yticklabels([LABEL[s] for s in reversed(SITES)])
    axA.set_xlabel("wind-bin edge [m/s]  (n=3)")
    axA.set_xlim(0, 18); axA.set_ylim(0.5, len(SITES) + 0.5)
    axA.set_title("A. Weather-only persistence selection\nreproduces equal-occupancy (gain +0.0%)")
    axA.legend(loc="lower left", fontsize=9); axA.grid(axis="x", alpha=0.3)

    # ---- Panel B: reward by scheme (3-bin), persistence == equal-occupancy ----
    x = np.arange(len(SITES)); w = 0.26
    eq = [reward(df, s, "chain_wind") for s in SITES]
    de = [reward(df, s, "chain_dec") for s in SITES]
    fa = [reward(df, s, "chain_fail") for s in SITES]
    axB.bar(x - w, eq, w, color=C_EQ, label="equal-occupancy = persistence")
    axB.bar(x,     de, w, color=C_DEC, label="decision-boundary (needs model w*)")
    axB.bar(x + w, fa, w, color=C_FAIL, label="failure-space")
    axB.axhline(0, color="k", lw=0.8)
    axB.set_xticks(x); axB.set_xticklabels([LABEL[s] for s in SITES])
    axB.set_ylabel("mean reward (whale observations)")
    axB.set_title("B. Only decision-aware placement beats equal-occupancy\n(3 bins; failure-space collapses to never-fly)")
    axB.legend(fontsize=9); axB.grid(axis="y", alpha=0.3)

    fig.suptitle("Wind-bin selection: persistence is a weather property (→ equal-occupancy); "
                 "the extra gain is a model property (→ decision boundary)", fontsize=12, y=1.02)
    fig.tight_layout()
    fig.savefig(OUT, dpi=150, bbox_inches="tight")
    print("wrote", OUT)


if __name__ == "__main__":
    main()

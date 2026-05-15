"""
Variance visualizations for battery sweep results, segmented by algorithm.

Produces three figures:
  1. Violin plots of scalar outcomes vs battery capacity (side-by-side per algorithm)
  2. Reward distribution heatmap (one subplot per algorithm)
  3. Scatter plot matrix with algorithm as marker/colour series
"""
import os
import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.colors import LogNorm

# ── Presentation-friendly font sizes ────────────────────────────────────────
mpl.rcParams.update({
    "font.size": 16,
    "axes.titlesize": 20,
    "axes.labelsize": 18,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "legend.fontsize": 14,
    "figure.titlesize": 22,
})

# ── Friendly labels ──────────────────────────────────────────────────────────

_SIM_TYPE_SHORT = {
    "OptimalContinuousAnalyticalPolicySimulation": "Optimal",
    "UnifiedThresholdContinuousSimulation": "Threshold",
}


def _sim_label(row):
    """Build a display label including threshold params when applicable."""
    short = _SIM_TYPE_SHORT.get(row["sim_type"], row["sim_type"])
    if short == "Threshold":
        obs = row.get("observation_threshold")
        wind = row.get("wind_threshold")
        parts = []
        if obs is not None and not np.isnan(obs):
            parts.append(f"obs={obs}")
        if wind is not None and not np.isnan(wind):
            parts.append(f"wind={wind}")
        if parts:
            short += f" ({', '.join(parts)})"
    return short


# ── Data extraction ──────────────────────────────────────────────────────────

def load_episode_scalars(h5_path: str) -> pd.DataFrame:
    """One row per episode with sim_type, threshold params, and scalar outcomes."""
    records = []
    with h5py.File(h5_path, "r") as f:
        for sim_name in f.keys():
            grp = f[sim_name]
            cap = grp.attrs.get("battery_capacity")
            if cap is None:
                cap = grp.attrs.get("capacity")
            if cap is not None:
                cap = float(cap)

            sim_type = grp.attrs.get("simulation_type", "")
            if isinstance(sim_type, bytes):
                sim_type = sim_type.decode()
            obs_t = grp.attrs.get("observation_threshold", np.nan)
            wind_t = grp.attrs.get("wind_threshold", np.nan)

            eps = grp.get("episodes")
            if eps is None:
                continue

            for ep_key in eps.keys():
                ep = eps[ep_key]
                rec = {
                    "battery_capacity": cap,
                    "sim_type": sim_type,
                    "observation_threshold": float(obs_t) if obs_t is not None else np.nan,
                    "wind_threshold": float(wind_t) if wind_t is not None else np.nan,
                }
                if "total_reward" in ep:
                    rec["total_reward"] = float(ep["total_reward"][()])
                if "flight_hrs" in ep:
                    rec["flight_hrs"] = float(ep["flight_hrs"][()])
                if "failure" in ep:
                    rec["failure"] = bool(ep["failure"][()])
                if "failure_step" in ep:
                    rec["failure_step"] = float(ep["failure_step"][()])
                records.append(rec)

    df = pd.DataFrame.from_records(records)
    df["algorithm"] = df.apply(_sim_label, axis=1)
    return df


# ── Plot 1: Side-by-side violin plots per algorithm ─────────────────────────

def plot_violins(df: pd.DataFrame, outdir: str):
    metrics = [
        ("total_reward", "Total Reward"),
        ("flight_hrs", "Flight Hours"),
        ("failure_step", "Failure Step"),
    ]
    metrics = [(col, label) for col, label in metrics if col in df.columns]

    algorithms = sorted(df["algorithm"].unique())
    caps = sorted(df["battery_capacity"].unique())
    n_alg = len(algorithms)
    colors = ["grey", "steelblue", "seagreen", "mediumpurple"][:n_alg]

    width = 0.35
    offsets = np.linspace(-width * (n_alg - 1) / 2,
                          width * (n_alg - 1) / 2, n_alg)

    fig, axes = plt.subplots(len(metrics), 1,
                             figsize=(max(12, 1.2 * len(caps) * n_alg),
                                      4 * len(metrics)),
                             sharex=True)
    if len(metrics) == 1:
        axes = [axes]

    for ax, (col, label) in zip(axes, metrics):
        for a_idx, (alg, colour) in enumerate(zip(algorithms, colors)):
            sub = df[df["algorithm"] == alg]
            data = [sub.loc[sub["battery_capacity"] == c, col].dropna().values
                    for c in caps]
            positions = np.arange(len(caps)) + offsets[a_idx]

            parts = ax.violinplot(data, positions=positions,
                                  widths=width * 0.9,
                                  showmedians=True, showextrema=False)
            for pc in parts["bodies"]:
                pc.set_facecolor(colour)
                pc.set_alpha(0.65)
            parts["cmedians"].set_color("black")

            # overlay mean as a marker
            means = [np.mean(d) if len(d) else np.nan for d in data]
            ax.scatter(positions, means, marker='D', color='firebrick',
                       s=30, zorder=5)

        ax.set_ylabel(label)
        ax.set_xticks(range(len(caps)))
        ax.set_xticklabels([f"{int(c)}" for c in caps])
        ax.grid(axis="y", linestyle=":", alpha=0.4)

    # legend
    handles = [plt.Line2D([0], [0], color=c, lw=8, alpha=0.65)
               for c in colors]
    labels_leg = list(algorithms)
    handles.append(plt.Line2D([0], [0], color="black", lw=2))
    labels_leg.append("Median")
    handles.append(plt.Line2D([0], [0], marker="D", color="firebrick",
                              lw=0, markersize=8))
    labels_leg.append("Mean")
    axes[0].legend(handles, labels_leg, loc="upper right")

    axes[-1].set_xlabel("Battery Capacity (Wh)")
    fig.suptitle("Distribution of Episode Outcomes vs Battery Capacity")
    fig.tight_layout()
    os.makedirs(outdir, exist_ok=True)
    fig.savefig(os.path.join(outdir, "violin_outcomes.png"),
                dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved violin plot: {outdir}/violin_outcomes.png")


# ── Plot 2: Reward distribution heatmap (one subplot per algorithm) ──────────

def plot_reward_heatmap(df: pd.DataFrame, outdir: str, n_bins: int = 50):
    if "total_reward" not in df.columns:
        print("Skipping heatmap - total_reward column missing")
        return

    algorithms = sorted(df["algorithm"].unique())
    caps = sorted(df["battery_capacity"].unique())

    reward_min = df["total_reward"].min()
    reward_max = df["total_reward"].max()
    bins = np.linspace(reward_min, reward_max, n_bins + 1)

    fig, axes = plt.subplots(1, len(algorithms),
                             figsize=(8 * len(algorithms), 6),
                             sharey=True)
    if len(algorithms) == 1:
        axes = [axes]

    for ax, alg in zip(axes, algorithms):
        sub = df[df["algorithm"] == alg]
        hist2d = np.zeros((n_bins, len(caps)))
        for j, c in enumerate(caps):
            vals = sub.loc[sub["battery_capacity"] == c, "total_reward"].values
            hist2d[:, j] = np.histogram(vals, bins=bins)[0]

        im = ax.pcolormesh(
            np.arange(len(caps) + 1) - 0.5, bins, hist2d,
            cmap="viridis",
            norm=LogNorm(vmin=1, vmax=max(hist2d.max(), 1)),
        )
        ax.set_facecolor("white")
        ax.set_xticks(range(len(caps)))
        ax.set_xticklabels([f"{int(c)}" for c in caps])
        ax.set_xlabel("Battery Capacity (Wh)")
        ax.set_title(alg)

    axes[0].set_ylabel("Total Reward")
    fig.colorbar(im, ax=axes, label="Episode Count", shrink=0.8)
    fig.suptitle("Episode Reward Distribution Heatmap")
    fig.tight_layout()
    os.makedirs(outdir, exist_ok=True)
    fig.savefig(os.path.join(outdir, "reward_heatmap.png"),
                dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved heatmap: {outdir}/reward_heatmap.png")


# ── Plot 3: Scatter plot matrix coloured by algorithm ────────────────────────

def plot_scatter_matrix(df: pd.DataFrame, outdir: str):
    metrics = ["total_reward", "flight_hrs", "failure_step"]
    metrics = [m for m in metrics if m in df.columns]
    labels = {
        "total_reward": "Total Reward",
        "flight_hrs": "Flight Hours",
        "failure_step": "Failure Step",
    }

    n = len(metrics)
    if n < 2:
        print("Skipping scatter matrix - need at least 2 metrics")
        return

    algorithms = sorted(df["algorithm"].unique())
    n_alg = len(algorithms)
    colors = ["grey", "steelblue", "seagreen", "mediumpurple"][:n_alg]

    fig, axes = plt.subplots(n, n, figsize=(5 * n, 5 * n))
    if n == 1:
        axes = np.array([[axes]])

    for i in range(n):
        for j in range(n):
            ax = axes[i, j]
            if i == j:
                # diagonal: overlaid histograms per algorithm
                for a_idx, (alg, colour) in enumerate(zip(algorithms, colors)):
                    vals = df.loc[df["algorithm"] == alg, metrics[i]].dropna()
                    ax.hist(vals, bins=40, alpha=0.5, color=colour,
                            density=True, label=alg)
                if i == 0:
                    ax.legend()
                ax.set_ylabel("Density" if j == 0 else "")
            else:
                # off-diagonal: scatter per algorithm
                for a_idx, (alg, colour) in enumerate(zip(algorithms, colors)):
                    sub = df[df["algorithm"] == alg]
                    ax.scatter(
                        sub[metrics[j]], sub[metrics[i]],
                        c=colour, s=6, alpha=0.25, edgecolors="none",
                        label=alg if (i == 0 and j == 1) else None,
                    )

            if i == n - 1:
                ax.set_xlabel(labels[metrics[j]])
            else:
                ax.set_xticklabels([])
            if j == 0:
                ax.set_ylabel(labels[metrics[i]])
            else:
                ax.set_yticklabels([])

    # single legend on top-right off-diagonal cell
    axes[0, n - 1].legend(markerscale=3)

    fig.suptitle("Pairwise Scalar Outcomes by Algorithm")
    fig.tight_layout()
    os.makedirs(outdir, exist_ok=True)
    fig.savefig(os.path.join(outdir, "scatter_matrix.png"),
                dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved scatter matrix: {outdir}/scatter_matrix.png")


# ── CLI entry point ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    H5_PATH = os.path.join(
        os.path.dirname(__file__), os.pardir,
        "Results", "BatterySweep",
        "battery_sweep_config_20250915_140523.h5",
    )
    OUT_DIR = os.path.join(
        os.path.dirname(__file__), os.pardir,
        "Results", "BatterySweep", "variance_plots",
    )

    print(f"Loading episodes from {H5_PATH}")
    df = load_episode_scalars(H5_PATH)
    print(f"  {len(df)} episodes across "
          f"{df['battery_capacity'].nunique()} capacity values")
    print(f"  Algorithms: {df['algorithm'].unique().tolist()}\n")

    plot_violins(df, OUT_DIR)
    plot_reward_heatmap(df, OUT_DIR)
    plot_scatter_matrix(df, OUT_DIR)

    print("\nDone.")

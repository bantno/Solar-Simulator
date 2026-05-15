"""
Violin plots for duration sweep results — 3 series only:
  Optimal, Threshold (obs=0.25, wind=4.0), Threshold (obs=0.25, wind=8.0)

X-axis is horizon converted to days (horizon * 15 / 1440).
"""
import os
import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl

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
            horizon = grp.attrs.get("horizon")
            if horizon is not None:
                horizon = int(horizon)

            sim_type = grp.attrs.get("simulation_type", "")
            if isinstance(sim_type, bytes):
                sim_type = sim_type.decode()
            obs_t = grp.attrs.get("observation_threshold", np.nan)
            wind_t = grp.attrs.get("wind_threshold", np.nan)

            # Filter: keep only Optimal or Threshold with obs=0.25, wind in {4,8}
            is_optimal = sim_type == "OptimalContinuousAnalyticalPolicySimulation"
            is_threshold = (
                sim_type == "UnifiedThresholdContinuousSimulation"
                and float(obs_t) == 0.25
                and float(wind_t) in (4.0, 8.0)
            )
            if not (is_optimal or is_threshold):
                continue

            eps = grp.get("episodes")
            if eps is None:
                continue

            for ep_key in eps.keys():
                ep = eps[ep_key]
                rec = {
                    "horizon": horizon,
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
    # Convert horizon (steps) to days: each step is 15 min
    df["horizon_days"] = df["horizon"] * 15 / 1440
    return df


# ── Violin plot ──────────────────────────────────────────────────────────────

def plot_violins(df: pd.DataFrame, outdir: str):
    metrics = [
        ("total_reward", "Total Reward"),
        ("flight_hrs", "Flight Hours"),
        ("failure_step", "Failure Step"),
    ]
    metrics = [(col, label) for col, label in metrics if col in df.columns]

    algorithms = sorted(df["algorithm"].unique())
    horizons = sorted(df["horizon_days"].unique())
    n_alg = len(algorithms)
    colors = ["grey", "steelblue", "seagreen", "mediumpurple"][:n_alg]

    width = 0.35
    offsets = np.linspace(-width * (n_alg - 1) / 2,
                          width * (n_alg - 1) / 2, n_alg)

    fig, axes = plt.subplots(len(metrics), 1,
                             figsize=(max(12, 1.2 * len(horizons) * n_alg),
                                      4 * len(metrics)),
                             sharex=True)
    if len(metrics) == 1:
        axes = [axes]

    for ax, (col, label) in zip(axes, metrics):
        for a_idx, (alg, colour) in enumerate(zip(algorithms, colors)):
            sub = df[df["algorithm"] == alg]
            data = [sub.loc[sub["horizon_days"] == h, col].dropna().values
                    for h in horizons]
            positions = np.arange(len(horizons)) + offsets[a_idx]

            parts = ax.violinplot(data, positions=positions,
                                  widths=width * 0.9,
                                  showmedians=True, showextrema=False)
            for pc in parts["bodies"]:
                pc.set_facecolor(colour)
                pc.set_alpha(0.65)
            parts["cmedians"].set_color("black")

            # overlay mean as red diamond
            means = [np.mean(d) if len(d) else np.nan for d in data]
            ax.scatter(positions, means, marker='D', color='firebrick',
                       s=30, zorder=5)

        ax.set_ylabel(label)
        ax.set_xticks(range(len(horizons)))
        ax.set_xticklabels([f"{h:.1f}" for h in horizons])
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

    axes[-1].set_xlabel("Horizon (days)")
    fig.suptitle("Distribution of Episode Outcomes vs Horizon Duration")
    fig.tight_layout()
    os.makedirs(outdir, exist_ok=True)
    fig.savefig(os.path.join(outdir, "violin_outcomes.png"),
                dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved violin plot: {outdir}/violin_outcomes.png")


# ── CLI entry point ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    H5_PATH = os.path.join(
        os.path.dirname(__file__), os.pardir,
        "Results", "DurationSweep",
        "duration_sweep_config_20250914_131054.h5",
    )
    OUT_DIR = os.path.join(
        os.path.dirname(__file__), os.pardir,
        "Results", "DurationSweep", "variance_plots",
    )

    print(f"Loading episodes from {H5_PATH}")
    df = load_episode_scalars(H5_PATH)
    print(f"  {len(df)} episodes across "
          f"{df['horizon_days'].nunique()} horizon values")
    print(f"  Horizons (days): {sorted(df['horizon_days'].unique())}")
    print(f"  Algorithms: {df['algorithm'].unique().tolist()}\n")

    plot_violins(df, OUT_DIR)

    print("\nDone.")

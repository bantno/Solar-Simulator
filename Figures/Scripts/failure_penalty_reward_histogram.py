"""Histogram of episode total rewards across failure penalties.

Shows Optimal and Threshold (obs=0.2, wind=6.0) as overlaid histograms,
one subplot per selected failure penalty value (phi).
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

_SIM_TYPE_SHORT = {
    "OptimalContinuousAnalyticalPolicySimulation": "Optimal",
    "UnifiedThresholdContinuousSimulation": "Threshold",
}


def _sim_label(row):
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

def load_episode_scalars(h5_path: str,
                         penalties: list[float],
                         obs_thresh: float = 0.2,
                         wind_thresh: float = 6.0) -> pd.DataFrame:
    """Load episodes, filtering to selected penalties and algorithm configs."""
    records = []
    with h5py.File(h5_path, "r") as f:
        for sim_name in f.keys():
            grp = f[sim_name]
            fp = grp.attrs.get("failure_penalty")
            if fp is None or float(fp) not in penalties:
                continue

            sim_type = grp.attrs.get("simulation_type", "")
            if isinstance(sim_type, bytes):
                sim_type = sim_type.decode()
            obs_t = grp.attrs.get("observation_threshold", np.nan)
            wind_t = grp.attrs.get("wind_threshold", np.nan)

            is_optimal = sim_type == "OptimalContinuousAnalyticalPolicySimulation"
            is_threshold = (
                sim_type == "UnifiedThresholdContinuousSimulation"
                and float(obs_t) == obs_thresh
                and float(wind_t) == wind_thresh
            )
            if not (is_optimal or is_threshold):
                continue

            base = {
                "failure_penalty": float(fp),
                "sim_type": sim_type,
                "observation_threshold": float(obs_t) if obs_t is not None else np.nan,
                "wind_threshold": float(wind_t) if wind_t is not None else np.nan,
            }

            # Columnar layout: one (episodes,) dataset per scalar field
            sc = grp.get("episode_scalars")
            if sc is not None and "total_reward" in sc:
                for r in np.asarray(sc["total_reward"][()], dtype=float):
                    records.append({**base, "total_reward": float(r)})
                continue

            # Legacy layout: one group of scalar datasets per episode
            eps = grp.get("episodes")
            if eps is None:
                continue

            for ep_key in eps.keys():
                ep = eps[ep_key]
                rec = dict(base)
                if "total_reward" in ep:
                    rec["total_reward"] = float(ep["total_reward"][()])
                records.append(rec)

    df = pd.DataFrame.from_records(records)
    df["algorithm"] = df.apply(_sim_label, axis=1)
    return df


# ── Histogram plot ───────────────────────────────────────────────────────────

def plot_reward_histogram(df: pd.DataFrame, penalties: list[float], outdir: str,
                          n_bins: int = 120):
    algorithms = sorted(df["algorithm"].unique())
    colors = {"Optimal": "grey",
              "Threshold (obs=0.2, wind=6.0)": "steelblue"}
    hatches = {"Optimal": "",
               "Threshold (obs=0.2, wind=6.0)": "//"}

    # Shared bin edges across all subplots
    rmin, rmax = df["total_reward"].min(), df["total_reward"].max()
    bins = np.linspace(rmin, rmax, n_bins + 1)

    n_rows = len(penalties)
    fig, axes = plt.subplots(n_rows, 1, figsize=(10, 3.5 * n_rows), sharex=True)
    if n_rows == 1:
        axes = [axes]

    for ax, phi in zip(axes, penalties):
        sub = df[df["failure_penalty"] == phi]
        for alg in algorithms:
            vals = sub.loc[sub["algorithm"] == alg, "total_reward"].dropna().values
            colour = colors.get(alg, "seagreen")
            hatch = hatches.get(alg, "")
            ax.hist(vals, bins=bins, alpha=0.5, color=colour,
                    edgecolor="black", linewidth=0.3, hatch=hatch,
                    label=alg)
        ax.set_ylabel("Count")
        ax.set_title(rf"$\phi = {int(phi)}$")
        ax.grid(axis="y", linestyle=":", alpha=0.4)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)

    axes[0].legend(loc="upper left", bbox_to_anchor=(0.0, 1.0),
                   framealpha=0.9)
    axes[-1].set_xlabel("Total Reward")
    fig.suptitle("Episode Reward Distribution by Failure Penalty")
    fig.tight_layout()

    os.makedirs(outdir, exist_ok=True)
    out_path = os.path.join(outdir, "reward_histogram_by_penalty.png")
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


# ── CLI entry point ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    PENALTIES = [0.0, 20.0, 40.0, 100.0]

    H5_PATH = os.path.join(
        os.path.dirname(__file__), os.pardir,
        "Results", "FailurePenaltySweep",
        "failure_penalty_sweep_config_20250915_164437.h5",
    )
    OUT_DIR = os.path.join(
        os.path.dirname(__file__), os.pardir,
        "Results", "FailurePenaltySweep", "variance_plots",
    )

    print(f"Loading episodes from {H5_PATH}")
    df = load_episode_scalars(H5_PATH, penalties=PENALTIES,
                              obs_thresh=0.2, wind_thresh=6.0)
    print(f"  {len(df)} episodes")
    print(f"  Penalties: {sorted(df['failure_penalty'].unique())}")
    print(f"  Algorithms: {df['algorithm'].unique().tolist()}\n")

    plot_reward_histogram(df, PENALTIES, OUT_DIR)
    print("\nDone.")

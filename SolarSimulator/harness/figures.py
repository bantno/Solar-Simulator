"""
Auto-generated figures for one experiment run set.

Three kinds of figure:
  * render_paper_figure  -- reproduce an exact journal-paper figure (matching
                           Figures/Scripts/generate_journal_paper_figures.py) from THIS run's
                           HDF5, by reusing that script's own plotting functions. Used when the
                           config names a `paper_figure`. This guarantees the harness plots match
                           the paper by construction.
  * plot_sweep_summary   -- generic fallback for non-paper configs: each metric vs the swept
                           parameter, drawn as a family of curves (one line per secondary
                           parameter, NO averaging) with the optimal policy overlaid.
  * plot_trajectory_replay -- a single-policy, deterministic replay of a representative sim on a
                           REAL historical weather window (wind / SoC / action over mission time).

The replay helpers mirror SolarSimulator/Scripts/compare_policies_episode.py; they are copied
here (rather than imported) to keep the harness free of that script's module-level wiring.
Matplotlib runs headless (Agg).
"""
import os
import sys
import warnings
from pathlib import Path
from typing import Dict, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

_METRICS = [
    ("average_reward", "avg reward"),
    ("failure_percentage", "failure fraction"),
    ("average_flight_hrs", "avg flight hrs"),
]
# Candidate sweep parameters (priority order; the one with the most distinct values becomes x).
_PARAM_LABELS = {
    "observation_threshold": "observation threshold",
    "battery_capacity": "battery capacity [Wh]",
    "wind_threshold": "wind threshold [m/s]",
    "horizon": "horizon [steps]",
    "failure_penalty": "failure penalty",
    "location_id": "location",
}
# Threshold-policy knobs: the optimal policy does not depend on these (its rows have NaN here),
# so when one is the x-axis the optimal is drawn as a horizontal reference line.
_KNOBS = {"observation_threshold", "wind_threshold"}


# --------------------------------------------------------------------------------------
# Exact paper-figure reproduction (reuses the journal figure script on this run's HDF5)
# --------------------------------------------------------------------------------------
# Maps a journal figure function -> the RESULTS_PATHS key it reads for its main HDF5. The harness
# points that key at the current run's HDF5 so the paper code renders from harness data.
_PAPER_FIGURE_H5_KEY = {
    "fig_threshold_sweep_combined": "threshold_h5",
    "fig_capacity_mean_reward": "loc_capacity_h5",
    "fig_optimal_capacity_by_location": "loc_capacity_h5",
}


def render_paper_figure(func_name: str, h5_path: str, out_dir: str) -> Optional[str]:
    """
    Reproduce the journal-paper figure `func_name` from this run's HDF5, writing it to `out_dir`.

    Reuses Figures/Scripts/generate_journal_paper_figures.py directly (its `load_summary` reads
    the same HDF5 attrs the harness writes), so the output matches the paper by construction.
    Returns the figures dir on success, or None (with a warning) if it can't be rendered.
    """
    os.makedirs(out_dir, exist_ok=True)
    fig_scripts = os.path.join(REPO_ROOT, "Figures", "Scripts")
    pkg_dir = os.path.join(REPO_ROOT, "SolarSimulator")
    for p in (fig_scripts, pkg_dir):
        if p not in sys.path:
            sys.path.insert(0, p)
    try:
        import generate_journal_paper_figures as gj  # noqa: E402
    except Exception as e:                            # noqa: BLE001 - figure is best-effort
        warnings.warn(f"Could not import journal figure script ({e}); skipping paper figure.")
        return None

    if not hasattr(gj, func_name):
        warnings.warn(f"Journal figure '{func_name}' not found; skipping.")
        return None
    h5_key = _PAPER_FIGURE_H5_KEY.get(func_name)
    if h5_key is None:
        warnings.warn(f"No HDF5 mapping for paper figure '{func_name}'; skipping.")
        return None

    # Point the paper code's artifact path + output dir at this run, then render.
    gj.RESULTS_PATHS[h5_key] = Path(h5_path)
    gj.OUT_DIR = Path(out_dir)
    try:
        gj.apply_paper_style()
        getattr(gj, func_name)()
    except Exception as e:                            # noqa: BLE001
        warnings.warn(f"Paper figure '{func_name}' failed to render ({e}); skipping.")
        return None
    return out_dir


# --------------------------------------------------------------------------------------
# Generic sweep-summary plots (fallback for configs without a named paper_figure)
# --------------------------------------------------------------------------------------
def plot_sweep_summary(df: pd.DataFrame, out_dir: str) -> list:
    """
    For each metric, plot it against the primary swept parameter as a FAMILY OF CURVES -- one
    line per value of the secondary swept parameter (no averaging) -- with the optimal policy
    overlaid (a line vs x when x is a global dim, or a horizontal reference when x is a
    threshold knob the optimal ignores). Returns the saved figure paths.
    """
    os.makedirs(out_dir, exist_ok=True)
    if df is None or df.empty:
        return []

    varying = [c for c in _PARAM_LABELS if c in df.columns and df[c].nunique(dropna=True) > 1]
    if not varying:
        warnings.warn("No swept parameter varies; skipping sweep-summary plots.")
        return []
    varying.sort(key=lambda c: df[c].nunique(dropna=True), reverse=True)
    x = varying[0]
    hue = varying[1] if len(varying) > 1 else None

    is_opt = df["simulation_type"].astype(str).str.contains("Optimal", case=False, na=False) \
        if "simulation_type" in df.columns else pd.Series(False, index=df.index)
    thr, opt = df[~is_opt], df[is_opt]
    saved = []

    for metric, mlabel in _METRICS:
        if metric not in df.columns or df[metric].isna().all():
            continue
        fig, ax = plt.subplots(figsize=(7, 5))

        # Threshold (or non-optimal) policy: one line per hue value, no averaging.
        if not thr.empty and thr[metric].notna().any():
            if hue is not None and thr[hue].nunique(dropna=True) > 1:
                for hv in sorted(thr[hue].dropna().unique()):
                    sub = thr[thr[hue] == hv].dropna(subset=[x, metric]).sort_values(x)
                    if not sub.empty:
                        ax.plot(sub[x], sub[metric], marker="o",
                                label=f"{_PARAM_LABELS[hue]}={hv:g}")
            else:
                sub = thr.dropna(subset=[x, metric]).sort_values(x)
                if not sub.empty:
                    ax.plot(sub[x], sub[metric], marker="o", label="threshold")

        # Optimal policy overlay.
        if not opt.empty and opt[metric].notna().any():
            if x in _KNOBS:
                for val in pd.to_numeric(opt[metric], errors="coerce").dropna().unique():
                    ax.axhline(val, ls="--", color="black", lw=1.0, label="optimal")
            else:
                sub = opt.dropna(subset=[x, metric]).sort_values(x)
                if not sub.empty:
                    ax.plot(sub[x], sub[metric], marker="s", ls="--", color="black",
                            label="optimal")

        ax.set_xlabel(_PARAM_LABELS[x])
        ax.set_ylabel(mlabel)
        ax.grid(True, alpha=0.3)
        handles, labels = ax.get_legend_handles_labels()
        uniq = dict(zip(labels, handles))   # dedupe repeated 'optimal'
        if uniq:
            ax.legend(uniq.values(), uniq.keys(), fontsize=8)
        ax.set_title(f"{mlabel} vs {_PARAM_LABELS[x]}")
        fig.tight_layout()
        out = os.path.join(out_dir, f"sweep_{metric}.png")
        fig.savefig(out, dpi=150)
        plt.close(fig)
        saved.append(out)

    return saved


# --------------------------------------------------------------------------------------
# Trajectory replay on real historical weather (single policy)
# --------------------------------------------------------------------------------------
def _historical_pkl_for(location: Dict) -> Optional[str]:
    """Map a config location to its HISTORICAL_DATA pickle, trying common naming conventions."""
    lat, lon = location.get("latitude"), location.get("longitude")
    if lat is None or lon is None:
        return None
    hist_dir = os.path.join(REPO_ROOT, "Data", "HISTORICAL_DATA")
    candidates = [
        f"data_{lat}_{lon}.pkl",
        f"data_{int(lat)}_{int(lon)}.pkl",
    ]
    for name in candidates:
        p = os.path.join(hist_dir, name)
        if os.path.exists(p):
            return p
    # Glob fallback: any pkl whose name contains both coordinate substrings.
    import glob as _glob
    lat_str, lon_str = f"{float(lat):g}", f"{float(lon):g}"
    matches = [p for p in _glob.glob(os.path.join(hist_dir, "*.pkl"))
               if lat_str in os.path.basename(p) and lon_str in os.path.basename(p)]
    return matches[0] if matches else os.path.join(hist_dir, f"data_{lat}_{lon}.pkl")


def _load_historical_window(hist_pkl, start_date, n_steps, interval_min=15):
    """Real wind [m/s] and GHI [W/m^2] for n_steps at the model timestep.

    Matches on calendar position (month/day/hour/minute) only — the year in start_date is
    ignored so that configs with future start_datetimes still produce figures from the most
    recent available historical year.
    """
    df = pd.read_pickle(hist_pkl)
    df = df[~((df.index.month == 2) & (df.index.day == 29))]
    res = (df[["wind_speed_10m", "shortwave_radiation"]]
           .resample(f"{interval_min}min").interpolate(method="linear"))

    ref = pd.Timestamp(start_date)
    mask = (
        (res.index.month  == ref.month) &
        (res.index.day    == ref.day)   &
        (res.index.hour   == ref.hour)  &
        (res.index.minute == ref.minute)
    )
    candidates = res.index[mask]
    if candidates.empty:
        raise ValueError(
            f"No historical data for {ref.month}/{ref.day} {ref.hour}:{ref.minute:02d}"
        )
    # Walk from the most recent matching date backwards until the window fits.
    for ts in reversed(candidates):
        i = res.index.get_loc(ts)
        if i + n_steps <= len(res):
            win = res.iloc[i:i + n_steps]
            return win["wind_speed_10m"].to_numpy(), win["shortwave_radiation"].to_numpy(), win.index[0]
    raise ValueError("No historical window long enough starting at the requested calendar date.")


def _replay(sim, wind, solar_energy, whale, edges, bin_aware):
    """
    Deterministic policy rollout on fixed real weather. Mechanical transitions are assumed to
    succeed (full intended trajectory); battery depletion (soc<0) terminates. Returns the
    trajectory plus the exact cumulative mechanical-failure prob 1 - prod_t p_success_t.
    """
    tl = sim.mdp.transition_logic
    H = len(wind)
    state = np.array([100.0, 0.0])
    energy = tl.soc_to_energy(state[0])
    socs, modes, actions = [state[0]], [int(state[1])], []
    log_surv = 0.0
    batt_fail_at = None
    for t in range(H):
        s2 = state[None, :]
        cur_bins = np.array([int(np.digitize(wind[t], edges[1:-1]))]) if bin_aware else None
        a = int(sim.choose_action_batch(
            s2, np.array([solar_energy[t]]), np.array([wind[t]]), np.array([whale[t]]), t,
            cur_bins=cur_bins)[0])
        actions.append(a)
        ec = tl._calculate_energy_consumption(s2, np.array([a]))
        nss, nse = tl._update_energy_and_state_continuous(
            energy, np.array([solar_energy[t]]), ec, np.array([a]))
        p_succ = float(tl.transition_model.compute_probability(
            np.array([wind[t]]), np.array([a]), s2)[0])
        log_surv += np.log(max(p_succ, 1e-12))
        cand = nss[0]
        if cand[1] == 2:                       # battery depletion (deterministic)
            state = np.array([-1.0, 2.0]); batt_fail_at = t + 1
            socs.append(state[0]); modes.append(2)
            break
        state, energy = cand, nse[0]
        socs.append(state[0]); modes.append(int(state[1]))
    return {"socs": np.array(socs), "modes": np.array(modes), "actions": np.array(actions),
            "batt_fail_at": batt_fail_at, "mech_fail_prob": 1.0 - np.exp(log_surv)}


def _read_episode_from_h5(h5_path: str, sim) -> Optional[dict]:
    """
    Read a representative full-history episode for `sim` from the HDF5.

    Matches the simulation group by class name and battery capacity, then picks
    the episode whose total reward is closest to the median of all saved
    full-history episodes (those with a wind_series dataset).

    Returns a dict with keys: wind, soc, mode, actions, episode_name, sim_group.
    Returns None if no matching full-history episode exists.
    """
    import h5py
    class_name = sim.__class__.__name__.lower()
    cap = getattr(getattr(sim, "mdp", None), "battery_capacity_wh", None)

    with h5py.File(h5_path, "r") as f:
        for grp_name in f.keys():
            grp = f[grp_name]
            sim_type = str(grp.attrs.get("simulation_type", "")).lower()
            if class_name not in sim_type:
                continue
            if cap is not None:
                grp_cap = grp.attrs.get("battery_capacity")
                if grp_cap is not None and abs(float(grp_cap) - float(cap)) > 1.0:
                    continue
            ep_grp = grp.get("episodes")
            if ep_grp is None:
                continue
            full_eps = [(name, ep_grp[name]) for name in ep_grp.keys()
                        if "wind_series" in ep_grp[name]]
            if not full_eps:
                continue
            # Pick the episode closest to median total reward.
            scored = []
            for name, ep in full_eps:
                r = ep.get("rewards")
                total = float(r[()].sum()) if r is not None else 0.0
                scored.append((total, name))
            scored.sort(key=lambda x: x[0])
            _, ep_name = scored[len(scored) // 2]
            ep = ep_grp[ep_name]
            return {
                "sim_group":   grp_name,
                "episode_name": ep_name,
                "wind":    ep["wind_series"][()],
                "soc":     ep["trajectory"][:, 0],
                "mode":    ep["trajectory"][:, 1].astype(int),
                "actions": ep["actions"][()],
            }
    return None


def plot_trajectory_replay(sim, h5_path: str,
                           out_dir: str, interval_min: int = 15) -> Optional[str]:
    """
    Plot wind / SoC / action for a representative saved episode of `sim`.

    Reads the median-reward full-history episode from the run's HDF5 file.
    Requires full_history_episodes > 0 (or save_states: true) in the config so
    that per-step time-series are stored. Returns the saved figure path or None.
    """
    os.makedirs(out_dir, exist_ok=True)
    try:
        ep = _read_episode_from_h5(h5_path, sim)
        if ep is None:
            warnings.warn(
                "No full-history episodes found for the representative sim; "
                "set full_history_episodes > 0 in config to enable trajectory figures."
            )
            return None

        wind    = ep["wind"]
        soc     = ep["soc"]
        actions = ep["actions"]
        H = len(wind)

        env       = sim.env_provider
        edges     = getattr(env, "wind_bin_edges", np.array([0.0, np.inf]))
        bin_aware = getattr(env, "use_wind_chain", False)
        label     = "chain" if bin_aware else "i.i.d."
        cap       = getattr(getattr(sim, "mdp", None), "battery_capacity_wh", "?")

        steps_per_day = 24 * 60 / interval_min
        days = np.arange(H) / steps_per_day

        fig, ax = plt.subplots(3, 1, figsize=(13, 8), sharex=True, constrained_layout=True)

        ax[0].plot(days, wind, color="0.35", lw=0.7)
        if np.isfinite(edges).all() and len(edges) > 2:
            for e in edges[1:-1]:
                ax[0].axhline(e, ls="--", color="0.6", lw=0.8)
            ax[0].fill_between(days, edges[-2], wind, where=wind >= edges[-2],
                               color="tab:red", alpha=0.25, label="high-wind bin")
            ax[0].legend(loc="upper right", fontsize=8)
        ax[0].set_ylabel("Wind [m/s]")
        ax[0].set_title(
            f"{ep['episode_name']} (median-reward) | {label} policy, {cap} Wh"
        )

        batt_fail_at = int(np.argmax(ep["mode"] == 2)) if 2 in ep["mode"] else None
        ax[1].step(days[:len(soc) - 1], soc[:-1], where="post", color="tab:green")
        if batt_fail_at is not None:
            ax[1].scatter(days[batt_fail_at], 0, marker="x", s=70, color="tab:green", zorder=5)
        ax[1].set_ylabel("State of charge [%]")

        ax[2].step(days[:len(actions)], actions, where="post", color="tab:green")
        ax[2].set_yticks([0, 1]); ax[2].set_yticklabels(["float/land", "fly"])
        ax[2].set_ylabel("Action"); ax[2].set_xlabel("Mission time [days]")

        flight_hrs = int(actions.sum()) * interval_min / 60
        batt_str = "none" if batt_fail_at is None else f"t{batt_fail_at}"
        fig.text(0.01, 0.01,
                 f"flight_hrs={flight_hrs:.1f}  batt_fail={batt_str}",
                 fontsize=8)

        out = os.path.join(out_dir, f"trajectory_{label}_{cap}Wh.png")
        fig.savefig(out, dpi=150)
        plt.close(fig)
        return out
    except Exception as e:                       # noqa: BLE001 - best-effort figure only
        warnings.warn(f"Trajectory replay figure failed ({e}); skipping.")
        return None

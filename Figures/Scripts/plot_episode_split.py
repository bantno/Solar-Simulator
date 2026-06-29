#!/usr/bin/env python3
"""Plot a single episode as two separate figures from a results HDF5.

Figure 1 – Environment (3 panels, shared x-axis):
  1) Collected solar radiation  →  y-label: e_k+ (Wh)
  2) Wind speed                 →  y-label: w_k (m/s)
  3) Whale observation prob.    →  y-label: O_k

Figure 2 – Vehicle (5 panels, shared x-axis):
  1) Stored energy              →  y-label: c_k (Wh)
  2) Mode                       →  y-label: m_k  (step plot)
  3) Cumulative reward          →  y-label: Cumulative Reward
  4) Cumulative flight time     →  y-label: Flight Time (hrs)
  5) Cumulative failure prob.   →  y-label: P(failure)

Usage:
  python plot_episode_split.py \
      --results path/to/results.h5 \
      --episode 1 \
      --outdir out/ \
      --combo 0.2,6.0 \
      --window 0,2880 \
      --transition-model moderate
"""

from __future__ import annotations
import argparse, os, sys
from typing import Dict, List, Tuple, Optional

import h5py
import numpy as np
import matplotlib.pyplot as plt
from cycler import cycler
from matplotlib.ticker import MaxNLocator

# Reuse helpers from the original plot_episode script
from plot_episode import (
    STYLE_NAME, RCPARAMS, COLOR_CYCLE, DT_MIN,
    stage_to_days, actions_to_cumulative_hours,
    select_sims, legend_label, align_lengths_per_sim,
    apply_window_indices, build_output_name,
    parse_combos, parse_window, parse_range,
    DATASETS,
)

# Transition-probability model for cumulative failure computation
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from SolarSimulator.BaseClasses.transition_model_base import ProbabilityModelFactory

# Datasets needed beyond DATASETS for the split plots
EXTRA_DATASETS = ['trajectory']
ALL_DATASETS = DATASETS + EXTRA_DATASETS

# ────────────────────────────────────────────
# Helpers
# ────────────────────────────────────────────

def _apply_style():
    plt.style.use(STYLE_NAME)
    plt.rcParams.update(RCPARAMS)
    plt.rcParams['axes.prop_cycle'] = cycler('color', COLOR_CYCLE)


def _common_ax_style(ax):
    ax.grid(True)
    for s in ax.spines.values():
        s.set_visible(True)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=8, steps=[1, 2, 2.5, 5, 10], min_n_ticks=4, prune='both'))


def _load_data(results_path, episode_num, combos, obs_filter, wind_filter,
               start_days, start_range):
    """Load and align episode data for selected sims."""
    episode_name = f"episode {episode_num}"

    with h5py.File(results_path, 'r') as f:
        all_groups = list(f.keys())
    selected = select_sims(all_groups, combos, obs_filter, wind_filter,
                           start_days, start_range)
    if not selected:
        raise ValueError("No matching sims")

    loaded: Dict[str, Dict[str, np.ndarray]] = {}
    with h5py.File(results_path, 'r') as f:
        for sim in selected:
            try:
                grp = f[sim]['episodes'][episode_name]
            except KeyError:
                continue
            sim_data = {ds: grp[ds][:] for ds in DATASETS}
            # trajectory is (N+1, state_dim); load separately
            if 'trajectory' in grp:
                sim_data['trajectory'] = grp['trajectory'][:]
            loaded[sim] = sim_data

    if not loaded:
        raise ValueError(f"No data for '{episode_name}'")

    # align_lengths_per_sim checks DATASETS only; also trim trajectory
    loaded = align_lengths_per_sim(loaded, episode_name)
    for sim, series in loaded.items():
        if 'trajectory' in series:
            L = len(series['actions'])
            traj = series['trajectory']
            if len(traj) == L + 1:
                series['trajectory'] = traj[:-1]

    return loaded


def _build_time_info(loaded, window):
    """Build per-sim time bases and window slices."""
    sim_time_bases: Dict[str, np.ndarray] = {}
    sim_slices: Dict[str, Tuple[int, int]] = {}
    tmin_days = None
    tmax_days = None
    for sim, series in loaded.items():
        L = len(series['actions'])
        sl = apply_window_indices(L, window)
        if sl is None:
            continue
        i0, i1 = sl
        sim_slices[sim] = (i0, i1)
        t = stage_to_days(np.arange(L))
        sim_time_bases[sim] = t
        if tmin_days is None:
            tmin_days = t[i0]; tmax_days = t[i1]
        else:
            tmin_days = min(tmin_days, t[i0]); tmax_days = max(tmax_days, t[i1])

    if not sim_slices:
        raise ValueError("Window does not overlap any selected simulation data.")
    return sim_time_bases, sim_slices, tmin_days, tmax_days


def _compute_failure_probability(wind_series, actions, trajectory, model):
    """Compute per-step failure probability (1 - P_success) at each timestep."""
    L = len(actions)
    p_fail = np.zeros(L)
    for k in range(L):
        state_k = trajectory[k]
        # If vehicle is already broken, failure prob is 1
        if state_k[1] == 2:
            p_fail[k] = 1.0
            continue
        p_fail[k] = 1.0 - model.compute_probability(wind_series[k], actions[k], state_k).item()
    return p_fail


# ────────────────────────────────────────────
# Figure 1: Environment
# ────────────────────────────────────────────

ENV_DATASETS = ['solar_series', 'wind_series', 'whale_series']
ENV_YLABELS  = [r'$\tilde{e}^+_k$ (Wh)', r'$w_k$ (m/s)', r'$O_k$']

def plot_environment(loaded, sim_time_bases, sim_slices, tmin_days, tmax_days,
                     figsize=(4, 4), dpi=300):
    """Create the 3-panel environment figure. Returns the Figure."""
    _apply_style()
    fig, axes = plt.subplots(3, 1, sharex=True, figsize=figsize, constrained_layout=True)

    for ax, ds, ylabel in zip(axes, ENV_DATASETS, ENV_YLABELS):
        for sim, data in loaded.items():
            if sim not in sim_slices:
                continue
            i0, i1 = sim_slices[sim]
            y = data[ds].astype(float)[i0:i1+1]
            t_days = sim_time_bases[sim][i0:i1+1]
            if ds == 'solar_series':
                y = y / 3600.0
            ax.plot(t_days, y, color='black', linewidth=0.5)
            break  # exogenous — same across sims, plot once
        ax.set_ylabel(ylabel)
        _common_ax_style(ax)

    axes[-1].set_xlabel('Time (days)')
    for ax in axes:
        ax.set_xlim(tmin_days, tmax_days)
    return fig


# ────────────────────────────────────────────
# Figure 2: Vehicle
# ────────────────────────────────────────────

VEH_YLABELS = [
    r'$\bar{c}_k$ (Wh)',
    r'$m_k$',
    r'$\Sigma\, r_k$',
    r'$t_{\mathrm{fly}}$ (hrs)',
    r'$P_f(k)$',
]

def plot_vehicle(loaded, sim_time_bases, sim_slices, tmin_days, tmax_days,
                 transition_model, figsize=(8, 7), dpi=300):
    """Create the 5-panel vehicle figure. Returns the Figure."""
    _apply_style()
    n_panels = 5
    fig, axes = plt.subplots(n_panels, 1, sharex=True, figsize=figsize, constrained_layout=True)

    for sim, data in loaded.items():
        if sim not in sim_slices:
            continue
        i0, i1 = sim_slices[sim]
        t_days = sim_time_bases[sim][i0:i1+1]
        lbl = legend_label(sim)

        # Panel 0: Stored energy
        y_energy = data['energy_series'].astype(float)[i0:i1+1] / 3600.0
        axes[0].plot(t_days, y_energy, label=lbl, linewidth=0.5)

        # Panel 1: Mode (step plot)
        mode = data['trajectory'][:, 1].astype(float)[i0:i1+1] if 'trajectory' in data else np.zeros(i1-i0+1)
        axes[1].step(t_days, mode, where='mid', label=lbl, linewidth=0.5)

        # Panel 2: Cumulative reward
        cum_reward = np.cumsum(data['rewards'].astype(float))[i0:i1+1]
        axes[2].plot(t_days, cum_reward, label=lbl, linewidth=0.5)

        # Panel 3: Cumulative flight time
        cum_flight = actions_to_cumulative_hours(data['actions'])[i0:i1+1]
        axes[3].plot(t_days, cum_flight, label=lbl, linewidth=0.5)

        # Panel 4: Per-step failure probability
        p_fail = _compute_failure_probability(
            data['wind_series'], data['actions'],
            data['trajectory'] if 'trajectory' in data else np.zeros((len(data['actions']), 2)),
            transition_model,
        )[i0:i1+1]
        axes[4].plot(t_days, p_fail, label=lbl, linewidth=0.5)

    for ax, ylabel in zip(axes, VEH_YLABELS):
        ax.set_ylabel(ylabel)
        _common_ax_style(ax)

    axes[-1].set_xlabel('Time (days)')
    for ax in axes:
        ax.set_xlim(tmin_days, tmax_days)

    # Figure-level legend
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels,
               loc='outside upper center',
               ncol=min(max(len(labels), 1), 4), frameon=True)
    return fig


# ────────────────────────────────────────────
# Main orchestration
# ────────────────────────────────────────────

def plot_episode_split(results_path, episode_num, outdir, combos, obs_filter,
                       wind_filter, start_days, start_range, window,
                       transition_model_name='moderate', dpi=300):
    loaded = _load_data(results_path, episode_num, combos, obs_filter,
                        wind_filter, start_days, start_range)
    sim_time_bases, sim_slices, tmin_days, tmax_days = _build_time_info(loaded, window)

    transition_model = ProbabilityModelFactory.select_probability_model(transition_model_name)

    os.makedirs(outdir, exist_ok=True)
    base = build_output_name(episode_num, combos, obs_filter, wind_filter,
                             start_days, start_range, window)
    stem = os.path.splitext(base)[0]

    # Figure 1: Environment
    fig_env = plot_environment(loaded, sim_time_bases, sim_slices, tmin_days, tmax_days, dpi=dpi)
    env_path = os.path.join(outdir, f"environment_{stem}.png")
    fig_env.savefig(env_path, dpi=dpi, bbox_inches='tight')
    plt.close(fig_env)
    print(f"Saved: {env_path}")

    # Figure 2: Vehicle
    fig_veh = plot_vehicle(loaded, sim_time_bases, sim_slices, tmin_days, tmax_days,
                           transition_model, dpi=dpi)
    veh_path = os.path.join(outdir, f"vehicle_{stem}.png")
    fig_veh.savefig(veh_path, dpi=dpi, bbox_inches='tight')
    plt.close(fig_veh)
    print(f"Saved: {veh_path}")

    return env_path, veh_path


# ────────────────────────────────────────────
# CLI
# ────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description="Create two-figure (environment + vehicle) episode plots from a results HDF5.")
    ap.add_argument("--results", required=True, help="Path to results HDF5 file.")
    ap.add_argument("--episode", required=True, type=int, help="Episode number.")
    ap.add_argument("--outdir", required=True, help="Directory to save PNGs.")
    ap.add_argument("--combo", dest="combos", metavar="OBS,WIND", nargs='*',
                    help="Exact threshold pairs, e.g., --combo 0.15,6.0 0.2,9.0")
    ap.add_argument("--obs-thresh", type=float, nargs='*',
                    help="Observation thresholds (space-separated).")
    ap.add_argument("--wind-thresh", type=float, nargs='*',
                    help="Wind thresholds (space-separated).")
    ap.add_argument("--start-day", type=int, nargs='*',
                    help="Mission start day-of-year(s) to include.")
    ap.add_argument("--start-range", type=parse_range,
                    help="Inclusive start day range, e.g., 150,170.")
    ap.add_argument("--window", type=parse_window, default=None,
                    help="Timestep window (inclusive), e.g., --window 0,2880")
    ap.add_argument("--transition-model", default="moderate",
                    help="Probability model name (default: moderate).")
    args = ap.parse_args()

    combos = parse_combos(args.combos)
    plot_episode_split(
        results_path=args.results,
        episode_num=args.episode,
        outdir=args.outdir,
        combos=combos,
        obs_filter=args.obs_thresh if not combos else None,
        wind_filter=args.wind_thresh if not combos else None,
        start_days=args.start_day,
        start_range=args.start_range,
        window=args.window,
        transition_model_name=args.transition_model,
    )


if __name__ == "__main__":
    main()

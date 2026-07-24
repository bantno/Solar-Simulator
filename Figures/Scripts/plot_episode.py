#!/usr/bin/env python3
"""Plot a single episode (multi-panel) from a results HDF5.

Panels (top→bottom):
  1) G_k (Wh)    [black]
  2) w_k (m/s)   [black]
  3) O_k         [black]
  4) E_k (Wh)    [colored by sim]
  5) a_k         [colored by sim]  (unitless)
  6) r_k         [colored by sim]  (unitless, y-axis uses symlog)
  7) Total Flight (hrs) [colored by sim]

Legend:
  - Figure-level legend at the very top (no title). Optimal + selected threshold combos.

Filtering:
  - Always include Optimal.
  - Exact threshold combos via:
        --combo 0.15,6.0 --combo 0.2,9.0
  - If no --combo, optional independent filters:
        --obs-thresh 0.1 0.15   --wind-thresh 6.0 9.0
  - Mission start day filters (parsed from name suffix like _d161):
        --start-day 161 162
        --start-range 150,170    (inclusive)

Windowing:
  - Use --window START,STOP where START and STOP are timestep indices (15-min stages).
  - Window is inclusive and applied per-sim; sims that end before START are skipped.

Usage:
  python plot_episode.py \
      --results path/to/results.h5 \
      --episode 1 \
      --outdir out/ \
      --combo 0.15,6.0 --combo 0.2,9.0 \
      --start-range 150,170 \
      --window 0,2880
"""

from __future__ import annotations
import argparse, os, re
from typing import Dict, List, Tuple, Optional

import h5py
import numpy as np
import matplotlib.pyplot as plt
from cycler import cycler
from matplotlib.ticker import MaxNLocator

# ----------------------------
# Config / styling
# ----------------------------
STYLE_NAME = 'seaborn-v0_8-whitegrid'
RCPARAMS = {
    'font.size':       10,
    'axes.titlesize':  12,
    'axes.labelsize':  11,
    'lines.linewidth': 0.5,   # all lines thin
    'figure.dpi':      300,
    'legend.fontsize': 9,
    'legend.frameon':  True,
    'legend.framealpha': 0.9,
    'legend.edgecolor':  'black',
    'grid.linestyle': '-',
    'grid.alpha': 0.35,
}
COLOR_CYCLE = [
    '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
    '#9fbd67', '#5c3e00', '#bd67b1', '#7e1603', '#3108d4'
]

DT_MIN = 15  # minutes

DATASETS = [
    'solar_series',
    'wind_series',
    'whale_series',
    'energy_series',
    'actions',
    'rewards',
]
YLABELS = [
    r'$\tilde{e}^+_k$ (Wh)',
    r'$w_k$ (m/s)',
    r'$O_k$',
    r'$\bar{c}_k$ (Wh)',
    r'$m_k$',
    r'$r_k$',
]

# ----------------------------
# Helpers
# ----------------------------
def stage_to_days(n: np.ndarray) -> np.ndarray:
    return np.asarray(n) * DT_MIN / (60.0 * 24.0)

def actions_to_cumulative_hours(actions: np.ndarray) -> np.ndarray:
    flags = (actions != 0).astype(int)
    return np.cumsum(flags) * (DT_MIN / 60.0)

def parse_thresholds_from_name(sim_name: str) -> Tuple[Optional[float], Optional[float]]:
    m = re.search(r"_t(?P<obs>[\d.]+)_w(?P<wind>[\d.]+)", sim_name, re.IGNORECASE)
    return (None, None) if not m else (float(m.group("obs")), float(m.group("wind")))

def parse_start_day_from_name(sim_name: str) -> Optional[int]:
    """Extract mission start day-of-year from suffix like '_d161'."""
    m = re.search(r"_d(?P<day>\d+)\b", sim_name)
    return int(m.group("day")) if m else None

def legend_label(sim_name: str) -> str:
    if sim_name.lower().startswith("optimal"):
        return "Optimal"
    obs, wind = parse_thresholds_from_name(sim_name)
    if obs is not None and wind is not None:
        return rf"$w_{{to}}={wind},\ O_{{th}}={obs}$"
    return sim_name

def select_sims(
    all_group_names: List[str],
    combos: List[Tuple[float, float]] | None,
    obs_filter: List[float] | None,
    wind_filter: List[float] | None,
    start_days: List[int] | None,
    start_range: Optional[Tuple[int,int]],
) -> List[str]:
    """
    Always include 'Optimal*'.
    Apply, in order:
      1) combo exact (if provided)
      2) else independent filters (if provided)
      3) else no threshold filtering
    AND also apply start-day filters (if provided).
    """
    selected: List[str] = []
    using_combos = combos is not None and len(combos) > 0
    have_thresh_filters = (obs_filter is not None and len(obs_filter) > 0) or \
                          (wind_filter is not None and len(wind_filter) > 0)
    have_day_filters = (start_days is not None and len(start_days) > 0) or (start_range is not None)

    for g in all_group_names:
        # Threshold checks
        if g.lower().startswith("optimal"):
            pass  # Optimal always allowed through threshold filters
        else:
            obs, wind = parse_thresholds_from_name(g)
            if using_combos:
                if not (obs is not None and wind is not None and any(
                        np.isclose(obs, o, atol=1e-9) and np.isclose(wind, w, atol=1e-9) for (o, w) in combos)):
                    continue
            elif have_thresh_filters:
                if obs is None or wind is None:
                    continue
                ok = True
                if obs_filter and len(obs_filter) > 0:
                    ok &= any(np.isclose(obs, of, rtol=0, atol=1e-9) for of in obs_filter)
                if wind_filter and len(wind_filter) > 0:
                    ok &= any(np.isclose(wind, wf, rtol=0, atol=1e-9) for wf in wind_filter)
                if not ok:
                    continue
            # else: no threshold filtering

        # Start-day filters (apply to Optimal and threshold sims)
        if have_day_filters:
            day = parse_start_day_from_name(g)
            if day is None:
                continue
            if start_days and len(start_days) > 0 and day not in set(start_days):
                continue
            if start_range is not None:
                a, b = start_range
                if not (a <= day <= b):
                    continue

        selected.append(g)

    # Optimal first, then by (wind, obs)
    optimal = [g for g in selected if g.lower().startswith("optimal")]
    rest = [g for g in selected if g not in optimal]

    def sort_key(name: str):
        o, w = parse_thresholds_from_name(name)
        return (1e9, 1e9) if (o is None or w is None) else (w, o)

    rest_sorted = sorted(rest, key=sort_key)
    return optimal + rest_sorted

def build_output_name(episode_num:int, combos, obs_filter, wind_filter, start_days, start_range, window)->str:
    if combos and len(combos) > 0:
        pairs = "__".join([f"obs-{o}_wind-{w}" for (o, w) in combos])
        filt = f"combos__{pairs}"
    elif obs_filter or wind_filter:
        obs_txt="obs-"+(",".join(str(v) for v in (obs_filter or [])) if obs_filter else "*")
        wind_txt="wind-"+(",".join(str(v) for v in (wind_filter or [])) if wind_filter else "*")
        filt=f"{obs_txt}__{wind_txt}"
    else:
        filt="all-thresholds"
    day_txt = ""
    if start_days and len(start_days) > 0:
        day_txt += "__days-" + ",".join(str(d) for d in start_days)
    if start_range:
        day_txt += f"__dayrange-{start_range[0]}-{start_range[1]}"
    win_txt = f"__win-{window[0]}-{window[1]}" if window else ""
    return f"episode_{episode_num}_{filt}{day_txt}{win_txt}.png"

def align_lengths_per_sim(loaded:Dict[str,Dict[str,np.ndarray]], episode_name:str)->Dict[str,Dict[str,np.ndarray]]:
    problems=[]; trim_notes=[]
    for sim,series in loaded.items():
        if 'actions' not in series:
            problems.append(f"{sim}: missing 'actions' in '{episode_name}'"); continue
        L=len(series['actions'])
        # Typical boundary (N+1) for energy; trim tail to align
        if 'energy_series' in series:
            Le=len(series['energy_series'])
            if Le==L+1:
                series['energy_series']=series['energy_series'][:-1]
                trim_notes.append(f"{sim}: trimmed energy_series {Le}->{L}")
        for ds in DATASETS:
            if ds not in series:
                problems.append(f"{sim}: missing '{ds}' in '{episode_name}'"); continue
            if len(series[ds])!=L:
                problems.append(f"{sim}: '{ds}' length {len(series[ds])} != actions length {L}")
    if problems: raise ValueError("Length mismatch:\n  - "+"\n  - ".join(problems))
    for note in trim_notes: print(note)
    return loaded

def apply_window_indices(L: int, window: Optional[Tuple[int,int]]) -> Optional[Tuple[int,int]]:
    """
    Given a series length L and a desired (start, stop) inclusive window (in timesteps),
    return a clipped (i0, i1) suitable for slicing [i0 : i1+1], or None if no overlap.
    """
    if not window:
        return (0, L-1) if L > 0 else None
    start, stop = window
    if start < 0 or stop < 0:
        raise ValueError("--window start and stop must be non-negative timestep indices")
    if stop < start:
        raise ValueError("--window stop must be >= start")
    i0 = max(0, start)
    i1 = min(stop, L-1)
    if i1 < i0:
        return None
    return (i0, i1)

# ----------------------------
# Plot
# ----------------------------
def plot_episode(results_path:str, episode_num:int, outdir:str,
                 combos:List[Tuple[float,float]]|None,
                 obs_filter:List[float]|None, wind_filter:List[float]|None,
                 start_days:List[int]|None, start_range:Optional[Tuple[int,int]],
                 window: Optional[Tuple[int,int]],
                 figsize=(8,10), dpi=300)->str:
    plt.style.use(STYLE_NAME); plt.rcParams.update(RCPARAMS)
    plt.rcParams['axes.prop_cycle']=cycler('color',COLOR_CYCLE)

    # Episode groups are lowercase: "episode {num}"
    episode_name=f"episode {episode_num}"

    with h5py.File(results_path,'r') as f: all_groups=list(f.keys())
    selected=select_sims(all_group_names=all_groups, combos=combos,
                         obs_filter=obs_filter, wind_filter=wind_filter,
                         start_days=start_days, start_range=start_range)
    if not selected: raise ValueError("No matching sims")

    loaded={}
    with h5py.File(results_path,'r') as f:
        for sim in selected:
            try: grp=f[sim]['episodes'][episode_name]
            except KeyError: continue
            loaded[sim]={ds:grp[ds][:] for ds in DATASETS}
    if not loaded: raise ValueError(f"No data for '{episode_name}'")

    loaded=align_lengths_per_sim(loaded,episode_name)

    # Time bases + windowing per sim
    sim_time_bases: Dict[str, np.ndarray] = {}
    sim_slices: Dict[str, Tuple[int,int]] = {}
    tmin_days = None
    tmax_days = None
    for sim,series in loaded.items():
        L=len(series['actions'])
        sl = apply_window_indices(L, window)
        if sl is None:
            continue  # this sim doesn't overlap the requested window
        i0, i1 = sl
        sim_slices[sim] = (i0, i1)
        t = stage_to_days(np.arange(L))
        sim_time_bases[sim] = t
        if tmin_days is None:
            tmin_days = t[i0]; tmax_days = t[i1]
        else:
            tmin_days = min(tmin_days, t[i0]); tmax_days = max(tmax_days, t[i1])

    if len(sim_slices) == 0:
        raise ValueError("Window does not overlap any selected simulation data.")

    n_panels=len(DATASETS)+1
    fig,axes=plt.subplots(n_panels,1,sharex=True,figsize=figsize,constrained_layout=False)

    # Data panels
    for idx,(ax,ds,ylabel) in enumerate(zip(axes,DATASETS,YLABELS)):
        use_black=(idx<3)
        for sim,data in loaded.items():
            if sim not in sim_slices:
                continue
            i0,i1 = sim_slices[sim]
            y_full = data[ds].astype(float)
            y = y_full[i0:i1+1]
            t_days = sim_time_bases[sim][i0:i1+1]
            if ds in ('energy_series','solar_series'): y=y/3600.0  # Wh

            if ds=='actions' or np.issubdtype(y.dtype,np.integer) or set(np.unique(y)).issubset({0.0,1.0}):
                ax.step(t_days,y,where='mid',
                        color='black' if use_black else None,
                        label=None if use_black else legend_label(sim),
                        linewidth=0.5)
            else:
                ax.plot(t_days,y,
                        color='black' if use_black else None,
                        label=None if use_black else legend_label(sim),
                        linewidth=0.5)

        ax.set_ylabel(ylabel)
        ax.grid(True)
        for s in ax.spines.values(): s.set_visible(True)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=8,steps=[1,2,2.5,5,10],min_n_ticks=4,prune='both'))

        # Rewards axis: nonlinear to show small positive rewards
        if ds == 'rewards':
            ax.set_yscale('symlog', linthresh=0.1, linscale=1.0, base=10)

    # Cumulative flight-time
    cf_ax=axes[-1]
    for sim,data in loaded.items():
        if sim not in sim_slices:
            continue
        i0,i1 = sim_slices[sim]
        cum_full = actions_to_cumulative_hours(data['actions'])
        cum = cum_full[i0:i1+1]
        t_days = sim_time_bases[sim][i0:i1+1]
        cf_ax.plot(t_days,cum,label=legend_label(sim),linewidth=0.5)
    cf_ax.set_ylabel('Total Flight (hrs)')
    cf_ax.set_xlabel('Time (days)')
    cf_ax.grid(True)
    cf_ax.xaxis.set_major_locator(MaxNLocator(nbins=8,steps=[1,2,2.5,5,10],min_n_ticks=4))

    # Figure-level legend at top (no title)
    handles,labels=cf_ax.get_legend_handles_labels()
    fig.legend(handles,labels,
               loc='upper center',bbox_to_anchor=(0.5,0.995),
               ncol=min(max(len(labels),1), 4),
               frameon=True)

    # Apply shared x-limits to the requested/available window (in days)
    for ax in axes: ax.set_xlim(tmin_days, tmax_days)
    # More headroom for the top legend; no plot title
    fig.subplots_adjust(top=0.92,bottom=0.08,left=0.12,right=0.95,hspace=0.30)

    os.makedirs(outdir,exist_ok=True)
    outname=build_output_name(episode_num, combos, obs_filter, wind_filter, start_days, start_range, window)
    outpath=os.path.join(outdir,outname)
    fig.savefig(outpath,dpi=dpi,bbox_inches='tight'); plt.close(fig)
    return outpath

# ----------------------------
# CLI
# ----------------------------
def parse_combos(values: List[str] | None) -> List[Tuple[float,float]]:
    if not values: return []
    combos=[]
    for v in values:
        try:
            obs_str, wind_str = v.split(",", 1)
            combos.append((float(obs_str), float(wind_str)))
        except Exception as e:
            raise argparse.ArgumentTypeError(f"Invalid --combo '{v}'. Use OBS,WIND (e.g., 0.15,6.0)") from e
    return combos

def parse_window(value: Optional[str]) -> Optional[Tuple[int,int]]:
    if value is None:
        return None
    try:
        a, b = value.split(",", 1)
        start = int(a.strip())
        stop  = int(b.strip())
        if start < 0 or stop < 0 or stop < start:
            raise ValueError
        return (start, stop)
    except Exception:
        raise argparse.ArgumentTypeError("Invalid --window. Use START,STOP with non-negative integers and STOP>=START.")

def parse_range(value: Optional[str]) -> Optional[Tuple[int,int]]:
    """Parse a simple inclusive integer range like '150,170'."""
    if value is None:
        return None
    try:
        a, b = value.split(",", 1)
        lo = int(a.strip()); hi = int(b.strip())
        if lo > hi: raise ValueError
        return (lo, hi)
    except Exception:
        raise argparse.ArgumentTypeError("Invalid --start-range. Use A,B with integers and B>=A.")

def main():
    ap=argparse.ArgumentParser(description="Create a multi-panel episode plot from a results HDF5.")
    ap.add_argument("--results", required=True, help="Path to results HDF5 file.")
    ap.add_argument("--episode", required=True, type=int, help="Episode number (maps to 'episode N').")
    ap.add_argument("--outdir", required=True, help="Directory to save the PNG.")
    # Threshold selection
    ap.add_argument("--combo", dest="combos", metavar="OBS,WIND", nargs='*',
                    help="Exact threshold pairs to include (repeatable), e.g., --combo 0.15,6.0 0.2,9.0")
    ap.add_argument("--obs-thresh", type=float, nargs='*',
                    help="Fallback filter: observation thresholds (space-separated).")
    ap.add_argument("--wind-thresh", type=float, nargs='*',
                    help="Fallback filter: wind thresholds (space-separated).")
    # Mission start day filters
    ap.add_argument("--start-day", type=int, nargs='*',
                    help="Mission start day-of-year(s) to include (parsed from sim name suffix like _d161).")
    ap.add_argument("--start-range", type=parse_range,
                    help="Inclusive mission start day-of-year range, e.g., 150,170.")
    # Time window
    ap.add_argument("--window", type=parse_window, default=None,
                    help="Time window in timesteps (inclusive), e.g., --window 0,2880")
    args=ap.parse_args()

    combos = parse_combos(args.combos)
    out = plot_episode(
        results_path=args.results,
        episode_num=args.episode,
        outdir=args.outdir,
        combos=combos,
        obs_filter=args.obs_thresh if not combos else None,
        wind_filter=args.wind_thresh if not combos else None,
        start_days=args.start_day,
        start_range=args.start_range,
        window=args.window,
        figsize=(4,6),
        dpi=300
    )
    print(f"Saved: {out}")

if __name__=="__main__":
    main()

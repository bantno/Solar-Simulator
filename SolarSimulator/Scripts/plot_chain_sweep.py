#!/usr/bin/env python3
"""Figures for the chain-vs-iid wind-persistence evaluation sweep.

Consumes the outputs of Scripts/compare_chain_sweep.py (comparison_cells.csv), the sweep
run directories (summary.csv, HDF5 episodes, solver_tables/*.npy), and the historical
weather pickles. All figures use one visual language: blue = iid arm, orange = chain arm
(colorblind-safe pair, reinforced by linestyle/marker), diverging colormaps centered at
zero for deltas.

Figures (select with --figs, default all):
  1 capacity-reliability frontier per location ("Wh saved at equal reliability")
  2 star-sweep delta panels (start month, duration, capacity x location heatmap)
  3 per-episode reward CDFs + paired delta distribution (tail / CVaR story)
  4 benefit vs wind-persistence statistics across locations (mechanism)
  5 calibration: self-predicted vs realized performance (iid overconfidence)
  6 event-aligned composites around storm onsets (anticipatory charging)
  7 value-table diagnostics (where in state space bin knowledge matters)
  8 three-way policy comparison: iid / chain optimal vs the best threshold policy

Figures 1, 3, and 8 include the best-threshold benchmark (green) when the threshold
configs have been run (generate with --thresholds).

Usage (pvlib conda env, from SolarSimulator/):
    conda run -n pvlib python Scripts/plot_chain_sweep.py [--smoke] [--figs 1,2,3]
"""
import argparse
import glob
import json
import os
import sys

import h5py
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

PKG_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # .../SolarSimulator
if PKG_DIR not in sys.path:
    sys.path.insert(0, PKG_DIR)
REPO_ROOT = os.path.dirname(PKG_DIR)

from Scripts.compare_chain_sweep import (  # noqa: E402
    load_manifest, load_summaries, read_episode_scalars, cvar,
    wh_saved_at_reliability, capacity_at_reliability, _find_solver_table, _n_soc_levels,
    chain_edges_for,
)

# Colorblind-safe triple (seaborn "colorblind" hues), reinforced by linestyle + marker.
COLOR = {"iid": "#0173B2", "chain": "#DE8F05", "thresh": "#029E73"}
LS = {"iid": "-", "chain": "--", "thresh": ":"}
MARKER = {"iid": "o", "chain": "s", "thresh": "^"}
ARM_LABEL = {"iid": "i.i.d. policy", "chain": "chain policy",
             "thresh": "best threshold policy"}
DPI = 150
TOP_BIN_EDGE = 10.0            # m/s, top interior cutpoint of [5, 10]


def _savefig(fig, out_dir, name):
    path = os.path.join(out_dir, name)
    fig.savefig(path, dpi=DPI)
    plt.close(fig)
    print(f"[fig] {path}")
    return path


def _binom_ci95(p_frac, n):
    return 196.0 * np.sqrt(np.maximum(p_frac * (1 - p_frac), 0) / max(n, 1))  # in pct


# ----------------------------------------------------------------------------------
# Fig 1: capacity-reliability frontier
# ----------------------------------------------------------------------------------

def fig1_frontier(cells, manifest, out_dir):
    sub = cells[(cells["sweep"] == "capgrid") & (cells["world"] == "hist")]
    locs = sorted(sub["location_id"].unique())
    if not len(locs):
        print("[skip] fig1: no capgrid hist cells")
        return
    n_eps = manifest["episodes"]
    ncols = min(2, len(locs))
    nrows = int(np.ceil(len(locs) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 4.2 * nrows),
                             sharex=True, squeeze=False, constrained_layout=True)
    for ax, loc in zip(axes.ravel(), locs):
        s = sub[sub["location_id"] == loc].sort_values("battery_capacity")
        caps = s["battery_capacity"].to_numpy(float)
        for arm in ("iid", "chain"):
            fail = s[f"failure_pct_{arm}"].to_numpy(float)
            err = _binom_ci95(fail / 100.0, n_eps)
            ax.errorbar(caps, fail, yerr=err, color=COLOR[arm], ls=LS[arm],
                        marker=MARKER[arm], ms=5, capsize=3, label=ARM_LABEL[arm])
        has_thresh = ("failure_pct_thresh_minfail" in s.columns
                      and s["failure_pct_thresh_minfail"].notna().any())
        if has_thresh:
            fail = s["failure_pct_thresh_minfail"].to_numpy(float)
            err = _binom_ci95(fail / 100.0, n_eps)
            ax.errorbar(caps, fail, yerr=err, color=COLOR["thresh"], ls=LS["thresh"],
                        marker=MARKER["thresh"], ms=5, capsize=3,
                        label="best threshold (reliability-tuned)")
        saved = wh_saved_at_reliability(s, target_failure_pct=5.0)
        if saved is not None:
            ax.axhline(5.0, color="0.75", lw=0.8, zorder=0)
            txt = (f"chain saves {saved:.0f} Wh at 5% failure" if saved >= 0
                   else f"chain needs {-saved:.0f} Wh more at 5% failure")
            if has_thresh:
                c_chain = capacity_at_reliability(s, "chain", target_failure_pct=5.0)
                c_thresh = capacity_at_reliability(s, "thresh", target_failure_pct=5.0)
                if c_chain is not None and c_thresh is not None:
                    txt += f"; {c_thresh - c_chain:+.0f} Wh vs threshold"
            ax.annotate(txt, xy=(caps.mean(), 5.0), fontsize=9, color="0.25",
                        ha="center", va="bottom")
        ax.set_title(loc, fontsize=10)
        ax.set_ylabel("Mission failure rate [%]")
        ax.set_xlabel("Battery capacity [Wh]")
    for ax in axes.ravel()[len(locs):]:
        ax.set_visible(False)
    axes[0, 0].legend(fontsize=9)
    fig.suptitle("Capacity-reliability frontier, historical weather", fontsize=11)
    _savefig(fig, out_dir, "fig1_capacity_reliability_frontier.png")


# ----------------------------------------------------------------------------------
# Fig 2: star-sweep delta panels
# ----------------------------------------------------------------------------------

def fig2_delta_panels(cells, out_dir):
    hist = cells[cells["world"] == "hist"]
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.4), constrained_layout=True)

    # (a) delta failure vs start month
    sd = hist[hist["sweep"] == "startdate"].copy()
    ax = axes[0]
    if len(sd):
        sd["month"] = pd.to_datetime(sd["start_time"]).dt.month
        sd = sd.sort_values("month")
        d = 100 * sd["d_failure"].to_numpy(float)
        lo = 100 * sd["d_failure_lo"].to_numpy(float)
        hi = 100 * sd["d_failure_hi"].to_numpy(float)
        ax.axhline(0, color="0.8", lw=0.8)
        ax.errorbar(sd["month"], d, yerr=[d - lo, hi - d], fmt="o",
                    color="0.2", ecolor="0.5", capsize=3)
        ax.set_xticks(sd["month"].unique())
    ax.set_xlabel("Mission start month")
    ax.set_ylabel(r"$\Delta$ failure rate [pp] (chain $-$ iid)")
    ax.set_title("(a) By season", fontsize=10)

    # (b) delta failure vs duration
    du = hist[hist["sweep"] == "duration"].copy()
    ax = axes[1]
    if len(du):
        du["days"] = du["horizon"].astype(int) * 15 / 1440
        du = du.sort_values("days")
        d = 100 * du["d_failure"].to_numpy(float)
        lo = 100 * du["d_failure_lo"].to_numpy(float)
        hi = 100 * du["d_failure_hi"].to_numpy(float)
        ax.axhline(0, color="0.8", lw=0.8)
        ax.errorbar(du["days"], d, yerr=[d - lo, hi - d], fmt="o",
                    color="0.2", ecolor="0.5", capsize=3)
    ax.set_xlabel("Mission duration [days]")
    ax.set_ylabel(r"$\Delta$ failure rate [pp]")
    ax.set_title("(b) By duration", fontsize=10)

    # (c) capacity x location heatmap
    cg = hist[hist["sweep"] == "capgrid"]
    ax = axes[2]
    if len(cg):
        pivot = cg.pivot_table(index="location_id", columns="battery_capacity",
                               values="d_failure", aggfunc="mean") * 100
        vmax = max(abs(np.nanmin(pivot.values)), abs(np.nanmax(pivot.values)), 1e-9)
        im = ax.imshow(pivot.values, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels([f"{c:g}" for c in pivot.columns])
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels(pivot.index, fontsize=8)
        for i in range(pivot.shape[0]):
            for j in range(pivot.shape[1]):
                v = pivot.values[i, j]
                if np.isfinite(v):
                    ax.text(j, i, f"{v:+.1f}", ha="center", va="center", fontsize=8,
                            color="white" if abs(v) > 0.6 * vmax else "black")
        fig.colorbar(im, ax=ax, label=r"$\Delta$ failure [pp] (red = chain worse)")
    ax.set_xlabel("Battery capacity [Wh]")
    ax.set_title("(c) Capacity x location", fontsize=10)

    fig.suptitle("Where the chain formulation helps (negative = fewer failures), "
                 "historical weather, paired episodes", fontsize=11)
    _savefig(fig, out_dir, "fig2_delta_panels.png")


# ----------------------------------------------------------------------------------
# Fig 3: reward CDFs + paired delta distribution
# ----------------------------------------------------------------------------------

def _best_thresh_summary_row(sub, cells, loc, cap):
    """Summary row of the per-cell reward-best threshold combo, or None."""
    cc = cells[(cells["sweep"] == "capgrid") & (cells["world"] == "hist")
               & (cells["location_id"] == loc)
               & (cells["battery_capacity"] == float(cap))]
    if cc.empty or "thresh_best_obs" not in cc.columns or pd.isna(cc.iloc[0]["thresh_best_obs"]):
        return None
    obs, wind = float(cc.iloc[0]["thresh_best_obs"]), float(cc.iloc[0]["thresh_best_wind"])
    t = sub[(sub["arm"] == "thresh") & (sub["battery_capacity"] == float(cap))]
    t = t[np.isclose(t["observation_threshold"].astype(float), obs)
          & np.isclose(t["wind_threshold"].astype(float), wind)]
    return None if t.empty else t.iloc[0]


def fig3_reward_cdf(cells, summaries, manifest, out_dir):
    base_loc = manifest["baseline"]["location_id"]
    sub = summaries[(summaries["sweep"] == "capgrid") & (summaries["world"] == "hist")
                    & (summaries["location_id"] == base_loc)]
    caps = sorted(sub["battery_capacity"].unique())
    if not caps:
        print("[skip] fig3: no capgrid hist runs at baseline location")
        return
    # middle three capacities (or all if fewer)
    mid = len(caps) // 2
    chosen = caps[max(0, mid - 1):mid + 2]

    fig, axes = plt.subplots(1, len(chosen) + 1, figsize=(4.4 * (len(chosen) + 1), 4.2),
                             constrained_layout=True)
    base_cap = manifest["baseline"]["capacity"]
    paired_delta_rewards = None
    for ax, cap in zip(axes[:-1], chosen):
        eps = {}
        for arm in ("iid", "chain", "thresh"):
            if arm == "thresh":
                r = _best_thresh_summary_row(sub, cells, base_loc, cap)
                if r is None:
                    continue
            else:
                row = sub[(sub["battery_capacity"] == cap) & (sub["arm"] == arm)]
                if row.empty:
                    continue
                r = row.iloc[0]
            eps[arm] = read_episode_scalars(r["run_dir"], r["group"])
            rewards = np.sort(eps[arm]["total_reward"].to_numpy(float))
            ecdf = np.arange(1, len(rewards) + 1) / len(rewards)
            ax.plot(rewards, ecdf, color=COLOR[arm], ls=LS[arm], lw=1.8,
                    label=f"{ARM_LABEL[arm]} (CVaR$_{{10}}$={cvar(rewards):.1f})")
        ax.axhline(0.10, color="0.8", lw=0.8)
        ax.text(ax.get_xlim()[0], 0.11, "worst decile", fontsize=8, color="0.4")
        ax.set_title(f"{cap:g} Wh", fontsize=10)
        ax.set_xlabel("Episode total reward")
        ax.set_ylabel("Empirical CDF")
        ax.legend(fontsize=8, loc="lower right")
        if cap == base_cap and "iid" in eps and "chain" in eps:
            joined = eps["iid"].join(eps["chain"], lsuffix="_iid", rsuffix="_chain",
                                     how="inner")
            paired_delta_rewards = (joined["total_reward_chain"]
                                    - joined["total_reward_iid"]).to_numpy(float)

    ax = axes[-1]
    if paired_delta_rewards is not None and len(paired_delta_rewards):
        d = paired_delta_rewards
        ax.hist(d, bins=60, color="0.6", edgecolor="none")
        ax.axvline(0, color="0.3", lw=0.8)
        ax.axvline(d.mean(), color=COLOR["chain"], lw=1.5,
                   label=f"mean {d.mean():+.2f}")
        ax.set_title(f"Paired per-episode $\\Delta$reward at {base_cap:g} Wh\n"
                     f"(same weather, chain $-$ iid)", fontsize=10)
        ax.set_xlabel(r"$\Delta$ total reward")
        ax.set_ylabel("Episodes")
        ax.legend(fontsize=8)
    else:
        ax.set_visible(False)
    fig.suptitle(f"Episode reward distributions on historical weather, {base_loc} "
                 "(tail improvement even where means match)", fontsize=11)
    _savefig(fig, out_dir, "fig3_reward_cdf_cvar.png")


# ----------------------------------------------------------------------------------
# Fig 4: benefit vs wind persistence
# ----------------------------------------------------------------------------------

def _persistence_stats(manifest, out_dir):
    """Per-location persistence statistics from the raw historical record (cached)."""
    cache = os.path.join(out_dir, "persistence_stats.json")
    if os.path.isfile(cache):
        with open(cache, "r") as f:
            return json.load(f)
    from Scripts.wind_persistence_precheck import analyze
    from harness.run_experiment import _find_historical_pkl, _hist_dir_from_data_path

    stats = {}
    loc_ids = sorted({c["location_id"] for c in manifest["cells"]})
    for loc_id in loc_ids:
        lat, lon = (float(v[3:]) for v in loc_id.split("_"))
        data_path = os.path.join(REPO_ROOT, "Data", "EXPECTED_DATA",
                                 f"data_expected_{loc_id}_15min.pkl")
        hist_pkl = _find_historical_pkl(_hist_dir_from_data_path(data_path), lat, lon)
        if hist_pkl is None:
            print(f"[warn] fig4: no historical pkl for {loc_id}")
            continue
        print(f"[fig4] persistence stats for {loc_id} ...")
        interior = chain_edges_for(data_path, manifest)
        # analyze()/make_bins expect the FULL edge array ([0, ..., inf]), unlike the
        # interior-cutpoint convention used in the YAML configs.
        full_edges = (np.concatenate(([0.0], interior, [np.inf]))
                      if interior is not None else None)
        r = analyze(hist_pkl, bin_edges=full_edges, n_perm=25, seed=0)
        stats[loc_id] = {
            "acf1": float(r["acf"][1]),
            "efold_hours": (float(r["efold"] * r["step_min"] / 60.0)
                            if r["efold"] else None),
            "mi_strat_corrected_bits": float(r["mi_strat_corrected_bits"]),
        }
    with open(cache, "w") as f:
        json.dump(stats, f, indent=2)
    return stats


def _spearman(x, y):
    rx = pd.Series(x).rank().to_numpy(float)
    ry = pd.Series(y).rank().to_numpy(float)
    if rx.std() == 0 or ry.std() == 0:
        return float("nan")
    return float(np.corrcoef(rx, ry)[0, 1])


def fig4_persistence(cells, manifest, out_dir):
    sub = cells[(cells["sweep"] == "capgrid") & (cells["world"] == "hist")]
    if sub.empty:
        print("[skip] fig4: no capgrid hist cells")
        return
    stats = _persistence_stats(manifest, out_dir)
    rows = []
    for loc, s in sub.groupby("location_id"):
        if loc not in stats:
            continue
        rows.append({
            "location_id": loc,
            "d_failure_pp": 100 * s["d_failure"].mean(),
            "d_reward": s["d_total_reward"].mean(),
            **stats[loc],
        })
    df = pd.DataFrame(rows)
    if len(df) < 2:
        print("[skip] fig4: need >= 2 locations with persistence stats")
        return

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), constrained_layout=True)
    for ax, xcol, xlabel in (
            (axes[0], "acf1", "Wind autocorrelation at 1 h (ACF(1))"),
            (axes[1], "efold_hours", "ACF e-folding time [h]")):
        ax.axhline(0, color="0.8", lw=0.8)
        ax.scatter(df[xcol], df["d_failure_pp"], color="0.2", s=40, zorder=3)
        for _, r in df.iterrows():
            ax.annotate(r["location_id"], (r[xcol], r["d_failure_pp"]),
                        textcoords="offset points", xytext=(6, 4), fontsize=8)
        rho = _spearman(df[xcol], df["d_failure_pp"])
        ax.set_title(f"Spearman $\\rho$ = {rho:.2f}", fontsize=10)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(r"Mean $\Delta$ failure rate [pp] (chain $-$ iid)")
    fig.suptitle("Chain benefit vs measured wind persistence "
                 "(benefit is predictable from weather statistics)", fontsize=11)
    _savefig(fig, out_dir, "fig4_benefit_vs_persistence.png")


# ----------------------------------------------------------------------------------
# Fig 5: calibration
# ----------------------------------------------------------------------------------

def fig5_calibration(cells, out_dir):
    hist = cells[(cells["world"] == "hist")].copy()
    if "native_failure_pct_iid" not in hist or hist["native_failure_pct_iid"].isna().all():
        print("[skip] fig5: no native-world runs joined (calibration unavailable)")
        return
    hist = hist.dropna(subset=["native_failure_pct_iid", "native_failure_pct_chain"])

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6), constrained_layout=True)

    ax = axes[0]
    lim = 0.0
    for arm in ("iid", "chain"):
        x = hist[f"native_failure_pct_{arm}"].to_numpy(float)
        y = hist[f"failure_pct_{arm}"].to_numpy(float)
        ax.scatter(x, y, color=COLOR[arm], marker=MARKER[arm], s=36, alpha=0.8,
                   label=ARM_LABEL[arm])
        lim = max(lim, x.max(initial=0), y.max(initial=0))
    lim *= 1.08
    ax.plot([0, lim], [0, lim], color="0.7", lw=0.9, zorder=0)
    ax.text(0.70 * lim, 0.72 * lim, "perfectly calibrated", fontsize=8,
            color="0.4", rotation=45, ha="center", va="bottom",
            rotation_mode="anchor", transform_rotates_text=True)
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.set_xlabel("Self-predicted failure rate [%] (rollout in own model)")
    ax.set_ylabel("Realized failure rate [%] (historical weather)")
    ax.set_title("Points above the line = model is overconfident", fontsize=10)
    ax.legend(fontsize=9)

    ax = axes[1]
    gap = hist.groupby("location_id")[
        ["calib_gap_failure_pct_iid", "calib_gap_failure_pct_chain"]].mean()
    xpos = np.arange(len(gap))
    w = 0.36
    ax.axhline(0, color="0.8", lw=0.8)
    ax.bar(xpos - w / 2, -gap["calib_gap_failure_pct_iid"], w, color=COLOR["iid"],
           label=ARM_LABEL["iid"])
    ax.bar(xpos + w / 2, -gap["calib_gap_failure_pct_chain"], w, color=COLOR["chain"],
           label=ARM_LABEL["chain"])
    ax.set_xticks(xpos)
    ax.set_xticklabels(gap.index, fontsize=8)
    ax.set_ylabel("Failure rate underestimation [pp]\n(realized minus self-predicted)")
    ax.set_title("How much each model underestimates real risk", fontsize=10)
    ax.legend(fontsize=9)

    fig.suptitle("Calibration: the i.i.d. model is not just weaker -- "
                 "it misjudges its own risk", fontsize=11)
    _savefig(fig, out_dir, "fig5_calibration.png")


# ----------------------------------------------------------------------------------
# Fig 6: event-aligned composites
# ----------------------------------------------------------------------------------

def _full_history_episodes(run_dir, group):
    h5s = glob.glob(os.path.join(run_dir, "*.h5"))
    out = {}
    with h5py.File(h5s[0], "r") as f:
        eps = f[group].get("episodes")
        if eps is None:
            return out
        for ename, ep in eps.items():
            if "wind_series" not in ep:
                continue
            idx = int(ep.attrs.get("episode_index", int(ename.split()[-1]) - 1))
            out[idx] = {
                "wind": np.asarray(ep["wind_series"][()], dtype=float),
                "soc": np.asarray(ep["trajectory"][()], dtype=float)[:, 0],
                "actions": np.asarray(ep["actions"][()], dtype=float),
            }
    return out


def _detect_onsets(wind, pre, post, top=TOP_BIN_EDGE, sustain=2):
    """Indices t where wind enters the top bin for >= sustain steps after a clean
    pre-window (no top-bin exposure in the previous `pre` steps)."""
    hi = wind >= top
    onsets = []
    t = pre
    while t < len(wind) - post:
        if hi[t] and hi[t:t + sustain].all() and not hi[t - pre:t].any():
            onsets.append(t)
            t += post  # non-overlapping windows
        else:
            t += 1
    return onsets


def fig6_event_aligned(summaries, manifest, out_dir, interval_min=15):
    b = manifest["baseline"]
    start_norm = pd.to_datetime(b["start"]).strftime("%Y-%m-%d %H:%M:%S")
    sub = summaries[(summaries["sweep"] == "capgrid") & (summaries["world"] == "hist")
                    & (summaries["location_id"] == b["location_id"])
                    & (summaries["battery_capacity"] == float(b["capacity"]))
                    & (summaries["start_time"] == start_norm)]
    rows = {r["arm"]: r for _, r in sub.iterrows()}
    if "iid" not in rows or "chain" not in rows:
        print("[skip] fig6: baseline cell hist pair not found")
        return
    eps = {arm: _full_history_episodes(rows[arm]["run_dir"], rows[arm]["group"])
           for arm in ("iid", "chain")}
    common = sorted(set(eps["iid"]) & set(eps["chain"]))
    if not common:
        print("[skip] fig6: no shared full-history episodes")
        return

    pre = int(24 * 60 / interval_min)    # 24 h before onset
    post = int(48 * 60 / interval_min)   # 48 h after
    windows = {"wind": [], "soc": {"iid": [], "chain": []},
               "fly": {"iid": [], "chain": []}}
    for i in common:
        wind = eps["iid"][i]["wind"]  # identical across arms (paired weather)
        for t in _detect_onsets(wind, pre, post):
            ok = all(len(eps[arm][i]["wind"]) >= t + post
                     and len(eps[arm][i]["soc"]) >= t + post
                     for arm in ("iid", "chain"))
            if not ok:
                continue  # an arm's episode ended (failure) before the window closes
            windows["wind"].append(wind[t - pre:t + post])
            for arm in ("iid", "chain"):
                windows["soc"][arm].append(eps[arm][i]["soc"][t - pre:t + post])
                windows["fly"][arm].append(eps[arm][i]["actions"][t - pre:t + post])
    n_ev = len(windows["wind"])
    if n_ev == 0:
        print("[skip] fig6: no storm-onset events with full surviving windows")
        return

    hrs = (np.arange(-pre, post) * interval_min) / 60.0
    fig, axes = plt.subplots(3, 1, figsize=(9, 8), sharex=True, constrained_layout=True)

    W = np.vstack(windows["wind"])
    axes[0].plot(hrs, W.mean(axis=0), color="0.35", lw=1.5)
    axes[0].fill_between(hrs, np.percentile(W, 25, axis=0), np.percentile(W, 75, axis=0),
                         color="0.35", alpha=0.2)
    axes[0].axhline(TOP_BIN_EDGE, ls="--", color="0.6", lw=0.8)
    axes[0].axvline(0, color="0.6", lw=0.8)
    axes[0].set_ylabel("Wind [m/s]")
    axes[0].set_title(f"Composite of {n_ev} storm onsets "
                      f"(identical weather for both policies)", fontsize=10)

    for k, (key, ylabel) in enumerate((("soc", "State of charge [%]"),
                                       ("fly", "Fly fraction")), start=1):
        ax = axes[k]
        for arm in ("iid", "chain"):
            M = np.vstack(windows[key][arm])
            mean = M.mean(axis=0)
            sem = M.std(axis=0) / np.sqrt(M.shape[0])
            ax.plot(hrs, mean, color=COLOR[arm], ls=LS[arm], lw=1.8, label=ARM_LABEL[arm])
            ax.fill_between(hrs, mean - sem, mean + sem, color=COLOR[arm], alpha=0.2)
        ax.axvline(0, color="0.6", lw=0.8)
        ax.set_ylabel(ylabel)
    axes[1].legend(fontsize=9)
    axes[2].set_xlabel("Hours relative to storm onset (wind enters top bin)")
    fig.suptitle("Anticipatory behavior: the chain policy charges before the storm",
                 fontsize=11)
    _savefig(fig, out_dir, "fig6_event_aligned_composites.png")


# ----------------------------------------------------------------------------------
# Fig 7: value-table diagnostics
# ----------------------------------------------------------------------------------

def _window_bin_masses(data_path, start_time, horizon, bin_edges):
    """Per-stage wind-bin masses (horizon, n_bins) from the stage Weibull parameters."""
    df = pd.read_pickle(data_path)
    ts = pd.to_datetime(start_time)
    mask = ((df["month"] == ts.month) & (df["day"] == ts.day)
            & (df["hour"] == ts.hour) & (df["minute"] == ts.minute))
    start_idx = int(mask.idxmax())
    idxs = np.arange(start_idx, start_idx + horizon) % len(df)
    window = df.iloc[idxs]
    k = window["weibull_k"].to_numpy(float)[:, None]
    scale = window["weibull_scale"].to_numpy(float)[:, None]
    edges = np.concatenate(([0.0], np.asarray(bin_edges, float), [np.inf]))[None, :]
    with np.errstate(over="ignore"):
        cdf = 1.0 - np.exp(-np.power(np.clip(edges / scale, 0, None), k))
    cdf[:, -1] = 1.0
    masses = np.diff(cdf, axis=1)
    return masses / masses.sum(axis=1, keepdims=True)


def fig7_value_tables(summaries, manifest, out_dir, interval_min=15):
    b = manifest["baseline"]
    start_norm = pd.to_datetime(b["start"]).strftime("%Y-%m-%d %H:%M:%S")
    cap, horizon = float(b["capacity"]), int(b["horizon"])
    sub = summaries[(summaries["sweep"] == "capgrid") & (summaries["world"] == "hist")
                    & (summaries["location_id"] == b["location_id"])
                    & (summaries["battery_capacity"] == cap)
                    & (summaries["start_time"] == start_norm)]
    rows = {r["arm"]: r for _, r in sub.iterrows()}
    if "iid" not in rows or "chain" not in rows:
        print("[skip] fig7: baseline cell pair not found")
        return
    paths = {arm: _find_solver_table(rows[arm]["run_dir"], cap, horizon, start_norm)
             for arm in ("iid", "chain")}
    if any(p is None for p in paths.values()):
        print(f"[skip] fig7: solver table missing ({paths})")
        return
    V_iid = np.load(paths["iid"])       # (num_states, H)
    V_chain = np.load(paths["chain"])   # (n_bins, num_states, H)
    if V_iid.ndim != 2 or V_chain.ndim != 3:
        print(f"[skip] fig7: unexpected table shapes {V_iid.shape} / {V_chain.shape}")
        return
    n_soc = _n_soc_levels(cap)
    data_path = os.path.join(REPO_ROOT, "Data", "EXPECTED_DATA",
                             f"data_expected_{b['location_id']}_15min.pkl")
    edges = chain_edges_for(data_path, manifest)
    if edges is None or V_chain.shape[0] != len(edges) + 1:
        print(f"[skip] fig7: wind-bin edges unavailable or stale for {data_path}")
        return
    masses = _window_bin_masses(data_path, start_norm, V_iid.shape[1], edges)

    # Mode-0 (moored) block, SoC ascending.
    Vc = V_chain[:, :n_soc, :]                            # (n_bins, n_soc, H)
    Vavg = np.einsum("tb,bst->st", masses, Vc)            # bin-mass-averaged chain value
    delta_model = Vavg - V_iid[:n_soc, :]                 # (a) chain vs iid
    delta_bins = Vc[-1] - Vc[0]                           # (b) top bin vs bottom bin

    days = V_iid.shape[1] * interval_min / 1440.0
    extent = (0, days, 0, 100)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6), constrained_layout=True)
    for ax, M, title in (
            (axes[0], delta_model,
             r"(a) $\sum_b p_b V_{chain}(b) - V_{iid}$ (model value gap)"),
            (axes[1], delta_bins,
             r"(b) $V_{chain}(top\ bin) - V_{chain}(bottom\ bin)$ (price of high wind)")):
        vmax = max(abs(np.nanmin(M)), abs(np.nanmax(M)), 1e-9)
        im = ax.imshow(M, origin="lower", aspect="auto", cmap="RdBu_r",
                       vmin=-vmax, vmax=vmax, extent=extent)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("Mission time [days]")
        ax.set_ylabel("State of charge [%] (moored mode)")
        fig.colorbar(im, ax=ax, label="Value difference")
    fig.suptitle(f"Value-table diagnostics, {b['location_id']}, {cap:g} Wh: "
                 "where knowing the wind bin matters", fontsize=11)
    _savefig(fig, out_dir, "fig7_value_table_diagnostics.png")


# ----------------------------------------------------------------------------------
# Fig 8: three-way policy comparison (iid / chain optimal vs best threshold)
# ----------------------------------------------------------------------------------

def fig8_threeway(cells, manifest, out_dir):
    hist = cells[cells["world"] == "hist"]
    if "avg_reward_thresh" not in hist.columns or hist["avg_reward_thresh"].isna().all():
        print("[skip] fig8: no threshold benchmark cells "
              "(generate configs with --thresholds and re-run the sweep)")
        return
    cg = hist[hist["sweep"] == "capgrid"]
    locs = sorted(cg["location_id"].unique())
    panels = len(locs) + 2
    ncols = 3
    nrows = int(np.ceil(panels / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.2 * ncols, 4.0 * nrows),
                             squeeze=False, constrained_layout=True)
    flat = axes.ravel()

    def plot_series(ax, s, xcol, xlabel):
        s = s.sort_values(xcol)
        for arm in ("iid", "chain", "thresh"):
            col = f"avg_reward_{arm}"
            if col in s.columns and s[col].notna().any():
                ax.plot(s[xcol], s[col], color=COLOR[arm], ls=LS[arm],
                        marker=MARKER[arm], ms=5, label=ARM_LABEL[arm])
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Mean episode reward")

    for ax, loc in zip(flat, locs):
        plot_series(ax, cg[cg["location_id"] == loc],
                    "battery_capacity", "Battery capacity [Wh]")
        ax.set_title(f"{loc} (by capacity)", fontsize=10)

    base_loc = manifest["baseline"]["location_id"]
    sd = hist[hist["sweep"] == "startdate"].copy()
    ax = flat[len(locs)]
    if len(sd):
        sd["month"] = pd.to_datetime(sd["start_time"]).dt.month
        plot_series(ax, sd, "month", "Mission start month")
        ax.set_xticks(sorted(sd["month"].unique()))
        ax.set_title(f"{base_loc} (by season)", fontsize=10)
    else:
        ax.set_visible(False)

    du = hist[hist["sweep"] == "duration"].copy()
    ax = flat[len(locs) + 1]
    if len(du):
        du["days"] = du["horizon"].astype(int) * 15 / 1440
        plot_series(ax, du, "days", "Mission duration [days]")
        ax.set_title(f"{base_loc} (by duration)", fontsize=10)
    else:
        ax.set_visible(False)

    for ax in flat[panels:]:
        ax.set_visible(False)
    flat[0].legend(fontsize=9)
    fig.suptitle("Optimal policies vs the best threshold policy, historical weather\n"
                 "(threshold = per-cell reward-best combo of the "
                 "observation x wind threshold grid)", fontsize=11)
    _savefig(fig, out_dir, "fig8_threeway_policy_comparison.png")


# ----------------------------------------------------------------------------------

FIGS = {
    "1": ("frontier", lambda c, s, m, o: fig1_frontier(c, m, o)),
    "2": ("delta panels", lambda c, s, m, o: fig2_delta_panels(c, o)),
    "3": ("reward CDFs", lambda c, s, m, o: fig3_reward_cdf(c, s, m, o)),
    "4": ("benefit vs persistence", lambda c, s, m, o: fig4_persistence(c, m, o)),
    "5": ("calibration", lambda c, s, m, o: fig5_calibration(c, o)),
    "6": ("event-aligned", lambda c, s, m, o: fig6_event_aligned(s, m, o)),
    "7": ("value tables", lambda c, s, m, o: fig7_value_tables(s, m, o)),
    "8": ("three-way policy comparison", lambda c, s, m, o: fig8_threeway(c, m, o)),
}


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--results", default=None)
    ap.add_argument("--manifest", default=None)
    ap.add_argument("--analysis", default=None,
                    help="Dir holding comparison_cells.csv (default: <results>/_analysis).")
    ap.add_argument("--figs", default="1,2,3,4,5,6,7,8",
                    help="Comma-separated figure numbers to draw.")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    suffix = "_smoke" if args.smoke else ""
    results_base = args.results or os.path.join(REPO_ROOT, "results",
                                                f"chain_vs_iid_sweep{suffix}")
    manifest_path = args.manifest or os.path.join(
        REPO_ROOT, "configs", f"chain_vs_iid_sweep{suffix}", "chain_vs_iid_sweep_manifest.json")
    analysis_dir = args.analysis or os.path.join(results_base, "_analysis")
    cells_path = os.path.join(analysis_dir, "comparison_cells.csv")
    if not os.path.isfile(cells_path):
        sys.exit(f"[error] {cells_path} not found -- run Scripts/compare_chain_sweep.py first")

    out_dir = os.path.join(analysis_dir, "figures")
    os.makedirs(out_dir, exist_ok=True)

    manifest = load_manifest(manifest_path)
    cells = pd.read_csv(cells_path)
    summaries = load_summaries(manifest, results_base)

    for key in [k.strip() for k in args.figs.split(",") if k.strip()]:
        if key not in FIGS:
            print(f"[warn] unknown figure '{key}'")
            continue
        name, fn = FIGS[key]
        try:
            fn(cells, summaries, manifest, out_dir)
        except Exception as e:  # noqa: BLE001 -- best-effort per figure
            print(f"[warn] fig{key} ({name}) failed: {e}")


if __name__ == "__main__":
    main()

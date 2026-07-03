#!/usr/bin/env python3
"""Aggregate and compare the chain-vs-iid sweep results.

Joins the paired runs emitted by Scripts/generate_chain_sweep_configs.py +
Scripts/run_chain_sweep.py and produces per-cell comparisons:

  * paired per-episode deltas (chain - iid) on historical weather with bootstrap CIs --
    valid because paired hist-world runs draw identical bootstrap weather (the provider
    RNG is reset with a fixed seed per batch; verified with --verify),
  * tail metrics (CVaR at the worst decile) per arm and as a paired delta,
  * calibration gaps: each arm's native-world (self-predicted) performance vs its
    historical-world (realized) performance -- is the iid model overconfident?
  * solver-predicted initial value V0 extracted from the saved value tables
    (chain tables are averaged over the stage-0 wind-bin masses),
  * best-threshold-policy benchmark per cell (when threshold configs were run):
    reward-best and failure-best combos, split-half selection check, and
    episode-paired deltas of each optimal arm against the best threshold.

Outputs (default results/chain_vs_iid_sweep/_analysis/):
    comparison_cells.csv   one row per (sweep, location, capacity, start, horizon) cell
    comparison_report.md   human-readable summary tables

Usage (pvlib conda env, from SolarSimulator/):
    conda run -n pvlib python Scripts/compare_chain_sweep.py [--smoke] [--verify]
"""
import argparse
import glob
import json
import os
import re
import sys

import h5py
import numpy as np
import pandas as pd

PKG_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # .../SolarSimulator
if PKG_DIR not in sys.path:
    sys.path.insert(0, PKG_DIR)
REPO_ROOT = os.path.dirname(PKG_DIR)

CVAR_ALPHA = 0.10  # worst-decile tail

CELL_KEYS = ["sweep", "location_id", "battery_capacity", "start_time", "horizon"]


# ----------------------------------------------------------------------------------
# Discovery / loading
# ----------------------------------------------------------------------------------

def load_manifest(path):
    with open(path, "r") as f:
        return json.load(f)


def latest_run_dir(results_base, basename):
    """Newest timestamped run dir for a config that contains a summary.csv, or None."""
    candidates = sorted(glob.glob(os.path.join(results_base, basename, "*")))
    for run_dir in reversed(candidates):
        if os.path.isfile(os.path.join(run_dir, "summary.csv")):
            return run_dir
    return None


def load_summaries(manifest, results_base):
    """Concat all cells' summary.csv rows, tagged with manifest fields.

    Optimal-arm cells keep their optimal-sim rows; threshold-benchmark cells
    (arm == "thresh") keep every threshold-combo row instead.
    """
    frames = []
    missing = []
    for cell in manifest["cells"]:
        run_dir = latest_run_dir(results_base, cell["config_basename"])
        if run_dir is None:
            missing.append(cell["config_basename"])
            continue
        df = pd.read_csv(os.path.join(run_dir, "summary.csv"))
        if cell["arm"] == "thresh":
            df = df[df["simulation_type"].str.contains("Threshold", case=False, na=False)].copy()
        else:
            df = df[df["simulation_type"].str.contains("Optimal", na=False)].copy()
        df["sweep"] = cell["sweep"]
        df["arm"] = cell["arm"]
        df["world"] = cell["world"]
        df["run_dir"] = run_dir
        df["pair_key"] = cell["pair_key"]
        frames.append(df)
    if missing:
        print(f"[warn] no completed run for {len(missing)} config(s): {', '.join(missing)}")
    if not frames:
        sys.exit("[error] no completed runs found -- run Scripts/run_chain_sweep.py first")
    out = pd.concat(frames, ignore_index=True)
    # Normalize the join keys.
    out["start_time"] = pd.to_datetime(out["start_time"]).dt.strftime("%Y-%m-%d %H:%M:%S")
    out["battery_capacity"] = out["battery_capacity"].astype(float)
    out["horizon"] = out["horizon"].astype(int)
    return out


_SCALARS_CACHE = {}


def read_all_episode_scalars(run_dir):
    """All sim groups' per-episode scalars for one run dir, cached to CSV and in memory."""
    if run_dir in _SCALARS_CACHE:
        return _SCALARS_CACHE[run_dir]
    cache = os.path.join(run_dir, "_episode_scalars.csv")
    if os.path.isfile(cache):
        df = pd.read_csv(cache)
    else:
        h5s = glob.glob(os.path.join(run_dir, "*.h5"))
        if not h5s:
            raise FileNotFoundError(f"no HDF5 in {run_dir}")
        rows = []
        with h5py.File(h5s[0], "r") as f:
            for gname in f.keys():
                eps = f[gname].get("episodes")
                if eps is None:
                    continue
                for ename, ep in eps.items():
                    rows.append({
                        "group": gname,
                        "episode_index": int(ep.attrs.get("episode_index",
                                                          int(ename.split()[-1]) - 1)),
                        "total_reward": float(ep["total_reward"][()]),
                        "failure": int(np.asarray(ep["failure"][()]).item()),
                        "failure_step": float(ep["failure_step"][()]),
                        "flight_hrs": float(ep["flight_hrs"][()]),
                    })
        df = pd.DataFrame(rows)
        df.to_csv(cache, index=False)
    _SCALARS_CACHE[run_dir] = df
    return df


def read_episode_scalars(run_dir, group):
    """Per-episode scalars for one sim group, cached to CSV inside the run dir.

    Returns a DataFrame indexed by episode_index with columns
    total_reward, failure, failure_step, flight_hrs.
    """
    df = read_all_episode_scalars(run_dir)
    df = df[df["group"] == group].set_index("episode_index").sort_index()
    return df


# ----------------------------------------------------------------------------------
# Statistics
# ----------------------------------------------------------------------------------

def cvar(rewards, alpha=CVAR_ALPHA):
    """Mean of the worst alpha-fraction of episode rewards."""
    r = np.sort(np.asarray(rewards, dtype=float))
    k = max(1, int(np.ceil(alpha * len(r))))
    return float(r[:k].mean())


def _boot_ci(stat_fn, rng, n_boot, *arrays):
    """Percentile bootstrap CI resampling episode indices (shared across arrays)."""
    n = len(arrays[0])
    stats = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        stats[b] = stat_fn(*[a[idx] for a in arrays])
    return float(np.percentile(stats, 2.5)), float(np.percentile(stats, 97.5))


def paired_delta(iid_eps, chain_eps, n_boot, rng, paired=True):
    """Delta metrics (chain - iid) with bootstrap CIs.

    paired=True joins on episode_index (valid for hist-world CRN pairs);
    paired=False uses independent two-sample bootstrap (native world).
    """
    out = {}
    if paired:
        joined = iid_eps.join(chain_eps, lsuffix="_iid", rsuffix="_chain", how="inner")
        n = len(joined)
        if n == 0:
            return None
        arrays = {m: (joined[f"{m}_iid"].to_numpy(float),
                      joined[f"{m}_chain"].to_numpy(float))
                  for m in ("total_reward", "failure", "flight_hrs")}
        for metric, (a_iid, a_chain) in arrays.items():
            d = a_chain - a_iid
            lo, hi = _boot_ci(lambda x: x.mean(), rng, n_boot, d)
            out[f"d_{metric}"] = float(d.mean())
            out[f"d_{metric}_lo"], out[f"d_{metric}_hi"] = lo, hi
        r_iid, r_chain = arrays["total_reward"]
        lo, hi = _boot_ci(lambda a, b: cvar(b) - cvar(a), rng, n_boot, r_iid, r_chain)
        out["d_cvar10"] = cvar(r_chain) - cvar(r_iid)
        out["d_cvar10_lo"], out["d_cvar10_hi"] = lo, hi
        out["n_paired_episodes"] = n
    else:
        for metric in ("total_reward", "failure", "flight_hrs"):
            a_iid = iid_eps[metric].to_numpy(float)
            a_chain = chain_eps[metric].to_numpy(float)
            boots = np.empty(n_boot)
            for b in range(n_boot):
                boots[b] = (a_chain[rng.integers(0, len(a_chain), len(a_chain))].mean()
                            - a_iid[rng.integers(0, len(a_iid), len(a_iid))].mean())
            out[f"d_{metric}"] = float(a_chain.mean() - a_iid.mean())
            out[f"d_{metric}_lo"] = float(np.percentile(boots, 2.5))
            out[f"d_{metric}_hi"] = float(np.percentile(boots, 97.5))
    return out


def _suffix_delta_keys(deltas, suffix):
    """Insert a suffix into delta keys while keeping `_lo`/`_hi` terminal, so
    `_fmt_ci`'s `<base>_lo`/`<base>_hi` lookup keeps working."""
    out = {}
    for k, v in deltas.items():
        for tail in ("_lo", "_hi"):
            if k.endswith(tail):
                out[f"{k[:-len(tail)]}{suffix}{tail}"] = v
                break
        else:
            out[f"{k}{suffix}"] = v
    return out


def best_threshold_for_cell(thresh_rows):
    """Select the best threshold combo for one cell.

    Selection is by full-sample mean reward (what a practitioner tuning on these
    episodes would pick). A split-half check -- select on even episode indices,
    evaluate on odd -- quantifies the winner's-curse bias of picking the best of
    ~35 combos on the same episodes. Also records the minimum-failure combo for
    the reliability-frontier envelope.

    Returns (best_row, best_eps, extras).
    """
    run_dir = thresh_rows.iloc[0]["run_dir"]
    best_row = thresh_rows.loc[thresh_rows["average_reward"].idxmax()]
    best_eps = read_episode_scalars(run_dir, best_row["group"])

    even_means = {}
    for _, row in thresh_rows.iterrows():
        eps = read_episode_scalars(run_dir, row["group"])
        even_means[row["group"]] = float(
            eps.loc[eps.index % 2 == 0, "total_reward"].mean())
    sel_group = max(even_means, key=even_means.get)
    sel_eps = read_episode_scalars(run_dir, sel_group)
    holdout = float(sel_eps.loc[sel_eps.index % 2 == 1, "total_reward"].mean())

    minfail_row = thresh_rows.sort_values(
        ["failure_percentage", "average_reward"], ascending=[True, False]).iloc[0]

    extras = {
        "thresh_best_obs": float(best_row["observation_threshold"]),
        "thresh_best_wind": float(best_row["wind_threshold"]),
        "avg_reward_thresh_holdout": holdout,
        "thresh_selection_stable": bool(sel_group == best_row["group"]),
        "thresh_minfail_obs": float(minfail_row["observation_threshold"]),
        "thresh_minfail_wind": float(minfail_row["wind_threshold"]),
        "failure_pct_thresh_minfail": 100 * float(minfail_row["failure_percentage"]),
        "avg_reward_thresh_minfail": float(minfail_row["average_reward"]),
        "n_thresh_combos": int(len(thresh_rows)),
    }
    return best_row, best_eps, extras


# ----------------------------------------------------------------------------------
# Solver-table V0 extraction
# ----------------------------------------------------------------------------------

def _n_soc_levels(capacity, energy_increment_wh=5.0):
    inc = (energy_increment_wh / capacity) * 100.0
    return len(np.arange(0, 100 + inc, inc))


def _find_solver_table(run_dir, capacity, horizon, start_time):
    """Locate the value-table .npy: {prefix}_{cap}Wh_{horizon}h_{penalty}p_{start[:12]}.npy."""
    start12 = re.escape(str(start_time)[:12])
    pat = re.compile(rf"_{capacity}Wh_{horizon}h_.*p_{start12}\.npy$")
    for p in glob.glob(os.path.join(run_dir, "solver_tables", "*.npy")):
        if pat.search(os.path.basename(p)):
            return p
    return None


def chain_edges_for(data_path, manifest=None):
    """Interior wind-bin edges for a location, read from its wind-chain artifact.

    Edges are per-location when quantile-derived, so they must come from the artifact,
    not the manifest. Falls back to a legacy manifest's global ``bin_edges`` (pre-quantile
    sweeps) and returns None when neither exists.
    """
    base, ext = os.path.splitext(data_path)
    path = f"{base}_windchain{ext or '.pkl'}"
    if os.path.exists(path):
        art = pd.read_pickle(path)
        return np.asarray(art["bin_edges"], dtype=float)[1:-1]
    if manifest is not None and manifest.get("bin_edges"):
        return np.asarray(manifest["bin_edges"], dtype=float)
    return None


def _stage0_bin_masses(data_path, start_time, bin_edges):
    """Wind-bin probability masses from the stage-0 Weibull at the mission start."""
    df = pd.read_pickle(data_path)
    ts = pd.to_datetime(start_time)
    mask = ((df["month"] == ts.month) & (df["day"] == ts.day)
            & (df["hour"] == ts.hour) & (df["minute"] == ts.minute))
    if not mask.any():
        raise ValueError(f"start {start_time} not found in {data_path}")
    row = df[mask].iloc[0]
    k, scale = float(row["weibull_k"]), float(row["weibull_scale"])
    edges = np.concatenate(([0.0], np.asarray(bin_edges, dtype=float), [np.inf]))
    cdf = 1.0 - np.exp(-np.power(np.clip(edges / scale, 0, None), k))
    cdf[-1] = 1.0
    masses = np.diff(cdf)
    return masses / masses.sum()


def v0_from_solver_table(run_dir, capacity, horizon, start_time, data_path, bin_edges):
    """Solver-predicted value of the initial state [SoC=100, mode=0] at t=0.

    2D (iid) tables index directly; 3D (chain) tables are averaged over the
    stage-0 wind-bin masses (the rollout's initial-bin distribution).
    """
    path = _find_solver_table(run_dir, capacity, horizon, start_time)
    if path is None:
        return None
    table = np.load(path)
    row = _n_soc_levels(capacity) - 1  # mode-0 block is SoC-ascending; initial SoC=100
    if table.ndim == 2:
        return float(table[row, 0])
    if bin_edges is None:
        raise ValueError("3D (chain) table but no wind-bin edges available")
    if table.shape[0] != len(bin_edges) + 1:
        raise ValueError(f"table has {table.shape[0]} bins but edges imply "
                         f"{len(bin_edges) + 1} (stale artifact?)")
    masses = _stage0_bin_masses(data_path, start_time, bin_edges)
    return float(masses @ table[:, row, 0])


# ----------------------------------------------------------------------------------
# CRN verification
# ----------------------------------------------------------------------------------

def verify_crn(iid_run_dir, chain_run_dir, iid_group, chain_group, max_eps=8):
    """Check paired hist-world runs saw identical weather: compare wind_series of the
    full-history episodes up to the earlier truncation point."""
    def _wind(run_dir, group):
        h5s = glob.glob(os.path.join(run_dir, "*.h5"))
        series = {}
        if not h5s:  # compacted scalars-only run: no full histories
            return series
        with h5py.File(h5s[0], "r") as f:
            eps = f[group].get("episodes")
            if eps is None:
                return series
            for ename, ep in eps.items():
                if "wind_series" in ep:
                    idx = int(ep.attrs.get("episode_index", int(ename.split()[-1]) - 1))
                    series[idx] = np.asarray(ep["wind_series"][()])
        return series

    a, b = _wind(iid_run_dir, iid_group), _wind(chain_run_dir, chain_group)
    common = sorted(set(a) & set(b))[:max_eps]
    if not common:
        return None, "no full-history episodes in common (inconclusive)"
    for i in common:
        n = min(len(a[i]), len(b[i]))
        if n and not np.allclose(a[i][:n], b[i][:n]):
            return False, f"episode {i}: wind series diverge"
    return True, f"{len(common)} episodes checked, weather identical"


# ----------------------------------------------------------------------------------
# Cell assembly
# ----------------------------------------------------------------------------------

def build_cells(summaries, manifest, n_boot, rng, do_verify):
    crn_checked = False
    crn_ok = None
    cells = []

    grouped = summaries.groupby(["pair_key"] + [k for k in CELL_KEYS if k != "sweep"])
    # pair_key already encodes sweep+location+world; remaining keys pin the cell params.
    for (pair_key, location_id, capacity, start_time, horizon), grp in grouped:
        arms = {row["arm"]: row for _, row in grp[grp["arm"] != "thresh"].iterrows()}
        if "iid" not in arms or "chain" not in arms:
            continue
        iid, chain = arms["iid"], arms["chain"]
        world = iid["world"]
        cell = {
            "sweep": iid["sweep"], "world": world,
            "location_id": location_id, "battery_capacity": capacity,
            "start_time": start_time, "horizon": horizon,
            "failure_pct_iid": 100 * iid["failure_percentage"],
            "failure_pct_chain": 100 * chain["failure_percentage"],
            "avg_reward_iid": iid["average_reward"],
            "avg_reward_chain": chain["average_reward"],
            "avg_flight_hrs_iid": iid["average_flight_hrs"],
            "avg_flight_hrs_chain": chain["average_flight_hrs"],
        }

        iid_eps = read_episode_scalars(iid["run_dir"], iid["group"])
        chain_eps = read_episode_scalars(chain["run_dir"], chain["group"])
        cell["cvar10_iid"] = cvar(iid_eps["total_reward"])
        cell["cvar10_chain"] = cvar(chain_eps["total_reward"])

        paired = world == "hist"
        if paired and do_verify and not crn_checked:
            crn_ok, msg = verify_crn(iid["run_dir"], chain["run_dir"],
                                     iid["group"], chain["group"])
            if crn_ok is None:
                print(f"[verify] CRN check ({pair_key}): skipped -- {msg}")
            else:
                crn_checked = True
                print(f"[verify] CRN check ({pair_key}): "
                      f"{'PASS' if crn_ok else 'FAIL'} -- {msg}")
        if paired and crn_ok is False:
            paired = False  # fall back to two-sample bootstrap
            cell["crn_fallback"] = True

        deltas = paired_delta(iid_eps, chain_eps, n_boot, rng, paired=paired)
        if deltas:
            cell.update(deltas)
        cell["paired"] = paired

        # Threshold-policy benchmark (hist world only; arm-agnostic, CRN-paired with both).
        thresh_rows = grp[grp["arm"] == "thresh"]
        if len(thresh_rows):
            best_row, best_eps, extras = best_threshold_for_cell(thresh_rows)
            cell.update(extras)
            cell["failure_pct_thresh"] = 100 * best_row["failure_percentage"]
            cell["avg_reward_thresh"] = best_row["average_reward"]
            cell["avg_flight_hrs_thresh"] = best_row["average_flight_hrs"]
            cell["cvar10_thresh"] = cvar(best_eps["total_reward"])
            for arm_name, arm_eps in (("iid", iid_eps), ("chain", chain_eps)):
                d = paired_delta(best_eps, arm_eps, n_boot, rng, paired=paired)
                if d:
                    cell.update(_suffix_delta_keys(d, f"_{arm_name}_vs_thresh"))

        data_path = _data_path_for_location(manifest, location_id)
        bin_edges = chain_edges_for(data_path, manifest)
        for arm_name, row in (("iid", iid), ("chain", chain)):
            try:
                cell[f"v0_{arm_name}"] = v0_from_solver_table(
                    row["run_dir"], capacity, horizon, start_time, data_path, bin_edges)
            except Exception as e:
                print(f"[warn] V0 extraction failed for {pair_key}/{arm_name}: {e}")
                cell[f"v0_{arm_name}"] = None
        cells.append(cell)
    return pd.DataFrame(cells)


def _data_path_for_location(manifest, location_id):
    return os.path.join(REPO_ROOT, "Data", "EXPECTED_DATA",
                        f"data_expected_{location_id}_15min.pkl")


def add_calibration(cells):
    """Join hist and native rows of each cell; gap = predicted (native) - realized (hist)."""
    keys = CELL_KEYS
    hist = cells[cells["world"] == "hist"].set_index(keys)
    native = cells[cells["world"] == "native"].set_index(keys)
    common = hist.index.intersection(native.index)
    for arm in ("iid", "chain"):
        hist.loc[common, f"calib_gap_failure_pct_{arm}"] = (
            native.loc[common, f"failure_pct_{arm}"] - hist.loc[common, f"failure_pct_{arm}"])
        hist.loc[common, f"calib_gap_reward_{arm}"] = (
            native.loc[common, f"avg_reward_{arm}"] - hist.loc[common, f"avg_reward_{arm}"])
        hist.loc[common, f"native_failure_pct_{arm}"] = native.loc[common, f"failure_pct_{arm}"]
        hist.loc[common, f"native_avg_reward_{arm}"] = native.loc[common, f"avg_reward_{arm}"]
    return pd.concat([hist.reset_index(), native.reset_index()], ignore_index=True)


# ----------------------------------------------------------------------------------
# Report
# ----------------------------------------------------------------------------------

def _fmt_ci(row, base):
    if pd.isna(row.get(base)):
        return ""
    lo, hi = row.get(f"{base}_lo"), row.get(f"{base}_hi")
    ci = f" [{lo:+.3f}, {hi:+.3f}]" if pd.notna(lo) and pd.notna(hi) else ""
    return f"{row[base]:+.3f}{ci}"


def write_report(cells, out_path, manifest):
    hist = cells[cells["world"] == "hist"].copy()
    lines = ["# Chain vs IID comparison report", ""]
    lines.append(f"Generated from manifest of {manifest['generated']} "
                 f"({'smoke' if manifest.get('smoke') else 'full'} sweep). "
                 f"All deltas are chain - iid; hist-world deltas are episode-paired "
                 f"(identical bootstrap weather) unless flagged.")
    lines.append("")

    has_thresh = "failure_pct_thresh" in hist.columns and hist["failure_pct_thresh"].notna().any()
    for sweep in hist["sweep"].unique():
        sub = hist[hist["sweep"] == sweep].sort_values(
            ["location_id", "battery_capacity", "start_time", "horizon"])
        lines.append(f"## Sweep: {sweep} (historical weather)")
        lines.append("")
        lines.append("| location | cap (Wh) | start | horizon | fail% iid | fail% chain | "
                     "Δreward [95% CI] | ΔCVaR₁₀ [95% CI] | Δflight hrs |")
        lines.append("|---|---|---|---|---|---|---|---|---|")
        for _, r in sub.iterrows():
            lines.append(
                f"| {r['location_id']} | {r['battery_capacity']:g} | {str(r['start_time'])[:10]} "
                f"| {r['horizon']} | {r['failure_pct_iid']:.2f} | {r['failure_pct_chain']:.2f} "
                f"| {_fmt_ci(r, 'd_total_reward')} | {_fmt_ci(r, 'd_cvar10')} "
                f"| {_fmt_ci(r, 'd_flight_hrs')} |")
        lines.append("")

    if has_thresh:
        lines.append("## Optimal arms vs best threshold policy (historical weather)")
        lines.append("")
        lines.append("The best threshold combo is selected per cell by full-sample mean "
                     "reward over the swept grid; deltas are episode-paired (identical "
                     "bootstrap weather across all three policies). The split-half column "
                     "re-selects on even episodes and evaluates on odd episodes -- if it is "
                     "close to the full-sample value, selection bias is negligible.")
        lines.append("")
        lines.append("| sweep | location | cap (Wh) | start | horizon | best (obs, wind) | "
                     "fail% thresh | fail% iid | fail% chain | Δreward iid−thresh [CI] | "
                     "Δreward chain−thresh [CI] | thresh reward (full / split-half) |")
        lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
        tsub = hist[hist["failure_pct_thresh"].notna()].sort_values(
            ["sweep", "location_id", "battery_capacity", "start_time", "horizon"])
        for _, r in tsub.iterrows():
            lines.append(
                f"| {r['sweep']} | {r['location_id']} | {r['battery_capacity']:g} "
                f"| {str(r['start_time'])[:10]} | {r['horizon']} "
                f"| ({r['thresh_best_obs']:g}, {r['thresh_best_wind']:g}) "
                f"| {r['failure_pct_thresh']:.2f} | {r['failure_pct_iid']:.2f} "
                f"| {r['failure_pct_chain']:.2f} "
                f"| {_fmt_ci(r, 'd_total_reward_iid_vs_thresh')} "
                f"| {_fmt_ci(r, 'd_total_reward_chain_vs_thresh')} "
                f"| {r['avg_reward_thresh']:.2f} / {r['avg_reward_thresh_holdout']:.2f} |")
        lines.append("")

    calib = hist.dropna(subset=["calib_gap_failure_pct_iid"], how="all") \
        if "calib_gap_failure_pct_iid" in hist else pd.DataFrame()
    if len(calib):
        lines.append("## Calibration (native-world predicted minus historical realized)")
        lines.append("")
        lines.append("Negative failure-gap = the model *underestimates* its real failure "
                     "rate (overconfident).")
        lines.append("")
        lines.append("| location | cap (Wh) | fail-gap iid | fail-gap chain | "
                     "reward-gap iid | reward-gap chain | V0 iid | V0 chain |")
        lines.append("|---|---|---|---|---|---|---|---|")
        for _, r in calib.sort_values(["location_id", "battery_capacity"]).iterrows():
            v0i = f"{r['v0_iid']:.2f}" if pd.notna(r.get("v0_iid")) else "-"
            v0c = f"{r['v0_chain']:.2f}" if pd.notna(r.get("v0_chain")) else "-"
            lines.append(
                f"| {r['location_id']} | {r['battery_capacity']:g} "
                f"| {r['calib_gap_failure_pct_iid']:+.2f} | {r['calib_gap_failure_pct_chain']:+.2f} "
                f"| {r['calib_gap_reward_iid']:+.2f} | {r['calib_gap_reward_chain']:+.2f} "
                f"| {v0i} | {v0c} |")
        lines.append("")

    # Wh saved at equal reliability, from the capgrid frontier.
    capgrid = hist[hist["sweep"] == "capgrid"]
    if len(capgrid):
        lines.append("## Battery capacity saved at equal reliability (5% failure)")
        lines.append("")
        for loc, sub in capgrid.groupby("location_id"):
            caps = {arm: capacity_at_reliability(sub, arm, target_failure_pct=5.0)
                    for arm in (("iid", "chain", "thresh") if has_thresh else ("iid", "chain"))}
            parts = []
            if caps["iid"] is not None and caps["chain"] is not None:
                parts.append(f"chain saves **{caps['iid'] - caps['chain']:+.0f} Wh** vs iid")
            if has_thresh and caps.get("thresh") is not None and caps["chain"] is not None:
                parts.append(f"**{caps['thresh'] - caps['chain']:+.0f} Wh** vs best threshold")
            if parts:
                lines.append(f"- {loc}: " + "; ".join(parts))
            else:
                lines.append(f"- {loc}: frontier does not cross 5% failure in the swept range")
        lines.append("")

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def capacity_at_reliability(capgrid_cells, arm, target_failure_pct=5.0):
    """Interpolate one arm's capacity at the target failure rate (None if out of range).

    For the threshold arm the reliability envelope (per-cell minimum-failure combo)
    is used rather than the reward-best combo.
    """
    col = "failure_pct_thresh_minfail" if arm == "thresh" else f"failure_pct_{arm}"
    if col not in capgrid_cells.columns:
        return None
    sub = capgrid_cells.dropna(subset=[col]).sort_values("battery_capacity")
    caps = sub["battery_capacity"].to_numpy(float)
    if len(caps) < 2:
        return None
    fail = sub[col].to_numpy(float)
    # failure decreases with capacity; interpolate capacity at the target
    order = np.argsort(fail)
    f_sorted, c_sorted = fail[order], caps[order]
    if target_failure_pct < f_sorted.min() or target_failure_pct > f_sorted.max():
        return None
    return float(np.interp(target_failure_pct, f_sorted, c_sorted))


def wh_saved_at_reliability(capgrid_cells, target_failure_pct=5.0):
    """Capacity iid needs minus capacity chain needs at the target failure rate."""
    c_iid = capacity_at_reliability(capgrid_cells, "iid", target_failure_pct)
    c_chain = capacity_at_reliability(capgrid_cells, "chain", target_failure_pct)
    if c_iid is None or c_chain is None:
        return None
    return c_iid - c_chain


# ----------------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--results", default=None,
                    help="Results base dir (default: results/chain_vs_iid_sweep[_smoke]).")
    ap.add_argument("--manifest", default=None,
                    help="Manifest JSON (default: configs/chain_vs_iid_sweep[_smoke]/...json).")
    ap.add_argument("--out", default=None, help="Analysis output dir (default: <results>/_analysis).")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--verify", action="store_true",
                    help="Check the common-random-numbers pairing on one hist pair.")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    suffix = "_smoke" if args.smoke else ""
    results_base = args.results or os.path.join(REPO_ROOT, "results", f"chain_vs_iid_sweep{suffix}")
    manifest_path = args.manifest or os.path.join(
        REPO_ROOT, "configs", f"chain_vs_iid_sweep{suffix}", "chain_vs_iid_sweep_manifest.json")
    out_dir = args.out or os.path.join(results_base, "_analysis")
    os.makedirs(out_dir, exist_ok=True)

    manifest = load_manifest(manifest_path)
    summaries = load_summaries(manifest, results_base)
    print(f"[load] {len(summaries)} optimal-sim rows across "
          f"{summaries['pair_key'].nunique()} pair groups")

    rng = np.random.default_rng(12345)
    cells = build_cells(summaries, manifest, args.n_boot, rng, args.verify)
    cells = add_calibration(cells)

    csv_path = os.path.join(out_dir, "comparison_cells.csv")
    cells.to_csv(csv_path, index=False)
    print(f"[out] {csv_path} ({len(cells)} cells)")

    report_path = os.path.join(out_dir, "comparison_report.md")
    write_report(cells, report_path, manifest)
    print(f"[out] {report_path}")


if __name__ == "__main__":
    main()

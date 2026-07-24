#!/usr/bin/env python3
"""Analyze the 5-arm Markov-ablation sweep (solar / wind / joint / iid / thresh).

Consumes runs produced by Scripts/generate_markov_ablation_configs.py +
Scripts/run_chain_sweep.py and produces, per cell (location x capacity x penalty x
season):

  * episode-paired deltas of each chain arm vs the iid arm (reward, failure,
    flight hrs, CVaR10) with percentile-bootstrap CIs, plus joint-vs-wind to isolate
    the solar increment on top of wind,
  * best-threshold benchmark with the threshold arm's rewards RE-WEIGHTED to the
    cell's failure penalty (threshold behavior is penalty-invariant; the arm is run
    at fp=5 only): reward(fp) = total_reward + (5 - fp) * failure,
  * fraction-of-gap-closed between the best threshold and the best optimal arm,
  * failure-rate / cost decomposition: penalty cost fp*failure_rate vs non-penalty
    reward mean(total_reward + fp*failure),
  * site persistence scores (mean diagonal mass of the wind/solar transition
    tensors) for the climatology-interaction readout.

Modes:
  default            full-sweep analysis -> <results>/_analysis/
  --solar-bins-study Phase 2 checkpoint: paired dReward vs IID per solar n_bins and
                     a knee-rule recommendation (smallest n_bins statistically
                     indistinguishable from the best).

Usage (pvlib env, from SolarSimulator/):
    python Scripts/compare_markov_ablation.py --verify
    python Scripts/compare_markov_ablation.py --solar-bins-study ../results/markov_ablation_smoke_solar
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

PKG_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # .../SolarSimulator
if PKG_DIR not in sys.path:
    sys.path.insert(0, PKG_DIR)
REPO_ROOT = os.path.dirname(PKG_DIR)

from Scripts.compare_chain_sweep import (  # noqa: E402
    _boot_ci, _fmt_ci, _suffix_delta_keys, cvar, latest_run_dir, load_manifest,
    paired_delta, read_episode_scalars, verify_crn,
)

OPTIMAL_ARMS = ("iid", "wind", "solar", "joint")
CHAIN_ARMS = ("wind", "solar", "joint")
CELL_KEYS = ["location_id", "battery_capacity", "failure_penalty", "start_time", "horizon"]


# ----------------------------------------------------------------------------------
# Loading
# ----------------------------------------------------------------------------------

def load_summaries(manifest, results_base):
    """Concat all configs' summary rows tagged with (arm, location_id) from the manifest."""
    frames, missing = [], []
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
        df["arm"] = cell["arm"]
        df["location_id"] = cell["location_id"]
        df["run_dir"] = run_dir
        df["config_basename"] = cell["config_basename"]
        if cell["arm"] == "solar" and cell.get("solar_bins") is not None:
            df["solar_bins"] = cell["solar_bins"]
        frames.append(df)
    if missing:
        print(f"[warn] no completed run for {len(missing)} config(s): {', '.join(missing)}")
    if not frames:
        sys.exit("[error] no completed runs found -- run Scripts/run_chain_sweep.py first")
    out = pd.concat(frames, ignore_index=True)
    out["start_time"] = pd.to_datetime(out["start_time"]).dt.strftime("%Y-%m-%d %H:%M:%S")
    out["battery_capacity"] = out["battery_capacity"].astype(float)
    out["failure_penalty"] = out["failure_penalty"].astype(float)
    out["horizon"] = out["horizon"].astype(int)
    return out


def reweighted_eps(eps, run_penalty, target_penalty):
    """Episode scalars with total_reward re-expressed at a different failure penalty.

    Valid only for penalty-invariant policies (threshold arms): the trajectory is
    unchanged, the penalty enters the reward once at failure.
    """
    if run_penalty == target_penalty:
        return eps
    out = eps.copy()
    out["total_reward"] = (out["total_reward"]
                           + (run_penalty - target_penalty) * out["failure"])
    return out


def best_threshold_at_penalty(thresh_rows, target_penalty, run_penalty):
    """Best threshold combo for one cell, selected on penalty-reweighted rewards.

    Mirrors compare_chain_sweep.best_threshold_for_cell (full-sample selection +
    split-half winner's-curse check) but selects on rewards recomputed at the cell's
    penalty instead of trusting summary.csv (only valid at the run penalty).
    """
    run_dir = thresh_rows.iloc[0]["run_dir"]
    all_eps, means, even_means = {}, {}, {}
    for _, row in thresh_rows.iterrows():
        eps = reweighted_eps(read_episode_scalars(run_dir, row["group"]),
                             run_penalty, target_penalty)
        all_eps[row["group"]] = (row, eps)
        means[row["group"]] = float(eps["total_reward"].mean())
        even_means[row["group"]] = float(
            eps.loc[eps.index % 2 == 0, "total_reward"].mean())
    best_group = max(means, key=means.get)
    best_row, best_eps = all_eps[best_group]
    sel_group = max(even_means, key=even_means.get)
    _, sel_eps = all_eps[sel_group]
    holdout = float(sel_eps.loc[sel_eps.index % 2 == 1, "total_reward"].mean())
    extras = {
        "thresh_best_obs": float(best_row["observation_threshold"]),
        "thresh_best_wind": float(best_row["wind_threshold"]),
        "avg_reward_thresh": means[best_group],
        "avg_reward_thresh_holdout": holdout,
        "thresh_selection_stable": bool(sel_group == best_group),
        "failure_pct_thresh": 100 * float(best_eps["failure"].mean()),
        "n_thresh_combos": int(len(thresh_rows)),
    }
    return best_eps, extras


def persistence_scores(manifest):
    """Mean diagonal mass of each site's wind/solar transition tensors (months 1-12)."""
    scores = {}
    for cell in manifest["cells"]:
        loc = cell["location_id"]
        scores.setdefault(loc, {})
        for kind in ("wind", "solar"):
            path = cell.get(f"{kind}_chain_path")
            if not path or f"{kind}_persistence" in scores[loc]:
                continue
            full = os.path.join(REPO_ROOT, path)
            try:
                art = pd.read_pickle(full)
                T = np.asarray(art["transition_by_month_hour"], dtype=float)
                diag = np.einsum("mhii->mhi", T[1:13])
                scores[loc][f"{kind}_persistence"] = float(np.nanmean(diag))
            except Exception as e:
                print(f"[warn] persistence score failed for {loc}/{kind}: {e}")
    return scores


# ----------------------------------------------------------------------------------
# Cell assembly
# ----------------------------------------------------------------------------------

def build_cells(summaries, manifest, n_boot, rng, do_verify):
    thresh_penalty = float(manifest.get("threshold_penalty", 5.0))
    opt = summaries[summaries["arm"] != "thresh"]
    thr = summaries[summaries["arm"] == "thresh"]
    scores = persistence_scores(manifest)
    crn_done = set()
    cells = []

    for keys, grp in opt.groupby(CELL_KEYS):
        location_id, capacity, penalty, start_time, horizon = keys
        arms = {row["arm"]: row for _, row in grp.iterrows()}
        if "iid" not in arms:
            print(f"[warn] cell {keys}: no iid arm -- skipped")
            continue
        cell = dict(zip(CELL_KEYS, keys))
        cell["season"] = "summer" if pd.to_datetime(start_time).month == 6 else "winter"
        cell.update(scores.get(location_id, {}))

        eps = {}
        for arm in OPTIMAL_ARMS:
            if arm not in arms:
                continue
            row = arms[arm]
            eps[arm] = read_episode_scalars(row["run_dir"], row["group"])
            cell[f"avg_reward_{arm}"] = float(eps[arm]["total_reward"].mean())
            cell[f"failure_pct_{arm}"] = 100 * float(eps[arm]["failure"].mean())
            cell[f"avg_flight_hrs_{arm}"] = float(eps[arm]["flight_hrs"].mean())
            cell[f"cvar10_{arm}"] = cvar(eps[arm]["total_reward"])
            # Decomposition: penalty cost vs everything else.
            cell[f"penalty_cost_{arm}"] = penalty * float(eps[arm]["failure"].mean())
            cell[f"nonpenalty_reward_{arm}"] = float(
                (eps[arm]["total_reward"] + penalty * eps[arm]["failure"]).mean())

        # CRN verification: once per (location, arm) pair on full-history episodes.
        if do_verify:
            t_first = thr[(thr["location_id"] == location_id)
                          & (thr["battery_capacity"] == capacity)
                          & (thr["start_time"] == start_time)
                          & (thr["horizon"] == horizon)]
            crn_targets = {a: (arms[a]["run_dir"], arms[a]["group"])
                           for a in CHAIN_ARMS if a in arms}
            if len(t_first):
                crn_targets["thresh"] = (t_first.iloc[0]["run_dir"], t_first.iloc[0]["group"])
            for arm, (a_dir, a_group) in crn_targets.items():
                pair = (location_id, arm)
                if pair in crn_done:
                    continue
                ok, msg = verify_crn(arms["iid"]["run_dir"], a_dir,
                                     arms["iid"]["group"], a_group)
                if ok is not None:
                    crn_done.add(pair)
                    print(f"[verify] CRN iid<->{arm} @ {location_id}: "
                          f"{'PASS' if ok else 'FAIL'} -- {msg}")
                    if not ok:
                        print("[verify] *** CRN FAIL: paired stats are invalid for "
                              "this pair -- investigate before trusting deltas ***")

        # Paired deltas vs iid, and joint vs wind (solar increment on top of wind).
        for arm in CHAIN_ARMS:
            if arm in eps:
                d = paired_delta(eps["iid"], eps[arm], n_boot, rng, paired=True)
                if d:
                    cell.update(_suffix_delta_keys(d, f"_{arm}"))
        if "wind" in eps and "joint" in eps:
            d = paired_delta(eps["wind"], eps["joint"], n_boot, rng, paired=True)
            if d:
                cell.update(_suffix_delta_keys(d, "_joint_vs_wind"))

        # Threshold benchmark at this cell's penalty (thresh arm ran at fp=5 only).
        t_rows = thr[(thr["location_id"] == location_id)
                     & (thr["battery_capacity"] == capacity)
                     & (thr["start_time"] == start_time)
                     & (thr["horizon"] == horizon)]
        if len(t_rows):
            t_eps, extras = best_threshold_at_penalty(t_rows, penalty, thresh_penalty)
            cell.update(extras)
            cell["cvar10_thresh"] = cvar(t_eps["total_reward"])
            for arm in OPTIMAL_ARMS:
                if arm not in eps:
                    continue
                d = paired_delta(t_eps, eps[arm], n_boot, rng, paired=True)
                if d:
                    cell.update(_suffix_delta_keys(d, f"_{arm}_vs_thresh"))

            # Fraction of gap closed: (arm - thresh*) / (best - thresh*), best fixed
            # on the full sample; CI by joint episode resampling. Degenerate gaps
            # (CI spanning 0) are flagged and left NaN.
            joined = t_eps[["total_reward"]].rename(columns={"total_reward": "r_thresh"})
            for arm in OPTIMAL_ARMS:
                if arm in eps:
                    joined = joined.join(
                        eps[arm][["total_reward"]].rename(columns={"total_reward": f"r_{arm}"}),
                        how="inner")
            present = [a for a in OPTIMAL_ARMS if f"r_{a}" in joined.columns]
            best_arm = max(present, key=lambda a: joined[f"r_{a}"].mean())
            cell["best_arm"] = best_arm
            r_t = joined["r_thresh"].to_numpy(float)
            r_b = joined[f"r_{best_arm}"].to_numpy(float)
            gap = float((r_b - r_t).mean())
            gap_lo, gap_hi = _boot_ci(lambda a, b: (b - a).mean(), rng, n_boot, r_t, r_b)
            cell["gap"], cell["gap_lo"], cell["gap_hi"] = gap, gap_lo, gap_hi
            cell["gap_degenerate"] = bool(gap_lo <= 0 <= gap_hi)
            for arm in present:
                r_a = joined[f"r_{arm}"].to_numpy(float)
                if cell["gap_degenerate"]:
                    cell[f"gap_closed_{arm}"] = np.nan
                    continue
                frac = float((r_a - r_t).mean() / gap)
                lo, hi = _boot_ci(
                    lambda t, a, b: (a - t).mean() / max((b - t).mean(), 1e-12),
                    rng, n_boot, r_t, r_a, r_b)
                cell[f"gap_closed_{arm}"] = frac
                cell[f"gap_closed_{arm}_lo"], cell[f"gap_closed_{arm}_hi"] = lo, hi
        cells.append(cell)
    return pd.DataFrame(cells)


# ----------------------------------------------------------------------------------
# Phase 2: solar bin-resolution study
# ----------------------------------------------------------------------------------

def solar_bins_study(manifest, results_base, n_boot, rng, out_dir):
    summaries = load_summaries(manifest, results_base)
    iid_rows = summaries[summaries["arm"] == "iid"]
    if iid_rows.empty:
        sys.exit("[error] no iid reference run found")
    iid = iid_rows.iloc[0]
    iid_eps = read_episode_scalars(iid["run_dir"], iid["group"])

    rows = []
    eps_by_bins = {}
    for _, r in summaries[summaries["arm"] == "solar"].sort_values("solar_bins").iterrows():
        eps = read_episode_scalars(r["run_dir"], r["group"])
        eps_by_bins[int(r["solar_bins"])] = eps
        d = paired_delta(iid_eps, eps, n_boot, rng, paired=True)
        rows.append({
            "solar_bins": int(r["solar_bins"]),
            "avg_reward": float(eps["total_reward"].mean()),
            "failure_pct": 100 * float(eps["failure"].mean()),
            "d_total_reward": d["d_total_reward"],
            "d_total_reward_lo": d["d_total_reward_lo"],
            "d_total_reward_hi": d["d_total_reward_hi"],
            "d_failure": d["d_failure"],
            "d_cvar10": d["d_cvar10"],
        })
    df = pd.DataFrame(rows).sort_values("solar_bins").reset_index(drop=True)
    if df.empty:
        sys.exit("[error] no solar-arm runs found")

    # Knee rule: smallest n_bins whose PAIRED delta against the best n_bins has a CI
    # containing 0 (statistically indistinguishable from the best). Tie -> 3.
    best_bins = int(df.loc[df["d_total_reward"].idxmax(), "solar_bins"])
    chosen = best_bins
    for g in sorted(eps_by_bins):
        if g == best_bins:
            chosen = min(chosen, g)
            break
        d = paired_delta(eps_by_bins[g], eps_by_bins[best_bins], n_boot, rng, paired=True)
        if d and d["d_total_reward_lo"] <= 0 <= d["d_total_reward_hi"]:
            chosen = g
            break
    df["is_best"] = df["solar_bins"] == best_bins
    df["is_chosen"] = df["solar_bins"] == chosen

    os.makedirs(out_dir, exist_ok=True)
    out_csv = os.path.join(out_dir, "solar_bins_study.csv")
    df.to_csv(out_csv, index=False)
    print(f"\n=== Solar bin-resolution study (paired vs IID, n={len(iid_eps)}) ===")
    print(df.to_string(index=False))
    print(f"\nBest n_bins by paired dReward: {best_bins}")
    print(f"CHOSEN (knee rule -- smallest within CI of best): {chosen}")
    print(f"-> regenerate later phases with: --solar-bins {chosen}")
    print(f"Wrote {out_csv}")
    return chosen


# ----------------------------------------------------------------------------------
# Phase 6: solar bin-resolution study across ALL locations
# ----------------------------------------------------------------------------------

MATCH_KEYS = ["battery_capacity", "failure_penalty", "start_time", "horizon"]


def _boot_mean_ci(values, rng, n_boot):
    """Percentile CI for the mean of a set of per-cell deltas (resample cells)."""
    v = np.asarray(values, dtype=float)
    if len(v) == 0:
        return float("nan"), float("nan"), float("nan")
    stats = np.array([v[rng.integers(0, len(v), len(v))].mean() for _ in range(n_boot)])
    return float(v.mean()), float(np.percentile(stats, 2.5)), float(np.percentile(stats, 97.5))


def solar_res_study(manifest, results_base, n_boot, rng, out_dir):
    """Per-cell paired Δreward vs IID for each solar n_bins, per location and pooled.

    Each solar-g sim is matched to the same-condition iid sim (cap, penalty, season,
    horizon) and paired on episode_index (identical bootstrap weather). Reports the
    mean over cells per (location, bins) and pooled over all 4 locations, with CIs
    bootstrapped over cells, then ranks bin counts and recommends one per the rule in
    docs/markov_ablation_experiment.md Phase 6.
    """
    summaries = load_summaries(manifest, results_base)
    bins_list = sorted({int(c["solar_bins"]) for c in manifest["cells"]
                        if c.get("solar_bins") is not None})
    locations = sorted(summaries["location_id"].unique())

    # Per-cell paired deltas: cell_deltas[(loc, g)] -> list of per-cell mean Δreward / Δfailure.
    cell_rows = []  # long form: one row per (loc, g, cell)
    for loc in locations:
        iid_rows = summaries[(summaries["location_id"] == loc) & (summaries["arm"] == "iid")]
        iid_by_key = {tuple(r[k] for k in MATCH_KEYS): r for _, r in iid_rows.iterrows()}
        for g in bins_list:
            s_rows = summaries[(summaries["location_id"] == loc)
                               & (summaries["arm"] == f"solar_g{g}")]
            for _, sr in s_rows.iterrows():
                key = tuple(sr[k] for k in MATCH_KEYS)
                ir = iid_by_key.get(key)
                if ir is None:
                    print(f"[warn] {loc} g{g}: no iid match for {key}")
                    continue
                iid_eps = read_episode_scalars(ir["run_dir"], ir["group"])
                s_eps = read_episode_scalars(sr["run_dir"], sr["group"])
                joined = iid_eps.join(s_eps, lsuffix="_iid", rsuffix="_s", how="inner")
                dR = float((joined["total_reward_s"] - joined["total_reward_iid"]).mean())
                dF = float((joined["failure_s"] - joined["failure_iid"]).mean())
                cell_rows.append({
                    "location_id": loc, "solar_bins": g,
                    "battery_capacity": sr["battery_capacity"],
                    "failure_penalty": sr["failure_penalty"],
                    "season": "summer" if pd.to_datetime(sr["start_time"]).month == 6 else "winter",
                    "d_reward": dR, "d_failure_pct": 100 * dF,
                    "failure_pct_solar": 100 * float(s_eps["failure"].mean()),
                })
    cell_df = pd.DataFrame(cell_rows)
    if cell_df.empty:
        sys.exit("[error] no matched solar/iid cells found")

    os.makedirs(out_dir, exist_ok=True)
    cell_df.to_csv(os.path.join(out_dir, "solar_res_cells.csv"), index=False)

    # Aggregate: mean Δreward per (location, bins) and pooled, CI bootstrapped over cells.
    agg_rows = []
    for g in bins_list:
        for loc in locations:
            sub = cell_df[(cell_df["solar_bins"] == g) & (cell_df["location_id"] == loc)]
            m, lo, hi = _boot_mean_ci(sub["d_reward"], rng, n_boot)
            agg_rows.append({"solar_bins": g, "scope": loc, "d_reward": m,
                             "lo": lo, "hi": hi, "n_cells": len(sub),
                             "d_failure_pct": float(sub["d_failure_pct"].mean())})
        sub = cell_df[cell_df["solar_bins"] == g]
        m, lo, hi = _boot_mean_ci(sub["d_reward"], rng, n_boot)
        agg_rows.append({"solar_bins": g, "scope": "POOLED", "d_reward": m,
                         "lo": lo, "hi": hi, "n_cells": len(sub),
                         "d_failure_pct": float(sub["d_failure_pct"].mean())})
    agg = pd.DataFrame(agg_rows)
    agg.to_csv(os.path.join(out_dir, "solar_res_summary.csv"), index=False)

    # Pivot for display: rows = bins, cols = location + POOLED.
    pooled = agg[agg["scope"] == "POOLED"].set_index("solar_bins")
    piv = agg.pivot(index="solar_bins", columns="scope", values="d_reward").round(3)

    print("\n=== Solar bin-resolution study: mean paired dReward vs IID ===")
    print("(per location and POOLED across all 4 locations; positive = solar beats IID)\n")
    print(piv.to_string())
    print("\nPOOLED with 95% CI (bootstrapped over cells):")
    for g in bins_list:
        r = pooled.loc[g]
        sig = "SIG+" if r["lo"] > 0 else ("SIG-" if r["hi"] < 0 else "ns")
        print(f"  g={g}: {r['d_reward']:+.3f} [{r['lo']:+.3f}, {r['hi']:+.3f}]  {sig}"
              f"   dfailure {r['d_failure_pct']:+.2f}pp")

    # --- Decision rule (see docs Phase 6) ---
    best_g = int(pooled["d_reward"].idxmax())
    best = pooled.loc[best_g]
    # Meaningful-benefit gate: pooled CI at best_g strictly above 0.
    significant = bool(best["lo"] > 0)
    # Clear winner: best is the unique bin whose CI does not overlap the runner-up's
    # mean, i.e., no OTHER bin has a mean within best's CI.
    tie_set = [g for g in bins_list
               if g != best_g and pooled.loc[g, "d_reward"] >= best["lo"]]
    clear_winner = significant and len(tie_set) == 0

    verdict = {"best_g": best_g, "pooled_d_reward": float(best["d_reward"]),
               "pooled_lo": float(best["lo"]), "pooled_hi": float(best["hi"]),
               "significant_benefit": significant,
               "clear_winner": clear_winner,
               "tie_set": sorted([best_g] + tie_set) if significant else []}
    with open(os.path.join(out_dir, "solar_res_verdict.json"), "w", encoding="utf-8") as f:
        json.dump(verdict, f, indent=2)

    print(f"\nBest pooled n_bins: {best_g} ({best['d_reward']:+.3f})")
    print(f"Significant pooled benefit (CI>0): {significant}")
    if significant:
        print(f"Clear winner: {clear_winner}"
              + ("" if clear_winner else f"  (statistically tied: {sorted([best_g]+tie_set)})"))
    else:
        print("No significant solar benefit at any resolution -> do NOT run joint rerun.")
    print(f"Wrote {out_dir}\\solar_res_summary.csv, solar_res_cells.csv, solar_res_verdict.json")

    _solar_res_figure(agg, bins_list, locations, out_dir)
    return verdict


def _solar_res_figure(agg, bins_list, locations, out_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 5))
    for loc in locations:
        s = agg[agg["scope"] == loc].sort_values("solar_bins")
        ax.plot(s["solar_bins"], s["d_reward"], marker="o", ms=4, lw=1, alpha=0.6, label=loc)
    p = agg[agg["scope"] == "POOLED"].sort_values("solar_bins")
    ax.errorbar(p["solar_bins"], p["d_reward"],
                yerr=[p["d_reward"] - p["lo"], p["hi"] - p["d_reward"]],
                fmt="o-", color="black", lw=2, ms=6, capsize=4, label="POOLED (95% CI)")
    ax.axhline(0, color="0.6", lw=0.8)
    ax.set_xlabel("solar chain n_bins")
    ax.set_ylabel("mean paired Δreward vs IID")
    ax.set_title("Solar bin-resolution study")
    ax.set_xticks(bins_list)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "solar_res_curve.png"), dpi=150)
    plt.close(fig)
    print(f"Wrote {out_dir}\\solar_res_curve.png")


# ----------------------------------------------------------------------------------
# Report + figures
# ----------------------------------------------------------------------------------

def write_report(cells, out_path, manifest):
    lines = ["# Markov ablation report (5 arms)", ""]
    lines.append(f"Manifest generated {manifest['generated']}; mode={manifest['mode']}; "
                 f"wind bins={manifest['wind_bins']['n_bins']}, "
                 f"solar bins={manifest['solar_bins']}. All deltas are episode-paired "
                 f"on identical bootstrap weather (arm minus iid unless noted); "
                 f"threshold rewards re-weighted per cell penalty from fp="
                 f"{manifest.get('threshold_penalty', 5.0)} runs.")
    lines.append("")
    for loc in cells["location_id"].unique():
        sub = cells[cells["location_id"] == loc].sort_values(
            ["season", "battery_capacity", "failure_penalty"])
        lines.append(f"## {loc}")
        lines.append("")
        lines.append("| season | cap | pen | fail% iid/wind/solar/joint | "
                     "Δwind [CI] | Δsolar [CI] | Δjoint [CI] | Δjoint−wind [CI] | "
                     "best arm | gap closed w/s/j |")
        lines.append("|---|---|---|---|---|---|---|---|---|---|")
        for _, r in sub.iterrows():
            fails = "/".join(
                f"{r.get(f'failure_pct_{a}', float('nan')):.1f}" for a in OPTIMAL_ARMS)
            gaps = "/".join(
                ("--" if pd.isna(r.get(f"gap_closed_{a}")) else f"{r[f'gap_closed_{a}']:.2f}")
                for a in CHAIN_ARMS)
            lines.append(
                f"| {r['season']} | {r['battery_capacity']:g} | {r['failure_penalty']:g} "
                f"| {fails} | {_fmt_ci(r, 'd_total_reward_wind')} "
                f"| {_fmt_ci(r, 'd_total_reward_solar')} | {_fmt_ci(r, 'd_total_reward_joint')} "
                f"| {_fmt_ci(r, 'd_total_reward_joint_vs_wind')} "
                f"| {r.get('best_arm', '--')} | {gaps} |")
        lines.append("")

    if "gap_degenerate" in cells.columns and cells["gap_degenerate"].any():
        n = int(cells["gap_degenerate"].sum())
        lines.append(f"**Note:** {n} cell(s) have a degenerate threshold gap "
                     f"(CI spans 0); gap-closed omitted there -- use absolute deltas.")
        lines.append("")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"Wrote {out_path}")


def write_figures(cells, fig_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(fig_dir, exist_ok=True)
    colors = {"wind": "#1f77b4", "solar": "#ff7f0e", "joint": "#2ca02c"}

    # Fig 1: forest plot of paired dReward vs iid, faceted by location.
    locs = list(cells["location_id"].unique())
    fig, axes = plt.subplots(1, len(locs), figsize=(4.2 * len(locs), 6), sharex=False)
    axes = np.atleast_1d(axes)
    for ax, loc in zip(axes, locs):
        sub = cells[cells["location_id"] == loc].sort_values(
            ["season", "battery_capacity", "failure_penalty"]).reset_index(drop=True)
        ylabels = [f"{r['season'][:3]} {r['battery_capacity']:g}Wh p{r['failure_penalty']:g}"
                   for _, r in sub.iterrows()]
        for j, arm in enumerate(CHAIN_ARMS):
            col = f"d_total_reward_{arm}"
            if col not in sub.columns:
                continue
            y = np.arange(len(sub)) + (j - 1) * 0.22
            ax.errorbar(sub[col], y,
                        xerr=[sub[col] - sub[f"{col}_lo"], sub[f"{col}_hi"] - sub[col]],
                        fmt="o", ms=3.5, lw=1, color=colors[arm], label=arm)
        ax.axvline(0, color="0.6", lw=0.8)
        ax.set_yticks(np.arange(len(sub)))
        ax.set_yticklabels(ylabels, fontsize=7)
        ax.set_title(loc)
        ax.set_xlabel("paired Δreward vs iid")
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "fig1_forest_dreward.png"), dpi=150)
    plt.close(fig)

    # Fig 2: gap closed per arm, bars per cell, faceted by location.
    fig, axes = plt.subplots(1, len(locs), figsize=(4.2 * len(locs), 5), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, loc in zip(axes, locs):
        sub = cells[(cells["location_id"] == loc)].sort_values(
            ["season", "battery_capacity", "failure_penalty"]).reset_index(drop=True)
        x = np.arange(len(sub))
        for j, arm in enumerate(CHAIN_ARMS):
            col = f"gap_closed_{arm}"
            if col in sub.columns:
                ax.bar(x + (j - 1) * 0.27, sub[col], width=0.25,
                       color=colors[arm], label=arm)
        ax.axhline(1.0, color="0.6", lw=0.8, ls="--")
        ax.axhline(0.0, color="0.6", lw=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(
            [f"{r['season'][:3]}\n{r['battery_capacity']:g}Wh\np{r['failure_penalty']:g}"
             for _, r in sub.iterrows()], fontsize=6)
        ax.set_title(loc)
    axes[0].set_ylabel("fraction of thresh→best gap closed")
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "fig2_gap_closed.png"), dpi=150)
    plt.close(fig)

    # Fig 3: interaction trends -- dReward vs capacity and vs penalty per arm/location.
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for ax, xkey, fixed_key, fixed_val in (
            (axes[0], "battery_capacity", "failure_penalty", 20.0),
            (axes[1], "failure_penalty", "battery_capacity", 300.0)):
        sub = cells[(cells[fixed_key] == fixed_val) & (cells["season"] == "summer")]
        for loc in locs:
            s = sub[sub["location_id"] == loc].sort_values(xkey)
            for arm in CHAIN_ARMS:
                col = f"d_total_reward_{arm}"
                if col in s.columns and s[col].notna().any():
                    ax.plot(s[xkey], s[col], marker="o", ms=3, lw=1,
                            color=colors[arm], alpha=0.7)
        ax.axhline(0, color="0.6", lw=0.8)
        ax.set_xscale("log", base=2)
        ax.set_xlabel(xkey)
        ax.set_ylabel("paired Δreward vs iid")
        ax.set_title(f"{fixed_key}={fixed_val:g}, summer")
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "fig3_interactions.png"), dpi=150)
    plt.close(fig)
    print(f"Wrote figures -> {fig_dir}")


# ----------------------------------------------------------------------------------


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--results", default=os.path.join(REPO_ROOT, "results", "markov_ablation"))
    ap.add_argument("--manifest", default=None,
                    help="Manifest path (default: <configs dir matching results>/markov_ablation_manifest.json).")
    ap.add_argument("--solar-bins-study", metavar="RESULTS_DIR", default=None,
                    help="Phase 2 mode: analyze the solar bin-resolution smoke in RESULTS_DIR.")
    ap.add_argument("--solar-res-study", metavar="RESULTS_DIR", default=None,
                    help="Phase 6 mode: analyze the all-location solar bin-resolution study in RESULTS_DIR.")
    ap.add_argument("--n-boot", type=int, default=10000)
    ap.add_argument("--verify", action="store_true",
                    help="Run the CRN identical-weather check on full-history cells.")
    ap.add_argument("--out", default=None, help="Analysis output dir override.")
    args = ap.parse_args()
    rng = np.random.default_rng(0)

    if args.solar_bins_study:
        results_base = os.path.abspath(args.solar_bins_study)
        manifest_path = args.manifest or os.path.join(
            REPO_ROOT, "configs", "markov_ablation_smoke_solar", "markov_ablation_manifest.json")
        manifest = load_manifest(manifest_path)
        out_dir = args.out or os.path.join(results_base, "_analysis")
        solar_bins_study(manifest, results_base, args.n_boot, rng, out_dir)
        return

    if args.solar_res_study:
        results_base = os.path.abspath(args.solar_res_study)
        manifest_path = args.manifest or os.path.join(
            REPO_ROOT, "configs", "markov_solar_res", "markov_ablation_manifest.json")
        manifest = load_manifest(manifest_path)
        out_dir = args.out or os.path.join(results_base, "_analysis")
        solar_res_study(manifest, results_base, args.n_boot, rng, out_dir)
        return

    results_base = os.path.abspath(args.results)
    manifest_path = args.manifest or os.path.join(
        REPO_ROOT, "configs", os.path.basename(results_base.rstrip("\\/")),
        "markov_ablation_manifest.json")
    manifest = load_manifest(manifest_path)
    out_dir = args.out or os.path.join(results_base, "_analysis")
    os.makedirs(out_dir, exist_ok=True)

    summaries = load_summaries(manifest, results_base)
    cells = build_cells(summaries, manifest, args.n_boot, rng, args.verify)
    if cells.empty:
        sys.exit("[error] no cells assembled")
    csv_path = os.path.join(out_dir, "markov_ablation_cells.csv")
    cells.to_csv(csv_path, index=False)
    print(f"Wrote {csv_path} ({len(cells)} cells)")
    write_report(cells, os.path.join(out_dir, "markov_ablation_report.md"), manifest)
    try:
        write_figures(cells, os.path.join(out_dir, "figures"))
    except Exception as e:
        print(f"[warn] figure generation failed: {e}")

    # Post-run sanity flags.
    if {"d_total_reward_joint", "d_total_reward_wind", "d_total_reward_solar"} <= set(cells.columns):
        worse = cells[
            (cells["d_total_reward_joint"] < cells[["d_total_reward_wind",
                                                    "d_total_reward_solar"]].max(axis=1))
            & (cells["d_total_reward_joint_hi"]
               < cells[["d_total_reward_wind_lo", "d_total_reward_solar_lo"]].max(axis=1))]
        if len(worse):
            print(f"[sanity] joint arm materially below max(wind, solar) in {len(worse)} "
                  f"cell(s) -- inspect these rows in the CSV")
    for arm in CHAIN_ARMS:
        col = f"d_total_reward_{arm}_hi"
        if col in cells.columns:
            n_bad = int((cells[col] < 0).sum())
            if n_bad:
                print(f"[sanity] iid materially beats {arm} in {n_bad} cell(s)")


if __name__ == "__main__":
    main()

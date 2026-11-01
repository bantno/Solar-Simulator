#!/usr/bin/env python3
"""Export the thesis-run results (runs 1-5) as small CSVs for the results dashboard.

Reads the analysis outputs of run_thesis_runs.py / summarize_thesis_runs.py and writes,
under <base>/thesis_summary/dashboard/:
    cells.csv     run 2: one row per headline cell (111), chain / i.i.d. / threshold
    worlds.csv    run 3 beside run 2: policy rewards per cell in the i.i.d. model's
                  weather, the chain model's weather and historical weather
    selection.csv run 1: bin-count scores on the inner split
    bins.csv      run 4: chain - threshold per cell and bin count at the extra sites
    blocks.csv    run 5 beside run 2: chain - threshold with 7-day and 30-day blocks

Usage (pvlib conda env, from SolarSimulator/):
    python Scripts/export_thesis_dashboard_data.py
"""
import argparse
import json
import os

import pandas as pd

BASE = "C:/Users/bepstein8/sim_results"


def lat_of(location_id):
    return float(location_id.split("_")[0][3:])


def x_of(r):
    """The swept parameter's value for a headline cell."""
    return {"capacity": r.battery_capacity, "duration": r.horizon,
            "penalty": r.failure_penalty, "location": lat_of(r.location_id),
            "startdate": pd.Timestamp(r.start_time).month}[r.sweep]


def diffs(df, pairs):
    """Rename d_total_reward[_<pair>]{,_lo,_hi} to <name>{,_lo,_hi}."""
    out = {}
    for src, name in pairs.items():
        for suf in ("", "_lo", "_hi"):
            out[name + suf] = df[src + suf].round(3)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--base", default=BASE)
    base = ap.parse_args().base
    out = f"{base}/thesis_summary/dashboard"
    os.makedirs(out, exist_ok=True)
    pairs = {"d_total_reward_chain_vs_thresh": "chain_vs_thresh",
             "d_total_reward_iid_vs_thresh": "iid_vs_thresh", "d_total_reward": "chain_vs_iid"}

    h = pd.read_csv(f"{base}/thesis_summary/thesis_holm_cells.csv")
    cells = pd.DataFrame({
        "cell": [f"{r.sweep}|{r.location_id}|{r.battery_capacity:g}Wh|{r.horizon}|p{r.failure_penalty:g}|"
                 f"{str(r.start_time)[:10]}" for r in h.itertuples()],
        "sweep": h.sweep, "x": [x_of(r) for r in h.itertuples()],
        "lat": h.location_id.map(lat_of), "capacity_wh": h.battery_capacity,
        "horizon": h.horizon, "penalty": h.failure_penalty,
        "start_month": pd.to_datetime(h.start_time).dt.month,
        **diffs(h, pairs),
        "reward_chain": h.avg_reward_chain.round(3), "reward_iid": h.avg_reward_iid.round(3),
        "reward_thresh": h.avg_reward_thresh.round(3),
        "fail_pct_chain": h.failure_pct_chain, "fail_pct_iid": h.failure_pct_iid,
        "fail_pct_thresh": h.failure_pct_thresh,
        "p_holm": h.p_holm.map(lambda p: float(f"{p:.3g}")), "holm_verdict": h.holm_verdict})
    cells.to_csv(f"{out}/cells.csv", index=False)

    w = pd.read_csv(f"{base}/thesis_worlds/_analysis/model_worlds_cells.csv")
    hist = h[h.sweep.isin(["capacity", "penalty"])]
    worlds = pd.concat([
        pd.DataFrame({"sweep": w.sweep, "world": w.world.str.replace("world", ""),
                      "x": w.battery_capacity.where(w.sweep == "capacity", w.failure_penalty),
                      "reward_iid": w.avg_reward_iid, "reward_chain": w.avg_reward_chain,
                      "reward_thresh": w.avg_reward_thresh}),
        pd.DataFrame({"sweep": hist.sweep, "world": "historical",
                      "x": hist.battery_capacity.where(hist.sweep == "capacity", hist.failure_penalty),
                      "reward_iid": hist.avg_reward_iid, "reward_chain": hist.avg_reward_chain,
                      "reward_thresh": hist.avg_reward_thresh})])
    worlds.round(3).sort_values(["sweep", "world", "x"]).to_csv(f"{out}/worlds.csv", index=False)

    sel = json.load(open(f"{base}/wind_bins_select/_analysis/bin_selection.json"))
    s = pd.DataFrame(sel["scores"])[["n_bins", "score", "diff_vs_best", "se_diff"]]
    s["chosen"] = s.n_bins == sel["chosen_n_bins"]
    s.round(4).to_csv(f"{out}/selection.csv", index=False)

    b = pd.read_csv(f"{base}/thesis_sites/_analysis/wind_bins_cells.csv")
    pd.DataFrame({"case": b.sweep, "capacity_wh": b.battery_capacity,
                  "start_month": pd.to_datetime(b.start_time).dt.month, "n_bins": b.n_bins,
                  "chain_vs_thresh": b.d_reward_vs_thresh.round(3),
                  "chain_vs_thresh_lo": b.d_reward_vs_thresh_lo.round(3),
                  "chain_vs_thresh_hi": b.d_reward_vs_thresh_hi.round(3)}
                 ).to_csv(f"{out}/bins.csv", index=False)

    b30 = pd.read_csv(f"{base}/thesis_block30/_analysis/comparison_cells.csv")
    rows = []
    for days, df in ((7, hist), (30, b30)):
        rows.append(pd.DataFrame({
            "sweep": df.sweep.values, "block_days": days,
            "x": df.battery_capacity.where(df.sweep == "capacity", df.failure_penalty).values,
            **{k: v.values for k, v in diffs(df, {k: pairs[k] for k in
                                                  ("d_total_reward_chain_vs_thresh",
                                                   "d_total_reward")}).items()}}))
    pd.concat(rows).sort_values(["sweep", "x", "block_days"]).to_csv(f"{out}/blocks.csv", index=False)
    for f in sorted(os.listdir(out)):
        print(f"[out] {out}/{f} ({len(pd.read_csv(f'{out}/{f}'))} rows)")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Thesis figures from the analyses of thesis runs 1-5 (PNG and PDF).

    fig1_model_worlds     reward of iid, chain and threshold in the i.i.d. model's weather,
                          in the chain model's weather and on held-out historical weather
                          (capacity and penalty sweeps; runs 3 and 2)
    fig2_chain_vs_thresh  chain minus fit-years-tuned threshold, every cell of run 2, with
                          95% CIs, one panel per sweep
    fig3_horizon          chain minus threshold and chain minus iid against planning
                          horizon (run 2, duration sweep)
    fig4_wind_bins        (a) run-1 selection score by bin count on 1986-2002, (b) chain
                          minus threshold by bin count on 2003-2022 (wind-bin studies),
                          (c) value-table solve time by bin count (benchmark)
    fig5_sites            chain minus threshold at the five sites, in reward and as a
                          percentage of the threshold's reward, one panel per sweep
                          (needs compare_thesis_sites.py's sites_cells.csv)

A figure whose inputs are missing is skipped with a note.

Usage (pvlib conda env, from SolarSimulator/):
    python Scripts/plot_thesis_figures.py
"""
import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

BASE = "C:/Users/bepstein8/sim_results"
# Reference categorical palette: slots 1-3 (validated all-pairs) are the policies; the
# wind-bin cases take slots 4-8 so they never borrow a policy's color.
COLORS = {"iid": "#2a78d6", "chain": "#eb6834", "thresh": "#1baf7a"}
SERIES = ["#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
MARKERS = {"iid": "o", "chain": "s", "thresh": "^"}
LABELS = {"iid": "i.i.d. optimal", "chain": "chain optimal", "thresh": "threshold"}
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"

plt.rcParams.update({
    "font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
    "ytick.color": MUTED, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False, "lines.linewidth": 2,
    "lines.markersize": 5, "legend.frameon": False, "savefig.dpi": 200,
})


def save(fig, out, name):
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(out, f"{name}.{ext}"), bbox_inches="tight")
    plt.close(fig)
    print(f"[fig] {name}")


def _x(sweep, df):
    if sweep == "capacity":
        return df["battery_capacity"], "battery capacity (Wh)"
    if sweep == "penalty":
        return df["failure_penalty"], "failure penalty"
    if sweep == "duration":
        return df["horizon"], "horizon (15-min stages)"
    if sweep == "startdate":
        return pd.to_datetime(df["start_time"]).dt.month, "start month"
    return df["battery_capacity"], "battery capacity (Wh)"


def fig_model_worlds(base, out):
    w_path = f"{base}/thesis_worlds/_analysis/model_worlds_cells.csv"
    h_path = f"{base}/thesis_headline/_analysis/comparison_cells.csv"
    if not (os.path.exists(w_path) and os.path.exists(h_path)):
        print("[skip] fig1: run 3 or run 2 analysis missing")
        return
    w, h = pd.read_csv(w_path), pd.read_csv(h_path)
    h = h[h["world"] == "hist"]
    sweeps = ["capacity", "penalty"]
    panels = [("iidworld", "i.i.d. model's weather"), ("chainworld", "chain model's weather"),
              ("hist", "historical weather 2003–2022")]
    fig, axes = plt.subplots(2, 3, figsize=(9.5, 5.6), sharey="row")
    for i, sweep in enumerate(sweeps):
        for j, (world, title) in enumerate(panels):
            ax = axes[i, j]
            d = (w[(w["sweep"] == sweep) & (w["world"] == world)] if world != "hist"
                 else h[h["sweep"] == sweep])
            d = d.assign(_x=_x(sweep, d)[0]).sort_values("_x")
            for arm in ("iid", "chain", "thresh"):
                ax.plot(d["_x"], d[f"avg_reward_{arm}"], color=COLORS[arm],
                        marker=MARKERS[arm], label=LABELS[arm])
            if i == 0:
                ax.set_title(title, color=INK, fontsize=9.5)
            ax.set_xlabel(_x(sweep, d)[1])
            if j == 0:
                ax.set_ylabel(f"{sweep} sweep\nmean episode reward")
    axes[0, 0].legend(loc="best")
    fig.suptitle("Each policy in its own model's weather and on real weather", color=INK)
    fig.tight_layout()
    save(fig, out, "fig1_model_worlds")


def fig_chain_vs_thresh(base, out):
    path = f"{base}/thesis_headline/_analysis/comparison_cells.csv"
    if not os.path.exists(path):
        print("[skip] fig2: run 2 analysis missing")
        return
    c = pd.read_csv(path)
    c = c[c["world"] == "hist"]
    col = "d_total_reward_chain_vs_thresh"
    sweeps = ["capacity", "duration", "penalty", "location", "startdate"]
    fig, axes = plt.subplots(1, 5, figsize=(12, 3.0))
    for ax, sweep in zip(axes, sweeps):
        d = c[c["sweep"] == sweep].copy()
        if sweep == "location":
            lat = d["location_id"].str.extract(r"lat(-?[\d.]+)")[0].astype(float)
            cap = d["battery_capacity"]
            d["_x"] = lat + (cap - 350) / 250 * 3.0  # capacities fan out around each latitude
            label = "latitude (capacities 100–600 Wh fanned out)"
        else:
            d["_x"], label = _x(sweep, d)
        err = np.vstack([d[col] - d[f"{col}_lo"], d[f"{col}_hi"] - d[col]])
        ax.errorbar(d["_x"], d[col], yerr=err, fmt="o", color=COLORS["chain"], ms=3.5,
                    elinewidth=1, capsize=0)
        ax.axhline(0, color=MUTED, lw=1)
        if sweep == "duration":
            ax.set_xscale("log")
        ax.set_title(sweep, color=INK, fontsize=9.5)
        ax.set_xlabel(label, fontsize=8)
    axes[0].set_ylabel("chain − threshold reward")
    fig.suptitle("Chain policy minus fit-years-tuned threshold, 95% CI, every cell", color=INK)
    fig.tight_layout()
    save(fig, out, "fig2_chain_vs_thresh")


def fig_horizon(base, out):
    path = f"{base}/thesis_headline/_analysis/comparison_cells.csv"
    if not os.path.exists(path):
        print("[skip] fig3: run 2 analysis missing")
        return
    c = pd.read_csv(path)
    d = c[(c["world"] == "hist") & (c["sweep"] == "duration")].sort_values("horizon")
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.0))
    for ax, (col, other, title) in zip(axes, [
            ("d_total_reward_chain_vs_thresh", "thresh", "chain − threshold"),
            ("d_total_reward", "iid", "chain − i.i.d.")]):
        ax.fill_between(d["horizon"], d[f"{col}_lo"], d[f"{col}_hi"], color=COLORS["chain"],
                        alpha=0.18, lw=0)
        ax.plot(d["horizon"], d[col], color=COLORS["chain"], marker="s")
        ax.axhline(0, color=MUTED, lw=1)
        ax.set_title(title, color=INK, fontsize=9.5)
        ax.set_xlabel("horizon (15-min stages)")
        ax.set_ylabel("reward difference")
    fig.suptitle("Chain advantage grows with planning horizon (95% CI band)", color=INK)
    fig.tight_layout()
    save(fig, out, "fig3_horizon")


def fig_wind_bins(base, out):
    sel_path = f"{base}/wind_bins_select/_analysis/bin_selection.json"
    studies = [f"{base}/wind_bins_study/_analysis/wind_bins_cells.csv",
               f"{base}/thesis_sites/_analysis/wind_bins_cells.csv"]
    bench_path = f"{base}/wind_bins_study/_analysis/solve_benchmark.json"
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.3))
    ax = axes[0]
    if os.path.exists(sel_path):
        sel = json.load(open(sel_path))
        r = pd.DataFrame(sel["scores"])
        ax.errorbar(r["n_bins"], r["diff_vs_best"], yerr=r["se_diff"], fmt="o-",
                    color=COLORS["chain"], capsize=0, elinewidth=1)
        ax.axhline(0, color=MUTED, lw=1)
        ax.axvline(sel["chosen_n_bins"], color=MUTED, lw=1, ls="--")
        ax.annotate(f"chosen: {sel['chosen_n_bins']}", (sel["chosen_n_bins"], 0),
                    xytext=(4, -14), textcoords="offset points", color=INK, fontsize=8)
        ho = sel["holdout"]
        ax.set_title(f"(a) selection on {ho['eval_years'][0]}–{ho['eval_years'][1]}",
                     color=INK, fontsize=9.5)
        ax.set_ylabel("score minus best (± 1 SE)")
    else:
        ax.text(0.5, 0.5, "run 1 missing", ha="center", transform=ax.transAxes)
    ax.set_xlabel("wind bins")

    ax = axes[1]
    frames = [pd.read_csv(p) for p in studies if os.path.exists(p)]
    if frames:
        s = pd.concat(frames, ignore_index=True)
        for k, (case, g) in enumerate(s.groupby("sweep", sort=False)):
            m = g.groupby("n_bins")["d_reward_vs_thresh"].mean()
            color = SERIES[k % len(SERIES)]
            ax.plot(m.index, m.values, marker="o", color=color, label=case)
        ax.axhline(0, color=MUTED, lw=1)
        ax.legend(loc="lower right", fontsize=7.5, ncol=2)
    ax.set_title("(b) chain − threshold on 2003–2022", color=INK, fontsize=9.5)
    ax.set_xlabel("wind bins")
    ax.set_ylabel("mean over the case's cells")

    ax = axes[2]
    if os.path.exists(bench_path):
        b = pd.DataFrame(json.load(open(bench_path))["rows"])
        ch = b[b["arm"] != "iid"]
        ax.plot(ch["n_bins"], ch["solve_s"], marker="o", color=COLORS["chain"], label="chain")
        iid = b[b["arm"] == "iid"]["solve_s"].iloc[0]
        ax.axhline(iid, color=COLORS["iid"], lw=1.5, ls="--", label="i.i.d.")
        ax.legend(loc="upper left", fontsize=7.5)
    ax.set_title("(c) value-table solve time", color=INK, fontsize=9.5)
    ax.set_xlabel("wind bins")
    ax.set_ylabel("seconds (300 Wh, 1000 stages)")
    fig.tight_layout()
    save(fig, out, "fig4_wind_bins")


def fig_sites(base, out):
    path = f"{base}/thesis_summary/sites/sites_cells.csv"
    if not os.path.exists(path):
        print("[skip] fig5: run compare_thesis_sites.py first")
        return
    h = pd.read_csv(path)
    h["d"] = h["avg_reward_chain"] - h["avg_reward_thresh"]
    h["pct"] = 100 * h["d"] / h["avg_reward_thresh"]
    # Sites in order of June-July mean wind (wind_persistence_by_site.py), calmest first.
    sites = [("lat30.0_lon-75.0", "lat 30, lon −75 (5.2 m/s)"),
             ("lat-20.0_lon-75.0", "lat −20, lon −75 (5.9 m/s)"),
             ("lat0.0_lon-25.0", "lat 0, lon −25 (6.1 m/s)"),
             ("lat-30.0_lon-75.0", "lat −30, lon −75 (6.8 m/s)"),
             ("lat-30.0_lon-25.0", "lat −30, lon −25 (7.3 m/s)")]
    sweeps = ["capacity", "duration", "penalty", "startdate"]
    rows = [("d", "chain − threshold\n(reward)"), ("pct", "chain − threshold\n(% of threshold reward)")]
    fig, axes = plt.subplots(2, 4, figsize=(11, 5.6), sharey="row")
    for i, (col, ylabel) in enumerate(rows):
        for j, sweep in enumerate(sweeps):
            ax = axes[i, j]
            g = h[h["sweep"] == sweep]
            if col == "pct":  # threshold reward ~0 at penalty 100: the ratio is meaningless
                g = g[g["failure_penalty"] < 100] if sweep == "penalty" else g
            for k, (loc, label) in enumerate(sites):
                d = g[g["location_id"] == loc].copy()
                d["_x"], xlabel = _x(sweep, d)
                d = d.sort_values("_x")
                ax.plot(d["_x"], d[col], color=SERIES[k], marker="o", ms=3.5, lw=1.6,
                        label=label)
            ax.axhline(0, color=MUTED, lw=1)
            if sweep == "startdate":
                ax.set_xticks([1, 3, 5, 7, 9], ["Jan", "Mar", "May", "Jul", "Sep"])
            if i == 0:
                ax.set_title(sweep, color=INK, fontsize=9.5)
            else:
                ax.set_xlabel(xlabel)
                if sweep == "penalty":
                    ax.annotate("penalty 100 omitted", xy=(0.98, 0.95), xycoords="axes fraction",
                                ha="right", va="top", color=MUTED, fontsize=7.5)
            if j == 0:
                ax.set_ylabel(ylabel)
    axes[0, 0].legend(title="site (June–July mean wind)", fontsize=7.5, title_fontsize=7.5,
                      loc="upper left")
    fig.suptitle("Chain's lead over the threshold at five sites: absolute (top) and relative "
                 "(bottom)", color=INK)
    fig.tight_layout()
    save(fig, out, "fig5_sites")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--base", default=BASE)
    ap.add_argument("--out", default=None, help="Default <base>/thesis_summary/figures.")
    args = ap.parse_args()
    out = args.out or f"{args.base}/thesis_summary/figures"
    os.makedirs(out, exist_ok=True)
    for fn in (fig_model_worlds, fig_chain_vs_thresh, fig_horizon, fig_wind_bins, fig_sites):
        fn(args.base, out)


if __name__ == "__main__":
    main()

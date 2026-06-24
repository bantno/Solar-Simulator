"""
wind_persistence_plots.py -- visualize the wind persistence / Markov-order pre-check.

Two modes, both built on Scripts.wind_persistence_precheck.analyze (single source of truth):

  single  (default): a 2x2 panel for one binning --
            ACF, PACF, the first-order transition matrix P(next|curr), and a bar chart of
            H(Next|Curr) vs H(Next|Curr,Prev) with the conditional-MI gap annotated.

  sweep            : run analyze over several bin resolutions and plot the (month,hour)-
            stratified higher-order signal -- I(Next;Prev|Curr,m,h) in bits and as % of
            H(Next|Curr) -- vs the number of bins. Optionally overlay an aircraft-threshold
            binning as a labeled reference point. This is the diagnostic that distinguishes
            "wind has real higher-order memory" from "the signal is a discretization artifact."

Usage (pvlib conda env):

    python Scripts/wind_persistence_plots.py single \\
        --historical Data/HISTORICAL_DATA/data_30_-90.pkl --n-bins 3 --out-dir ./precheck_out

    python Scripts/wind_persistence_plots.py sweep \\
        --historical Data/HISTORICAL_DATA/data_30_-90.pkl \\
        --bins-list 3 4 6 8 10 --aircraft-edges 5.0 10.0 --out-dir ./precheck_out
"""
import argparse
import os
import sys

import numpy as np

# Make this script importable-agnostic: add its own dir so `wind_persistence_precheck`
# resolves whether run from the repo root or the SolarSimulator dir.
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
from wind_persistence_precheck import analyze  # noqa: E402


def _mpl():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def _bin_labels(edges):
    """Human-readable bin ranges, e.g. ['0-3.1', '3.1-4.7', '4.7+'] (m/s)."""
    labels = []
    for i in range(len(edges) - 1):
        lo, hi = edges[i], edges[i + 1]
        labels.append(f"{lo:.1f}-{hi:.1f}" if np.isfinite(hi) else f"{lo:.1f}+")
    return labels


# --------------------------------------------------------------------------------------
def plot_single(results, out_path):
    """2x2 summary panel for one binning."""
    plt = _mpl()
    r = results
    acf, pacf = r["acf"], r["pacf"]
    step_min, ci = r["step_min"], r["ci"]
    hrs = np.arange(acf.size) * step_min / 60.0
    labels = _bin_labels(r["edges"])

    fig, ax = plt.subplots(2, 2, figsize=(13, 9))

    # (0,0) ACF
    a = ax[0, 0]
    a.stem(hrs, acf, basefmt=" ")
    a.axhspan(-ci, ci, color="tab:blue", alpha=0.12, label="95% CI (white noise)")
    a.axhline(1 / np.e, color="grey", ls="--", lw=0.8, label="1/e")
    a.set_title("Continuous wind ACF"); a.set_xlabel("lag [h]"); a.set_ylabel("ACF")
    a.legend(loc="upper right", fontsize=8)

    # (0,1) PACF
    a = ax[0, 1]
    a.stem(hrs, pacf, basefmt=" ")
    a.axhspan(-ci, ci, color="tab:orange", alpha=0.15)
    a.set_title("Continuous wind PACF (memory beyond lag 1 if spikes exit band)")
    a.set_xlabel("lag [h]"); a.set_ylabel("PACF")
    a.set_xlim(0, min(hrs[-1], 24))  # PACF detail lives at short lags

    # (1,0) first-order transition matrix
    a = ax[1, 0]
    T = r["transition"]
    im = a.imshow(T, cmap="viridis", vmin=0, vmax=1, aspect="auto")
    a.set_xticks(range(r["n_bins"])); a.set_yticks(range(r["n_bins"]))
    a.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
    a.set_yticklabels(labels, fontsize=8)
    a.set_xlabel("next bin [m/s]"); a.set_ylabel("current bin [m/s]")
    a.set_title("First-order transition  P(next | curr)")
    for i in range(r["n_bins"]):
        for j in range(r["n_bins"]):
            if np.isfinite(T[i, j]):
                a.text(j, i, f"{T[i, j]:.2f}", ha="center", va="center",
                       color="white" if T[i, j] < 0.6 else "black", fontsize=8)
    fig.colorbar(im, ax=a, fraction=0.046, pad=0.04)

    # (1,1) entropy reduction bar
    a = ax[1, 1]
    pooled, strat = r["pooled"], r["stratified"]
    h_curr = pooled["h_next_given_curr_bits"]
    h_both = pooled["h_next_given_both_bits"]
    bars = a.bar([0, 1], [h_curr, h_both], width=0.55,
                 color=["tab:gray", "tab:green"])
    a.set_xticks([0, 1])
    a.set_xticklabels(["H(Next | Curr)\n(chain order)", "H(Next | Curr, Prev)\n(2nd order)"])
    a.set_ylabel("conditional entropy of next bin [bits]")
    a.set_title("Uncertainty reduced by adding the previous bin")
    for b, v in zip(bars, [h_curr, h_both]):
        a.text(b.get_x() + b.get_width() / 2, v, f"{v:.3f}", ha="center", va="bottom", fontsize=9)
    txt = (f"pooled I(Next;Prev|Curr) = {pooled['mi_bits']:.4f} bits\n"
           f"(month,hour)-stratified  = {strat['mi_bits']:.4f} bits\n"
           f"  = {r['pct_entropy']:.2f}% of H(Next|Curr)   <- fair vs BI-chain")
    a.text(0.5, 0.82, txt, transform=a.transAxes, ha="center", va="top", fontsize=9,
           bbox=dict(boxstyle="round", fc="lightyellow", ec="goldenrod"))

    title = (f"Wind persistence pre-check  --  {os.path.basename(r['historical'])}  "
             f"({r['year_min']}-{r['year_max']}, ~{r['step_min']} min, {r['n_bins']} bins)")
    fig.suptitle(title, fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    return out_path


# --------------------------------------------------------------------------------------
def run_sweep(historical, bins_list, aircraft_edges, dt, wind_col):
    """Call analyze for each bin count (+ optional aircraft edges). Returns list of rows."""
    rows = []
    for nb in bins_list:
        r = analyze(historical, n_bins=nb, bin_edges=None, dt=dt, wind_col=wind_col)
        rows.append({
            "label": f"{nb}q", "n_bins": r["n_bins"], "is_aircraft": False,
            "mi_strat": r["mi_strat_bits"], "mi_pooled": r["mi_pooled_bits"],
            "pct": r["pct_entropy"], "h_curr": r["h_next_given_curr_bits"],
            "edges": r["edges"],
        })
        print(f"  n_bins={r['n_bins']:>2} (quantile): "
              f"I_strat={r['mi_strat_bits']:.4f} bits ({r['pct_entropy']:.2f}%), "
              f"edges={np.round(r['edges'], 2)}")
    if aircraft_edges:
        full = np.concatenate(([0.0], np.asarray(aircraft_edges, float), [np.inf]))
        r = analyze(historical, bin_edges=full, dt=dt, wind_col=wind_col)
        rows.append({
            "label": "aircraft", "n_bins": r["n_bins"], "is_aircraft": True,
            "mi_strat": r["mi_strat_bits"], "mi_pooled": r["mi_pooled_bits"],
            "pct": r["pct_entropy"], "h_curr": r["h_next_given_curr_bits"],
            "edges": r["edges"],
        })
        print(f"  aircraft edges {aircraft_edges}: "
              f"I_strat={r['mi_strat_bits']:.4f} bits ({r['pct_entropy']:.2f}%)")
    return rows


def plot_sweep(rows, historical, out_path):
    plt = _mpl()
    q = [r for r in rows if not r["is_aircraft"]]
    x = [r["n_bins"] for r in q]
    mi = [r["mi_strat"] for r in q]
    pct = [r["pct"] for r in q]

    fig, ax1 = plt.subplots(figsize=(9, 6))
    ax1.plot(x, mi, "o-", color="tab:red", label="I(Next;Prev|Curr,m,h)  [bits]")
    ax1.set_xlabel("number of wind bins (equal-occupancy quantile)")
    ax1.set_ylabel("higher-order memory beyond chain  [bits]", color="tab:red")
    ax1.tick_params(axis="y", labelcolor="tab:red")
    ax1.set_ylim(bottom=0)

    ax2 = ax1.twinx()
    ax2.plot(x, pct, "s--", color="tab:blue", label="% of H(Next|Curr)")
    ax2.set_ylabel("% of remaining next-bin entropy", color="tab:blue")
    ax2.tick_params(axis="y", labelcolor="tab:blue")
    ax2.set_ylim(bottom=0)

    for r in rows:
        if r["is_aircraft"]:
            ax1.scatter([r["n_bins"]], [r["mi_strat"]], marker="*", s=240,
                        color="tab:green", zorder=5,
                        label=f"aircraft edges {np.round(r['edges'][1:-1], 1)}")
            ax1.annotate("aircraft", (r["n_bins"], r["mi_strat"]),
                         textcoords="offset points", xytext=(8, 8), color="tab:green")

    lines = ax1.get_lines() + ax2.get_lines()
    ax1.legend(lines, [l.get_label() for l in lines], loc="upper left", fontsize=9)
    ax1.set_title(f"Does higher-order wind memory grow with bin resolution?\n"
                  f"{os.path.basename(historical)} -- if it rises with bins, the signal is "
                  f"lost to coarse discretization, not absent")
    ax1.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    return out_path


# --------------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Visualize the wind persistence pre-check.")
    sub = ap.add_subparsers(dest="mode", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--historical", required=True, help="Path to hourly HISTORICAL_DATA pickle.")
    common.add_argument("--wind-col", default="wind_speed_10m")
    common.add_argument("--dt", type=int, default=None,
                        help="Resample to N minutes first (default: native; resampling inflates "
                             "short-lag persistence).")
    common.add_argument("--out-dir", default=".")

    s = sub.add_parser("single", parents=[common], help="2x2 panel for one binning.")
    s.add_argument("--bin-edges", type=float, nargs="+", metavar="M_S",
                   help="Interior cutpoints in m/s. Overrides --n-bins.")
    s.add_argument("--n-bins", type=int, default=3)

    w = sub.add_parser("sweep", parents=[common], help="MI vs bin resolution.")
    w.add_argument("--bins-list", type=int, nargs="+", default=[3, 4, 6, 8, 10],
                   help="Quantile bin counts to sweep.")
    w.add_argument("--aircraft-edges", type=float, nargs="+", default=None, metavar="M_S",
                   help="Optional interior cutpoints to overlay as a reference point.")

    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    if args.mode == "single":
        r = analyze(args.historical, n_bins=args.n_bins, bin_edges=args.bin_edges,
                    dt=args.dt, wind_col=args.wind_col)
        out = plot_single(r, os.path.join(args.out_dir, "wind_persistence_single.png"))
        print(f"[fig] single-run panel -> {out}")
    else:
        print(f"[sweep] {os.path.basename(args.historical)}  bins={args.bins_list}")
        rows = run_sweep(args.historical, args.bins_list, args.aircraft_edges,
                         args.dt, args.wind_col)
        out = plot_sweep(rows, args.historical,
                         os.path.join(args.out_dir, "wind_persistence_sweep.png"))
        print(f"[fig] bin-resolution sweep -> {out}")


if __name__ == "__main__":
    main()

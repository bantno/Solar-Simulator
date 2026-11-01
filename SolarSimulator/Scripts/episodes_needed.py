#!/usr/bin/env python3
"""Episodes needed per cell for a paired policy comparison to reach significance.

Every arm of a cell sees the same weather and crash draws (common random numbers), so
each policy comparison is a paired difference. Its per-episode standard deviation is
backed out of the 95 % bootstrap confidence interval that compare_chain_sweep.py writes
into comparison_cells.csv:

    sd = (hi - lo) / (2 * 1.96) * sqrt(n_paired_episodes)

and the episodes needed for a two-sided paired test at level alpha with the given power
to detect an effect delta are

    n = ((z_{1 - alpha / (2 m)} + z_{power}) * sd / delta) ** 2

where m is the number of comparisons in the family (Bonferroni). By default delta is the
observed effect; --effect fixes the smallest effect you care about instead, which is the
better basis for choosing a run size. --half-width reports the episodes needed for a 95 %
interval of that half-width.

Caveats: the CVaR delta is a tail statistic, so its back-calculated sd is approximate.
More episodes shrink Monte Carlo noise only; they do not add evaluation years.

Usage (pvlib conda env, from SolarSimulator/):
    python Scripts/episodes_needed.py ../results/apl_holdout_2026-10-20/_analysis/comparison_cells.csv
    python Scripts/episodes_needed.py <csv> [<csv> ...] --tests 30 --power 0.9 --effect 1.0
    python Scripts/episodes_needed.py --sd 25 --effect 1.0 --tests 30      # manual, no CSV
"""
import argparse
import math
import os
import sys
from statistics import NormalDist

import pandas as pd

Z = NormalDist().inv_cdf
Z95 = 1.959963984540054

# Delta columns written by compare_chain_sweep.py. Each base has <base>_lo / <base>_hi.
METRICS = ("total_reward", "cvar10", "failure", "flight_hrs")
COMPARISONS = {
    "": "chain - iid",
    "_iid_vs_thresh": "iid - thresh",
    "_chain_vs_thresh": "chain - thresh",
}


def sd_from_ci(lo: float, hi: float, n: int) -> float:
    """Per-episode sd of a paired difference from its 95 % CI and episode count."""
    return (hi - lo) / (2.0 * Z95) * math.sqrt(n)


def n_for_power(sd: float, delta: float, alpha: float = 0.05, power: float = 0.8,
                tests: int = 1) -> float:
    """Episodes for a two-sided paired test (Bonferroni over `tests`) to detect `delta`."""
    if delta == 0 or not math.isfinite(sd):
        return math.inf
    z = Z(1.0 - alpha / (2.0 * tests)) + Z(power)
    return math.ceil((z * sd / abs(delta)) ** 2)


def n_for_half_width(sd: float, half_width: float) -> float:
    """Episodes for a 95 % confidence interval of the given half-width."""
    return math.ceil((Z95 * sd / half_width) ** 2)


def min_detectable_effect(sd: float, n: int, alpha: float = 0.05, power: float = 0.8,
                          tests: int = 1) -> float:
    """Smallest effect detectable with `n` episodes at the given alpha, power and family size."""
    return (Z(1.0 - alpha / (2.0 * tests)) + Z(power)) * sd / math.sqrt(n)


def rows_from_csv(path: str, default_n: int):
    """Yield one record per (cell, comparison, metric) that has a delta and a CI."""
    df = pd.read_csv(path)
    label = os.path.basename(os.path.dirname(os.path.dirname(os.path.abspath(path)))) or path
    id_cols = [c for c in ("world", "location_id", "battery_capacity", "start_time",
                           "horizon") if c in df.columns]
    for _, r in df.iterrows():
        n0 = int(r["n_paired_episodes"]) if pd.notna(r.get("n_paired_episodes")) else default_n
        cell = " ".join(str(r[c]) for c in id_cols)
        for suffix, comp in COMPARISONS.items():
            for metric in METRICS:
                base = f"d_{metric}{suffix}"
                lo_c, hi_c = f"{base}_lo", f"{base}_hi"
                if base not in df.columns or lo_c not in df.columns or hi_c not in df.columns:
                    continue
                d, lo, hi = r[base], r[lo_c], r[hi_c]
                if pd.isna(d) or pd.isna(lo) or pd.isna(hi):
                    continue
                yield {"source": label, "cell": cell, "comparison": comp, "metric": metric,
                       "delta": float(d), "ci_lo": float(lo), "ci_hi": float(hi),
                       "n_run": n0, "sd": sd_from_ci(float(lo), float(hi), n0)}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("csv", nargs="*", help="comparison_cells.csv file(s) from compare_chain_sweep.py")
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--power", type=float, default=0.8)
    ap.add_argument("--tests", type=int, default=1,
                    help="Comparisons in the family, for a Bonferroni correction (default 1).")
    ap.add_argument("--effect", type=float, default=None,
                    help="Smallest effect worth detecting; default is each cell's observed delta.")
    ap.add_argument("--half-width", type=float, default=None,
                    help="Also report episodes for a 95%% CI of this half-width.")
    ap.add_argument("--metric", choices=METRICS, action="append",
                    help="Restrict to these metrics (repeatable). Default: all present.")
    ap.add_argument("--n-run", type=int, default=3000,
                    help="Episode count to assume when a CSV lacks n_paired_episodes.")
    ap.add_argument("--sd", type=float, default=None,
                    help="Manual mode: per-episode sd of the paired difference (no CSV).")
    ap.add_argument("--out", default=None, help="Optional CSV path for the full table.")
    args = ap.parse_args(argv)

    if args.sd is not None:
        if args.effect is None:
            ap.error("--sd needs --effect")
        n = n_for_power(args.sd, args.effect, args.alpha, args.power, args.tests)
        print(f"sd={args.sd:g}  effect={args.effect:g}  alpha={args.alpha:g}  "
              f"power={args.power:g}  tests={args.tests}:  n = {n}")
        if args.half_width:
            print(f"95% CI half-width {args.half_width:g}: n = "
                  f"{n_for_half_width(args.sd, args.half_width)}")
        return 0

    if not args.csv:
        ap.error("give at least one comparison_cells.csv, or use --sd/--effect")

    recs = [r for p in args.csv for r in rows_from_csv(p, args.n_run)]
    if args.metric:
        recs = [r for r in recs if r["metric"] in args.metric]
    if not recs:
        print("no delta columns with confidence intervals found", file=sys.stderr)
        return 1

    for r in recs:
        delta = args.effect if args.effect is not None else r["delta"]
        r["n_needed"] = n_for_power(r["sd"], delta, args.alpha, args.power, args.tests)
        r["significant_now"] = r["ci_lo"] > 0 or r["ci_hi"] < 0
        r["mde_at_n_run"] = min_detectable_effect(r["sd"], r["n_run"], args.alpha,
                                                  args.power, args.tests)
        if args.half_width:
            r["n_half_width"] = n_for_half_width(r["sd"], args.half_width)
    table = pd.DataFrame(recs)

    basis = f"effect={args.effect:g}" if args.effect is not None else "observed effect"
    print(f"alpha={args.alpha:g}, power={args.power:g}, tests={args.tests} (Bonferroni), "
          f"basis: {basis}\n")
    cols = ["cell", "comparison", "metric", "delta", "ci_lo", "ci_hi", "sd", "n_run",
            "significant_now", "mde_at_n_run", "n_needed"]
    if args.half_width:
        cols.append("n_half_width")
    with pd.option_context("display.max_rows", None, "display.width", 200,
                           "display.float_format", "{:.2f}".format):
        print(table[cols].to_string(index=False))

    finite = table.loc[table["n_needed"] != math.inf, "n_needed"]
    print(f"\nper-episode sd range: {table['sd'].min():.1f} to {table['sd'].max():.1f}")
    if len(finite):
        print(f"n_needed: median {finite.median():.0f}, max {finite.max():.0f}")
    if args.out:
        table.to_csv(args.out, index=False)
        print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""Wind variability and persistence per site, beside the chain's margin over the threshold.

For each site, from the raw hourly record (Data/HISTORICAL_DATA/data_<lat>_<lon>.pkl) over the
fit years (1950-2002, so nothing from the evaluation years) and in a chosen season window:
    mean_wind        mean 10 m wind speed (m/s)
    anom_sd          std of the daily-mean wind anomaly (daily mean minus that calendar day's
                     1950-2002 mean, smoothed over +/- 15 days): how far a day departs from what
                     the calendar alone predicts, i.e. what an i.i.d. model cannot anticipate
    acf_1d, acf_3d   autocorrelation of that anomaly 1 and 3 days apart (persistence)
    predictable_sd   anom_sd * acf_1d: the part of tomorrow's departure that today's wind
                     predicts (m/s), which only a policy that tracks the wind state can use
and, from the thesis results under <base>, the mean chain - threshold and chain - iid reward over
the capacity sweep in historical weather (run 2) and in the chain model's weather (run 3).

Usage (pvlib conda env, from SolarSimulator/):
    python Scripts/wind_persistence_by_site.py
    python Scripts/wind_persistence_by_site.py --months 6 7
"""
import argparse
import os

import numpy as np
import pandas as pd

PKG_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HIST_DIR = os.path.join(os.path.dirname(PKG_DIR), "Data", "HISTORICAL_DATA")
BASE = "C:/Users/bepstein8/sim_results"
FIT_YEARS = (1950, 2002)
SITES = {"": (30.0, -75.0), "latm20_lonm75": (-20.0, -75.0), "latm30_lonm75": (-30.0, -75.0),
         "lat0_lonm25": (0.0, -25.0), "latm30_lonm25": (-30.0, -25.0)}


def wind_stats(lat, lon, months):
    w = pd.read_pickle(os.path.join(HIST_DIR, f"data_{lat}_{lon}.pkl"))["wind_speed_10m"]
    w = w[(w.index.year >= FIT_YEARS[0]) & (w.index.year <= FIT_YEARS[1])]
    daily = w.groupby(w.index.date).mean()
    daily.index = pd.to_datetime(daily.index)
    doy = np.minimum(daily.index.dayofyear, 365)
    clim = daily.groupby(doy).mean()
    clim = pd.concat([clim.iloc[-15:], clim, clim.iloc[:15]]).rolling(31, center=True).mean().iloc[15:-15]
    anom = daily - clim.reindex(doy).values
    keep = daily.index.month.isin(months)
    a = anom.values
    pairs = lambda k: (a[:-k][keep[:-k]], a[k:][keep[:-k]])  # noqa: E731
    acf = {k: np.corrcoef(*pairs(k))[0, 1] for k in (1, 3)}
    sd = anom[keep].std()
    return {"mean_wind": w[w.index.month.isin(months)].mean(), "anom_sd": sd,
            "acf_1d": acf[1], "acf_3d": acf[3], "predictable_sd": sd * acf[1]}


def margins(base, tag):
    sfx = f"_{tag}" if tag else ""
    h = pd.read_csv(f"{base}/thesis_headline{sfx}/_analysis/comparison_cells.csv")
    h = h[(h.world == "hist") & (h.sweep == "capacity")]
    w = pd.read_csv(f"{base}/thesis_worlds{sfx}/_analysis/model_worlds_cells.csv")
    w = w[(w.world == "chainworld") & (w.sweep == "capacity")]
    return {"hist_chain_thresh": h.d_total_reward_chain_vs_thresh.mean(),
            "hist_chain_iid": h.d_total_reward.mean(),
            "chainworld_chain_thresh": w.d_reward_chain_vs_thresh.mean(),
            "chainworld_chain_iid": w.d_reward_chain_vs_iid.mean()}


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--base", default=BASE)
    ap.add_argument("--months", nargs="+", type=int, default=[6, 7],
                    help="Season window (default 6 7: the capacity sweep starts 10 June, 3000 "
                         "stages = 31 days).")
    args = ap.parse_args()
    rows = []
    for tag, (lat, lon) in SITES.items():
        rows.append({"site": f"lat{lat:g}_lon{lon:g}", **wind_stats(lat, lon, args.months),
                     **margins(args.base, tag)})
    df = pd.DataFrame(rows).sort_values("hist_chain_thresh", ascending=False)
    out = f"{args.base}/thesis_summary/sites"
    os.makedirs(out, exist_ok=True)
    df.round(3).to_csv(f"{out}/wind_persistence_by_site.csv", index=False)
    print(df.round(2).to_string(index=False))
    num = df.drop(columns="site")
    print("\nCorrelation with mean historical chain - threshold (capacity sweep, 5 sites):")
    print(num.corr()["hist_chain_thresh"].drop("hist_chain_thresh").round(2).to_string())
    print(f"[out] {out}/wind_persistence_by_site.csv")


if __name__ == "__main__":
    main()

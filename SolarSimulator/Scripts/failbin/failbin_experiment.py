#!/usr/bin/env python3
"""failbin_experiment.py -- wind-chain vs IID-optimal vs threshold validation,
comparing wind-space vs failure-space wind-bin edges.

Pipeline
--------
  provision : for each location, fetch historical weather (Open-Meteo, cached),
              build expected-data + histcube artifacts, build TWO wind-chain
              artifacts (wind-space equal-occupancy quantile bins, and
              failure-space equal-failure-mass bins), and emit the per-arm YAML
              configs.  Writes failbin_edges.json summarizing the bin edges.

Arms (per location), all evaluated on the SAME historical block-bootstrap weather:
  iid            optimal MDP solved assuming i.i.d. wind         (wind_chain off)
  chain_wind     chain-optimal, bins = wind terciles            (equal occupancy)
  chain_fail     chain-optimal, bins = equal expected-failure mass
  threshold      best-of-grid threshold policy                  (no solve)

Failure-space edges (equal-failure-mass)
----------------------------------------
Let f(w) = P(takeoff failure | wind w) from the configured transition model
(`moderate`: 1/(1.1+exp(8-0.5 w))).  Using the empirical 15-min historical wind
samples w_i as a draw from p(w), we place the two interior edges so each of the
three bins carries one third of the total expected failure risk
    M = sum_i f(w_i)  ~  integral f(w) p(w) dw.
This concentrates bin resolution where failure risk actually accumulates.  The
wind-space arm instead splits occupancy into thirds (wind terciles).
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
import yaml

PKG_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # .../SolarSimulator
if PKG_DIR not in sys.path:
    sys.path.insert(0, PKG_DIR)
REPO_DIR = os.path.dirname(PKG_DIR)

from Scripts.create_weather_distributions import (  # noqa: E402
    WeatherDataProcessor,
    build_wind_chain_artifact,
    build_historical_cube_artifact,
)
from BaseClasses.weather_processor_cs_normalization import build_expected_data_artifact  # noqa: E402
from BaseClasses.transition_model_base import ProbabilityModelFactory  # noqa: E402
from BaseClasses.run_sim import _derive_histcube_path  # noqa: E402

# ----------------------------------------------------------------------------------
# Experiment definition
# ----------------------------------------------------------------------------------
LOCATIONS = [
    dict(name="bering",     lat=65.0,  lon=-169.0, blurb="Bering Strait (cold, windy, persistent)"),
    dict(name="hawaii",     lat=21.0,  lon=-158.0, blurb="Hawaii offshore (steady trade winds)"),
    dict(name="florida",    lat=27.0,  lon=-79.5,  blurb="Florida Atlantic coast (calm, variable)"),
    dict(name="natlantic",  lat=45.0,  lon=-45.0,  blurb="N. Atlantic storm track (windy, storm persistence)"),
]

N_BINS = 3
INTERVAL_MIN = 15
TRANSITION_MODEL = "moderate"

# Full-scale experiment parameters (30-day mission, 10k episodes).
FULL = dict(
    horizon=30 * 24 * 60 // INTERVAL_MIN,   # 2880 stages
    episodes=10000,
    battery_capacities=[300.0],
    failure_penalties=[20.0],
    start_datetime="2025-06-10T00:00:00",
    threshold_values=[0.0, 0.1, 0.2, 0.3],
    wind_thresholds=[0.0, 4.0, 8.0, 12.0],
    block_length_days=7,
)
# Smoke parameters (fast end-to-end validation of the whole pipeline).
SMOKE = dict(
    horizon=2 * 24 * 60 // INTERVAL_MIN,    # 2 days = 192 stages
    episodes=50,
    battery_capacities=[300.0],
    failure_penalties=[20.0],
    start_datetime="2025-06-10T00:00:00",
    threshold_values=[0.0, 0.2],
    wind_thresholds=[0.0, 8.0],
    block_length_days=7,
)

DATA_DIR = os.path.join(REPO_DIR, "Data")
HIST_DIR = os.path.join(DATA_DIR, "HISTORICAL_DATA")
EXP_DIR = os.path.join(DATA_DIR, "EXPECTED_DATA")
CONFIG_DIR = os.path.join(REPO_DIR, "configs", "failbin_validation")


# ----------------------------------------------------------------------------------
# Path helpers (relative paths in configs so the harness resolves them from repo root)
# ----------------------------------------------------------------------------------
def _rel(p):
    return os.path.relpath(p, REPO_DIR).replace("\\", "/")


def loc_paths(loc):
    lat, lon = loc["lat"], loc["lon"]
    hist = os.path.join(HIST_DIR, f"data_{lat}_{lon}.pkl")
    exp = os.path.join(EXP_DIR, f"data_expected_lat{lat}_lon{lon}_15min.pkl")
    base, ext = os.path.splitext(exp)
    return dict(
        hist=hist,
        exp=exp,
        chain_wind=f"{base}_windchain_windspace{ext}",
        chain_fail=f"{base}_windchain_failspace{ext}",
        histcube=_derive_histcube_path(exp),
    )


# ----------------------------------------------------------------------------------
# Failure model f(w) and edge computation
# ----------------------------------------------------------------------------------
def takeoff_failure(wind, model_name=TRANSITION_MODEL):
    """f(w) = P(failure | takeoff from moored, wind w) for the configured model."""
    model = ProbabilityModelFactory.select_probability_model(model_name)
    wind = np.atleast_1d(np.asarray(wind, dtype=float))
    state = np.tile([100.0, 0.0], (len(wind), 1))   # moored, full battery
    action = np.ones(len(wind), dtype=int)          # takeoff
    success = np.asarray(model.compute_probability(wind, action, state), dtype=float)
    return 1.0 - success


def failure_mass_edges(wind_vals, n_bins=N_BINS):
    """Interior edges (len n_bins-1) splitting total expected failure mass into n_bins equal parts."""
    w = np.asarray(wind_vals, dtype=float)
    w = w[np.isfinite(w)]
    order = np.argsort(w)
    w_sorted = w[order]
    fw = takeoff_failure(w_sorted)
    cum = np.cumsum(fw)
    total = cum[-1]
    edges = []
    for k in range(1, n_bins):
        target = total * k / n_bins
        i = int(np.searchsorted(cum, target))
        i = min(i, len(w_sorted) - 1)
        edges.append(float(w_sorted[i]))
    # Enforce strictly increasing edges (guard against mass concentrated at a point).
    for k in range(1, len(edges)):
        if edges[k] <= edges[k - 1]:
            edges[k] = np.nextafter(edges[k - 1], np.inf)
    return edges, float(total)


def quantile_edges(wind_vals, n_bins=N_BINS):
    """Wind-space equal-occupancy interior edges (terciles)."""
    w = np.asarray(wind_vals, dtype=float)
    w = w[np.isfinite(w)]
    qs = np.linspace(0.0, 1.0, n_bins + 1)[1:-1]
    return [float(x) for x in np.quantile(w, qs)]


def bin_occupancy(wind_vals, interior_edges):
    w = np.asarray(wind_vals, dtype=float)
    w = w[np.isfinite(w)]
    idx = np.digitize(w, interior_edges)
    n = len(interior_edges) + 1
    return [float(np.mean(idx == b)) for b in range(n)]


# ----------------------------------------------------------------------------------
# Provisioning
# ----------------------------------------------------------------------------------
def resampled_wind(hist_pkl):
    """15-min historical wind series (matches how the chain artifacts bin wind)."""
    hist = pd.read_pickle(hist_pkl)
    hist = hist[~((hist.index.month == 2) & (hist.index.day == 29))]
    return hist["wind_speed_10m"].resample(f"{INTERVAL_MIN}min").interpolate(method="linear")


def ensure_historical(loc, paths):
    if os.path.exists(paths["hist"]):
        print(f"  [hist] cached: {_rel(paths['hist'])}")
        return
    os.makedirs(HIST_DIR, exist_ok=True)
    print(f"  [hist] fetching Open-Meteo 1950-2022 for lat={loc['lat']} lon={loc['lon']} ...")
    proc = WeatherDataProcessor()
    proc.fetch_weather_data(
        loc["lat"], loc["lon"], "1950-01-01", "2022-12-31",
        ["wind_speed_10m", "wind_direction_10m", "shortwave_radiation"],
    )
    proc.process_hourly_data()
    proc.hourly_dataframe.to_pickle(paths["hist"])
    print(f"  [hist] saved {_rel(paths['hist'])}")


def provision_location(loc):
    paths = loc_paths(loc)
    print(f"[{loc['name']}] {loc['blurb']}")
    ensure_historical(loc, paths)

    os.makedirs(EXP_DIR, exist_ok=True)
    if not os.path.exists(paths["exp"]):
        print(f"  [exp] building expected-data artifact ...")
        build_expected_data_artifact(paths["hist"], paths["exp"], loc["lat"], loc["lon"], INTERVAL_MIN)
    else:
        print(f"  [exp] cached")

    if not os.path.exists(paths["histcube"]):
        print(f"  [cube] building histcube ...")
        build_historical_cube_artifact(paths["hist"], paths["histcube"], interval_minutes=INTERVAL_MIN)
    else:
        print(f"  [cube] cached")

    wind = resampled_wind(paths["hist"]).values

    # Wind-space arm: equal-occupancy quantile bins.
    q_edges = quantile_edges(wind)
    print(f"  [wind-space] tercile edges (m/s) = {np.round(q_edges, 3)}")
    build_wind_chain_artifact(paths["hist"], paths["chain_wind"], interval_minutes=INTERVAL_MIN, n_bins=N_BINS)

    # Failure-space arm: equal-failure-mass bins.
    f_interior, total_mass = failure_mass_edges(wind)
    print(f"  [fail-space] equal-failure-mass edges (m/s) = {np.round(f_interior, 3)}")
    full_edges = np.concatenate(([0.0], np.asarray(f_interior, float), [np.inf]))
    build_wind_chain_artifact(paths["hist"], paths["chain_fail"], interval_minutes=INTERVAL_MIN,
                              bin_edges=full_edges)

    summary = dict(
        name=loc["name"], lat=loc["lat"], lon=loc["lon"], blurb=loc["blurb"],
        mean_wind=float(np.nanmean(wind)),
        wind_p50=float(np.nanpercentile(wind, 50)),
        wind_p90=float(np.nanpercentile(wind, 90)),
        wind_p99=float(np.nanpercentile(wind, 99)),
        wind_max=float(np.nanmax(wind)),
        windspace_edges=q_edges,
        windspace_occupancy=bin_occupancy(wind, q_edges),
        failspace_edges=f_interior,
        failspace_occupancy=bin_occupancy(wind, f_interior),
        total_failure_mass_per_sample=float(total_mass / np.isfinite(wind).sum()),
    )
    return summary


# ----------------------------------------------------------------------------------
# Config generation
# ----------------------------------------------------------------------------------
def _base_cfg(loc, paths, params, name):
    return dict(
        experiment_name=name,
        description=f"failbin validation: {name} @ {loc['blurb']}",
        start_datetime=params["start_datetime"],
        battery_capacities=params["battery_capacities"],
        horizons=[params["horizon"]],
        failure_penalties=params["failure_penalties"],
        episodes=params["episodes"],
        transition_model=TRANSITION_MODEL,
        solar_panel_model="constant",
        whale_series="real",
        energy_increment_wh=5,
        delta_t=INTERVAL_MIN,
        save_states=False,
        full_history_episodes=0,
        locations=[dict(latitude=loc["lat"], longitude=loc["lon"], data_path=_rel(paths["exp"]))],
        historical_weather=dict(enabled=True, block_length_days=params["block_length_days"]),
    )


def write_configs(loc, paths, params, out_dir, tag):
    os.makedirs(out_dir, exist_ok=True)
    written = []
    nm = loc["name"]

    # IID optimal (no chain), no threshold sims.
    c = _base_cfg(loc, paths, params, f"{tag}_{nm}_iid")
    c.update(include_optimal=True, threshold_values=[], wind_thresholds=[],
             wind_chain=dict(enabled=False))
    written.append(_dump(c, out_dir, f"{tag}_{nm}_iid.yaml"))

    # Chain optimal, wind-space bins.
    c = _base_cfg(loc, paths, params, f"{tag}_{nm}_chain_wind")
    c.update(include_optimal=True, threshold_values=[], wind_thresholds=[],
             wind_chain=dict(enabled=True, path=_rel(paths["chain_wind"]), n_bins=N_BINS))
    written.append(_dump(c, out_dir, f"{tag}_{nm}_chain_wind.yaml"))

    # Chain optimal, failure-space bins.
    c = _base_cfg(loc, paths, params, f"{tag}_{nm}_chain_fail")
    c.update(include_optimal=True, threshold_values=[], wind_thresholds=[],
             wind_chain=dict(enabled=True, path=_rel(paths["chain_fail"]), n_bins=N_BINS))
    written.append(_dump(c, out_dir, f"{tag}_{nm}_chain_fail.yaml"))

    # Threshold benchmark (grid; best-of reported downstream).
    c = _base_cfg(loc, paths, params, f"{tag}_{nm}_threshold")
    c.update(include_optimal=False,
             threshold_values=params["threshold_values"],
             wind_thresholds=params["wind_thresholds"],
             wind_chain=dict(enabled=False))
    written.append(_dump(c, out_dir, f"{tag}_{nm}_threshold.yaml"))
    return written


def _dump(cfg, out_dir, fname):
    path = os.path.join(out_dir, fname)
    with open(path, "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    return path


# ----------------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="failbin experiment provisioning + config generation.")
    ap.add_argument("--only", nargs="+", help="Restrict to these location names.")
    ap.add_argument("--smoke", action="store_true", help="Also emit smoke configs (1st location).")
    args = ap.parse_args()

    locs = LOCATIONS
    if args.only:
        locs = [l for l in LOCATIONS if l["name"] in args.only]

    summaries = []
    for loc in locs:
        s = provision_location(loc)
        summaries.append(s)
        paths = loc_paths(loc)
        write_configs(loc, paths, FULL, os.path.join(CONFIG_DIR, "full"), "full")
        if args.smoke and loc is locs[0]:
            write_configs(loc, paths, SMOKE, os.path.join(CONFIG_DIR, "smoke"), "smoke")

    os.makedirs(CONFIG_DIR, exist_ok=True)
    edges_path = os.path.join(CONFIG_DIR, "failbin_edges.json")
    with open(edges_path, "w") as f:
        json.dump(summaries, f, indent=2)
    print(f"\nEdge summary -> {_rel(edges_path)}")
    print(f"Configs -> {_rel(os.path.join(CONFIG_DIR, 'full'))}"
          + ("  (+ smoke/)" if args.smoke else ""))
    for s in summaries:
        print(f"  {s['name']:10s} mean_wind={s['mean_wind']:.1f}  "
              f"wind-edges={np.round(s['windspace_edges'],1)} occ={np.round(s['windspace_occupancy'],2)}  "
              f"fail-edges={np.round(s['failspace_edges'],1)} occ={np.round(s['failspace_occupancy'],2)}")


if __name__ == "__main__":
    main()

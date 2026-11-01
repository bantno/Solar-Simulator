"""Verification for the year-split hold-out (standard practice since 2026-10-24).

Checks:
  1. holdout_spec rules: default-on spec, the in-sample switches (false / null /
     enabled: false), custom ranges, reverse splits, and the errors for overlapping or
     backwards ranges, a bad rollout_years, and hold-out combined with explicit paths.
     Explicit paths without a holdout key stay in-sample (legacy hand-built artifacts).
  2. resolve_artifact_paths naming: in-sample paths are unchanged from the original
     pipeline; hold-out paths carry _fit<a>-<b> / _eval<a>-<b> tags and compose with the
     clear-sky suffix; rollout_years: fit points the cube at the fit years.
  3. Provisioning on synthetic data: the wind chain is fitted on the fit years only and
     the historical cube holds the evaluation years only; a year outside the record and
     a missing explicitly named artifact both raise instead of building.
  4. Real record (skipped when absent): the default split of the Gulf hourly record has
     whole 365-day years on each side (53 + 20 years).

Run (any cwd) with the pvlib conda env:
    conda run -n pvlib python SolarSimulator/Tests/verify_holdout.py
"""
import os
import sys
import tempfile

import numpy as np
import pandas as pd

TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
PKG_DIR = os.path.dirname(TESTS_DIR)
REPO_ROOT = os.path.dirname(PKG_DIR)
sys.path.insert(0, PKG_DIR)

from BaseClasses.run_sim import holdout_spec, resolve_artifact_paths, DEFAULT_HOLDOUT  # noqa: E402
from harness.run_experiment import _ensure_location_data, _year_slice_pkl  # noqa: E402


def check(name, cond):
    status = "PASS" if cond else "FAIL"
    print(f"[{status}] {name}")
    if not cond:
        sys.exit(1)


def raises(fn, exc=ValueError):
    try:
        fn()
    except exc:
        return True
    return False


def synthetic_hourly(years, lat_tz="Etc/GMT+6"):
    """Hourly record shaped like HISTORICAL_DATA: Feb 29 dropped, tz-aware index.
    Wind is ~2 m/s in years < 2002 and ~10 m/s from 2002 on, so a fit that leaks
    evaluation years shows up in the quantile edges."""
    idx = pd.date_range(f"{years[0]}-01-01", f"{years[-1]}-12-31 23:00", freq="h", tz=lat_tz)
    idx = idx[~((idx.month == 2) & (idx.day == 29))]
    rng = np.random.default_rng(0)
    base = np.where(idx.year < 2002, 2.0, 10.0)
    return pd.DataFrame({
        "wind_speed_10m": base + rng.uniform(0, 1, len(idx)),
        "wind_direction_10m": rng.uniform(0, 360, len(idx)),
        "shortwave_radiation": np.clip(500 * np.sin(np.pi * (idx.hour - 6) / 12), 0, None),
    }, index=idx)


def main():
    loc = {"latitude": 30.0, "longitude": -90.0,
           "data_path": "Data/EXPECTED_DATA/data_expected_lat30.0_lon-90.0_15min.pkl"}

    # ------------------------------------------------------------ 1. spec rules
    print("1. holdout_spec rules")
    d = holdout_spec({})
    check("no key -> default spec",
          d == {"fit_years": DEFAULT_HOLDOUT["fit_years"],
                "eval_years": DEFAULT_HOLDOUT["eval_years"], "rollout_years": "eval"})
    check("default split is 1950-2002 / 2003-2022",
          d["fit_years"] == [1950, 2002] and d["eval_years"] == [2003, 2022])
    check("holdout: false -> in-sample", holdout_spec({"holdout": False}) is None)
    check("holdout: null -> in-sample", holdout_spec({"holdout": None}) is None)
    check("enabled: false -> in-sample", holdout_spec({"holdout": {"enabled": False}}) is None)
    check("holdout: true -> default spec", holdout_spec({"holdout": True}) == d)
    custom = holdout_spec({"holdout": {"fit_years": [1979, 2004], "eval_years": [2005, 2022]}})
    check("custom ranges kept", custom["fit_years"] == [1979, 2004]
          and custom["eval_years"] == [2005, 2022])
    rev = holdout_spec({"holdout": {"fit_years": [1990, 2022], "eval_years": [1950, 1989]}})
    check("reverse split allowed", rev["eval_years"] == [1950, 1989])
    check("overlapping ranges rejected", raises(lambda: holdout_spec(
        {"holdout": {"fit_years": [1950, 2002], "eval_years": [2002, 2022]}})))
    check("backwards range rejected", raises(lambda: holdout_spec(
        {"holdout": {"fit_years": [2002, 1950]}})))
    check("bad rollout_years rejected", raises(lambda: holdout_spec(
        {"holdout": {"rollout_years": "all"}})))
    explicit = {"historical_weather": {"enabled": True, "path": "x_histcube.pkl"}}
    check("explicit path, no key -> in-sample (legacy)", holdout_spec(explicit) is None)
    check("explicit path + holdout enabled -> error",
          raises(lambda: holdout_spec({**explicit, "holdout": True})))
    check("explicit path + holdout: false -> in-sample",
          holdout_spec({**explicit, "holdout": False}) is None)

    # ------------------------------------------------------------ 2. path naming
    print("2. artifact path naming")
    p_in = resolve_artifact_paths({"holdout": False}, loc)
    check("in-sample paths unchanged",
          p_in["expected_data"] == loc["data_path"]
          and p_in["wind_chain"].endswith("_15min_windchain.pkl")
          and p_in["histcube"].endswith("_15min_histcube.pkl")
          and p_in["fit_years"] is None and p_in["cube_years"] is None)
    p_ho = resolve_artifact_paths({}, loc)
    check("hold-out expected data tagged with fit years",
          p_ho["expected_data"].endswith("_15min_fit1950-2002.pkl"))
    check("hold-out chains derive from the fit table (quantile chain named by bin count)",
          p_ho["wind_chain"].endswith("_15min_fit1950-2002_windchain_q3.pkl")
          and p_ho["solar_chain"].endswith("_15min_fit1950-2002_solarchain.pkl"))
    p_q12 = resolve_artifact_paths({"wind_chain": {"enabled": True, "n_bins": 12}}, loc)
    p_edges = resolve_artifact_paths(
        {"wind_chain": {"enabled": True, "bin_edges": [3.0, 6.0]}}, loc)
    check("bin counts get separate chain files; explicit edges keep the plain name",
          p_q12["wind_chain"].endswith("_15min_fit1950-2002_windchain_q12.pkl")
          and p_edges["wind_chain"].endswith("_15min_fit1950-2002_windchain.pkl"))
    check("hold-out cube tagged with eval years",
          p_ho["histcube"].endswith("_15min_eval2003-2022_histcube.pkl")
          and p_ho["cube_years"] == [2003, 2022])
    p_cs = resolve_artifact_paths({"clearsky_model": "ineichen"}, loc)
    check("clear-sky suffix composes with hold-out tags",
          p_cs["expected_data"].endswith("_15min_ineichen_fit1950-2002.pkl")
          and p_cs["histcube"].endswith("_15min_ineichen_eval2003-2022_histcube.pkl"))
    p_fit = resolve_artifact_paths({"holdout": {"rollout_years": "fit"}}, loc)
    check("rollout_years: fit -> cube on fit years",
          p_fit["histcube"].endswith("_15min_fit1950-2002_histcube.pkl")
          and p_fit["cube_years"] == [1950, 2002])
    p_apl = resolve_artifact_paths({
        "wind_chain": {"enabled": True, "path": "a_windchain.pkl"},
        "historical_weather": {"enabled": True, "path": "b_histcube.pkl"}}, loc)
    check("legacy explicit paths used as given",
          p_apl["wind_chain"] == "a_windchain.pkl" and p_apl["histcube"] == "b_histcube.pkl"
          and p_apl["holdout"] is None)

    # ------------------------------------------------------------ 3. provisioning
    print("3. provisioning on synthetic data")
    with tempfile.TemporaryDirectory(prefix="verify_holdout_") as root:
        exp_dir = os.path.join(root, "Data", "EXPECTED_DATA")
        hist_dir = os.path.join(root, "Data", "HISTORICAL_DATA")
        os.makedirs(exp_dir)
        os.makedirs(hist_dir)
        hist = synthetic_hourly(range(2000, 2004))
        hist.to_pickle(os.path.join(hist_dir, "data_30.0_-90.0.pkl"))
        sloc = dict(loc, data_path=os.path.join(exp_dir, "data_expected_lat30.0_lon-90.0_15min.pkl"))
        cfg = {"delta_t": 15,
               "holdout": {"fit_years": [2000, 2001], "eval_years": [2002, 2003]},
               "wind_chain": {"enabled": True, "n_bins": 3},
               "historical_weather": {"enabled": True}}
        paths = resolve_artifact_paths(cfg, sloc)
        # The expected-data fit (per-slot Weibull/Beta, pvlib) is slow and is the same
        # builder as in-sample; stand in a placeholder so this check exercises the
        # year routing of the chain and the cube.
        pd.to_pickle(pd.DataFrame(), paths["expected_data"])
        _ensure_location_data(cfg, sloc)
        chain = pd.read_pickle(paths["wind_chain"])
        cube = pd.read_pickle(paths["histcube"])
        check("wind chain fitted on fit years only (edges in the ~2-3 m/s regime)",
              float(chain["bin_edges"][-2]) < 4.0)
        check("cube holds the evaluation years only", list(cube["years"]) == [2002, 2003])
        check("cube values are evaluation-year wind (~10-11 m/s)",
              float(np.nanmin(cube["wind_cube"])) > 9.0)

        sl = _year_slice_pkl(os.path.join(hist_dir, "data_30.0_-90.0.pkl"), [2001, 2002], root)
        check("year slice keeps whole years",
              sorted(set(pd.read_pickle(sl).index.year)) == [2001, 2002]
              and len(pd.read_pickle(sl)) == 2 * 8760)
        check("year outside the record rejected", raises(lambda: _year_slice_pkl(
            os.path.join(hist_dir, "data_30.0_-90.0.pkl"), [2003, 2004], root)))

        missing_explicit = {"delta_t": 15, "holdout": False,
                            "historical_weather": {"enabled": True,
                                                   "path": os.path.join(exp_dir, "nope_histcube.pkl")}}
        pd.to_pickle(pd.DataFrame(), sloc["data_path"])
        check("missing explicit artifact raises instead of building",
              raises(lambda: _ensure_location_data(missing_explicit, sloc), FileNotFoundError))

    # ------------------------------------------------------------ 4. real record
    print("4. default split of the real Gulf record")
    real = os.path.join(REPO_ROOT, "Data", "HISTORICAL_DATA", "data_30_-90.pkl")
    if not os.path.exists(real):
        print("[SKIP] Data/HISTORICAL_DATA/data_30_-90.pkl not present")
    else:
        with tempfile.TemporaryDirectory(prefix="verify_holdout_real_") as tmp:
            fit = pd.read_pickle(_year_slice_pkl(real, DEFAULT_HOLDOUT["fit_years"], tmp))
            ev = pd.read_pickle(_year_slice_pkl(real, DEFAULT_HOLDOUT["eval_years"], tmp))
            check("fit slice is 53 whole years (1950-2002)",
                  len(fit) == 53 * 8760 and fit.index.year.min() == 1950
                  and fit.index.year.max() == 2002)
            check("eval slice is 20 whole years (2003-2022)",
                  len(ev) == 20 * 8760 and ev.index.year.min() == 2003
                  and ev.index.year.max() == 2022)
            check("slices are disjoint", fit.index.max() < ev.index.min())

    print("All hold-out checks passed.")


if __name__ == "__main__":
    main()

"""solar_persistence_precheck.py -- does the clear-sky index have memory worth modeling?

The solar analogue of wind_persistence_precheck.py, run BEFORE building the solar
Markov chain. The random variable is the clear-sky index

    K = GHI / G_cs,        G_cs = A * cos(apparent zenith),  A = 1150 W/m^2,

the same normalized variable the solver's per-stage Beta(alpha, beta) describes
(weather_processor_cs_normalization._fit_beta). K is only defined when the clear-sky
envelope is meaningful; slots with G_cs <= 50 W/m^2 (night / deep twilight -- the same
gate _fit_beta uses) are masked to NaN and drop out of every binned statistic
automatically (make_bins maps NaN to an invalid bin; consecutive_triples requires all
three bins valid and exact step spacing, so cross-night triples never form).

Two questions, at two timescales:

  1. INTRA-DAY (hour-to-hour): given the current index bin, does the previous bin add
     information about the next? Same (month, hour)-stratified conditional mutual
     information + permutation bias floor as the wind precheck. This decides whether a
     first-order chain suffices *within* the day, and the n-bin sweep picks the bin
     resolution.

  2. DAY-SCALE (the prize): multi-day cloud spells, not hour-to-hour flicker, drive
     battery-depletion failure. The chain carries day-to-day memory through a fitted
     dusk->dawn matrix (yesterday's last valid bin -> this morning's first valid bin).
     We report (a) that matrix's diagonal strength per month, (b) a month-stratified
     first- vs second-order test on the daily-regime series (does the day-before-
     yesterday help beyond yesterday?), and (c) ACF/AR(1) of daily-mean K.

Usage (pvlib conda env, from the SolarSimulator dir or repo root):

    python Scripts/solar_persistence_precheck.py \\
        --historical Data/HISTORICAL_DATA/data_30_-90.pkl \\
        --lat 30.0 --lon -90.0 \\
        --n-bins 3 \\
        --out-dir ./precheck_out

    # sweep bin resolutions (intra-day CMI vs n_bins):
    python Scripts/solar_persistence_precheck.py \\
        --historical Data/HISTORICAL_DATA/data_30_-90.pkl \\
        --lat 30.0 --lon -90.0 --sweep-bins 2 3 4 5

NOTE: like the wind precheck, run at native hourly resolution for the honest test;
--dt 15 interpolates and inflates short-lag persistence (the dawn-matrix / day-scale
numbers are unaffected by interpolation).
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
PKG_DIR = os.path.dirname(SCRIPTS_DIR)
if PKG_DIR not in sys.path:
    sys.path.insert(0, PKG_DIR)

from Scripts.wind_persistence_precheck import (  # noqa: E402
    autocorrelation,
    pacf_from_acf,
    make_bins,
    consecutive_triples,
    cmi_first_vs_second,
    cmi_stratified,
    permutation_bias_floor,
    chi2_sf,
)

LN2 = np.log(2.0)
A_CLEAR = 1150.0          # Fatemi-Kuh-Fripp clear-sky scale [W/m^2]
VALID_CS_WM2 = 50.0       # same validity gate as _fit_beta (clearsky > 50)
K_CLIP_HI = 0.999999      # same saturation clip as the Beta fitting pipeline


# --------------------------------------------------------------------------------------
# Clear-sky index series
# --------------------------------------------------------------------------------------
def fatemi_clearsky_series(index: pd.DatetimeIndex, lat: float, lon: float,
                           A: float = A_CLEAR) -> np.ndarray:
    """Vectorized Fatemi clear-sky GHI, A*max(cos(apparent zenith), 0), for a full index.

    (clearsky_ghi_fatemi in weather_processor_cs_normalization.py is scalar-only;
    this is its series counterpart, used for fitting/diagnostics.)
    """
    import pvlib
    sp = pvlib.solarposition.get_solarposition(index, lat, lon)
    cosz = np.cos(np.deg2rad(sp["apparent_zenith"].to_numpy()))
    return A * np.clip(cosz, 0.0, None)


def clearsky_index_series(hist: pd.DataFrame, lat: float, lon: float,
                          solar_col: str = "shortwave_radiation",
                          dt: int | None = None,
                          valid_threshold: float = VALID_CS_WM2) -> pd.Series:
    """Clear-sky index K on the analysis timestep; invalid (G_cs <= threshold) slots -> NaN.

    The hour-averaged GHI record biases K low near sunrise/sunset (part of the averaging
    window is dark), so a higher threshold (e.g. 200 W/m^2) anchors the day-scale
    dusk/dawn statistics at well-established daylight.
    """
    ghi = hist[solar_col].astype(float)
    if dt is not None:
        ghi = ghi.resample(f"{dt}min").interpolate(method="linear")
    idx = pd.DatetimeIndex(ghi.index)
    g_cs = fatemi_clearsky_series(idx, lat, lon)
    valid = g_cs > valid_threshold
    with np.errstate(divide="ignore", invalid="ignore"):
        k = np.where(valid, ghi.values / g_cs, np.nan)
    k = np.where(valid, np.clip(k, 0.0, K_CLIP_HI), np.nan)
    return pd.Series(k, index=idx, name="clearsky_index")


# --------------------------------------------------------------------------------------
# Stage-relative (per month/hour quantile) binning
# --------------------------------------------------------------------------------------
def stage_quantile_bins(K: pd.Series, n_bins: int) -> np.ndarray:
    """Bin each valid sample by its rank within its own (month, hour) slot population.

    This is the binning the solar chain actually uses: the runtime bin is the stage
    Beta's quantile band (bin masses exactly 1/n_bins), so the fit-time analogue is
    the empirical within-(month,hour) rank tercile. GLOBAL edges are unusable for
    solar: the hour-averaged GHI biases K low near sunrise/sunset (part of the
    averaging window is dark), so global bins degenerate at dusk/dawn and bottleneck
    the day-to-day channel; rank bins are comparable across hours by construction.

    Returns an int array aligned with K (invalid/NaN slots -> -1).
    """
    idx = pd.DatetimeIndex(K.index)
    out = np.full(K.size, -1, dtype=int)
    valid = ~np.isnan(K.values)
    slot = idx.month.values[valid] * 100 + idx.hour.values[valid]
    ranks = (
        pd.Series(K.values[valid])
        .groupby(slot)
        .rank(pct=True, method="average")
        .values
    )
    out[valid] = np.minimum((ranks * n_bins).astype(int), n_bins - 1)
    return out


# --------------------------------------------------------------------------------------
# Day-scale series (dusk / dawn / daily mean)
# --------------------------------------------------------------------------------------
def daily_regime_series(K: pd.Series):
    """Per calendar day: first-valid (dawn), last-valid (dusk), and mean of valid K.

    Returns a DataFrame indexed by normalized date with columns
    ['dawn_k', 'dusk_k', 'mean_k', 'n_valid'] (days with no valid slot dropped).
    """
    valid = K.dropna()
    if valid.empty:
        raise ValueError("no valid clear-sky-index samples")
    by_day = valid.groupby(valid.index.normalize())
    out = pd.DataFrame({
        "dawn_k": by_day.first(),
        "dusk_k": by_day.last(),
        "mean_k": by_day.mean(),
        "n_valid": by_day.size(),
    })
    return out


def fit_dusk_dawn_matrix(bins: np.ndarray, idx: pd.DatetimeIndex, n_bins: int):
    """Month-stratified dusk(d)-bin -> dawn(d+1)-bin transition -> (13, n_bins, n_bins).

    `bins` is the stage-relative bin series (-1 = invalid); dusk/dawn are the last/first
    valid bins of each calendar day.
    """
    valid = bins >= 0
    vdates = idx[valid].normalize()
    vbins = pd.Series(bins[valid], index=vdates)
    g = vbins.groupby(level=0)
    dusk = g.last()
    dawn = g.first()
    dates = dusk.index
    gap_days = np.diff(dates.values).astype("timedelta64[D]").astype(int)
    src = dusk.values[:-1]
    dst = dawn.values[1:]
    month = dates.month.values[1:]          # stratify by the DAWN day's month
    ok = gap_days == 1
    counts = np.zeros((13, n_bins, n_bins))
    np.add.at(counts, (month[ok], src[ok], dst[ok]), 1.0)
    row = counts.sum(axis=-1, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        P = np.where(row > 0, counts / row, np.nan)
    return P, counts


def day_scale_order_test(daily: pd.DataFrame, n_bins: int, n_perm: int, seed):
    """First- vs second-order Markov test on the daily-regime (daily-mean K bin) series.

    Consecutive-day triples (d-1, d, d+1), month-stratified (hour := 0). Answers: does
    the day before yesterday inform tomorrow's regime beyond yesterday's? This is the
    day-scale analogue of the intra-day CMI, i.e. whether ONE dusk->dawn matrix (first
    order at day scale) is the right complexity.
    """
    vals = daily["mean_k"].values
    bins, edges = make_bins(vals, n_bins, None)
    n_bins = len(edges) - 1
    idx = pd.DatetimeIndex(daily.index)
    # consecutive_triples wants exact step spacing in minutes: days are 1440 min apart.
    prev, curr, nxt, month, hour = consecutive_triples(bins, idx, 1440, n_bins)
    hour = np.zeros_like(hour)  # stratify by month only
    pooled = cmi_first_vs_second(prev, curr, nxt, n_bins)
    strat = cmi_stratified(prev, curr, nxt, month, hour, n_bins)
    if n_perm and prev.size:
        perm = permutation_bias_floor(prev, curr, nxt, month, hour, n_bins,
                                      strat["mi_bits"], n_perm=n_perm, seed=seed)
    else:
        perm = dict(bias_floor_bits=float("nan"), shuffle_std_bits=float("nan"),
                    mi_strat_corrected_bits=float("nan"), p_emp=float("nan"), n_perm=0)
    # First-order day-scale persistence itself: I(next ; curr) via entropy difference.
    c2 = np.zeros((n_bins, n_bins))
    np.add.at(c2, (curr, nxt), 1.0)
    N = c2.sum()
    p_n = c2.sum(axis=0) / N
    h_next = -np.sum(p_n[p_n > 0] * np.log(p_n[p_n > 0])) / LN2
    mi_first = h_next - pooled["h_next_given_curr_bits"]
    return dict(edges=edges, n_bins=n_bins, n_triples=int(prev.size),
                pooled=pooled, stratified=strat, perm=perm,
                h_next_bits=h_next, mi_first_order_bits=mi_first)


# --------------------------------------------------------------------------------------
# Analysis driver
# --------------------------------------------------------------------------------------
def analyze(historical, lat, lon, n_bins=3, bin_edges=None, dt=None,
            solar_col="shortwave_radiation", max_lag_days=21, n_perm=100, seed=None,
            valid_threshold=VALID_CS_WM2):
    """Full solar-persistence pre-check; returns a structured results dict."""
    hist = pd.read_pickle(historical)
    if not isinstance(hist.index, pd.DatetimeIndex):
        raise ValueError("historical pickle must have a DatetimeIndex.")
    hist = hist[~((hist.index.month == 2) & (hist.index.day == 29))]  # 365-day alignment

    K = clearsky_index_series(hist, lat, lon, solar_col=solar_col, dt=dt,
                              valid_threshold=valid_threshold)
    idx = K.index
    vals = K.values
    step_min = int(np.median(np.diff(idx.values).astype("timedelta64[m]").astype(int)))
    n_valid = int(np.isfinite(vals).sum())

    # --- 1. Intra-day binned first- vs second-order Markov (valid slots only) ---
    # Stage-relative (per month/hour rank-quantile) bins: the binning the chain uses.
    # bin_edges (global) kept for comparison runs only.
    if bin_edges is not None:
        full_edges = np.concatenate(([0.0], np.asarray(bin_edges, dtype=float), [np.inf]))
        bins, edges = make_bins(vals, n_bins, full_edges)
        n_bins = len(edges) - 1
    else:
        bins = stage_quantile_bins(K, n_bins)
        edges = None  # per-stage quantile bins have no single global edge array
    prev, curr, nxt, month, hour = consecutive_triples(bins, idx, step_min, n_bins)
    pooled = cmi_first_vs_second(prev, curr, nxt, n_bins)
    strat = cmi_stratified(prev, curr, nxt, month, hour, n_bins)
    if n_perm and n_perm > 0:
        perm = permutation_bias_floor(prev, curr, nxt, month, hour, n_bins,
                                      strat["mi_bits"], n_perm=n_perm, seed=seed)
    else:
        perm = dict(bias_floor_bits=float("nan"), shuffle_std_bits=float("nan"),
                    mi_strat_corrected_bits=float("nan"), p_emp=float("nan"), n_perm=0)

    # Pooled intra-day transition + occupancy (for reporting/plots).
    c2 = np.zeros((n_bins, n_bins))
    np.add.at(c2, (curr, nxt), 1.0)
    row = c2.sum(axis=1, keepdims=True)
    transition = np.divide(c2, row, out=np.full_like(c2, np.nan), where=row > 0)
    occupancy = np.bincount(curr, minlength=n_bins) / max(curr.size, 1)

    # --- 2. Day scale ---
    daily = daily_regime_series(K)
    dawn_P, dawn_counts = fit_dusk_dawn_matrix(bins, idx, n_bins)
    day_test = day_scale_order_test(daily, n_bins, n_perm, seed)

    dm = daily["mean_k"].values
    nlags = min(max_lag_days, dm.size // 4)
    acf_daily = autocorrelation(dm, nlags)
    pacf_daily = pacf_from_acf(acf_daily)
    n_days = int(np.isfinite(dm).sum())

    h_curr = pooled["h_next_given_curr_bits"]
    return {
        "historical": historical, "lat": lat, "lon": lon, "dt": dt,
        "valid_threshold": valid_threshold,
        "step_min": step_min, "n_samples": int(vals.size), "n_valid": n_valid,
        "year_min": int(idx.year.min()), "year_max": int(idx.year.max()),
        "mean_k": float(np.nanmean(vals)),
        "edges": edges, "n_bins": n_bins, "n_triples": int(prev.size),
        "pooled": pooled, "stratified": strat,
        "transition": transition, "occupancy": occupancy,
        "mi_pooled_bits": pooled["mi_bits"],
        "mi_strat_bits": strat["mi_bits"],
        "pct_entropy": 100.0 * strat["mi_bits"] / max(h_curr, 1e-9),
        "h_next_given_curr_bits": h_curr,
        "mi_strat_corrected_bits": perm["mi_strat_corrected_bits"],
        "bias_floor_bits": perm["bias_floor_bits"],
        "shuffle_std_bits": perm["shuffle_std_bits"],
        "p_emp": perm["p_emp"], "n_perm": perm["n_perm"],
        "pct_entropy_corrected":
            100.0 * perm["mi_strat_corrected_bits"] / max(h_curr, 1e-9),
        # day scale
        "daily": daily, "dawn_P": dawn_P, "dawn_counts": dawn_counts,
        "day_test": day_test,
        "acf_daily": acf_daily, "pacf_daily": pacf_daily, "n_days": n_days,
    }


def print_report(r):
    print(f"[data] {r['historical']}  (lat {r['lat']}, lon {r['lon']})")
    if r["dt"] is not None:
        print(f"[note] resampled to {r['dt']} min (interpolated) -- "
              "intra-day short-lag persistence inflated.")
    print(f"[data] {r['n_samples']} slots at ~{r['step_min']} min, "
          f"{r['year_min']}-{r['year_max']}; {r['n_valid']} solar-valid "
          f"(G_cs > {r.get('valid_threshold', VALID_CS_WM2):.0f} W/m^2), "
          f"mean K {r['mean_k']:.3f}")

    pooled, strat = r["pooled"], r["stratified"]
    print("\n=== 1. Intra-day Markov-order test on the clear-sky index ===")
    if r["edges"] is None:
        print(f"  bins                     : stage-relative rank quantiles "
              f"(n={r['n_bins']}, per month/hour; the chain's binning)")
    else:
        edges_disp = np.array(r["edges"], dtype=float).copy()
        edges_disp[-1] = 1.0  # top edge is 1.0 for an index in (0,1); inf only internal
        print(f"  bins (global K edges)    : {np.round(edges_disp, 3)}")
    print(f"  bin occupancy            : {np.round(r['occupancy'], 3)}")
    print(f"  pooled 1st-order matrix  :\n{np.round(r['transition'], 3)}")
    print(f"  valid consecutive triples: {r['n_triples']}")
    print("\n  -- pooled (NOT conditioned on month/hour) --")
    print(f"    I(Next ; Prev | Curr)        : {pooled['mi_bits']:.4f} bits")
    print(f"    H(Next | Curr)               : {pooled['h_next_given_curr_bits']:.4f} bits")
    print(f"    G^2 / dof                    : {pooled['g2']:.1f} / {pooled['dof']}  "
          f"(p={chi2_sf(pooled['g2'], pooled['dof']):.2e})")
    print("\n  -- (month, hour)-stratified  <-- FAIR test vs a (m,h)-conditioned chain --")
    print(f"    I(Next ; Prev | Curr, m, h)  : {strat['mi_bits']:.4f} bits   "
          f"({r['pct_entropy']:.1f}% of H(Next|Curr))   [raw, plug-in biased]")
    if r["n_perm"]:
        print(f"    bias floor (perm null)       : {r['bias_floor_bits']:.4f} "
              f"+/- {r['shuffle_std_bits']:.4f} bits   ({r['n_perm']} shuffles)")
        print(f"    bias-corrected CMI           : {r['mi_strat_corrected_bits']:.4f} bits   "
              f"({r['pct_entropy_corrected']:.1f}% of H(Next|Curr))   <-- drives verdict")
        print(f"    empirical p (shuffled >= obs): {r['p_emp']:.3f}")

    dt_ = r["day_test"]
    dpool, dstrat, dperm = dt_["pooled"], dt_["stratified"], dt_["perm"]
    print("\n=== 2. Day-scale regime persistence (the multi-day-spell channel) ===")
    print(f"  daily-mean-K bins (edges): {np.round(np.clip(dt_['edges'], 0, 1), 3)}")
    print(f"  consecutive-day triples  : {dt_['n_triples']}")
    print(f"  H(NextDay)               : {dt_['h_next_bits']:.4f} bits")
    print(f"  I(NextDay ; CurrDay)     : {dt_['mi_first_order_bits']:.4f} bits   "
          f"({100.0 * dt_['mi_first_order_bits'] / max(dt_['h_next_bits'], 1e-9):.1f}% "
          f"of H)  <-- FIRST-order day-scale persistence (what the dawn matrix carries)")
    print(f"  I(NextDay ; PrevDay | CurrDay, month) : {dstrat['mi_bits']:.4f} bits [raw]")
    if dperm["n_perm"]:
        print(f"  bias-corrected               : {dperm['mi_strat_corrected_bits']:.4f} bits "
              f"({100.0 * dperm['mi_strat_corrected_bits'] / max(dpool['h_next_given_curr_bits'], 1e-9):.1f}% "
              f"of H(Next|Curr))  <-- higher-order day memory beyond first order")
    acf_d = r["acf_daily"]
    efold = next((k for k in range(1, acf_d.size) if acf_d[k] < 1.0 / np.e), None)
    print(f"  daily-mean-K ACF(1)      : {acf_d[1]:.3f}   (AR(1) strength)")
    print(f"  daily-mean-K ACF e-fold  : "
          f"{efold} days" if efold else "  daily-mean-K ACF e-fold  : > window")
    print(f"  daily-mean-K PACF(2)     : {r['pacf_daily'][2]:.3f}   "
          f"(|CI|={1.96 / np.sqrt(max(r['n_days'], 1)):.3f})")

    months_with_data = [m for m in range(1, 13)
                        if np.isfinite(r["dawn_P"][m]).all()]
    if months_with_data:
        diags = np.array([np.diag(r["dawn_P"][m]) for m in months_with_data])
        print("\n  dusk->dawn matrix diagonals (persistence of yesterday's regime):")
        print(f"    mean over months         : {np.round(diags.mean(axis=0), 3)} "
              f"(per bin; 1/n_bins = {1.0 / r['n_bins']:.3f} would be memoryless)")
        m_lo, m_hi = months_with_data[int(diags.mean(axis=1).argmin())], \
            months_with_data[int(diags.mean(axis=1).argmax())]
        print(f"    weakest / strongest month: {m_lo} / {m_hi}")

    print("\n=== verdict ===")
    hi_intra = r["mi_strat_corrected_bits"] if r["n_perm"] else strat["mi_bits"]
    hi_day = (dperm["mi_strat_corrected_bits"] if dperm["n_perm"]
              else dstrat["mi_bits"])
    first_day = dt_["mi_first_order_bits"]
    pct_day = 100.0 * first_day / max(dt_["h_next_bits"], 1e-9)
    if pct_day >= 5.0:
        print(f"  Day-scale FIRST-order persistence is material ({first_day:.3f} bits, "
              f"{pct_day:.0f}% of daily entropy) -> the dusk->dawn channel is worth modeling.")
    else:
        print(f"  Day-scale first-order persistence is weak ({first_day:.3f} bits, "
              f"{pct_day:.0f}%) -> the solar chain may not pay; reconsider before building.")
    for name, mi in (("intra-day", hi_intra), ("day-scale", hi_day)):
        if not np.isfinite(mi) or mi < 0.005:
            print(f"  Higher-order {name} memory beyond first order: negligible "
                  f"({mi:.4f} bits) -> first order is the right complexity.")
        elif mi < 0.02:
            print(f"  Higher-order {name} memory: small ({mi:.4f} bits) -> first order "
                  "probably fine; note in docs.")
        else:
            print(f"  Higher-order {name} memory: substantial ({mi:.4f} bits) -> a deeper "
                  "state may be justified; investigate before committing to first order.")
    print("\n  NOTE: judge by bits / %-entropy, not p-values (huge N makes p ~ 0 always).")


def plot_day_scale(r, out_path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"[warn] plotting skipped ({e})")
        return None
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    acf_d, pacf_d = r["acf_daily"], r["pacf_daily"]
    lags = np.arange(acf_d.size)
    ci = 1.96 / np.sqrt(max(r["n_days"], 1))
    ax[0].stem(lags, acf_d, basefmt=" ")
    ax[0].axhline(0, color="k", lw=0.6)
    ax[0].axhspan(-ci, ci, color="tab:blue", alpha=0.12, label="95% CI (white noise)")
    ax[0].set_xlabel("lag [days]"); ax[0].set_ylabel("ACF")
    ax[0].set_title("Daily-mean clear-sky index: autocorrelation")
    ax[0].legend(loc="upper right")
    months = [m for m in range(1, 13) if np.isfinite(r["dawn_P"][m]).all()]
    diags = np.array([np.diag(r["dawn_P"][m]) for m in months])
    for b in range(diags.shape[1]):
        ax[1].plot(months, diags[:, b], marker="o", label=f"bin {b}")
    ax[1].axhline(1.0 / r["n_bins"], color="k", ls="--", lw=0.8, label="memoryless")
    ax[1].set_xlabel("month"); ax[1].set_ylabel("P(dawn bin = dusk bin)")
    ax[1].set_title("Dusk->dawn self-transition by month")
    ax[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    return out_path


# --------------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Solar (clear-sky index) persistence pre-check.")
    ap.add_argument("--historical", required=True, help="Path to hourly HISTORICAL_DATA pickle.")
    ap.add_argument("--lat", type=float, required=True, help="Latitude [deg].")
    ap.add_argument("--lon", type=float, required=True, help="Longitude [deg, east +].")
    ap.add_argument("--solar-col", default="shortwave_radiation")
    ap.add_argument("--bin-edges", type=float, nargs="+", metavar="K",
                    help="Interior K cutpoints in (0,1), e.g. 0.4 0.8. Overrides --n-bins.")
    ap.add_argument("--n-bins", type=int, default=3,
                    help="Equal-occupancy quantile bins over valid K (default 3).")
    ap.add_argument("--sweep-bins", type=int, nargs="+",
                    help="Run the intra-day + day-scale CMI for each of these bin counts "
                         "and print a summary table (e.g. --sweep-bins 2 3 4 5).")
    ap.add_argument("--dt", type=int, default=None,
                    help="Resample to this many minutes first (default: native hourly). "
                         "Interpolation inflates intra-day persistence; day-scale is safe.")
    ap.add_argument("--max-lag-days", type=int, default=21)
    ap.add_argument("--n-perm", type=int, default=100,
                    help="Permutation shuffles for the CMI bias floor (0 disables).")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--valid-threshold", type=float, default=VALID_CS_WM2,
                    help="G_cs validity gate in W/m^2 (default 50, matching _fit_beta). "
                         "Raise (e.g. 200) to exclude twilight slots whose hour-averaged "
                         "GHI biases K low and degenerates the dusk/dawn anchors.")
    ap.add_argument("--out-dir", default=".", help="Directory for figures.")
    args = ap.parse_args()

    if args.sweep_bins:
        rows = []
        for nb in args.sweep_bins:
            r = analyze(args.historical, args.lat, args.lon, n_bins=nb,
                        dt=args.dt, solar_col=args.solar_col,
                        max_lag_days=args.max_lag_days,
                        n_perm=args.n_perm, seed=args.seed,
                        valid_threshold=args.valid_threshold)
            d = r["day_test"]
            rows.append((nb, r["mi_strat_corrected_bits"], r["pct_entropy_corrected"],
                         d["mi_first_order_bits"],
                         100.0 * d["mi_first_order_bits"] / max(d["h_next_bits"], 1e-9),
                         d["perm"]["mi_strat_corrected_bits"]))
            print(f"[sweep] n_bins={nb} done")
        print("\n n_bins | intra-day 2nd-order CMI (bits, corr.) | % of H | "
              "day 1st-order MI (bits) | % of H | day 2nd-order CMI (bits, corr.)")
        for nb, mi2, pct2, mi1d, pct1d, mi2d in rows:
            print(f"   {nb:2d}   |            {mi2:8.4f}              | "
                  f"{pct2:5.1f} |        {mi1d:8.4f}        | {pct1d:5.1f} |     {mi2d:8.4f}")
        return

    r = analyze(args.historical, args.lat, args.lon, n_bins=args.n_bins,
                bin_edges=args.bin_edges, dt=args.dt, solar_col=args.solar_col,
                max_lag_days=args.max_lag_days, n_perm=args.n_perm, seed=args.seed,
                valid_threshold=args.valid_threshold)
    print_report(r)
    os.makedirs(args.out_dir, exist_ok=True)
    figp = plot_day_scale(r, os.path.join(args.out_dir, "solar_day_scale.png"))
    if figp:
        print(f"\n  day-scale figure -> {figp}")


if __name__ == "__main__":
    main()

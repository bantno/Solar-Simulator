"""
wind_persistence_precheck.py -- does the wind have memory beyond a first-order chain?

A diagnostic run BEFORE building any higher-order / history-augmented model. It answers
two separate questions about the historical wind series:

  1. (descriptive)  What is the timescale of wind persistence?
        -> continuous-wind autocorrelation (ACF) and partial autocorrelation (PACF).
           PACF is the key: if it collapses into the noise band after lag 1, an AR(1)-like
           (first-order) structure already captures the linear memory.

  2. (decision-relevant)  Given the wind *bin* you're in now, does the *previous* bin tell
     you anything more about the *next* bin?
        -> first-order vs second-order Markov test on the binned series, reported as
           conditional mutual information  I(Next ; Prev | Curr)  in bits, plus a G^2
           likelihood-ratio test.

     This is the question that decides whether the BI-chain (a first-order Markov chain on
     wind bins, conditioned on (month, hour)) is leaving structure on the table. Because the
     chain ALREADY conditions on (month, hour), we report two variants:
        - pooled:                I(Next ; Prev | Curr)
        - (month,hour)-stratified I(Next ; Prev | Curr, month, hour)   <-- the fair test
     The stratified version removes the diurnal/seasonal persistence the chain models, so
     any remaining mutual information is *genuine higher-order memory* the chain cannot see.

Binning mirrors Scripts/create_weather_distributions.fit_wind_transition_chain exactly:
quantile (equal-occupancy) bins via --n-bins, or explicit interior cutpoints via --bin-edges.

Usage (pvlib conda env, from the SolarSimulator dir or repo root):

    python Scripts/wind_persistence_precheck.py \\
        --historical Data/HISTORICAL_DATA/data_30_-90.pkl \\
        --n-bins 3 \\
        --out-dir ./precheck_out

    # mirror an aircraft-threshold chain:
    python Scripts/wind_persistence_precheck.py \\
        --historical Data/HISTORICAL_DATA/data_30_-90.pkl \\
        --bin-edges 5.0 10.0
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

LN2 = np.log(2.0)


# --------------------------------------------------------------------------------------
# Continuous-wind autocorrelation
# --------------------------------------------------------------------------------------
def autocorrelation(x: np.ndarray, nlags: int) -> np.ndarray:
    """Biased ACF estimator (acf[0] == 1) on a mean-removed series, lags 0..nlags."""
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    n = x.size
    x = x - x.mean()
    denom = np.dot(x, x)
    acf = np.empty(nlags + 1)
    for k in range(nlags + 1):
        acf[k] = np.dot(x[: n - k], x[k:]) / denom
    return acf


def pacf_from_acf(acf: np.ndarray) -> np.ndarray:
    """Partial autocorrelation via the Levinson-Durbin recursion. Returns pacf[0..nlags]."""
    nlags = acf.size - 1
    pacf = np.empty(nlags + 1)
    pacf[0] = 1.0
    phi = np.zeros((nlags + 1, nlags + 1))
    if nlags >= 1:
        phi[1, 1] = acf[1]
        pacf[1] = acf[1]
    for k in range(2, nlags + 1):
        num = acf[k] - np.dot(phi[k - 1, 1:k], acf[1:k][::-1])
        den = 1.0 - np.dot(phi[k - 1, 1:k], acf[1:k])
        rk = num / den if den != 0 else 0.0
        phi[k, k] = rk
        phi[k, 1:k] = phi[k - 1, 1:k] - rk * phi[k - 1, 1:k][::-1]
        pacf[k] = rk
    return pacf


# --------------------------------------------------------------------------------------
# Binned Markov-order analysis
# --------------------------------------------------------------------------------------
def make_bins(vals: np.ndarray, n_bins: int, bin_edges):
    """Return (bin_index_array, full_edges) mirroring fit_wind_transition_chain."""
    if bin_edges is None:
        qs = np.linspace(0.0, 1.0, n_bins + 1)[1:-1]
        interior = np.quantile(vals[~np.isnan(vals)], qs)
        edges = np.concatenate(([0.0], interior, [np.inf]))
    else:
        edges = np.asarray(bin_edges, dtype=float)
        n_bins = len(edges) - 1
    interior = edges[1:-1]
    bins = np.digitize(vals, interior)  # 0..n_bins-1 ; NaN -> n_bins
    bins = np.where(np.isnan(vals), -1, bins)
    return bins, edges


def consecutive_triples(bins: np.ndarray, idx: pd.DatetimeIndex, step_min: int, n_bins: int):
    """
    Return (prev, curr, nxt, month, hour) for every position where the three consecutive
    samples (i-1, i, i+1) are each exactly `step_min` apart and all bins are valid.
    month/hour are taken at the *current* step i (the source of the curr->next transition,
    matching how the chain keys its transition matrix).
    """
    step = np.diff(idx.values).astype("timedelta64[m]").astype(int)  # gap from i to i+1
    # position i (1..n-2): need gap(i-1->i)==step and gap(i->i+1)==step
    n = bins.size
    i = np.arange(1, n - 1)
    prev, curr, nxt = bins[i - 1], bins[i], bins[i + 1]
    ok = (
        (step[i - 1] == step_min) & (step[i] == step_min)
        & (prev >= 0) & (prev < n_bins)
        & (curr >= 0) & (curr < n_bins)
        & (nxt >= 0) & (nxt < n_bins)
    )
    sel = i[ok]
    return (prev[ok], curr[ok], nxt[ok],
            idx.month.values[sel], idx.hour.values[sel])


def cmi_first_vs_second(prev, curr, nxt, n_bins):
    """
    Conditional mutual information I(Next ; Prev | Curr) and the G^2 likelihood-ratio
    statistic for first- vs second-order Markov, from triple counts.

    Returns dict with: mi_bits, g2, dof, n, h_next_given_curr_bits, h_next_given_both_bits.
    """
    c3 = np.zeros((n_bins, n_bins, n_bins), dtype=np.float64)
    np.add.at(c3, (prev, curr, nxt), 1.0)
    N = c3.sum()
    if N == 0:
        return dict(mi_bits=0.0, g2=0.0, dof=0, n=0,
                    h_next_given_curr_bits=0.0, h_next_given_both_bits=0.0)

    c_pc = c3.sum(axis=2)                     # (prev, curr)
    c_cn = c3.sum(axis=0)                     # (curr, next)
    c_c = c3.sum(axis=(0, 2))                 # (curr,)

    g2 = 0.0
    mi_nats = 0.0
    for p in range(n_bins):
        for c in range(n_bins):
            if c_pc[p, c] == 0:
                continue
            for nx in range(n_bins):
                cnt = c3[p, c, nx]
                if cnt == 0:
                    continue
                p_both = cnt / c_pc[p, c]          # P(next | prev, curr)
                p_curr = c_cn[c, nx] / c_c[c]       # P(next | curr)
                ratio = p_both / p_curr
                g2 += 2.0 * cnt * np.log(ratio)
                mi_nats += (cnt / N) * np.log(ratio)

    # Conditional entropies of Next (bits) under the two models.
    def cond_entropy(joint_counts, cond_counts):
        h = 0.0
        it = np.ndindex(joint_counts.shape)
        for ix in it:
            cnt = joint_counts[ix]
            if cnt == 0:
                continue
            denom = cond_counts[ix[:-1]]
            ph = cnt / denom
            h -= (cnt / N) * np.log(ph) / LN2
        return h

    h_curr = cond_entropy(c_cn, c_c)               # H(Next | Curr)
    h_both = cond_entropy(c3, c_pc)                # H(Next | Prev, Curr)

    dof = n_bins * (n_bins - 1) * (n_bins - 1)     # 2nd-order minus 1st-order free params
    return dict(mi_bits=mi_nats / LN2, g2=g2, dof=dof, n=int(N),
                h_next_given_curr_bits=h_curr, h_next_given_both_bits=h_both)


def cmi_stratified(prev, curr, nxt, month, hour, n_bins):
    """
    (month, hour)-stratified I(Next ; Prev | Curr, month, hour): sum the per-stratum G^2 and
    MI. This is the FAIR test against the chain, which already conditions on (month, hour).
    """
    g2 = 0.0
    mi_bits_weighted = 0.0
    dof = 0
    N_total = prev.size
    n_strata = 0
    key = month.astype(int) * 24 + hour.astype(int)
    for k in np.unique(key):
        m = key == k
        if m.sum() < n_bins * n_bins:  # too sparse to estimate a 2nd-order table
            continue
        r = cmi_first_vs_second(prev[m], curr[m], nxt[m], n_bins)
        g2 += r["g2"]
        dof += r["dof"]
        mi_bits_weighted += r["mi_bits"] * r["n"]
        n_strata += 1
    mi_bits = mi_bits_weighted / N_total if N_total else 0.0
    return dict(mi_bits=mi_bits, g2=g2, dof=dof, n=N_total, n_strata=n_strata)


def chi2_sf(stat: float, dof: int) -> float:
    """Upper-tail p-value; uses scipy if available, else a normal (Wilson-Hilferty) approx."""
    try:
        from scipy.stats import chi2
        return float(chi2.sf(stat, dof))
    except Exception:
        if dof <= 0:
            return float("nan")
        import math
        z = ((stat / dof) ** (1.0 / 3.0) - (1.0 - 2.0 / (9.0 * dof))) / np.sqrt(2.0 / (9.0 * dof))
        return float(0.5 * math.erfc(z / np.sqrt(2.0)))


# --------------------------------------------------------------------------------------
# Plotting
# --------------------------------------------------------------------------------------
def plot_acf_pacf(acf, pacf, n_eff, step_min, out_path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"[warn] plotting skipped ({e})")
        return None
    lags = np.arange(acf.size)
    hrs = lags * step_min / 60.0
    ci = 1.96 / np.sqrt(n_eff)
    fig, ax = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
    ax[0].stem(hrs, acf, basefmt=" ")
    ax[0].axhline(0, color="k", lw=0.6)
    ax[0].axhspan(-ci, ci, color="tab:blue", alpha=0.12, label="95% CI (white noise)")
    ax[0].set_ylabel("ACF"); ax[0].legend(loc="upper right")
    ax[0].set_title("Continuous wind-speed autocorrelation (raw resolution)")
    ax[1].stem(hrs, pacf, basefmt=" ")
    ax[1].axhline(0, color="k", lw=0.6)
    ax[1].axhspan(-ci, ci, color="tab:orange", alpha=0.12)
    ax[1].set_ylabel("PACF"); ax[1].set_xlabel("lag [hours]")
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    return out_path


# --------------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Wind persistence / Markov-order pre-check.")
    ap.add_argument("--historical", required=True, help="Path to hourly HISTORICAL_DATA pickle.")
    ap.add_argument("--wind-col", default="wind_speed_10m")
    ap.add_argument("--bin-edges", type=float, nargs="+", metavar="M_S",
                    help="Interior cutpoints in m/s (e.g. 5.0 10.0). Overrides --n-bins.")
    ap.add_argument("--n-bins", type=int, default=3, help="Equal-occupancy quantile bins (default 3).")
    ap.add_argument("--max-lag-hours", type=int, default=72, help="Max ACF/PACF lag in hours.")
    ap.add_argument("--dt", type=int, default=None,
                    help="Resample to this many minutes BEFORE analysis (default: native "
                         "resolution). NOTE: resampling interpolates and inflates short-lag "
                         "persistence; leave unset for the honest test.")
    ap.add_argument("--out-dir", default=".", help="Directory for the ACF/PACF figure.")
    args = ap.parse_args()

    results = analyze(
        args.historical, n_bins=args.n_bins, bin_edges=args.bin_edges,
        dt=args.dt, wind_col=args.wind_col, max_lag_hours=args.max_lag_hours,
    )
    print_report(results)

    os.makedirs(args.out_dir, exist_ok=True)
    figp = plot_acf_pacf(
        results["acf"], results["pacf"], results["n_eff"], results["step_min"],
        os.path.join(args.out_dir, "wind_acf_pacf.png"),
    )
    if figp:
        print(f"\n  ACF/PACF figure -> {figp}")


def analyze(historical, n_bins=3, bin_edges=None, dt=None,
            wind_col="wind_speed_10m", max_lag_hours=72):
    """
    Run the full persistence pre-check and return a structured results dict.

    Importable single source of truth: the CLI and the visualization/sweep scripts all
    call this so the numbers and the figures can never drift apart.
    """
    hist = pd.read_pickle(historical)
    if not isinstance(hist.index, pd.DatetimeIndex):
        raise ValueError("historical pickle must have a DatetimeIndex.")
    hist = hist[~((hist.index.month == 2) & (hist.index.day == 29))]  # 365-day alignment
    w = hist[wind_col].astype(float)
    if dt is not None:
        w = w.resample(f"{dt}min").interpolate(method="linear")
    idx = pd.DatetimeIndex(w.index)
    vals = w.values
    step_min = int(np.median(np.diff(idx.values).astype("timedelta64[m]").astype(int)))

    # --- 1. Continuous ACF / PACF ---
    nlags = max(1, int(max_lag_hours * 60 / step_min))
    nlags = min(nlags, vals.size // 4)
    acf = autocorrelation(vals, nlags)
    pacf = pacf_from_acf(acf)
    n_eff = int(np.isfinite(vals).sum())
    ci = 1.96 / np.sqrt(n_eff)
    efold = next((k for k in range(1, nlags + 1) if acf[k] < 1.0 / np.e), None)
    pacf_cut = next((k for k in range(1, nlags + 1) if abs(pacf[k]) < ci), None)

    # --- 2. Binned first- vs second-order Markov ---
    bins, edges = make_bins(vals, n_bins, bin_edges)
    n_bins = len(edges) - 1
    prev, curr, nxt, month, hour = consecutive_triples(bins, idx, step_min, n_bins)
    pooled = cmi_first_vs_second(prev, curr, nxt, n_bins)
    strat = cmi_stratified(prev, curr, nxt, month, hour, n_bins)

    # Pooled first-order transition matrix P(next|curr) and bin occupancy, for plotting.
    c2 = np.zeros((n_bins, n_bins), dtype=np.float64)
    np.add.at(c2, (curr, nxt), 1.0)
    row = c2.sum(axis=1, keepdims=True)
    transition = np.divide(c2, row, out=np.full_like(c2, np.nan), where=row > 0)
    occupancy = np.bincount(curr, minlength=n_bins) / max(curr.size, 1)

    h_curr = pooled["h_next_given_curr_bits"]
    return {
        "historical": historical,
        "wind_col": wind_col, "dt": dt,
        "step_min": step_min, "n_samples": int(vals.size),
        "year_min": int(idx.year.min()), "year_max": int(idx.year.max()),
        "mean_wind": float(np.nanmean(vals)),
        "acf": acf, "pacf": pacf, "n_eff": n_eff, "ci": ci,
        "efold": efold, "pacf_cut": pacf_cut, "nlags": nlags,
        "edges": edges, "n_bins": n_bins, "n_triples": int(prev.size),
        "pooled": pooled, "stratified": strat,
        "transition": transition, "occupancy": occupancy,
        # headline scalars for sweeps:
        "mi_pooled_bits": pooled["mi_bits"],
        "mi_strat_bits": strat["mi_bits"],
        "pct_entropy": 100.0 * strat["mi_bits"] / max(h_curr, 1e-9),
        "h_next_given_curr_bits": h_curr,
    }


def print_report(r):
    """Pretty-print the analyze() results (mirrors the original CLI output)."""
    print(f"[data] {r['historical']}")
    if r["dt"] is not None:
        print(f"[note] resampled to {r['dt']} min (interpolated) -- short-lag persistence inflated.")
    print(f"[data] {r['n_samples']} samples at ~{r['step_min']} min, "
          f"{r['year_min']}-{r['year_max']}, mean wind {r['mean_wind']:.2f} m/s")

    acf, pacf, step_min = r["acf"], r["pacf"], r["step_min"]
    print("\n=== 1. Continuous-wind autocorrelation (descriptive) ===")
    efold = r["efold"]
    print(f"  ACF e-folding time      : "
          f"{efold} steps ({efold * step_min / 60.0:.1f} h)" if efold
          else "  ACF e-folding time      : > window")
    print(f"  ACF(lag 1)              : {acf[1]:.3f}")
    print(f"  PACF(lag 1)             : {pacf[1]:.3f}")
    print(f"  PACF(lag 2)             : {pacf[2]:.3f}   (|CI|={r['ci']:.3f})")
    if r["pacf_cut"] is not None:
        k = r["pacf_cut"]
        print(f"  PACF enters noise band  : lag {k} ({k * step_min / 60.0:.1f} h)")
        print(f"  -> linear memory beyond lag 1: {'YES' if k > 2 else 'no (≈AR(1))'}")

    pooled, strat = r["pooled"], r["stratified"]
    print("\n=== 2. Binned Markov-order test (does Prev help beyond Curr?) ===")
    print(f"  bins (m/s edges)        : {np.round(r['edges'], 2)}")
    print(f"  valid consecutive triples: {r['n_triples']}")
    print("\n  -- pooled (NOT conditioned on month/hour) --")
    print(f"    I(Next ; Prev | Curr)        : {pooled['mi_bits']:.4f} bits")
    print(f"    H(Next | Curr)               : {pooled['h_next_given_curr_bits']:.4f} bits")
    print(f"    H(Next | Curr, Prev)         : {pooled['h_next_given_both_bits']:.4f} bits")
    print(f"    G^2 / dof                    : {pooled['g2']:.1f} / {pooled['dof']}  "
          f"(p={chi2_sf(pooled['g2'], pooled['dof']):.2e})")
    print("    (this mixes in diurnal/seasonal memory the chain already models)")
    print("\n  -- (month, hour)-stratified  <-- FAIR test vs BI-chain --")
    print(f"    I(Next ; Prev | Curr, m, h)  : {strat['mi_bits']:.4f} bits   "
          f"({r['pct_entropy']:.1f}% of H(Next|Curr))")
    print(f"    G^2 / dof                    : {strat['g2']:.1f} / {strat['dof']}  "
          f"(p={chi2_sf(strat['g2'], strat['dof']):.2e}, {strat['n_strata']} strata)")

    print("\n=== verdict ===")
    mi = strat["mi_bits"]
    if mi < 0.005:
        print(f"  Genuine higher-order memory is negligible ({mi:.4f} bits beyond the chain).")
        print("  -> A first-order (month,hour)-conditioned chain is ~sufficient; a history-")
        print("     augmented model is unlikely to beat BI-chain. That is itself a clean result.")
    elif mi < 0.02:
        print(f"  Small residual higher-order memory ({mi:.4f} bits). A history-augmented model")
        print("  may yield a marginal gain over BI-chain; weigh against added complexity.")
    else:
        print(f"  Substantial higher-order memory ({mi:.4f} bits beyond Curr, within month/hour).")
        print("  -> A history-augmented (k-step) state is justified; BI-chain likely leaves")
        print("     persistence structure on the table. Proceed with the richer-state learner.")
    print("\n  NOTE: statistical significance (tiny p) is near-certain with this much data; judge")
    print("  PRACTICAL importance by the bits / %-entropy-reduction, not by p alone.")


if __name__ == "__main__":
    main()

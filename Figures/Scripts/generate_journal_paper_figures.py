#!/usr/bin/env python3
"""
Journal-paper figure generator.

Single entry point that re-renders every figure used in the journal paper
into ``Figures/Journal_Paper_Figures/`` with one uniform style
(sans-serif, >=8 pt fonts, 1.0 pt data lines, 0.75 pt axes, dpi=600),
rendered at final print width so LaTeX does not rescale fonts.

Usage::

    python Figures/Scripts/generate_journal_paper_figures.py            # render all
    python Figures/Scripts/generate_journal_paper_figures.py --only fig_threshold_sweep_combined
    python Figures/Scripts/generate_journal_paper_figures.py --list     # print figure names

Figures whose generating scripts were not found in the original code base
(iso, locations, failure_probability, Whale_observation_probability,
wind_month) are intentionally omitted; the user regenerates those
separately.
"""
from __future__ import annotations

import argparse
import calendar
import os
import re
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import h5py
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import AutoMinorLocator, MaxNLocator, PercentFormatter
from scipy.special import gamma

# ---------------------------------------------------------------------------
# Make sibling helpers importable when running from any cwd.
# ---------------------------------------------------------------------------
_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

_SOLAR_SIM_PKG = _THIS_DIR.parent.parent / "SolarSimulator"
if str(_SOLAR_SIM_PKG) not in sys.path:
    sys.path.insert(0, str(_SOLAR_SIM_PKG))

from results_io import load_summary as _load_summary_raw, savefig_all_formats  # noqa: E402
from BaseClasses.transition_model_base import ProbabilityModelFactory  # noqa: E402
from BaseClasses.whale_base import WhaleRewardSeriesFactory  # noqa: E402
import contextlib
import io


def load_summary(paths):
    """Wrapper around results_io.load_summary that silences its debug print."""
    with contextlib.redirect_stdout(io.StringIO()):
        return _load_summary_raw(paths)


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO_ROOT = _THIS_DIR.parent.parent
FIGURES_DIR = REPO_ROOT / "Figures"
RESULTS_DIR = FIGURES_DIR / "Results"
OUT_DIR = FIGURES_DIR / "Journal_Paper_Figures"

# Canonical HDF5 / NPY artifacts used by each figure.
RESULTS_PATHS = {
    "threshold_h5": RESULTS_DIR / "ThresholdSweep" /
        "observation_and_windspeed_threshold_config_20250917_142509.h5",
    "threshold_value_npy": RESULTS_DIR / "ThresholdSweep" /
        "observation_and_windspeed_threshold_config_300.0Wh_3000h_5.0p_2025-06-10 0.npy",

    "battery_h5": RESULTS_DIR / "BatterySweep" /
        "battery_sweep_config_20250915_140523.h5",

    "duration_h5": RESULTS_DIR / "DurationSweep" /
        "combined_duration_sweep_filtered2.h5",
    "duration_value_npy_short": RESULTS_DIR / "DurationSweep" /
        "duration_sweep_config_300.0Wh_3000h_20.0p_2025-01-01 0.npy",
    "duration_value_npy_long": RESULTS_DIR / "DurationSweep" /
        "duration_sweep_config 2_300.0Wh_18000h_20.0p_2025-01-01 0.npy",

    "failure_h5": RESULTS_DIR / "FailurePenaltySweep" /
        "failure_penalty_sweep_config_20250915_164437.h5",

    "loc_capacity_h5": RESULTS_DIR / "LocationCapacitySweep" /
        "location_threshold_comparison_config_20250912_022625.h5",

    "wind_solar_data_dir": REPO_ROOT / "Data" / "EXPECTED_DATA",
}


# ---------------------------------------------------------------------------
# Universal paper style
# ---------------------------------------------------------------------------
PAPER_DPI = 1200

PAPER_STYLE = {
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "mathtext.fontset": "dejavusans",

    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,

    "axes.linewidth": 0.75,
    "xtick.major.width": 0.75,
    "ytick.major.width": 0.75,
    "xtick.minor.width": 0.5,
    "ytick.minor.width": 0.5,
    "lines.linewidth": 1.0,
    "lines.markersize": 3.0,

    "figure.dpi": PAPER_DPI,
    "savefig.dpi": PAPER_DPI,
    "savefig.bbox": "tight",

    "legend.frameon": True,
    "legend.framealpha": 0.9,
    "legend.edgecolor": "black",

    "axes.grid": True,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
}


def apply_paper_style() -> None:
    mpl.rcParams.update(PAPER_STYLE)


# ---------------------------------------------------------------------------
# Common helpers
# ---------------------------------------------------------------------------
SINGLE_COL_IN = 3.25
DOUBLE_COL_IN = 6.5
DT_MINUTES = 15  # all sweeps use 15-minute stages


def _ensure_outdir() -> Path:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    return OUT_DIR


def _save(fig: plt.Figure, basename: str, formats: Sequence[str] = ("png",)) -> None:
    outdir = _ensure_outdir()
    savefig_all_formats(fig, str(outdir), basename, formats=formats, dpi=PAPER_DPI)
    plt.close(fig)
    print(f"[OK] {outdir / basename}.png")


PANEL_LABEL_X = -0.18  # axes-fraction; shared by panel label and y-axis label


def _panel_label(ax: plt.Axes, label: str) -> None:
    """Place a bottom-left panel label aligned horizontally with the y-axis
    label (both sit at PANEL_LABEL_X in axes coords)."""
    ax.yaxis.set_label_coords(PANEL_LABEL_X, 0.5)
    ax.text(
        PANEL_LABEL_X, -0.08, label,
        transform=ax.transAxes,
        ha="center", va="top",
        fontsize=9, fontweight="bold",
    )


def _legend_label(obs: float, wind: float) -> str:
    return rf"$w_{{to}}={wind},\ O_{{th}}={obs}$"


def _split_optimal(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    is_opt = df["sim_type"].astype(str).str.contains("optimal", case=False, na=False)
    return df[~is_opt].copy(), df[is_opt].copy()


def _normalize_percent(df: pd.DataFrame, col: str = "failure_percentage") -> pd.DataFrame:
    if col in df.columns:
        vals = pd.to_numeric(df[col], errors="coerce")
        if not vals.dropna().empty and vals.dropna().max() <= 1.01:
            df = df.copy()
            df[col] = vals * 100.0
    return df


def _filter(
    df: pd.DataFrame,
    *,
    capacities: Optional[Sequence[float]] = None,
    obs_thresholds: Optional[Sequence[float]] = None,
    wind_thresholds: Optional[Sequence[float]] = None,
    penalties: Optional[Sequence[float]] = None,
    locations: Optional[Sequence[Tuple[float, float]]] = None,
) -> pd.DataFrame:
    out = df
    if capacities:
        out = out[out["battery_capacity"].isin(capacities)]
    if obs_thresholds:
        out = out[out["observation_threshold"].isin(obs_thresholds)]
    if wind_thresholds:
        out = out[out["wind_threshold"].isin(wind_thresholds)]
    if penalties and "failure_penalty" in out.columns:
        out = out[out["failure_penalty"].isin(penalties)]
    if locations:
        mask = False
        for (lat, lon) in locations:
            mask = mask | ((out["latitude"] == lat) & (out["longitude"] == lon))
        out = out[mask]
    return out


def _check_path(p: Path, what: str) -> None:
    if not p.exists():
        raise FileNotFoundError(f"Missing {what}: {p}")


# Re-used npy block layout (shared by surface_plot_alternatives & threshold_plots)
def _split_value_blocks(data: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    rows, _T = data.shape
    n_soc = rows // 2 if rows % 2 == 0 else (rows - 1) // 2
    moored = data[:n_soc, :]
    flying = data[n_soc:2 * n_soc, :]
    soc = np.linspace(0, 100, n_soc)
    return moored, flying, soc


def _infer_start_dt(path: Path) -> datetime:
    pat = r"(?P<date>\d{4}-\d{2}-\d{2})(?:[ _](?P<hour>\d{1,2})(?::(?P<minute>\d{2}))?)?"
    m = re.search(pat, path.as_posix())
    if not m:
        return datetime(2000, 1, 1)
    y, mo, dd = map(int, m.group("date").split("-"))
    hh = int(m.group("hour") or 0)
    mm = int(m.group("minute") or 0)
    return datetime(y, mo, dd, hh, mm)


def _compute_thresholds(moored: np.ndarray, flying: np.ndarray, soc: np.ndarray
                        ) -> Tuple[np.ndarray, np.ndarray]:
    _n_soc, T = moored.shape
    takeoff = np.full(T, np.nan)
    cliff = np.full(T, np.nan)
    jump_threshold = 1.0
    soc_cutoff = 20.0
    for t in range(T):
        diff = flying[:, t] - moored[:, t]
        idx = np.where(diff > 0)[0]
        if idx.size:
            takeoff[t] = soc[idx[0]]
        delta = np.diff(flying[:, t])
        mask = soc[:-1] < soc_cutoff
        idx = np.where((delta > jump_threshold) & mask)[0]
        if idx.size:
            cliff[t] = soc[idx[0] + 1]
    return takeoff, cliff


# ---------------------------------------------------------------------------
# Figure functions
# ---------------------------------------------------------------------------
# Stubs below — each is implemented in its own commit/turn.
PAPER_LOCATIONS: Sequence[Tuple[float, float]] = (
    (20.0, -159.0),
    (30.0, -75.0),
    (35.0, 14.0),
    (40.0, 138.0),
    (58.0, -161.0),
)


def _load_paper_location_pkls() -> List[Tuple[Tuple[float, float], pd.DataFrame]]:
    data_dir = RESULTS_PATHS["wind_solar_data_dir"]
    _check_path(data_dir, "EXPECTED_DATA directory")
    out = []
    for (lat, lon) in PAPER_LOCATIONS:
        # The pkl filenames stored as 30.0 / -75.0 etc.
        fname = f"data_expected_lat{lat}_lon{lon}_15min.pkl"
        path = data_dir / fname
        if not path.exists():
            raise FileNotFoundError(f"Expected pkl missing: {path}")
        out.append(((lat, lon), pd.read_pickle(path)))
    return out


def _location_label(lat: float, lon: float) -> str:
    lat_h = "N" if lat >= 0 else "S"
    lon_h = "E" if lon >= 0 else "W"
    return f"{abs(lat):.0f}°{lat_h}, {abs(lon):.0f}°{lon_h}"


def fig_total_monthly_solar_energy() -> None:
    months = list(range(1, 13))
    tick_months = months[::2]
    tick_labels = [calendar.month_abbr[m] for m in tick_months]
    dt_seconds = DT_MINUTES * 60.0

    fig, ax = plt.subplots(figsize=(SINGLE_COL_IN, 2.8))
    for (lat, lon), df in _load_paper_location_pkls():
        monthly = df.groupby("month")["expected_solar_rad"].sum() * dt_seconds / 1e6
        ax.plot(monthly.index, monthly.values, marker=".",
                label=_location_label(lat, lon))
    ax.set_xlabel("Month")
    ax.set_ylabel(r"Monthly Insolation (MJ/m$^2$)")
    ax.set_xticks(tick_months)
    ax.set_xticklabels(tick_labels)
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.30),
        ncol=2,
        borderaxespad=0.0,
    )
    fig.subplots_adjust(top=0.96, bottom=0.42, left=0.18, right=0.97)
    _save(fig, "total_monthly_solar_energy")


WIND_MONTHS_LOCATION: Tuple[float, float] = (30.0, -75.0)
WIND_MONTHS_PANELS: Tuple[int, int, int, int] = (1, 4, 7, 10)
WIND_MONTHS_SIGMA: float = 3.0


def _weibull_std(k: np.ndarray, lam: np.ndarray) -> np.ndarray:
    k = np.clip(np.asarray(k, float), 1e-6, None)
    lam = np.clip(np.asarray(lam, float), 1e-12, None)
    g1 = gamma(1.0 + 1.0 / k)
    g2 = gamma(1.0 + 2.0 / k)
    return np.sqrt(np.maximum((lam ** 2) * (g2 - g1 ** 2), 0.0))


def fig_wind_months_mean_plus_3sigma() -> None:
    lat, lon = WIND_MONTHS_LOCATION
    pkl = RESULTS_PATHS["wind_solar_data_dir"] / (
        f"data_expected_lat{lat}_lon{lon}_15min.pkl"
    )
    _check_path(pkl, "wind months pkl")
    df = pd.read_pickle(pkl)

    ref_year = 2001
    dt_idx = pd.to_datetime(
        dict(
            year=ref_year,
            month=df["month"].astype(int),
            day=df["day"].astype(int),
            hour=df["hour"].astype(int),
            minute=df["minute"].astype(int),
        )
    )
    df = df.assign(_dt=dt_idx).sort_values("_dt").set_index("_dt")
    mu = df["expected_wind_speed"].to_numpy()
    sd = _weibull_std(df["weibull_k"].to_numpy(), df["weibull_scale"].to_numpy())
    df = df.assign(_mu=mu, _sd=sd)

    sigma = WIND_MONTHS_SIGMA
    panels = WIND_MONTHS_PANELS

    fig, axes = plt.subplots(
        len(panels), 1, figsize=(SINGLE_COL_IN, 6.4),
        sharey=True,
        gridspec_kw={"hspace": 0.35},
    )

    for i, (ax, m) in enumerate(zip(axes, panels)):
        sub = df[df.index.month == m]
        if sub.empty:
            ax.text(0.5, 0.5, f"No data for month {m}",
                    transform=ax.transAxes, ha="center", va="center")
            continue

        upper = sub["_mu"].to_numpy() + sigma * sub["_sd"].to_numpy()
        lower = np.clip(sub["_mu"].to_numpy() - sigma * sub["_sd"].to_numpy(), 0.0, None)

        h_mean, = ax.plot(sub.index, sub["_mu"].to_numpy(),
                          linewidth=1.0, color="C0", label="Mean")
        h_band = ax.fill_between(
            sub.index, lower, upper,
            alpha=0.22, color="C0", linewidth=0,
            label=rf"$\pm{int(sigma)}\sigma$",
        )

        ax.set_ylim(0, 20)
        ax.set_ylabel("Wind (m/s)")
        ax.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=3, maxticks=6))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))

        if i == 0:
            ax.legend(
                [h_mean, h_band], ["Mean", rf"$\pm{int(sigma)}\sigma$"],
                loc="upper right", bbox_to_anchor=(1.0, 0.75),
                ncol=2, fontsize=8,
                frameon=True, framealpha=0.9, edgecolor="black",
            )

    axes[-1].set_xlabel("Date")
    fig.subplots_adjust(top=0.97, left=0.22, right=0.97, bottom=0.07)
    _save(fig, "fig_wind_months_mean_plus_3sigma")


def _beta_std(alpha: np.ndarray, beta: np.ndarray, scale: np.ndarray) -> np.ndarray:
    a = np.clip(np.asarray(alpha, float), 1e-12, None)
    b = np.clip(np.asarray(beta, float), 1e-12, None)
    s = np.asarray(scale, float)
    var_unit = (a * b) / (((a + b) ** 2) * (a + b + 1.0))
    return s * np.sqrt(np.maximum(var_unit, 0.0))


def fig_solar_months_mean_plus_3sigma() -> None:
    lat, lon = WIND_MONTHS_LOCATION
    pkl = RESULTS_PATHS["wind_solar_data_dir"] / (
        f"data_expected_lat{lat}_lon{lon}_15min.pkl"
    )
    _check_path(pkl, "solar months pkl")
    df = pd.read_pickle(pkl)

    ref_year = 2001
    dt_idx = pd.to_datetime(
        dict(
            year=ref_year,
            month=df["month"].astype(int),
            day=df["day"].astype(int),
            hour=df["hour"].astype(int),
            minute=df["minute"].astype(int),
        )
    )
    df = df.assign(_dt=dt_idx).sort_values("_dt").set_index("_dt")
    mu = df["expected_solar_rad"].to_numpy()
    sd = _beta_std(
        df["beta_alpha"].to_numpy(),
        df["beta_beta"].to_numpy(),
        df["clearsky_irradiance"].to_numpy(),
    )
    df = df.assign(_mu=mu, _sd=sd)

    sigma = WIND_MONTHS_SIGMA
    panels = WIND_MONTHS_PANELS

    fig, axes = plt.subplots(
        len(panels), 1, figsize=(SINGLE_COL_IN, 6.4),
        sharey=True,
        gridspec_kw={"hspace": 0.35},
    )

    for i, (ax, m) in enumerate(zip(axes, panels)):
        sub = df[df.index.month == m]
        if sub.empty:
            ax.text(0.5, 0.5, f"No data for month {m}",
                    transform=ax.transAxes, ha="center", va="center")
            continue

        mu_v = sub["_mu"].to_numpy()
        sd_v = sub["_sd"].to_numpy()
        cs = sub["clearsky_irradiance"].to_numpy()
        upper = np.clip(mu_v + sigma * sd_v, 0.0, cs)
        lower = np.clip(mu_v - sigma * sd_v, 0.0, None)

        h_mean, = ax.plot(sub.index, mu_v,
                          linewidth=1.0, color="C0", label="Mean")
        h_band = ax.fill_between(
            sub.index, lower, upper,
            alpha=0.22, color="C0", linewidth=0,
            label=rf"$\pm{int(sigma)}\sigma$",
        )

        ax.set_ylabel(r"Irradiance (W/m$^2$)")
        ax.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=3, maxticks=6))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))

        if i == 0:
            ax.legend(
                [h_mean, h_band], ["Mean", rf"$\pm{int(sigma)}\sigma$"],
                loc="upper left",
                ncol=2, fontsize=8,
                frameon=True, framealpha=0.9, edgecolor="black",
            )

    axes[-1].set_xlabel("Date")
    fig.subplots_adjust(top=0.97, left=0.22, right=0.97, bottom=0.07)
    _save(fig, "fig_solar_months_mean_plus_3sigma")


def fig_mean_wind_speed() -> None:
    months = list(range(1, 13))
    tick_months = months[::2]
    tick_labels = [calendar.month_abbr[m] for m in tick_months]

    fig, ax = plt.subplots(figsize=(SINGLE_COL_IN, 2.8))
    for (lat, lon), df in _load_paper_location_pkls():
        monthly = df.groupby("month")["expected_wind_speed"].mean()
        ax.plot(monthly.index, monthly.values, marker=".",
                label=_location_label(lat, lon))
    ax.set_xlabel("Month")
    ax.set_ylabel("Wind Speed (m/s)")
    ax.set_xticks(tick_months)
    ax.set_xticklabels(tick_labels)
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.30),
        ncol=2,
        borderaxespad=0.0,
    )
    fig.subplots_adjust(top=0.96, bottom=0.42, left=0.18, right=0.97)
    _save(fig, "mean_wind_speed")


_EPISODE_DATASETS = [
    "solar_series",
    "wind_series",
    "whale_series",
    "energy_series",
    "actions",
    "rewards",
]
_EPISODE_YLABELS = [
    r"$\tilde{e}^+_k$ (Wh)",
    r"$w_k$ (m/s)",
    r"$O_k$",
    r"$\bar{c}_k$ (Wh)",
    r"$m_k$",
    r"$r_k$",
]


def _episode_sim_label(sim_name: str) -> str:
    if sim_name.lower().startswith("optimal"):
        return "Optimal"
    m = re.search(r"_t(?P<obs>[\d.]+)_w(?P<wind>[\d.]+)", sim_name, re.IGNORECASE)
    if m:
        return rf"$w_{{to}}={float(m.group('wind'))},\ O_{{th}}={float(m.group('obs'))}$"
    return sim_name


def _episode_thresholds(sim_name: str) -> Tuple[Optional[float], Optional[float]]:
    m = re.search(r"_t(?P<obs>[\d.]+)_w(?P<wind>[\d.]+)", sim_name, re.IGNORECASE)
    if not m:
        return None, None
    return float(m.group("obs")), float(m.group("wind"))


def fig_episode_4_combos() -> None:
    h5 = RESULTS_PATHS["threshold_h5"]
    _check_path(h5, "threshold sweep H5 (for episode plot)")

    target_combo = (0.15, 6.0)
    episode_num = 4
    episode_name = f"episode {episode_num}"

    with h5py.File(str(h5), "r") as f:
        all_groups = list(f.keys())

    selected: List[str] = []
    for g in all_groups:
        if g.lower().startswith("optimal"):
            selected.append(g)
            continue
        obs, wind = _episode_thresholds(g)
        if obs is None:
            continue
        if np.isclose(obs, target_combo[0]) and np.isclose(wind, target_combo[1]):
            selected.append(g)

    if not selected:
        raise RuntimeError(
            f"No matching sims in {h5} for combo {target_combo} in {episode_name}")

    loaded = {}
    with h5py.File(str(h5), "r") as f:
        for sim in selected:
            try:
                grp = f[sim]["episodes"][episode_name]
            except KeyError:
                continue
            loaded[sim] = {ds: grp[ds][:] for ds in _EPISODE_DATASETS}

    if not loaded:
        raise RuntimeError(f"No data for {episode_name} in {h5}")

    # Optimal first
    sims_ordered = (
        [s for s in loaded if s.lower().startswith("optimal")]
        + [s for s in loaded if not s.lower().startswith("optimal")]
    )

    n_panels = len(_EPISODE_DATASETS) + 1
    fig, axes = plt.subplots(
        n_panels, 1, sharex=True,
        figsize=(DOUBLE_COL_IN, 8.5),
        constrained_layout=False,
    )

    # Align lengths: trim energy_series tail if N+1 vs N actions
    for sim, series in loaded.items():
        L = len(series["actions"])
        if "energy_series" in series and len(series["energy_series"]) == L + 1:
            series["energy_series"] = series["energy_series"][:-1]

    # Data panels
    for idx, (ax, ds, ylabel) in enumerate(zip(axes, _EPISODE_DATASETS, _EPISODE_YLABELS)):
        use_black = idx < 3
        for sim in sims_ordered:
            data = loaded[sim]
            if ds not in data:
                continue
            y = data[ds].astype(float)
            if ds in ("energy_series", "solar_series"):
                y = y / 3600.0  # to Wh
            L = len(data["actions"])
            t_days = np.arange(L) * DT_MINUTES / (60.0 * 24.0)
            y = y[:L]
            label = None if use_black else _episode_sim_label(sim)
            kwargs = dict(linewidth=0.6,
                          color="black" if use_black else None,
                          label=label)
            if ds == "actions" or set(np.unique(y)).issubset({0.0, 1.0}):
                ax.step(t_days, y, where="mid", **kwargs)
            else:
                ax.plot(t_days, y, **kwargs)
        ax.set_ylabel(ylabel)
        ax.grid(True, linewidth=0.4, alpha=0.35)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=8, prune="both"))
        if ds == "rewards":
            ax.set_yscale("symlog", linthresh=0.1, linscale=1.0, base=10)

    # Cumulative flight time
    cf_ax = axes[-1]
    handles_legend = None
    for sim in sims_ordered:
        data = loaded[sim]
        actions = data["actions"]
        cum = np.cumsum((actions != 0).astype(int)) * (DT_MINUTES / 60.0)
        L = len(actions)
        t_days = np.arange(L) * DT_MINUTES / (60.0 * 24.0)
        cf_ax.plot(t_days, cum, linewidth=0.8,
                   label=_episode_sim_label(sim))
    cf_ax.set_ylabel("Total Flight (hrs)")
    cf_ax.set_xlabel("Time (days)")
    cf_ax.grid(True, linewidth=0.4, alpha=0.35)
    cf_ax.xaxis.set_major_locator(MaxNLocator(nbins=8))

    handles, labels = cf_ax.get_legend_handles_labels()
    if handles:
        fig.legend(
            handles, labels,
            loc="lower center", bbox_to_anchor=(0.5, 0.0),
            ncol=min(len(labels), 4),
            frameon=True, framealpha=0.9, edgecolor="black",
        )

    fig.subplots_adjust(top=0.97, bottom=0.10, left=0.10, right=0.97, hspace=0.30)
    basename = f"episode_{episode_num}_combos__obs-{target_combo[0]}_wind-{target_combo[1]}"
    _save(fig, basename)


def _draw_soc_slice_panel(
    ax: plt.Axes,
    npy_path: Path,
    soc_levels: Sequence[float] = (100.0,),
    *,
    show_xlabel: bool = True,
    show_ylabel: bool = True,
    line_width: float = 1.1,
    date_fmt: str = "%b %d",
) -> Tuple[List, List[str]]:
    _check_path(npy_path, "value-function .npy")
    data = np.load(npy_path)
    moored, flying, soc = _split_value_blocks(data)
    _, T = moored.shape
    start_dt = _infer_start_dt(npy_path)
    times = pd.date_range(start=start_dt, periods=T, freq=f"{DT_MINUTES}min")

    soc_arr = np.array(soc_levels, dtype=float)
    handles: List = []
    labels: List[str] = []
    for s_target in soc_arr:
        idx = int(np.argmin(np.abs(soc - s_target)))
        s_disp = soc[idx]
        ln_fly, = ax.plot(times, flying[idx, :], linestyle="-",
                          linewidth=0.4, color="black",
                          label=f"Flying {s_disp:.0f}%")
        ln_float, = ax.plot(times, moored[idx, :], linestyle="--",
                            linewidth=line_width,
                            label=f"Floating {s_disp:.0f}%")
        handles.extend([ln_fly, ln_float])
        labels.extend([ln_fly.get_label(), ln_float.get_label()])

    ax.xaxis.set_major_locator(mdates.AutoDateLocator(maxticks=6, minticks=3))
    ax.xaxis.set_major_formatter(mdates.DateFormatter(date_fmt))
    ax.set_xlim(times[0], times[-1])
    if show_xlabel:
        ax.set_xlabel("Time")
    if show_ylabel:
        ax.set_ylabel("Value")
    return handles, labels


def _draw_thresholds_panel(
    ax: plt.Axes,
    npy_path: Path,
    *,
    show_xlabel: bool = True,
    show_ylabel: bool = True,
    line_width: float = 1.2,
    date_fmt: str = "%b %d",
) -> Tuple[List, List[str]]:
    _check_path(npy_path, "value-function .npy")
    data = np.load(npy_path)
    moored, flying, soc = _split_value_blocks(data)
    _, T = moored.shape
    start_dt = _infer_start_dt(npy_path)
    times = [start_dt + timedelta(minutes=i * DT_MINUTES) for i in range(T)]

    takeoff, cliff = _compute_thresholds(moored, flying, soc)

    h1, = ax.plot(times, takeoff, color="tab:blue", linewidth=line_width,
                  label="Takeoff Threshold")
    h2, = ax.plot(times, cliff, color="tab:red", linewidth=line_width,
                  label="Landing Threshold")
    no_takeoff = np.isnan(takeoff)
    handles = [h1, h2]
    labels = [h1.get_label(), h2.get_label()]
    if np.any(no_takeoff):
        h3 = ax.scatter(
            np.array(times, dtype="object")[no_takeoff],
            [100] * int(np.sum(no_takeoff)),
            color="gray", s=10, label="No Takeoff", zorder=3,
        )
        handles.append(h3)
        labels.append("No Takeoff")

    ax.xaxis.set_major_locator(mdates.AutoDateLocator(maxticks=6, minticks=3))
    ax.xaxis.set_major_formatter(mdates.DateFormatter(date_fmt))
    ax.set_xlim(times[0], times[-1])
    if show_xlabel:
        ax.set_xlabel("Time")
    if show_ylabel:
        ax.set_ylabel("SoC (%)")
    return handles, labels


def fig_threshold_example_value_combined() -> None:
    npy = RESULTS_PATHS["threshold_value_npy"]
    fig, axes = plt.subplots(
        2, 1, figsize=(SINGLE_COL_IN, 4.2),
        gridspec_kw={"hspace": 0.35},
    )

    h_top, l_top = _draw_soc_slice_panel(
        axes[0], npy, soc_levels=(100,), show_xlabel=False)
    axes[0].legend(h_top, l_top, loc="best", ncol=2, fontsize=8)
    _panel_label(axes[0], "(a)")

    h_bot, l_bot = _draw_thresholds_panel(axes[1], npy, show_xlabel=True)
    axes[1].legend(
        h_bot, l_bot,
        loc="lower left", bbox_to_anchor=(0.0, 0.15),
        fontsize=8,
    )
    _panel_label(axes[1], "(b)")

    fig.subplots_adjust(top=0.93, left=0.18, right=0.97, bottom=0.12)
    _save(fig, "threshold_example_value_combined")


def _draw_threshold_lines(
    ax: plt.Axes,
    df_main: pd.DataFrame,
    df_opt: pd.DataFrame,
    value_col: str,
    *,
    percent: bool = False,
) -> Tuple[List, List[str]]:
    main = df_main.copy()
    if percent:
        main = _normalize_percent(main, value_col)

    pivot = (
        main.pivot_table(index="observation_threshold",
                         columns="wind_threshold",
                         values=value_col,
                         aggfunc="mean")
            .sort_index()
    )
    handles: List = []
    labels: List[str] = []
    for w in pivot.columns:
        h, = ax.plot(pivot.index, pivot[w], marker="o",
                     label=fr"$w_{{to}} = {w}$")
        handles.append(h)
        labels.append(h.get_label())

    if not df_opt.empty and value_col in df_opt.columns:
        opt = df_opt.copy()
        if percent:
            opt = _normalize_percent(opt, value_col)
        opt_val = pd.to_numeric(opt[value_col], errors="coerce").dropna().mean()
        if pd.notna(opt_val):
            h_opt = ax.axhline(opt_val, linestyle="--", color="black",
                               linewidth=1.0, label="Optimal")
            for x in pivot.index:
                ax.scatter(x, opt_val, color="black", s=14, marker="x", zorder=3)
            handles.append(h_opt)
            labels.append("Optimal")

    if percent:
        ax.yaxis.set_major_formatter(PercentFormatter(xmax=100))
        y0, y1 = ax.get_ylim()
        ax.set_ylim(max(0, y0 - 5), min(105, y1 + 5))

    return handles, labels


def fig_threshold_sweep_combined() -> None:
    h5 = RESULTS_PATHS["threshold_h5"]
    _check_path(h5, "threshold sweep H5")
    df = load_summary([str(h5)])
    df_main, df_opt = _split_optimal(df)
    df_main = df_main[
        pd.to_numeric(df_main["wind_threshold"], errors="coerce") != 3.0
    ]

    fig, axes = plt.subplots(
        3, 1, figsize=(SINGLE_COL_IN, 5.8),
        sharex=True,
        gridspec_kw={"hspace": 0.30},
    )

    handles, labels = _draw_threshold_lines(
        axes[0], df_main, df_opt, "mean_reward")
    axes[0].set_ylabel("Mean Total Reward")
    _panel_label(axes[0], "(a)")

    _draw_threshold_lines(axes[1], df_main, df_opt, "mean_failure_step")
    axes[1].set_ylabel("Mean Failure Stage")
    _panel_label(axes[1], "(b)")

    _draw_threshold_lines(axes[2], df_main, df_opt, "failure_percentage",
                          percent=True)
    axes[2].set_ylabel("Failure Percentage")
    axes[2].set_xlabel(r"Observation Threshold $O_{th}$")
    _panel_label(axes[2], "(c)")

    fig.legend(
        handles, labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.02),
        ncol=3,
        frameon=True, framealpha=0.9, edgecolor="black",
    )
    fig.subplots_adjust(top=0.97, left=0.20, right=0.97, bottom=0.17)
    _save(fig, "threshold_sweep_combined")


def fig_capacity_mean_reward() -> None:
    h5 = RESULTS_PATHS["battery_h5"]
    _check_path(h5, "battery sweep H5")
    df = load_summary([str(h5)])
    df_main, df_opt = _split_optimal(df)

    obs = 0.2
    sub = df_main[df_main["observation_threshold"] == obs]
    wind_vals = sorted(sub["wind_threshold"].dropna().unique())

    fig, ax = plt.subplots(figsize=(SINGLE_COL_IN, 2.5))
    for w in wind_vals:
        ser = sub[sub["wind_threshold"] == w].sort_values("battery_capacity")
        if ser.empty:
            continue
        ax.plot(
            ser["battery_capacity"], ser["mean_reward"],
            marker="o",
            label=fr"$w_{{to}}={w},\ O_{{th}}={obs}$",
        )

    if not df_opt.empty:
        opt_cap = (
            df_opt.groupby("battery_capacity")["mean_reward"]
                  .mean()
                  .reset_index()
                  .sort_values("battery_capacity")
        )
        if not opt_cap.empty:
            ax.plot(
                opt_cap["battery_capacity"], opt_cap["mean_reward"],
                color="black", marker="o", linestyle="-",
                label="Optimal",
            )

    ax.set_xlabel("Battery Capacity (Wh)")
    ax.set_ylabel("Mean Total Reward")
    ax.legend(loc="best")
    fig.tight_layout()
    _save(fig, "capacity_mean_reward_obs0.2")


def fig_world_map() -> None:
    locations = list(PAPER_LOCATIONS)
    use_cartopy = False
    try:
        import cartopy.crs as ccrs
        import cartopy.feature as cfeature
        use_cartopy = True
    except Exception:
        use_cartopy = False

    if use_cartopy:
        fig = plt.figure(figsize=(SINGLE_COL_IN, 2.0))
        ax = plt.axes(projection=ccrs.Robinson())
        ax.set_global()
        ax.coastlines(linewidth=0.5)
        ax.add_feature(cfeature.LAND, facecolor="0.93")
        ax.gridlines(draw_labels=False, linewidth=0.25, linestyle=":")
        handles, labels = [], []
        for (lat, lon) in locations:
            h = ax.plot(lon, lat, "o", markersize=4,
                        transform=ccrs.PlateCarree())[0]
            handles.append(Line2D([], [], marker="o", linestyle="none",
                                  color=h.get_color(), markersize=5))
            labels.append(_location_label(lat, lon))
    else:
        fig, ax = plt.subplots(figsize=(SINGLE_COL_IN, 2.0))
        ax.set_xlim(-180, 180)
        ax.set_ylim(-60, 80)
        ax.grid(True, linestyle=":", linewidth=0.25)
        ax.set_xticks([])
        ax.set_yticks([])
        handles, labels = [], []
        for (lat, lon) in locations:
            h, = ax.plot(lon, lat, "o", markersize=4)
            handles.append(Line2D([], [], marker="o", linestyle="none",
                                  color=h.get_color(), markersize=5))
            labels.append(_location_label(lat, lon))

    fig.legend(
        handles, labels,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.02),
        ncol=min(len(labels), 3),
        frameon=True, framealpha=0.9, edgecolor="black",
    )
    fig.subplots_adjust(bottom=0.30, top=0.97, left=0.03, right=0.97)
    _save(fig, "subfig_a_world_map")


def fig_optimal_capacity_by_location() -> None:
    h5 = RESULTS_PATHS["loc_capacity_h5"]
    _check_path(h5, "location capacity sweep H5")
    df = load_summary([str(h5)])
    _, df_opt = _split_optimal(df)
    if df_opt.empty:
        raise RuntimeError("No optimal rows found in location capacity sweep H5")

    fig, ax = plt.subplots(figsize=(SINGLE_COL_IN, 2.7))
    for (lat, lon) in PAPER_LOCATIONS:
        sub = df_opt[(df_opt["latitude"] == lat) & (df_opt["longitude"] == lon)]
        if sub.empty:
            continue
        agg = (
            sub.groupby("battery_capacity")["mean_reward"]
               .mean()
               .reset_index()
               .sort_values("battery_capacity")
        )
        if agg.empty:
            continue
        ax.plot(agg["battery_capacity"], agg["mean_reward"], marker="o",
                label=_location_label(lat, lon))

    ax.set_xlabel("Battery Capacity (Wh)")
    ax.set_ylabel("Mean Total Reward")
    ax.legend(loc="best")
    fig.tight_layout()
    _save(fig, "optimal_capacity_by_location")


def _draw_penalty_lines(
    ax: plt.Axes,
    df_main: pd.DataFrame,
    df_opt: pd.DataFrame,
    metric: str,
    obs: float,
    *,
    percent: bool = False,
) -> Tuple[List, List[str]]:
    sub = df_main[df_main["observation_threshold"] == obs].copy()
    if percent:
        sub = _normalize_percent(sub, metric)

    agg = (
        sub.groupby(["failure_penalty", "wind_threshold"], as_index=False)[metric]
           .mean()
           .sort_values(["wind_threshold", "failure_penalty"])
    )
    handles: List = []
    labels: List[str] = []
    for w in sorted(agg["wind_threshold"].dropna().unique()):
        ser = agg[agg["wind_threshold"] == w]
        if ser.empty:
            continue
        h, = ax.plot(
            ser["failure_penalty"], ser[metric], marker="o",
            label=fr"$w_{{to}}={w},\ O_{{th}}={obs}$",
        )
        handles.append(h)
        labels.append(h.get_label())

    if not df_opt.empty and metric in df_opt.columns:
        opt = df_opt.copy()
        if percent:
            opt = _normalize_percent(opt, metric)
        opt_pen = (
            opt.groupby("failure_penalty")[metric].mean()
               .reset_index()
               .sort_values("failure_penalty")
        )
        if not opt_pen.empty:
            h_opt, = ax.plot(
                opt_pen["failure_penalty"], opt_pen[metric],
                color="black", marker="o", linestyle="-",
                label="Optimal",
            )
            handles.append(h_opt)
            labels.append("Optimal")

    if percent:
        ax.yaxis.set_major_formatter(PercentFormatter(xmax=100))
        y0, y1 = ax.get_ylim()
        ax.set_ylim(max(0, y0), min(105, y1))

    return handles, labels


def fig_failure_penalty_combined() -> None:
    h5 = RESULTS_PATHS["failure_h5"]
    _check_path(h5, "failure penalty H5")
    df = load_summary([str(h5)])
    df_main, df_opt = _split_optimal(df)

    obs = 0.2
    fig, axes = plt.subplots(
        4, 1, figsize=(SINGLE_COL_IN, 7.2),
        sharex=True,
        gridspec_kw={"hspace": 0.30},
    )

    handles, labels = _draw_penalty_lines(axes[0], df_main, df_opt, "mean_reward", obs)
    axes[0].set_ylabel("Mean Total Reward")
    _panel_label(axes[0], "(a)")

    _draw_penalty_lines(axes[1], df_main, df_opt, "failure_percentage", obs, percent=True)
    axes[1].set_ylabel("Failure Percentage")
    _panel_label(axes[1], "(b)")

    _draw_penalty_lines(axes[2], df_main, df_opt, "mean_failure_step", obs)
    axes[2].set_ylabel("Mean Failure Stage")
    _panel_label(axes[2], "(c)")

    _draw_penalty_lines(axes[3], df_main, df_opt, "average_flight_hrs", obs)
    axes[3].set_ylabel("Mean Flight Hours")
    axes[3].set_xlabel(r"Failure Penalty $\phi$")
    _panel_label(axes[3], "(d)")

    axes[0].legend(
        handles, labels,
        loc="best",
        ncol=1,
        fontsize=8,
        frameon=True, framealpha=0.9, edgecolor="black",
    )
    fig.subplots_adjust(top=0.98, left=0.20, right=0.97, bottom=0.08)
    _save(fig, "failure_penalty_combined")


def _draw_horizon_lines(
    ax: plt.Axes,
    df_main: pd.DataFrame,
    df_opt: pd.DataFrame,
    x_main: pd.Series,
    x_opt: pd.Series,
    y_col: str,
    *,
    percent: bool = False,
) -> Tuple[List, List[str]]:
    agg = (
        pd.DataFrame({
            "x": pd.to_numeric(x_main.loc[df_main.index], errors="coerce"),
            "observation_threshold": pd.to_numeric(df_main["observation_threshold"], errors="coerce"),
            "wind_threshold": pd.to_numeric(df_main["wind_threshold"], errors="coerce"),
            y_col: pd.to_numeric(df_main[y_col], errors="coerce"),
        })
        .dropna(subset=["x", "observation_threshold", "wind_threshold", y_col])
        .groupby(["x", "observation_threshold", "wind_threshold"], as_index=False)[y_col]
        .mean()
        .sort_values(["observation_threshold", "wind_threshold", "x"])
    )
    handles: List = []
    labels: List[str] = []
    pairs = sorted(agg.groupby(["observation_threshold", "wind_threshold"]).groups.keys())
    for (obs, w) in pairs:
        ser = agg[(agg["observation_threshold"] == obs) & (agg["wind_threshold"] == w)]
        if ser.empty:
            continue
        h, = ax.plot(ser["x"], ser[y_col], marker="o",
                     label=_legend_label(obs, w))
        handles.append(h)
        labels.append(h.get_label())

    if not df_opt.empty and y_col in df_opt.columns:
        opt_tbl = pd.DataFrame({
            "x": pd.to_numeric(x_opt.loc[df_opt.index], errors="coerce"),
            y_col: pd.to_numeric(df_opt[y_col], errors="coerce"),
        }).dropna()
        if not opt_tbl.empty:
            opt_mean = opt_tbl.groupby("x", as_index=False)[y_col].mean().sort_values("x")
            h_opt, = ax.plot(
                opt_mean["x"], opt_mean[y_col],
                color="black", marker="o", linewidth=1.2,
                label="Optimal",
            )
            handles.append(h_opt)
            labels.append("Optimal")

    ax.xaxis.set_major_locator(MaxNLocator(nbins=6))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
    if percent:
        ax.yaxis.set_major_formatter(PercentFormatter(xmax=100))

    return handles, labels


def fig_horizon_sweep_combined() -> None:
    h5 = RESULTS_PATHS["duration_h5"]
    _check_path(h5, "duration sweep H5")
    df = load_summary([str(h5)])
    df_main, df_opt = _split_optimal(df)

    # Restrict to the (obs, wind) pairs shown in the original paper figure.
    paper_pairs = {(0.25, 4.0), (0.25, 8.0)}
    pair_mask = df_main.apply(
        lambda r: (
            pd.notna(r["observation_threshold"])
            and pd.notna(r["wind_threshold"])
            and (float(r["observation_threshold"]), float(r["wind_threshold"])) in paper_pairs
        ),
        axis=1,
    )
    df_main = df_main[pair_mask].copy()

    # Convert horizon (timesteps of 15 min) to days
    days_main = pd.to_numeric(df_main["horizon"], errors="coerce") * (DT_MINUTES / 1440.0)
    days_opt = pd.to_numeric(df_opt["horizon"], errors="coerce") * (DT_MINUTES / 1440.0)

    # Derived columns
    df_main = df_main.copy()
    df_opt = df_opt.copy()
    horizon_main = pd.to_numeric(df_main["horizon"], errors="coerce").replace(0, np.nan)
    df_main["mean_reward_per_ts"] = (
        pd.to_numeric(df_main["mean_reward"], errors="coerce") / horizon_main
    )
    horizon_opt = pd.to_numeric(df_opt["horizon"], errors="coerce").replace(0, np.nan)
    df_opt["mean_reward_per_ts"] = (
        pd.to_numeric(df_opt["mean_reward"], errors="coerce") / horizon_opt
    )
    df_main["flight_hours_per_day"] = (
        pd.to_numeric(df_main["average_flight_hrs"], errors="coerce") / days_main
    )
    df_opt["flight_hours_per_day"] = (
        pd.to_numeric(df_opt["average_flight_hrs"], errors="coerce") / days_opt
    )

    # Scale failure percentage to 0..100
    if "failure_percentage" in df_main.columns:
        df_main["failure_percentage"] = pd.to_numeric(
            df_main["failure_percentage"], errors="coerce") * 100.0
    if "failure_percentage" in df_opt.columns and not df_opt.empty:
        df_opt["failure_percentage"] = pd.to_numeric(
            df_opt["failure_percentage"], errors="coerce") * 100.0

    fig, axes = plt.subplots(
        4, 1, figsize=(SINGLE_COL_IN, 7.6),
        sharex=True, gridspec_kw={"hspace": 0.30},
    )

    handles, labels = _draw_horizon_lines(
        axes[0], df_main, df_opt, days_main, days_opt, "mean_reward_per_ts")
    axes[0].set_ylabel("Mean Reward per Stage")
    axes[0].ticklabel_format(axis="y", style="sci", scilimits=(-3, 3),
                             useMathText=True)
    _panel_label(axes[0], "(a)")

    _draw_horizon_lines(
        axes[1], df_main, df_opt, days_main, days_opt, "mean_failure_step")
    axes[1].set_ylabel("Mean Failure Stage")
    _panel_label(axes[1], "(b)")

    _draw_horizon_lines(
        axes[2], df_main, df_opt, days_main, days_opt,
        "failure_percentage", percent=True)
    axes[2].set_ylabel("Failure Percentage")
    _panel_label(axes[2], "(c)")

    _draw_horizon_lines(
        axes[3], df_main, df_opt, days_main, days_opt, "flight_hours_per_day")
    axes[3].set_ylabel("Flight Hours per Day")
    axes[3].set_xlabel("Mission Duration (days)")
    _panel_label(axes[3], "(d)")

    axes[0].legend(
        handles, labels,
        loc="best",
        ncol=1,
        fontsize=8,
        frameon=True, framealpha=0.9, edgecolor="black",
    )
    fig.subplots_adjust(top=0.97, left=0.20, right=0.97, bottom=0.08)
    _save(fig, "horizon_sweep_combined")


def _duration_soc_slice_figure(
    npy: Path, basename: str, date_fmt: str,
    legend_loc: str = "upper left",
    legend_ncol: int = 2,
    legend_bbox: Optional[Tuple[float, float]] = None,
) -> None:
    fig, ax = plt.subplots(figsize=(3.2, 2.4))
    h, l = _draw_soc_slice_panel(
        ax, npy, soc_levels=(100,), show_xlabel=True, show_ylabel=True,
        date_fmt=date_fmt,
    )
    kwargs = dict(loc=legend_loc, fontsize=8, ncol=legend_ncol)
    if legend_bbox is not None:
        kwargs["bbox_to_anchor"] = legend_bbox
    ax.legend(h, l, **kwargs)
    fig.tight_layout()
    _save(fig, basename)


def fig_duration_value_soc_slices_short() -> None:
    npy = RESULTS_PATHS["duration_value_npy_short"]
    _duration_soc_slice_figure(
        npy,
        "duration_sweep_config_300.0Wh_3000h_20.0p_2025-01-01_soc_slices",
        date_fmt="%b %d",
        legend_loc="upper right",
        legend_ncol=1,
        legend_bbox=(1.0, 0.90),
    )


def fig_whale_observation_probability() -> None:
    horizon = 96  # one day at 15-min resolution
    series = WhaleRewardSeriesFactory.create_series("real", horizon)
    hours = np.arange(horizon) * (DT_MINUTES / 60.0)

    fig, ax = plt.subplots(figsize=(SINGLE_COL_IN, 2.4))
    ax.step(hours, series, where="post", linewidth=1.0, color="C0")
    ax.set_xlim(0, 24)
    ax.set_ylim(-0.02, None)
    ax.set_xticks(np.arange(0, 25, 4))
    ax.set_xlabel("Hour of Day")
    ax.set_ylabel(r"Whale Observation Probability ($O_k$)")
    fig.subplots_adjust(top=0.96, bottom=0.22, left=0.18, right=0.97)
    _save(fig, "Whale_observation_probability")


def fig_failure_probability(model_name: str = "moderate") -> None:
    model = ProbabilityModelFactory.select_probability_model(model_name)

    wind_speeds = np.linspace(0, 40, 400)
    cases = [
        (0, (0, 0), "Float", "-"),
        (0, (0, 1), "Land",  "-"),
        (1, (0, 0), "Takeoff", "-"),
        (1, (0, 1), "Fly",   "-."),
    ]

    fig, ax = plt.subplots(figsize=(SINGLE_COL_IN, 2.4))
    for action, state, label, ls in cases:
        probs = model.compute_probability(
            wind_speeds, action,
            np.array([state] * len(wind_speeds)),
        )
        ax.plot(wind_speeds, 1.0 - probs, label=label, linestyle=ls)

    ax.set_xlim(0, 40)
    ax.yaxis.set_ticks(np.linspace(0, 1, 6))
    ax.set_xlabel("Wind Speed (m/s)")
    ax.set_ylabel(r"P($z_k$=1)")
    ax.legend(loc="best", ncol=1)
    fig.subplots_adjust(top=0.96, bottom=0.22, left=0.18, right=0.97)
    _save(fig, "failure_probability")


def fig_duration_value_soc_slices_long() -> None:
    npy = RESULTS_PATHS["duration_value_npy_long"]
    _duration_soc_slice_figure(
        npy,
        "duration_sweep_config2_300.0Wh_18000h_20.0p_2025-01-01_soc_slices",
        date_fmt="%b %Y",
        legend_loc="lower right",
        legend_ncol=1,
    )


# Ordered dispatch table; key is the --only name.
FIGURES = {
    "fig_total_monthly_solar_energy": fig_total_monthly_solar_energy,
    "fig_mean_wind_speed": fig_mean_wind_speed,
    "fig_wind_months_mean_plus_3sigma": fig_wind_months_mean_plus_3sigma,
    "fig_solar_months_mean_plus_3sigma": fig_solar_months_mean_plus_3sigma,
    "fig_episode_4_combos": fig_episode_4_combos,
    "fig_threshold_example_value_combined": fig_threshold_example_value_combined,
    "fig_threshold_sweep_combined": fig_threshold_sweep_combined,
    "fig_capacity_mean_reward": fig_capacity_mean_reward,
    "fig_world_map": fig_world_map,
    "fig_optimal_capacity_by_location": fig_optimal_capacity_by_location,
    "fig_failure_penalty_combined": fig_failure_penalty_combined,
    "fig_horizon_sweep_combined": fig_horizon_sweep_combined,
    "fig_duration_value_soc_slices_short": fig_duration_value_soc_slices_short,
    "fig_duration_value_soc_slices_long": fig_duration_value_soc_slices_long,
    "fig_whale_observation_probability": fig_whale_observation_probability,
    "fig_failure_probability": fig_failure_probability,
}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--only", nargs="+", default=None,
        help="Render only these figure names (see --list).",
    )
    ap.add_argument(
        "--list", action="store_true",
        help="Print the known figure names and exit.",
    )
    ap.add_argument(
        "--continue-on-error", action="store_true",
        help="Skip a failing figure and continue with the rest.",
    )
    args = ap.parse_args(argv)

    if args.list:
        for name in FIGURES:
            print(name)
        return 0

    apply_paper_style()
    _ensure_outdir()

    selected = list(FIGURES.keys()) if args.only is None else args.only
    unknown = [n for n in selected if n not in FIGURES]
    if unknown:
        print(f"Unknown figure(s): {unknown}", file=sys.stderr)
        print("Run with --list to see valid names.", file=sys.stderr)
        return 2

    failed: List[str] = []
    for name in selected:
        print(f"--- {name} ---")
        try:
            FIGURES[name]()
        except NotImplementedError:
            print(f"[SKIP] {name} not implemented yet")
        except Exception as exc:  # noqa: BLE001
            failed.append(name)
            msg = f"[FAIL] {name}: {exc}"
            print(msg, file=sys.stderr)
            if not args.continue_on_error:
                raise

    if failed:
        print(f"\nFailed figures: {failed}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

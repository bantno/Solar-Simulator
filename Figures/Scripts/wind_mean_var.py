#!/usr/bin/env python3
"""Wind mean and ±3σ panels (Jan/Apr/Jul/Oct) using expected_wind_speed + Weibull spread.

- Mean comes from column: expected_wind_speed  (m/s)
- Spread comes from either:
    (A) Weibull(k, λ) columns -> std = sqrt(λ² [Γ(1+2/k) - Γ(1+1/k)²])
    (B) A direct std column (override with --std-col)
The script auto-builds a DatetimeIndex from month/day/hour/minute.
"""

from __future__ import annotations
import argparse
from typing import Optional, Sequence
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy.special import gamma

# ---------- time handling ----------
def build_datetime_index_from_parts(
    df: pd.DataFrame,
    month_col: str = "month",
    day_col: str = "day",
    hour_col: str = "hour",
    minute_col: str = "minute",
    ref_year: int = 2001,  # non-leap
) -> pd.DataFrame:
    missing = [c for c in (month_col, day_col, hour_col, minute_col) if c not in df.columns]
    if missing:
        raise KeyError(f"Missing time-part columns: {missing}")
    parts = df[[month_col, day_col, hour_col, minute_col]].astype(int)
    dt = pd.to_datetime(
        dict(year=ref_year,
             month=parts[month_col], day=parts[day_col],
             hour=parts[hour_col], minute=parts[minute_col])
    )
    out = df.copy()
    out.index = dt
    return out.sort_index()

# ---------- stats ----------
def weibull_std_from_params(k: np.ndarray, lam: np.ndarray) -> np.ndarray:
    k = np.clip(np.asarray(k, float), 1e-6, None)
    lam = np.clip(np.asarray(lam, float), 1e-12, None)
    g1 = gamma(1.0 + 1.0 / k)
    g2 = gamma(1.0 + 2.0 / k)
    var = (lam ** 2) * (g2 - g1 ** 2)
    return np.sqrt(np.maximum(var, 0.0))

# ---------- plot ----------
def plot_monthly_panels(
    df: pd.DataFrame,
    mean_col: str = "expected_wind_speed",
    std_col: Optional[str] = None,           # if you already have a std column
    shape_col: Optional[str] = "weibull_k",   # else compute std from Weibull
    scale_col: Optional[str] = "weibull_lambda",
    months: Sequence[int] = (1, 4, 7, 10),
    sigma: float = 3.0,
    ylabel: str = "Wind speed (m/s)",
    figsize: tuple[float, float] = (10, 8),
    outfile: Optional[str] = None,
) -> None:

    if mean_col not in df.columns:
        raise KeyError(f"Mean column '{mean_col}' not found. Available: {list(df.columns)}")

    # derive std
    if std_col and std_col in df.columns:
        std = df[std_col].to_numpy()
    elif shape_col and scale_col and (shape_col in df.columns) and (scale_col in df.columns):
        std = weibull_std_from_params(df[shape_col].to_numpy(), df[scale_col].to_numpy())
    else:
        raise KeyError(
            "No way to compute spread: provide --std-col OR Weibull columns "
            f"('{shape_col}', '{scale_col}')."
        )

    df = df.copy()
    df["mean_wind"] = df[mean_col].to_numpy()
    df["std_wind"]  = std

    fig, axes = plt.subplots(len(months), 1, figsize=figsize, sharex=False)
    if len(months) == 1:
        axes = [axes]

    for ax in axes:
        ax.grid(True, which="both", alpha=0.25, linestyle="--")
        ax.set_ylim(bottom=0,top=20)

    for ax, m in zip(axes, months):
        sub = df[df.index.month == m]
        if sub.empty:
            ax.text(0.5, 0.5, f"No data for month {m}", transform=ax.transAxes,
                    ha="center", va="center")
            continue

        mu = sub["mean_wind"].to_numpy()
        sd = sub["std_wind"].to_numpy()
        upper = mu + sigma * sd
        lower = np.clip(mu - sigma * sd, 0.0, None)

        ax.plot(sub.index, mu, linewidth=1.5, label="Mean")
        ax.fill_between(sub.index, lower, upper, alpha=0.18,
                        label=f"±{int(sigma)}σ")

        # ax.set_title(f"{sub.index[0]:%b} (representative month)", loc="left", fontsize=11)
        ax.set_ylabel(ylabel)
        ax.legend(loc="upper right", frameon=True)

        ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=1))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))

    axes[-1].set_xlabel("Date")
    fig.tight_layout()
    if outfile:
        fig.savefig(outfile, dpi=300, bbox_inches="tight")
    plt.show()

# ---------- CLI ----------
def main():
    ap = argparse.ArgumentParser(description="Wind mean and ±3σ monthly panels from expected_wind_speed.")
    ap.add_argument("--infile", required=True, help="Pickle with 15-min timesteps and columns")
    ap.add_argument("--mean-col", default="expected_wind_speed")
    ap.add_argument("--std-col", default=None, help="Optional: direct std column name")
    ap.add_argument("--shape-col", default="weibull_k")
    ap.add_argument("--scale-col", default="weibull_scale")
    ap.add_argument("--months", nargs="+", type=int, default=[1, 4, 7, 10])
    ap.add_argument("--sigma", type=float, default=3.0)
    ap.add_argument("--ylabel", default="Wind speed (m/s)")
    ap.add_argument("--outfile", default="fig_wind_months_mean_plus_3sigma.png")
    ap.add_argument("--month-col", default="month")
    ap.add_argument("--day-col", default="day")
    ap.add_argument("--hour-col", default="hour")
    ap.add_argument("--minute-col", default="minute")
    args = ap.parse_args()

    df = pd.read_pickle(args.infile)
    df = build_datetime_index_from_parts(
        df,
        month_col=args.month_col, day_col=args.day_col,
        hour_col=args.hour_col, minute_col=args.minute_col,
        ref_year=2001,
    )

    plot_monthly_panels(
        df=df,
        mean_col=args.mean_col,
        std_col=args.std_col,
        shape_col=args.shape_col,
        scale_col=args.scale_col,
        months=args.months,
        sigma=args.sigma,
        ylabel=args.ylabel,
        outfile=args.outfile,
    )

if __name__ == "__main__":
    main()

# results_io.py
"""
Utilities for loading simulation HDF5 results into a tidy Pandas DataFrame
and a few small plotting helpers shared by standalone results scripts.

This module is **independent** of any GUI code. It mirrors the summarization
logic used in the GUI reference: prefer group-level summary attributes when
available and fall back to aggregating across episodes.
"""
from __future__ import annotations

import re
import h5py
import numpy as np
import pandas as pd
from typing import Iterable, List, Tuple, Optional, Dict, Any
import matplotlib.pyplot as plt

_COORD_RE = re.compile(r"lat(?P<lat>[-\d\.]+)_lon(?P<lon>[-\d\.]+)")


def _safe_attr(grp: h5py.Group, key: str, default=None):
    val = grp.attrs.get(key, default)
    if isinstance(val, bytes):
        try:
            return val.decode("utf-8")
        except Exception:
            return str(val)
    if isinstance(val, np.generic):
        return np.asarray(val).item()
    return val


def _parse_location(grp: h5py.Group) -> Tuple[Optional[float], Optional[float]]:
    loc_id = _safe_attr(grp, "location_id")
    if isinstance(loc_id, str):
        m = _COORD_RE.search(loc_id)
        if m:
            return float(m.group("lat")), float(m.group("lon"))
    lat = _safe_attr(grp, "latitude")
    lon = _safe_attr(grp, "longitude")
    if lat is not None and lon is not None:
        return float(lat), float(lon)
    return None, None


def _aggregate_from_episodes(grp: h5py.Group) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    episodes = grp.get("episodes")
    if episodes is None:
        return None, None, None

    rewards: List[float] = []
    failures: List[int] = []
    failure_steps: List[float] = []

    for ep in episodes.values():
        if "total_reward" in ep:
            try:
                rewards.append(float(ep["total_reward"][()]))
            except Exception:
                pass
        if "failure" in ep:
            try:
                failures.append(int(bool(ep["failure"][()])))
            except Exception:
                pass
        if "failure_step" in ep:
            try:
                failure_steps.append(float(ep["failure_step"][()]))
            except Exception:
                pass

    mean_reward = float(np.mean(rewards)) if len(rewards) else None
    failure_percentage = float(100.0 * np.mean(failures)) if len(failures) else None
    mean_failure_step = float(np.mean(failure_steps)) if len(failure_steps) else None
    return mean_reward, failure_percentage, mean_failure_step


def load_summary(h5_paths: Iterable[str]) -> pd.DataFrame:
    records: List[Dict[str, Any]] = []

    for path in h5_paths:
        with h5py.File(path, "r") as f:
            for sim_name in f.keys():
                grp = f[sim_name]
                sim_type = _safe_attr(grp, "simulation_type", "")
                obs_t = _safe_attr(grp, "observation_threshold")
                wind_t = _safe_attr(grp, "wind_threshold")
                cap = _safe_attr(grp, "battery_capacity", _safe_attr(grp, "capacity"))
                horizon = _safe_attr(grp, "horizon")
                fp = _safe_attr(grp, "failure_penalty")
                avg_reward = _safe_attr(grp, "average_reward")
                avg_fail_step = _safe_attr(grp, "average_failure_step")
                avg_flt_hrs = _safe_attr(grp, "average_flight_hrs")
                fail_pct = _safe_attr(grp, "failure_percentage")
                start_time = _safe_attr(grp, "start_time")
                lat, lon = _parse_location(grp)

                if avg_reward is None or fail_pct is None or avg_fail_step is None:
                    m_reward, m_fail_pct, m_fail_step = _aggregate_from_episodes(grp)
                    if avg_reward is None:
                        avg_reward = m_reward
                    if fail_pct is None:
                        fail_pct = m_fail_pct
                    if avg_fail_step is None:
                        avg_fail_step = m_fail_step

                if (
                    sim_type is None and obs_t is None and wind_t is None
                    and cap is None and horizon is None and fp is None
                ):
                    continue

                records.append(
                    {
                        "sim_type": sim_type,
                        "observation_threshold": obs_t,
                        "wind_threshold": wind_t,
                        "battery_capacity": cap,
                        "horizon": horizon,
                        "failure_penalty": fp,
                        "mean_reward": avg_reward,
                        "failure_percentage": fail_pct,
                        "mean_failure_step": avg_fail_step,
                        "average_flight_hrs": avg_flt_hrs,
                        "latitude": lat,
                        "longitude": lon,
                        "start_time": start_time,
                        "source_file": path,
                    }
                )

    df = pd.DataFrame.from_records(records)
    numeric_cols = [
        "observation_threshold",
        "wind_threshold",
        "battery_capacity",
        "horizon",
        "failure_penalty",
        "mean_reward",
        "failure_percentage",
        "mean_failure_step",
        "average_flight_hrs",
        "latitude",
        "longitude",
    ]
    for c in numeric_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    if "start_time" in df.columns:
        try:
            df["start_time"] = pd.to_datetime(df["start_time"], errors="coerce")
        except Exception:
            pass
    
    print(df)
    return df


def steps_to_days(steps: pd.Series | np.ndarray | float, minutes_per_step: float = 15.0) -> pd.Series:
    return (pd.Series(steps, dtype=float) * minutes_per_step) / (60.0 * 24.0)


def apply_style(style: Optional[str] = None, rc: Optional[Dict[str, Any]] = None):
    if style:
        try:
            plt.style.use(style)
        except Exception:
            pass
    if rc:
        import matplotlib as mpl
        mpl.rcParams.update(rc)


def savefig_all_formats(fig, outdir: str, basename: str, formats: Iterable[str] = ("png", "pdf"), dpi: int = 300):
    import os
    os.makedirs(outdir, exist_ok=True)
    for ext in formats:
        fig.savefig(os.path.join(outdir, f"{basename}.{ext}"), dpi=dpi, bbox_inches="tight")


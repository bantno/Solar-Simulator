"""The metrics seam: turn one experiment's HDF5 output into a tidy, self-describing CSV.

`summarize_hdf5` reads the per-simulation metadata that SimulationRunManager already writes
as HDF5 group attributes (see BaseClasses/simulation_run_manager.py:_run_one_sim) and emits
one row per simulation, augmented with experiment-level columns from the config so every row
stands alone without a join.

This single function is the stable artifact a future regression mode can snapshot and diff
(within Monte-Carlo tolerance) -- keep its output schema stable.
"""
from typing import Dict, Optional

import h5py
import numpy as np
import pandas as pd

# Per-simulation metrics written as HDF5 group attrs by _run_one_sim. Pulled verbatim when present.
_SIM_ATTR_COLUMNS = [
    "simulation_type",
    "battery_capacity",
    "horizon",
    "failure_penalty",
    "observation_threshold",
    "wind_threshold",
    "location_id",
    "start_time",
    "episodes_count",
    "failure_percentage",
    "average_failure_step",
    "average_reward",
    "average_flight_hrs",
]


def _experiment_columns(config: Dict) -> Dict:
    """Experiment-level (config-wide) columns added to every row so each row is self-describing."""
    wind_chain = config.get("wind_chain") or {}
    hist = config.get("historical_weather") or {}
    if hist.get("enabled", False):
        weather_mode = "historical"
    elif wind_chain.get("enabled", False):
        weather_mode = "chain"
    else:
        weather_mode = "iid"
    return {
        "config_basename": config.get("_config_basename"),
        "weather_mode": weather_mode,
        "wind_chain_enabled": bool(wind_chain.get("enabled", False)),
        "transition_model": config.get("transition_model", "moderate"),
        "solar_panel_model": config.get("solar_panel_model", "constant"),
        "whale_series": config.get("whale_series", "real"),
        "energy_increment_wh": config.get("energy_increment_wh"),
        # Backend selectors: only vectorized/batched are implemented today. Reserved so a future
        # scalar (old-solver) run is distinguishable in the CSV without a schema change.
        "solver_backend": config.get("solver_backend", "vectorized"),
        "rollout_backend": config.get("rollout_backend", "batched"),
    }


def _attr(grp, key):
    """Read an HDF5 attr, decoding bytes and unwrapping 0-d numpy scalars to plain Python."""
    if key not in grp.attrs:
        return None
    val = grp.attrs[key]
    if isinstance(val, bytes):
        return val.decode("utf-8", "replace")
    if isinstance(val, np.generic):
        return val.item()
    return val


def summarize_hdf5(h5_path: str, config: Optional[Dict] = None) -> pd.DataFrame:
    """Build a tidy one-row-per-simulation DataFrame from an experiment's HDF5 file.

    Args:
        h5_path: path to the batch HDF5 written by SimulationRunManager.
        config:  the (resolved) experiment config; supplies the experiment-level columns.

    Returns:
        DataFrame with the sim-attr columns followed by the experiment-level columns.
    """
    config = config or {}
    exp_cols = _experiment_columns(config)

    rows = []
    with h5py.File(h5_path, "r") as f:
        for group_name in f.keys():
            grp = f[group_name]
            row = {"group": group_name}
            for col in _SIM_ATTR_COLUMNS:
                row[col] = _attr(grp, col)
            row.update(exp_cols)
            rows.append(row)

    df = pd.DataFrame(rows)
    if not df.empty:
        # Stable, human-scannable order: identity -> sim attrs -> experiment columns.
        ordered = ["group"] + _SIM_ATTR_COLUMNS + list(exp_cols.keys())
        df = df[[c for c in ordered if c in df.columns]]
        df = df.sort_values(by=[c for c in ["simulation_type", "battery_capacity",
                                            "wind_threshold", "observation_threshold"]
                                if c in df.columns]).reset_index(drop=True)
    return df


def write_summary_csv(h5_path: str, config: Dict, out_csv: str) -> pd.DataFrame:
    """Summarize `h5_path` and write the CSV to `out_csv`; returns the DataFrame."""
    df = summarize_hdf5(h5_path, config)
    df.to_csv(out_csv, index=False)
    return df

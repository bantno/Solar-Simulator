"""Verification for the configurable whale series and the takeoff/landing cruise fractions.

Checks:
  1. Backward compatibility of the block-series refactor: ``block_series`` with the
     built-in profile, delta_t=15 and a midnight start reproduces the original
     ``np.repeat(block_values, 8)`` / ``np.tile`` construction exactly, for a range of
     horizons; ``create_series("real", horizon)`` (legacy call) and
     ``create_series("custom", ...)`` with the same values agree.
  2. New options: delta_t=30 gives 4 steps per block; a 06:00 / 06:15 start rolls the
     series so stage 0 holds the 06:00-08:00 value; invalid inputs are rejected.
  3. Observation gain: default MDP has gain [[0,1],[0,1],[0,0]] and ``reward`` equals the
     original ``actions * samples``; custom fractions scale only the takeoff and landing
     stages.
  4. Value-table identity: the analytical solver on harness/examples/iid_small.yaml with
     defaults, with explicit (1.0, 0.0), and with ``whale_series: custom`` carrying the
     built-in values, produces bit-identical tables; (0.5, 0.0) changes it.
  5. ``"auto"`` fractions come from the seaplane climb/descent times and lie in (0, 1).
  6. ``summarize._fmt_list`` formats list-valued config entries.

Run (any cwd) with the pvlib conda env:
    conda run -n pvlib python SolarSimulator/Tests/verify_whale_model.py
"""
import copy
import os
import sys
import tempfile

import numpy as np
import yaml

TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
PKG_DIR = os.path.dirname(TESTS_DIR)
REPO_ROOT = os.path.dirname(PKG_DIR)
sys.path.insert(0, PKG_DIR)

from BaseClasses.whale_base import (  # noqa: E402
    DEFAULT_BLOCK_VALUES, WhaleRewardSeriesFactory, block_series,
)
from BaseClasses.run_sim import SimulationFactory  # noqa: E402
from BaseClasses.backward_induction_base import mdpAnalyticalBackwardSolver  # noqa: E402
from harness.summarize import _fmt_list  # noqa: E402

CONFIG_PATH = os.path.join(PKG_DIR, "harness", "examples", "iid_small.yaml")
CUSTOM_CONFIG_PATH = os.path.join(PKG_DIR, "harness", "examples", "whale_custom_small.yaml")


def legacy_real_series(horizon):
    """The original RealWhaleRewardSeries.create_series body, verbatim."""
    block_values = np.array([
        0.00, 0.00, 0.00, 0.082, 0.098, 0.095,
        0.217, 0.215, 0.183, 0.278, 0.000, 0.000,
    ])
    daily_series = np.repeat(block_values, 8)
    full_series = np.tile(daily_series, int(np.ceil(horizon / 96)))
    return full_series[:horizon]


def check(cond, msg):
    if not cond:
        raise AssertionError(msg)
    print("  ok -", msg)


# ---------------------------------------------------------------- 1. legacy equivalence
print("1. block-series refactor reproduces the original series")
for h in (1, 8, 95, 96, 97, 150, 672, 1000):
    new = WhaleRewardSeriesFactory.create_series("real", h)
    check(np.array_equal(new, legacy_real_series(h)), f"real == legacy for horizon {h}")
    custom = WhaleRewardSeriesFactory.create_series(
        "custom", h, block_values=list(DEFAULT_BLOCK_VALUES), block_hours=2)
    check(np.array_equal(custom, legacy_real_series(h)), f"custom(defaults) == legacy for horizon {h}")
explicit = WhaleRewardSeriesFactory.create_series("real", 150, delta_t=15, start_offset_min=0)
check(np.array_equal(explicit, legacy_real_series(150)), "explicit delta_t=15/offset=0 == legacy")

# ---------------------------------------------------------------- 2. new options
print("2. delta_t and start-offset handling")
s30 = block_series(DEFAULT_BLOCK_VALUES, 96, delta_t=30)
check(s30.size == 96 and np.all(s30[:4] == 0.0) and np.all(s30[12:16] == 0.082),
      "delta_t=30 -> 4 steps per block (block 3 at stages 12-15)")
s06 = block_series(DEFAULT_BLOCK_VALUES, 96, start_offset_min=360)
check(s06[0] == 0.082 and s06[8] == 0.098 and s06[72] == 0.0 and s06[96 - 24] == 0.0,
      "06:00 start -> stage 0 holds the 06:00-08:00 value, wraps past midnight")
s0615 = block_series(DEFAULT_BLOCK_VALUES, 96, start_offset_min=375)
check(s0615[0] == 0.082 and s0615[6] == 0.082 and s0615[7] == 0.098,
      "06:15 start -> 7 stages of block 3 remain before block 4")
for bad_kwargs, why in (
    (dict(block_values=[0.1] * 11), "11 blocks x 2 h != 24 h"),
    (dict(block_values=list(DEFAULT_BLOCK_VALUES), delta_t=25), "2 h not a multiple of 25 min"),
    (dict(block_values=[1.5] + [0.0] * 11), "value > 1"),
    (dict(block_values=list(DEFAULT_BLOCK_VALUES), start_offset_min=7), "offset not a multiple of delta_t"),
):
    try:
        block_series(horizon=96, **bad_kwargs)
    except ValueError:
        check(True, f"rejects {why}")
    else:
        raise AssertionError(f"should have rejected {why}")
try:
    WhaleRewardSeriesFactory.create_series("custom", 96)
except ValueError:
    check(True, "custom without block_values is rejected")
else:
    raise AssertionError("custom without block_values should fail")

# ---------------------------------------------------------------- 3. observation gain
print("3. observation gain and reward")
with open(CONFIG_PATH) as fh:
    base_cfg = yaml.safe_load(fh)
location = dict(base_cfg["locations"][0])
location["data_path"] = os.path.join(REPO_ROOT, location["data_path"])
HORIZON = 40          # midnight start -> covers the 06:00-10:00 non-zero blocks
PENALTY = 5.0
CAP = 300.0


def make_factory(cfg):
    return SimulationFactory(cfg, location, HORIZON, PENALTY)


def gain_config(takeoff, landing):
    cfg = copy.deepcopy(base_cfg)
    cfg["takeoff_cruise_fraction"] = takeoff
    cfg["landing_cruise_fraction"] = landing
    return cfg


mdp_default = make_factory(base_cfg).build_mdp(CAP)
check(np.array_equal(mdp_default.observation_gain, np.array([[0.0, 1.0], [0.0, 1.0], [0.0, 0.0]])),
      "default gain table is [[0,1],[0,1],[0,0]]")
states = np.array([[100.0, 0], [100.0, 1], [100.0, 1], [100.0, 0]])
actions = np.array([1, 1, 0, 0])
next_ok = np.array([[90.0, 1], [90.0, 1], [95.0, 0], [100.0, 0]])
t = 28  # 07:00 -> 0.082
p = mdp_default.env_provider.whale_reward_series[t]
check(p == 0.082, "stage 28 of a midnight start is 07:00 -> p = 0.082")
r_default = mdp_default.reward(states, actions, next_ok, t)
check(np.array_equal(r_default, actions * p), "default reward == actions * p(t) (original formula)")
next_fail = next_ok.copy()
next_fail[1, 1] = 2
r_fail = mdp_default.reward(states, actions, next_fail, t)
check(r_fail[1] == p - PENALTY and np.array_equal(np.delete(r_fail, 1), np.delete(r_default, 1)),
      "failure penalty still applied per lane")

mdp_half = make_factory(gain_config(0.5, 0.25)).build_mdp(CAP)
r_half = mdp_half.reward(states, actions, next_ok, t)
check(np.allclose(r_half, [0.5 * p, p, 0.25 * p, 0.0]),
      "takeoff 0.5 / landing 0.25 scale only the takeoff and landing stages")
try:
    make_factory(gain_config(1.5, 0.0)).build_mdp(CAP)
except ValueError:
    check(True, "fraction outside [0, 1] is rejected")
else:
    raise AssertionError("fraction 1.5 should be rejected")

# ---------------------------------------------------------------- 4. value-table identity
print("4. analytical solver value tables")
out_dir = tempfile.mkdtemp(prefix="verify_whale_")


def solve_table(cfg):
    mdp = make_factory(cfg).build_mdp(CAP)
    solver = mdpAnalyticalBackwardSolver(mdp, HORIZON, sim_name_prefix="verify", output_dir=out_dir)
    solver.set_start_date("2020-04-01 00:00:00")  # required before solve(), as in run_sim
    solver.solve()
    return solver.future_value_table.copy(), solver.optimal_action_table.copy()


V_default, A_default = solve_table(base_cfg)
V_explicit, A_explicit = solve_table(gain_config(1.0, 0.0))
check(np.array_equal(V_default, V_explicit) and np.array_equal(A_default, A_explicit),
      "explicit (1.0, 0.0) == defaults (values and actions)")
with open(CUSTOM_CONFIG_PATH) as fh:
    custom_cfg = yaml.safe_load(fh)
V_custom, A_custom = solve_table(custom_cfg)
check(np.array_equal(V_default, V_custom) and np.array_equal(A_default, A_custom),
      "whale_series: custom with the built-in values == real (values and actions)")
V_half, _ = solve_table(gain_config(0.5, 0.0))
check(not np.array_equal(V_default, V_half) and np.all(V_half <= V_default + 1e-12),
      "takeoff fraction 0.5 changes the table and never raises a value")
check(np.isfinite(V_default).all(), "value table is finite")

# ---------------------------------------------------------------- 5. auto fractions
print("5. auto cruise fractions")
fac_auto = make_factory(gain_config("auto", "auto"))
f_to, f_ld = fac_auto._resolve_cruise_fractions(CAP)
check(0.0 < f_to < 1.0 and 0.0 < f_ld < 1.0, f"auto fractions in (0,1): takeoff={f_to:.3f}, landing={f_ld:.3f}")
mdp_auto = fac_auto.build_mdp(CAP)
check(mdp_auto.observation_gain[0, 1] == f_to and mdp_auto.observation_gain[1, 0] == f_ld,
      "auto fractions land in the MDP gain table")
f_to2, _ = make_factory(gain_config("auto", 0.0))._resolve_cruise_fractions(CAP)
check(f_to2 == f_to, "auto takeoff with fixed landing resolves identically")

# ---------------------------------------------------------------- 6. summary formatting
print("6. summary column formatting")
check(_fmt_list(None) is None and _fmt_list([0.0, 0.082]) == "0.0;0.082", "_fmt_list")

# ---------------------------------------------------------------- 7. start-time roll via factory
print("7. start_datetime alignment through SimulationFactory")
cfg_0600 = copy.deepcopy(base_cfg)
cfg_0600["start_datetime"] = "2020-04-01T06:00:00"
series_0600 = make_factory(cfg_0600).env_provider.whale_reward_series
check(series_0600[0] == 0.082 and series_0600[8] == 0.098,
      "06:00 start -> provider series begins with the 06:00-08:00 value")

print("\nAll whale-model checks passed.")

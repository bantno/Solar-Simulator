"""Verification for the whale series (JoA2026 / custom / factored) and the cruise fractions.

Checks:
  1. Backward compatibility of the block-series refactor: ``block_series`` with the
     built-in profile, delta_t=15 and a midnight start reproduces the original
     ``np.repeat(block_values, 8)`` / ``np.tile`` construction exactly, for a range of
     horizons; ``create_series("JoA2026", horizon)`` (legacy call), the deprecated
     ``"real"`` alias, and ``create_series("custom", ...)`` with the same values agree.
  2. New options: delta_t=30 gives 4 steps per block; a 06:00 / 06:15 start rolls the
     series so stage 0 holds the 06:00-08:00 value; invalid inputs are rejected.
  3. Observation gain: default MDP has gain [[0,1],[0,1],[0,0]] and ``reward`` equals the
     original ``actions * samples``; custom fractions scale only the takeoff and landing
     stages.
  4. Value-table identity: the analytical solver on harness/examples/iid_small.yaml with
     defaults, with explicit (1.0, 0.0), with ``whale_series: custom`` carrying the
     built-in values, and with ``whale_series: encounter`` in its focal form
     (kogia_spp card, g0 = 1, no density), produces bit-identical tables; (0.5, 0.0) changes it.
  5. ``"auto"`` fractions come from the seaplane climb/descent times and lie in (0, 1).
  6. ``summarize._fmt_list`` formats list-valued config entries.
  7. ``start_datetime`` alignment through SimulationFactory.
  8. Encounter model: focal form with g0 = 1 == JoA2026; species cards (kogia_spp, humpback,
     overrides, bare); swept area; density forms (scalar / monthly list / month map);
     lambda -> 1 - exp(-lambda) against hand-computed values; fixed-hours and clear-sky light
     gates; through SimulationFactory with a location density and the clear-sky gate.

Run (any cwd) with the pvlib conda env:
    conda run -n pvlib python SolarSimulator/Tests/verify_whale_model.py
"""
import copy
import os
import sys
import tempfile
import warnings

import numpy as np
import yaml

TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
PKG_DIR = os.path.dirname(TESTS_DIR)
REPO_ROOT = os.path.dirname(PKG_DIR)
sys.path.insert(0, PKG_DIR)

from BaseClasses.whale_base import (  # noqa: E402
    JOA2026_BLOCK_VALUES, SCOTT2001_SURFACE_PROPORTIONS, SPECIES_CARDS, WhaleRewardSeriesFactory,
    block_series, resolve_density, stage_time_of_day_min, swept_area_km2,
)
from BaseClasses.run_sim import SimulationFactory  # noqa: E402
from BaseClasses.backward_induction_base import mdpAnalyticalBackwardSolver  # noqa: E402
from harness.summarize import _fmt_list  # noqa: E402

CONFIG_PATH = os.path.join(PKG_DIR, "harness", "examples", "iid_small.yaml")
CUSTOM_CONFIG_PATH = os.path.join(PKG_DIR, "harness", "examples", "whale_custom_small.yaml")
ENCOUNTER_CONFIG_PATH = os.path.join(PKG_DIR, "harness", "examples", "whale_encounter_small.yaml")


def legacy_series(horizon):
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
    new = WhaleRewardSeriesFactory.create_series("JoA2026", h)
    check(np.array_equal(new, legacy_series(h)), f"JoA2026 == legacy for horizon {h}")
    custom = WhaleRewardSeriesFactory.create_series(
        "custom", h, block_values=list(JOA2026_BLOCK_VALUES), block_hours=2)
    check(np.array_equal(custom, legacy_series(h)), f"custom(JoA2026 values) == legacy for horizon {h}")
explicit = WhaleRewardSeriesFactory.create_series("JoA2026", 150, delta_t=15, start_offset_min=0)
check(np.array_equal(explicit, legacy_series(150)), "explicit delta_t=15/offset=0 == legacy")
with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    aliased = WhaleRewardSeriesFactory.create_series("real", 150)
check(np.array_equal(aliased, legacy_series(150))
      and any(issubclass(w.category, DeprecationWarning) for w in caught),
      "'real' alias resolves to JoA2026 with a DeprecationWarning")

# ---------------------------------------------------------------- 2. new options
print("2. delta_t and start-offset handling")
s30 = block_series(JOA2026_BLOCK_VALUES, 96, delta_t=30)
check(s30.size == 96 and np.all(s30[:4] == 0.0) and np.all(s30[12:16] == 0.082),
      "delta_t=30 -> 4 steps per block (block 3 at stages 12-15)")
s06 = block_series(JOA2026_BLOCK_VALUES, 96, start_offset_min=360)
check(s06[0] == 0.082 and s06[8] == 0.098 and s06[72] == 0.0 and s06[96 - 24] == 0.0,
      "06:00 start -> stage 0 holds the 06:00-08:00 value, wraps past midnight")
s0615 = block_series(JOA2026_BLOCK_VALUES, 96, start_offset_min=375)
check(s0615[0] == 0.082 and s0615[6] == 0.082 and s0615[7] == 0.098,
      "06:15 start -> 7 stages of block 3 remain before block 4")
for bad_kwargs, why in (
    (dict(block_values=[0.1] * 11), "11 blocks x 2 h != 24 h"),
    (dict(block_values=list(JOA2026_BLOCK_VALUES), delta_t=25), "2 h not a multiple of 25 min"),
    (dict(block_values=[1.5] + [0.0] * 11), "value > 1"),
    (dict(block_values=list(JOA2026_BLOCK_VALUES), start_offset_min=7), "offset not a multiple of delta_t"),
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


def make_factory(cfg, horizon=HORIZON, loc=None):
    """SimulationFactory takes the location separately; density_per_km2 lives on it."""
    return SimulationFactory(cfg, location if loc is None else loc, horizon, PENALTY)


def location_of(cfg):
    loc = dict(cfg["locations"][0])
    loc["data_path"] = os.path.join(REPO_ROOT, loc["data_path"])
    return loc


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
      "whale_series: custom with the built-in values == JoA2026 (values and actions)")
with open(ENCOUNTER_CONFIG_PATH) as fh:
    encounter_cfg = yaml.safe_load(fh)
focal_cfg = copy.deepcopy(encounter_cfg)
focal_cfg["whale_species"] = {"name": "kogia_spp", "g0": 1.0}
focal_cfg["locations"][0].pop("density_per_km2", None)
V_focal, A_focal = solve_table(focal_cfg)
check(np.array_equal(V_default, V_focal) and np.array_equal(A_default, A_focal),
      "whale_series: encounter, focal form with g0 = 1 == JoA2026 (values and actions)")
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
check(_fmt_list(None) is None and _fmt_list([0.0, 0.082]) == "0.0;0.082"
      and _fmt_list("scott2001") == "scott2001" and _fmt_list(1.0) == 1.0, "_fmt_list")

# ---------------------------------------------------------------- 7. start-time roll via factory
print("7. start_datetime alignment through SimulationFactory")
cfg_0600 = copy.deepcopy(base_cfg)
cfg_0600["start_datetime"] = "2020-04-01T06:00:00"
series_0600 = make_factory(cfg_0600).env_provider.whale_reward_series
check(series_0600[0] == 0.082 and series_0600[8] == 0.098,
      "06:00 start -> provider series begins with the 06:00-08:00 value")

# ---------------------------------------------------------------- 8. encounter model
print("8. encounter model")
H = 4 * 96
FOCAL = {"name": "kogia_spp", "g0": 1.0}
enc = WhaleRewardSeriesFactory.create_series("encounter", H, species=FOCAL)
check(np.array_equal(enc, legacy_series(H)), "focal kogia_spp with g0 = 1 == JoA2026 (4 days)")
enc_off = WhaleRewardSeriesFactory.create_series("encounter", H, species=FOCAL, start_offset_min=375)
check(np.array_equal(enc_off, block_series(JOA2026_BLOCK_VALUES, H, start_offset_min=375)),
      "focal form == JoA2026 with a 06:15 start")
raw = WhaleRewardSeriesFactory.create_series("encounter", 96, species=FOCAL, light={"mode": "none"})
check(np.array_equal(raw, np.repeat(SCOTT2001_SURFACE_PROPORTIONS, 8)),
      "light none -> the full Scott 2001 profile (night blocks 0.073/0.093/0.065/0.204/0.090)")

# Species cards.
kog = WhaleRewardSeriesFactory.create_series("encounter", 96)   # default card, card g0
check(np.allclose(kog, SPECIES_CARDS["kogia_spp"]["g0"] * legacy_series(96)),
      f"default card kogia_spp applies its g0 = {SPECIES_CARDS['kogia_spp']['g0']}")
hb = WhaleRewardSeriesFactory.create_series("encounter", 96, species="humpback")
day = legacy_series(96) > 0
check(np.allclose(hb[day], 0.63 * 0.91) and np.all(hb[~day] == 0.0),
      "humpback card: constant availability 0.63 x g0 0.91 by day, gated at night")
bare = WhaleRewardSeriesFactory.create_series("encounter", 96, species={"availability_profile": 0.5, "g0": 0.5})
check(np.allclose(bare[day], 0.25), "bare species mapping (availability 0.5, g0 0.5)")
ovr = WhaleRewardSeriesFactory.create_series("encounter", 96, species={"name": "humpback", "g0": 1.0})
check(np.allclose(ovr[day], 0.63), "card override replaces g0 only")
for bad, why in (({"name": "orca"}, "unknown card"), ({"g0": 0.5}, "bare mapping without availability"),
                 ({"name": "kogia_spp", "g0": 1.5}, "g0 > 1"),
                 ({"availability_profile": "tag2020", "g0": 1.0}, "unknown availability profile")):
    try:
        WhaleRewardSeriesFactory.create_series("encounter", 96, species=bad)
    except ValueError:
        check(True, f"rejects {why}")
    else:
        raise AssertionError(f"should reject {why}")

# Swept area and density forms.
A = swept_area_km2(0.4, 18.8, 15)
check(np.isclose(A, 0.4 * 18.8 * 900 / 1000), f"swept area 0.4 km x 18.8 m/s x 15 min = {A:.3f} km^2")
check(resolve_density(0.01) == 0.01 and resolve_density(None) is None, "scalar / None density")
check(resolve_density(list(range(1, 13)), start_month=4) == 4.0, "12-entry monthly density picks the start month")
check(resolve_density({4: 0.02, "5": 0.03}, start_month=5) == 0.03, "{month: value} density (string keys accepted)")
for bad, kw, why in (([1, 2, 3], {"start_month": 4}, "monthly list of the wrong length"),
                     ({4: 0.02}, {"start_month": 5}, "missing month"),
                     ([0.1] * 12, {}, "monthly list without start_month"),
                     (-0.1, {}, "negative density")):
    try:
        resolve_density(bad, **kw)
    except ValueError:
        check(True, f"rejects {why}")
    else:
        raise AssertionError(f"should reject {why}")

# Poisson form against hand computation.
PLAT = {"strip_width_km": 0.4, "ground_speed_ms": 18.8}
D = 0.01
srch = WhaleRewardSeriesFactory.create_series("encounter", 96, species=FOCAL, platform=PLAT, density_per_km2=D)
lam = D * A * legacy_series(96)
check(np.allclose(srch, 1.0 - np.exp(-lam)), "p = 1 - exp(-D A a g0 light) matches hand computation")
check(srch[28] < 0.01 and np.all(srch <= legacy_series(96) + 1e-12),
      f"search form is small (07:00: {srch[28]:.4f} vs focal 0.082) and never exceeds the focal form")
srch2 = WhaleRewardSeriesFactory.create_series("encounter", 96, species=FOCAL, platform=PLAT, density_per_km2=2 * D)
check(np.allclose(-np.log(1 - srch2), 2 * lam), "lambda is linear in density")
try:
    WhaleRewardSeriesFactory.create_series("encounter", 96, species=FOCAL, density_per_km2=D)
except ValueError:
    check(True, "density without a platform block is rejected")
else:
    raise AssertionError("density without platform should fail")

# Light gates.
tod = stage_time_of_day_min(96)
g7 = WhaleRewardSeriesFactory.create_series("encounter", 96, species=FOCAL,
                                            light={"mode": "fixed_hours", "day_start": 7, "day_end": 19})
check(np.all(g7[(tod < 420) | (tod >= 1140)] == 0.0) and np.all(g7[(tod >= 420) & (tod < 1140)] > 0.0),
      "fixed_hours 07-19 gates at stage resolution")
cs = np.where((tod >= 6 * 60 + 40) & (tod < 19 * 60 + 5), 500.0, 0.0)   # synthetic sun 06:40-19:05
gcs = WhaleRewardSeriesFactory.create_series("encounter", 96, species=FOCAL, light={"mode": "clearsky_gate"}, solar_cs=cs)
check(np.array_equal(gcs > 0, cs > 0) and np.array_equal(gcs[cs > 0], raw[cs > 0]),
      "clearsky_gate follows the clear-sky series and leaves daytime values untouched")
for bad, why in ((dict(light={"mode": "clearsky_gate"}), "clearsky_gate without solar_cs"),
                 (dict(light={"mode": "lunar"}), "unknown light mode")):
    try:
        WhaleRewardSeriesFactory.create_series("encounter", 96, species=FOCAL, **bad)
    except ValueError:
        check(True, f"rejects {why}")
    else:
        raise AssertionError(f"should reject {why}")

# Through SimulationFactory: location density + clear-sky gate on real location data (4 days).
cfg_cs = copy.deepcopy(encounter_cfg)
cfg_cs["whale_light"] = {"mode": "clearsky_gate", "threshold_wm2": 0.0}
fac_cs = make_factory(cfg_cs, horizon=H, loc=location_of(cfg_cs))
prov = fac_cs.env_provider
ser_cs = prov.whale_reward_series
expect = 1.0 - np.exp(-encounter_cfg["locations"][0]["density_per_km2"] * A
                      * SPECIES_CARDS["kogia_spp"]["g0"]
                      * block_series(SCOTT2001_SURFACE_PROPORTIONS, H) * (prov.solar_cs[:H] > 0))
check(np.allclose(ser_cs, expect) and np.all(ser_cs[prov.solar_cs[:H] <= 0] == 0.0),
      "SimulationFactory composes D (location) x A (platform) x a (card) x g0 x clear-sky gate")
print(f"     (Gulf Kogia example: D={encounter_cfg['locations'][0]['density_per_km2']}, A={A:.2f} km^2, "
      f"midday p = {ser_cs[52]:.4f} vs JoA2026 {legacy_series(96)[52]:.3f})")
cfg_focal = copy.deepcopy(encounter_cfg)
cfg_focal["locations"][0].pop("density_per_km2")
check(np.allclose(make_factory(cfg_focal, horizon=H, loc=location_of(cfg_focal)).env_provider.whale_reward_series,
                  SPECIES_CARDS["kogia_spp"]["g0"] * legacy_series(H)),
      "location without density -> focal form through SimulationFactory")

print("\nAll whale-model checks passed.")

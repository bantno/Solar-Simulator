"""Verification for cross-model rollouts (config key `synthetic_world`).

Checks, on the lat 30 / lon -75 hold-out files with a 4-bin chain and a short horizon:
  1. Weather identity: every policy rolled out in one world sees the same weather. The
     i.i.d. policy's rollout provider in the chain world draws exactly the wind the chain
     policy's own provider draws for the same seed, and likewise in the i.i.d. world.
  2. Bin recovery: wrapping the chain model's own provider in CrossModelEnvironmentProvider
     (bins read back from realized wind, as for historical weather) reproduces the chain
     policy's native rollout episode for episode, so the wrapper reports the bins the
     policy would have seen.
  3. The i.i.d. policy in its own world and in the chain world, and the chain policy in
     the i.i.d. world, all run; worlds differ (different weather).
  4. A threshold config with the chain enabled and no historical weather rolls out in
     the chain world (same weather as the chain policy).

Run (any cwd) with the pvlib conda env:
    conda run -n pvlib python SolarSimulator/Tests/verify_model_worlds.py
"""
import copy
import os
import sys

import numpy as np

TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
PKG_DIR = os.path.dirname(TESTS_DIR)
sys.path.insert(0, PKG_DIR)

from BaseClasses.environment_provider_base import CrossModelEnvironmentProvider  # noqa: E402
from BaseClasses.run_sim import SimulationFactory  # noqa: E402
from harness.run_experiment import _ensure_location_data, _resolve_paths_in_place  # noqa: E402

N_EP = 400
H = 672  # 7 days


def check(name, cond):
    print(f"[{'PASS' if cond else 'FAIL'}] {name}")
    if not cond:
        sys.exit(1)


def config(chain, world=None):
    cfg = {
        "transition_model": "moderate", "solar_panel_model": "constant",
        "whale_series": "JoA2026", "energy_increment_wh": 5, "delta_t": 15,
        "start_datetime": "2025-06-10T00:00:00",
        "locations": [{"latitude": 30.0, "longitude": -75.0,
                       "data_path": "Data/EXPECTED_DATA/data_expected_lat30.0_lon-75.0_15min.pkl"}],
        "wind_chain": {"enabled": chain, "n_bins": 4},
    }
    if world:
        cfg["synthetic_world"] = world
    _resolve_paths_in_place(cfg)
    _ensure_location_data(cfg, cfg["locations"][0])
    return cfg


def factory(cfg):
    return SimulationFactory(cfg, cfg["locations"][0], H, 20.0, config_name="verify_worlds")


def rollout(sim):
    eps = list(sim.simulate_multiple_episodes(N_EP))
    return (np.array([e["total_reward"] for e in eps]),
            np.array([e.get("failure_step", -1) for e in eps]))


def winds(provider, steps=50):
    provider.reset(0)
    return np.stack([provider.sample_wind_speed(t, 64) for t in range(steps)])


def main():
    f_chain = factory(config(True))                    # chain policy, chain world
    f_iid = factory(config(False))                     # iid policy, iid world
    f_iid_cw = factory(config(False, "chain"))         # iid policy, chain world
    f_chain_iw = factory(config(True, "iid"))          # chain policy, iid world
    f_thr_cw = factory(config(True))                   # threshold config, chain enabled

    print("1. weather identity within a world")
    check("own-world runs build no cross-model provider",
          f_chain.sim_env_provider is None and f_iid.sim_env_provider is None)
    check("iid policy in chain world: wrapper reports no chain to the policy",
          isinstance(f_iid_cw.sim_env_provider, CrossModelEnvironmentProvider)
          and f_iid_cw.sim_env_provider.use_exo_chain is False)
    check("chain world: iid policy's rollout wind equals the chain policy's",
          np.array_equal(winds(f_iid_cw.sim_env_provider), winds(f_chain.env_provider)))
    check("iid world: chain policy's rollout wind equals the iid policy's",
          np.array_equal(winds(f_chain_iw.sim_env_provider), winds(f_iid.env_provider)))
    check("the two worlds differ",
          not np.array_equal(winds(f_chain.env_provider), winds(f_iid.env_provider)))

    print("2. bin recovery through the wrapper")
    native = f_chain.create_simulation("optimal", 300.0)
    r_native, s_native = rollout(native)
    wrapped = copy.copy(native)
    wrapped.env_provider = CrossModelEnvironmentProvider(f_chain.env_provider,
                                                         f_chain.env_provider)
    r_wrap, s_wrap = rollout(wrapped)
    check(f"chain policy through the wrapper reproduces its native rollout "
          f"({N_EP} episodes)",
          np.array_equal(r_native, r_wrap) and np.array_equal(s_native, s_wrap))

    print("3. cross-world rollouts run")
    r_iid, _ = rollout(f_iid.create_simulation("optimal", 300.0))
    r_iid_cw, _ = rollout(f_iid_cw.create_simulation("optimal", 300.0))
    r_chain_iw, _ = rollout(f_chain_iw.create_simulation("optimal", 300.0))
    print(f"    mean reward  chain@chain {r_native.mean():.3f}  iid@iid {r_iid.mean():.3f}  "
          f"iid@chain {r_iid_cw.mean():.3f}  chain@iid {r_chain_iw.mean():.3f}")
    check("cross-world rewards are finite and differ from own-world rewards",
          np.isfinite([r_iid_cw.mean(), r_chain_iw.mean()]).all()
          and not np.array_equal(r_iid, r_iid_cw)
          and not np.array_equal(r_native, r_chain_iw))

    print("4. threshold in the chain world")
    thr = f_thr_cw.create_simulation("threshold", 300.0, threshold=0.1, wind_threshold=5.0)
    check("threshold rollout provider is the chain model's",
          np.array_equal(winds(thr.env_provider), winds(f_chain.env_provider)))
    print("All model-world checks passed.")


if __name__ == "__main__":
    main()

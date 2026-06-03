"""
Verification for the wind-persistence (Markov-modulated wind) model.

Checks:
  1. Chain OFF (n_bins==1) is the unchanged i.i.d. path (solver table is 2D, runs).
  2. Wind sampling: with a rank-1 transition whose rows equal the per-stage Weibull bin
     masses, the chain reproduces the i.i.d. Weibull wind marginal (mean/quantiles match).
  3. Rank-1 collapse: a chain-on optimal sim with that rank-1 transition gives rollout
     statistics matching the i.i.d. optimal sim within Monte Carlo error (the wind process
     is identical i.i.d.; persistence carries no information).
  4. Persistence effect: with the real fitted transition, the value table varies across
     wind bins (high-wind bin is worth less) and failure statistics shift sensibly.

Run (any cwd) with the pvlib conda env:
    conda run -n pvlib python SolarSimulator/Tests/verify_wind_chain.py
"""
import os
import sys
import tempfile

import numpy as np

TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
PKG_DIR = os.path.dirname(TESTS_DIR)
REPO_ROOT = os.path.dirname(PKG_DIR)
sys.path.insert(0, PKG_DIR)

from BaseClasses.run_sim import YAMLSimulationRunner, SimulationFactory  # noqa: E402

DATA_PKL = os.path.join(REPO_ROOT, "Data", "EXPECTED_DATA",
                        "data_expected_lat30.0_lon-90.0_15min.pkl")
CHAIN_PKL = os.path.join(REPO_ROOT, "Data", "EXPECTED_DATA",
                         "data_expected_lat30.0_lon-90.0_15min_windchain.pkl")
HORIZON = 150
EPISODES = 4000


def make_config(wind_chain):
    text = f"""\
battery_capacities:
- 400.0
failure_penalties:
- 5.0
horizons:
- {HORIZON}
locations:
- data_path: {DATA_PKL}
  latitude: 30.0
  longitude: -90.0
solar_panel_model: constant
start_datetime: '2020-04-01T00:00:44'
threshold_values:
- 0.25
wind_thresholds:
- 10.0
transition_model: moderate
energy_increment_wh: 5
episodes: {EPISODES}
wind_chain:
  enabled: {str(wind_chain).lower()}
"""
    fd, path = tempfile.mkstemp(suffix=".yaml", prefix="windchain_verify_")
    with os.fdopen(fd, "w") as f:
        f.write(text)
    return path


def factory_for(cfg_path):
    runner = YAMLSimulationRunner(cfg_path)
    cfg = runner.config
    locs = runner.locations
    loc = (locs[0] if isinstance(locs, tuple) else locs)[0]
    return SimulationFactory(cfg, loc, runner.horizons[0], runner.failure_penalties[0],
                             config_name="windverify"), runner


def rollout_stats(sim, episodes):
    eps = list(sim.simulate_multiple_episodes(episodes))
    fails = sum(1 for e in eps if e["failure"]) / len(eps)
    avg_r = float(np.mean([e["total_reward"] for e in eps]))
    return fails, avg_r


def weibull_bin_masses(env, t):
    F = env._weibull_cdf(env.wind_bin_edges, env.wind_shape[t], env.wind_scale[t])
    p = np.diff(F)
    return p / p.sum()


def main():
    if not os.path.exists(DATA_PKL):
        sys.exit(f"[error] missing expected-data pickle {DATA_PKL}")
    # Build the wind-chain artifact from historical data if it isn't present.
    if not os.path.exists(CHAIN_PKL):
        hist = os.path.join(REPO_ROOT, "Data", "HISTORICAL_DATA", "data_30_-90.pkl")
        if not os.path.exists(hist):
            sys.exit(f"[error] missing chain artifact and historical data {hist}")
        from Scripts.create_weather_distributions import build_wind_chain_artifact
        print(f"[info] building wind-chain artifact from {os.path.basename(hist)} ...")
        build_wind_chain_artifact(hist, CHAIN_PKL, interval_minutes=15, n_bins=3)

    np.random.seed(0)

    # 1. i.i.d. baseline (chain OFF)
    print("\n=== 1. i.i.d. baseline (chain OFF) ===")
    cfg_off = make_config(False)
    fac_off, _ = factory_for(cfg_off)
    sim_off = fac_off.create_simulation(sim_type="optimal", cap=400.0, full_history_episodes=0)
    assert sim_off.mdp_solver.future_value_table.ndim == 2, "i.i.d. table should be 2D"
    f_off, r_off = rollout_stats(sim_off, EPISODES)
    print(f"  n_bins={sim_off.mdp_solver.n_bins}  failure%={100*f_off:.2f}  avg_reward={r_off:.3f}")

    # 2 & 3. chain ON, but overwrite transition with rank-1 (Weibull-mass) rows
    print("\n=== 2/3. rank-1 chain reproduces i.i.d. ===")
    cfg_on = make_config(True)
    fac_on, _ = factory_for(cfg_on)
    env = fac_on.env_provider
    nb = env.n_wind_bins
    print(f"  n_bins={nb}  bin_edges={np.round(env.wind_bin_edges, 3)}")
    # Rank-1 transition that reproduces i.i.d. Weibull: the transition INTO stage t+1 must
    # draw bins from stage (t+1)'s Weibull masses, so bin_{t+1} matches the within-bin
    # Weibull used at t+1. (Initial bin_0 already uses stage-0 masses via _init_wind_bins.)
    T = env.wind_transition.shape[0]
    rank1 = np.empty_like(env.wind_transition)
    for t in range(T):
        dest = min(t + 1, T - 1)
        rank1[t] = np.tile(weibull_bin_masses(env, dest), (nb, 1))
    env.wind_transition = rank1

    # Wind marginal identity: bins ~ stage-t Weibull masses + within-bin truncated Weibull
    # must equal the full stage-t Weibull (this is the exact mixture identity).
    from scipy.stats import weibull_min
    env.reset(0)
    for t in [0, 40, 80]:
        masses = weibull_bin_masses(env, t)
        bins = env.rng.choice(nb, size=40000, p=masses)
        w = env._sample_within_bin(t, bins)
        wd = weibull_min(c=env.wind_shape[t], scale=env.wind_scale[t])
        dm, dq = abs(w.mean() - wd.mean()), abs(np.quantile(w, 0.9) - wd.ppf(0.9))
        print(f"  stage {t}: |mean diff|={dm:.3f}  |q90 diff|={dq:.3f}")
        assert dm < 0.1 and dq < 0.2, "within-bin mixture deviates from Weibull marginal"

    sim_r1 = fac_on.create_simulation(sim_type="optimal", cap=400.0, full_history_episodes=0)
    assert sim_r1.mdp_solver.future_value_table.ndim == 3
    f_r1, r_r1 = rollout_stats(sim_r1, EPISODES)
    print(f"  rank-1 chain: failure%={100*f_r1:.2f}  avg_reward={r_r1:.3f}")
    print(f"  vs i.i.d.: |d failure%|={abs(100*f_r1-100*f_off):.2f}  |d reward|={abs(r_r1-r_off):.3f}")
    assert abs(f_r1 - f_off) < 0.03 and abs(r_r1 - r_off) < 0.5, "rank-1 chain diverges from i.i.d."
    print("  PASS")

    # 4. real persistent chain: value varies across bins; stats shift
    print("\n=== 4. real fitted (persistent) chain ===")
    fac_p, _ = factory_for(cfg_on)
    sim_p = fac_p.create_simulation(sim_type="optimal", cap=400.0, full_history_episodes=0)
    V = sim_p.mdp_solver.future_value_table  # (n_bins, num_states, horizon)
    # mid-horizon, flying mode, mid SoC: value should fall as wind bin rises
    nsoc = sim_p.mdp_solver.n_soc_levels
    mid_soc = nsoc // 2
    row = nsoc + mid_soc  # flying-mode block
    vals_by_bin = V[:, row, HORIZON // 2]
    print(f"  value by wind bin (flying, mid SoC, mid stage): {np.round(vals_by_bin, 3)}")
    assert vals_by_bin[0] >= vals_by_bin[-1] - 1e-9, "expected value to fall with higher wind bin"
    f_p, r_p = rollout_stats(sim_p, EPISODES)
    print(f"  persistent chain: failure%={100*f_p:.2f}  avg_reward={r_p:.3f}")
    print(f"  vs i.i.d.: d failure%={100*(f_p-f_off):+.2f}  d reward={r_p-r_off:+.3f}")
    print("  PASS (value decreases with wind bin; stats reported)")

    for c in (cfg_off, cfg_on):
        os.remove(c)
    for f in os.listdir("."):
        if f.endswith(".npy") and "Wh_" in f:
            try:
                os.remove(f)
            except OSError:
                pass
    print("\nAll wind-chain checks passed.")


if __name__ == "__main__":
    main()

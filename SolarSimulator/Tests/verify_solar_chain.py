"""Verification for the solar-persistence (Markov-modulated clear-sky index) model.

Checks:
  1. Chain OFF (n_solar_bins==1) is the unchanged i.i.d. path (solver table is 2D, runs).
  2. Rank-1 solar chain reproduces i.i.d.: solar bins are the stage Beta's quantile
     bands (mass exactly 1/n), so the memoryless chain is the UNIFORM transition; with
     it, within-bin sampling must reproduce the full stage-Beta marginal (mixture
     identity) and rollout statistics must match the i.i.d. run within MC error.
  3. Persistence effect: with the real fitted chain, the value table RISES with the
     solar bin at a solar-valid mid-day stage (a clear regime is worth more) and the
     failure/reward statistics shift.
  4. Joint wind+solar: the value table is (n_wind*n_solar, |S|, T); with the solar
     factor overwritten to uniform (solar-memoryless) the joint rollout matches the
     wind-only chain within MC error - validates the Kronecker/joint-index plumbing.
  5. Night/dawn structure: the per-stage solar transition stack is identity exactly at
     the held (night) stages except one dawn matrix per night gap; degenerate
     (saturated, kappa ~ 1e6) stage Betas give finite conditional CDFs and samples.

Run (any cwd) with the pvlib conda env:
    conda run -n pvlib python SolarSimulator/Tests/verify_solar_chain.py
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
SOLAR_CHAIN_PKL = os.path.join(REPO_ROOT, "Data", "EXPECTED_DATA",
                               "data_expected_lat30.0_lon-90.0_15min_solarchain.pkl")
WIND_CHAIN_PKL = os.path.join(REPO_ROOT, "Data", "EXPECTED_DATA",
                              "data_expected_lat30.0_lon-90.0_15min_windchain.pkl")
HORIZON = 150
EPISODES = 4000


def make_config(solar_chain, wind_chain=False):
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
solar_chain:
  enabled: {str(solar_chain).lower()}
wind_chain:
  enabled: {str(wind_chain).lower()}
"""
    fd, path = tempfile.mkstemp(suffix=".yaml", prefix="solarchain_verify_")
    with os.fdopen(fd, "w") as f:
        f.write(text)
    return path


def factory_for(cfg_path):
    runner = YAMLSimulationRunner(cfg_path)
    cfg = runner.config
    locs = runner.locations
    loc = (locs[0] if isinstance(locs, tuple) else locs)[0]
    return SimulationFactory(cfg, loc, runner.horizons[0], runner.failure_penalties[0],
                             config_name="solarverify"), runner


def rollout_stats(sim, episodes):
    eps = list(sim.simulate_multiple_episodes(episodes))
    fails = sum(1 for e in eps if e["failure"]) / len(eps)
    avg_r = float(np.mean([e["total_reward"] for e in eps]))
    return fails, avg_r


def main():
    if not os.path.exists(DATA_PKL):
        sys.exit(f"[error] missing expected-data pickle {DATA_PKL}")
    hist = os.path.join(REPO_ROOT, "Data", "HISTORICAL_DATA", "data_30_-90.pkl")
    from Scripts.create_weather_distributions import (
        build_solar_chain_artifact, build_wind_chain_artifact,
    )
    if not os.path.exists(SOLAR_CHAIN_PKL):
        if not os.path.exists(hist):
            sys.exit(f"[error] missing chain artifact and historical data {hist}")
        print(f"[info] building solar-chain artifact from {os.path.basename(hist)} ...")
        build_solar_chain_artifact(hist, SOLAR_CHAIN_PKL, latitude=30.0, longitude=-90.0,
                                   interval_minutes=15, n_bins=3)
    if not os.path.exists(WIND_CHAIN_PKL):
        print(f"[info] building wind-chain artifact from {os.path.basename(hist)} ...")
        build_wind_chain_artifact(hist, WIND_CHAIN_PKL, interval_minutes=15, n_bins=3)

    np.random.seed(0)

    # 1. i.i.d. baseline (chain OFF)
    print("\n=== 1. i.i.d. baseline (solar chain OFF) ===")
    cfg_off = make_config(False)
    fac_off, _ = factory_for(cfg_off)
    sim_off = fac_off.create_simulation(sim_type="optimal", cap=400.0, full_history_episodes=0)
    assert sim_off.mdp_solver.future_value_table.ndim == 2, "i.i.d. table should be 2D"
    f_off, r_off = rollout_stats(sim_off, EPISODES)
    print(f"  n_bins={sim_off.mdp_solver.n_bins}  failure%={100*f_off:.2f}  avg_reward={r_off:.3f}")

    # 2. rank-1 (uniform = memoryless quantile chain) reproduces i.i.d.
    print("\n=== 2. rank-1 (uniform) solar chain reproduces i.i.d. ===")
    cfg_on = make_config(True)
    fac_on, _ = factory_for(cfg_on)
    env = fac_on.env_provider
    ng = env.n_solar_bins
    print(f"  n_solar_bins={ng}  valid stages={int(env.solar_valid.sum())}/{HORIZON}")
    # Quantile bins have mass exactly 1/n at every stage, so the memoryless chain is
    # the uniform transition at EVERY stage (dawn and identity slots included).
    env.solar_transition = np.full_like(env.solar_transition, 1.0 / ng)

    # Mixture identity at solar-valid stages: uniform bins + within-bin quantile
    # sampling must reproduce the full stage-Beta marginal.
    from scipy.stats import beta as beta_dist
    env.reset(0)
    valid_stages = np.nonzero(env.solar_valid)[0]
    for t in valid_stages[[0, len(valid_stages) // 2, -1]]:
        bins = env.rng.integers(0, ng, size=40000)
        k = env._sample_solar_index_within_bin(int(t), bins)
        bd = beta_dist(env.solar_alpha[t], env.solar_beta[t])
        dm = abs(k.mean() - bd.mean())
        dq = abs(np.quantile(k, 0.9) - bd.ppf(0.9))
        print(f"  stage {t}: |mean diff|={dm:.4f}  |q90 diff|={dq:.4f}")
        assert dm < 0.01 and dq < 0.02, "within-bin mixture deviates from Beta marginal"

    sim_r1 = fac_on.create_simulation(sim_type="optimal", cap=400.0, full_history_episodes=0)
    assert sim_r1.mdp_solver.future_value_table.ndim == 3
    assert sim_r1.mdp_solver.future_value_table.shape[0] == ng
    f_r1, r_r1 = rollout_stats(sim_r1, EPISODES)
    print(f"  rank-1 chain: failure%={100*f_r1:.2f}  avg_reward={r_r1:.3f}")
    print(f"  vs i.i.d.: |d failure%|={abs(100*f_r1-100*f_off):.2f}  |d reward|={abs(r_r1-r_off):.3f}")
    assert abs(f_r1 - f_off) < 0.03 and abs(r_r1 - r_off) < 0.5, "rank-1 chain diverges from i.i.d."
    print("  PASS")

    # 3. real persistent chain: value rises with the solar bin; stats shift
    print("\n=== 3. real fitted (persistent) solar chain ===")
    fac_p, _ = factory_for(cfg_on)
    envp = fac_p.env_provider
    sim_p = fac_p.create_simulation(sim_type="optimal", cap=400.0, full_history_episodes=0)
    V = sim_p.mdp_solver.future_value_table  # (n_g, num_states, horizon)
    nsoc = sim_p.mdp_solver.n_soc_levels
    row = nsoc + nsoc // 2  # flying-mode block, mid SoC
    # a solar-valid mid-day stage in the middle of the horizon
    valid_stages = np.nonzero(envp.solar_valid)[0]
    t_mid = int(valid_stages[len(valid_stages) // 2])
    vals_by_bin = V[:, row, t_mid]
    print(f"  value by solar bin (flying, mid SoC, stage {t_mid}): {np.round(vals_by_bin, 3)}")
    assert vals_by_bin[-1] >= vals_by_bin[0] - 1e-9, \
        "expected value to rise with higher (clearer) solar bin"
    f_p, r_p = rollout_stats(sim_p, EPISODES)
    print(f"  persistent chain: failure%={100*f_p:.2f}  avg_reward={r_p:.3f}")
    print(f"  vs i.i.d.: d failure%={100*(f_p-f_off):+.2f}  d reward={r_p-r_off:+.3f}")
    print("  PASS (value rises with solar bin; stats reported)")

    # 4. joint wind+solar: table shape + solar-memoryless joint == wind-only chain
    print("\n=== 4. joint wind+solar chain ===")
    cfg_joint = make_config(True, wind_chain=True)
    cfg_windonly = make_config(False, wind_chain=True)
    fac_w, _ = factory_for(cfg_windonly)
    sim_w = fac_w.create_simulation(sim_type="optimal", cap=400.0, full_history_episodes=0)
    f_w, r_w = rollout_stats(sim_w, EPISODES)
    print(f"  wind-only chain: n_bins={sim_w.mdp_solver.n_bins}  "
          f"failure%={100*f_w:.2f}  avg_reward={r_w:.3f}")

    fac_j, _ = factory_for(cfg_joint)
    envj = fac_j.env_provider
    nb, ngj = envj.n_wind_bins, envj.n_solar_bins
    envj.solar_transition = np.full_like(envj.solar_transition, 1.0 / ngj)  # solar memoryless
    sim_j = fac_j.create_simulation(sim_type="optimal", cap=400.0, full_history_episodes=0)
    assert sim_j.mdp_solver.future_value_table.shape[0] == nb * ngj, \
        f"joint table should have {nb * ngj} regime rows"
    f_j, r_j = rollout_stats(sim_j, EPISODES)
    print(f"  joint (solar memoryless): n_bins={nb * ngj}  "
          f"failure%={100*f_j:.2f}  avg_reward={r_j:.3f}")
    print(f"  vs wind-only: |d failure%|={abs(100*f_j-100*f_w):.2f}  |d reward|={abs(r_j-r_w):.3f}")
    assert abs(f_j - f_w) < 0.03 and abs(r_j - r_w) < 0.5, \
        "solar-memoryless joint chain diverges from wind-only chain"
    print("  PASS")

    # 5. structure + degenerate-slot guards
    print("\n=== 5. night/dawn structure & degenerate slots ===")
    envs = fac_p.env_provider
    ident = np.eye(envs.n_solar_bins)
    n_dawn = 0
    for t in range(HORIZON - 1):
        is_ident = np.allclose(envs.solar_transition[t], ident)
        if envs.solar_valid[t] and envs.solar_valid[t + 1]:
            pass  # fitted (month,hour) matrix (may occasionally equal identity; no assert)
        elif (not envs.solar_valid[t]) and envs.solar_valid[t + 1]:
            if not is_ident:
                n_dawn += 1
        else:
            assert is_ident, f"stage {t}: held stage should carry the identity"
    n_dawns_expected = int(np.sum(~envs.solar_valid[:-1] & envs.solar_valid[1:]))
    print(f"  dawn matrices applied: {n_dawn} (dawn crossings in window: {n_dawns_expected})")
    assert 1 <= n_dawn <= n_dawns_expected, "dawn matrix should fire once per in-window dawn"

    # Degenerate (saturated) Beta: conditional CDF and sampler stay finite and valid.
    solver = sim_p.mdp_solver
    t_v = int(valid_stages[0])
    a_save, b_save = envs.solar_alpha[t_v], envs.solar_beta[t_v]
    envs.solar_alpha[t_v], envs.solar_beta[t_v] = 1e6 * (1 - 1e-6), 1e6 * 1e-6  # point mass ~1
    u = np.linspace(0, 1, 11)
    for g in range(envs.n_solar_bins):
        F = solver._solar_cdf(t_v, g, u)
        assert np.all(np.isfinite(F)) and np.all(F >= -1e-12) and np.all(F <= 1 + 1e-12)
        k = envs._sample_solar_index_within_bin(t_v, np.full(64, g))
        assert np.all(np.isfinite(k)) and np.all((k >= 0) & (k <= 1))
    envs.solar_alpha[t_v], envs.solar_beta[t_v] = a_save, b_save
    print("  degenerate-slot conditional CDF/sampler: finite and in-range")
    print("  PASS")

    for c in (cfg_off, cfg_on, cfg_joint, cfg_windonly):
        try:
            os.remove(c)
        except OSError:
            pass
    for f in os.listdir("."):
        if f.endswith(".npy") and "Wh_" in f:
            try:
                os.remove(f)
            except OSError:
                pass
    print("\nAll solar-chain checks passed.")


if __name__ == "__main__":
    main()

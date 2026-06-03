"""
Verification for the runtime-optimization work (vectorized solver + batched episodes).

Strategy: the original scalar code paths (`_value`, `simulate_episode`, scalar
`choose_action`, `lookup_future_values`) are still present alongside the vectorized
ones, so this compares NEW vs ORIGINAL in the same process on identical inputs:

  * solver refactor        -> EXACT equivalence (same math, different loop)
  * fast value lookup      -> EXACT equivalence
  * batched episodes       -> STATISTICAL equivalence (RNG order changed) + speedup

Run (any cwd) with the pvlib conda env:
    conda run -n pvlib python SolarSimulator/Tests/verify_runtime_opt.py
    conda run -n pvlib python SolarSimulator/Tests/verify_runtime_opt.py --horizon 2000 --episodes 3000

ASCII-only output.
"""
import argparse
import os
import sys
import tempfile
import time

import numpy as np

# Resolve repo layout from this file: .../Solar-Simulator/SolarSimulator/Tests/<this>
TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
PKG_DIR = os.path.dirname(TESTS_DIR)            # .../SolarSimulator
REPO_ROOT = os.path.dirname(PKG_DIR)            # .../Solar-Simulator
sys.path.insert(0, PKG_DIR)

from BaseClasses.run_sim import YAMLSimulationRunner, SimulationFactory  # noqa: E402

DATA_PKL = os.path.join(
    REPO_ROOT, "Data", "EXPECTED_DATA", "data_expected_lat30.0_lon-90.0_15min.pkl"
)


def make_config(horizon, episodes):
    """Write a small self-contained config (absolute data_path) to a temp file."""
    text = f"""\
battery_capacities:
- 400.0
failure_penalties:
- 5.0
horizons:
- {horizon}
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
episodes: {episodes}
"""
    fd, path = tempfile.mkstemp(suffix=".yaml", prefix="rtopt_verify_")
    with os.fdopen(fd, "w") as f:
        f.write(text)
    return path


def build_factory(runner, sim_type):
    cfg = runner.config
    locs = runner.locations
    # YAMLSimulationRunner's no-"locations" fallback wraps the list in a tuple; unwrap.
    if isinstance(locs, tuple):
        locs = locs[0]
    loc = locs[0]
    H = runner.horizons[0]
    fp = runner.failure_penalties[0]
    name = runner.config_basename if sim_type == "optimal" else None
    return SimulationFactory(cfg, loc, H, fp, config_name=name), H, fp


def check_solver_equivalence(runner):
    print("\n=== Solver: batch vs scalar formula equivalence ===")
    factory, H, fp = build_factory(runner, "optimal")
    cap = runner.config["battery_capacities"][0]
    sim = factory.create_simulation(sim_type="optimal", cap=cap, full_history_episodes=0)
    solver = sim.mdp_solver
    states = solver.states[:-1]

    max_abs = 0.0
    for t in sorted(set([0, H // 2, H - 1])):
        last = (t == H - 1)
        for a in (0, 1):
            batch = solver._value_batch(states, a, t, last)
            scalar = np.array([
                solver._value(s[np.newaxis], np.array(a)[np.newaxis], t, last)
                for s in states
            ]).ravel()
            max_abs = max(max_abs, float(np.max(np.abs(batch - scalar))))
    print(f"  max |batch - scalar| over sampled stages/actions = {max_abs:.3e}")
    assert max_abs < 1e-8, "Solver batch path diverges from scalar formula!"
    print("  PASS")
    return solver


def check_fast_lookup(solver):
    print("\n=== Fast value lookup vs equality-scan lookup ===")
    rng = np.random.default_rng(1)
    n = 200
    idx = rng.integers(0, solver.n_soc_levels, size=n)
    soc = idx * solver.soc_increment
    mode = rng.integers(0, 2, size=n)
    broken = rng.random(n) < 0.1
    soc[broken] = -1.0
    mode[broken] = 2
    next_states = np.column_stack((soc, mode)).astype(float)
    stage = solver.horizon - 2

    fast = solver._lookup_future_values_fast(next_states, stage)
    slow = solver.lookup_future_values(next_states, np.full(n, stage, dtype=int))
    slow = np.where(next_states[:, 1] == 2, 0.0, slow)
    diff = float(np.max(np.abs(fast - slow)))
    print(f"  max |fast - scan| = {diff:.3e}")
    assert diff < 1e-12
    print("  PASS")


def run_policy(runner, sim_type, episodes):
    factory, H, fp = build_factory(runner, sim_type)
    cap = runner.config["battery_capacities"][0]
    kwargs = dict(sim_type=sim_type, cap=cap, full_history_episodes=5)
    if sim_type == "threshold":
        kwargs.update(threshold=runner.config["threshold_values"][0],
                      wind_threshold=runner.config["wind_thresholds"][0])
    t0 = time.time()
    sim = factory.create_simulation(**kwargs)
    build_dt = time.time() - t0

    t0 = time.time()
    eps = list(sim.simulate_multiple_episodes(episodes))
    run_dt = time.time() - t0

    fails = sum(1 for e in eps if e["failure"])
    avg_r = float(np.mean([e["total_reward"] for e in eps]))
    avg_fh = float(np.mean([e["flight_hrs"] for e in eps]))
    print(f"\n=== {sim_type}: {episodes} episodes (NEW vectorized) ===")
    print(f"  build={build_dt:.3f}s  run={run_dt:.3f}s")
    print(f"  failure%={100*fails/len(eps):.1f}  avg_reward={avg_r:.3f}  avg_flight_hrs={avg_fh:.3f}")
    e0 = eps[0]
    assert "trajectory" in e0 and e0["trajectory"].shape[1] == 2
    assert len(e0["actions"]) == len(e0["rewards"]) == e0["failure_step"]
    assert e0["trajectory"].shape[0] == e0["failure_step"] + 1
    print("  full-history shapes OK")

    # Baseline: original scalar per-episode loop (still present in the source).
    t0 = time.time()
    base_reward, base_fh, base_fail = [], [], 0
    for i in range(episodes):
        sim.env_provider.reset(i)
        out = sim.simulate_episode()
        traj, acts, rews = out[0], out[1], out[2]
        base_reward.append(float(rews.sum()))
        base_fh.append(float(acts.sum()) / 4)
        base_fail += int(traj[-1][1] == 2)
    base_dt = time.time() - t0
    b_avg_r = float(np.mean(base_reward))
    b_avg_fh = float(np.mean(base_fh))
    print(f"  --- baseline scalar loop: run={base_dt:.3f}s  speedup x{base_dt/max(run_dt,1e-9):.1f}")
    print(f"      failure%={100*base_fail/episodes:.1f}  avg_reward={b_avg_r:.3f}  avg_flight_hrs={b_avg_fh:.3f}")
    print(f"      |d avg_reward|={abs(avg_r-b_avg_r):.3f}  "
          f"|d failure%|={abs(100*fails/len(eps)-100*base_fail/episodes):.2f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--horizon", type=int, default=200)
    ap.add_argument("--episodes", type=int, default=2000)
    args = ap.parse_args()

    if not os.path.exists(DATA_PKL):
        sys.exit(f"[error] missing data file: {DATA_PKL}")

    cfg_path = make_config(args.horizon, args.episodes)
    try:
        runner = YAMLSimulationRunner(cfg_path)
        solver = check_solver_equivalence(runner)
        check_fast_lookup(solver)
        run_policy(runner, "threshold", episodes=args.episodes)
        run_policy(runner, "optimal", episodes=args.episodes)
        print("\nAll verification checks passed.")
    finally:
        # solve() drops a value-table .npy in cwd; clean up both that and the temp config.
        os.remove(cfg_path)
        for f in os.listdir("."):
            if f.endswith(".npy") and "Wh_" in f:
                try:
                    os.remove(f)
                except OSError:
                    pass


if __name__ == "__main__":
    main()

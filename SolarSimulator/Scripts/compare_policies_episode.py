"""
Same-weather, different-policy comparison over a REAL historical window.

Both the i.i.d.-optimal policy and the wind-persistence (Markov-chain) optimal policy are
replayed through one actual 2-week historical weather window (wind + solar) pulled from
HISTORICAL_DATA, resampled to the 15-min model step. Neither model "authors" the weather.

Trajectories are deterministic (the policy picks an action from the state; we assume the
mechanical transition succeeds unless the battery depletes), so any divergence is purely
the policy. Mechanical-failure RISK is reported exactly as 1 - prod_t p_success_t along
each policy's flown trajectory (no single coin draw needed).

Run (any cwd) with the pvlib conda env:
    conda run -n pvlib python SolarSimulator/Scripts/compare_policies_episode.py
    conda run -n pvlib python SolarSimulator/Scripts/compare_policies_episode.py --days 14 --start 2018-07-15
"""
import argparse
import os
import sys
import tempfile

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
PKG_DIR = os.path.dirname(SCRIPTS_DIR)
REPO_ROOT = os.path.dirname(PKG_DIR)
sys.path.insert(0, PKG_DIR)

from BaseClasses.run_sim import YAMLSimulationRunner, SimulationFactory  # noqa: E402
from Scripts.create_weather_distributions import build_wind_chain_artifact  # noqa: E402

DATA_PKL = os.path.join(REPO_ROOT, "Data", "EXPECTED_DATA",
                        "data_expected_lat30.0_lon-90.0_15min.pkl")
CHAIN_PKL = os.path.join(REPO_ROOT, "Data", "EXPECTED_DATA",
                         "data_expected_lat30.0_lon-90.0_15min_windchain.pkl")
HIST_PKL = os.path.join(REPO_ROOT, "Data", "HISTORICAL_DATA", "data_30_-90.pkl")


def make_config(horizon, start_iso, chain):
    text = f"""\
battery_capacities: [400.0]
failure_penalties: [5.0]
horizons: [{horizon}]
locations:
- {{data_path: {DATA_PKL}, latitude: 30.0, longitude: -90.0}}
solar_panel_model: constant
start_datetime: '{start_iso}'
threshold_values: [0.25]
wind_thresholds: [10.0]
transition_model: moderate
energy_increment_wh: 5
episodes: 1
wind_chain: {{enabled: {str(chain).lower()}}}
"""
    fd, path = tempfile.mkstemp(suffix=".yaml", prefix="cmp_")
    with os.fdopen(fd, "w") as f:
        f.write(text)
    return path


def make_optimal_sim(horizon, start_iso, chain):
    cfg = make_config(horizon, start_iso, chain)
    try:
        runner = YAMLSimulationRunner(cfg)
        locs = runner.locations
        loc = (locs[0] if isinstance(locs, tuple) else locs)[0]
        fac = SimulationFactory(runner.config, loc, runner.horizons[0],
                                runner.failure_penalties[0], config_name="cmp")
        return fac.create_simulation(sim_type="optimal", cap=400.0, full_history_episodes=0)
    finally:
        os.remove(cfg)


def load_historical_window(hist_pkl, start_date, n_steps, interval_min=15):
    """Real wind [m/s] and GHI [W/m^2] for n_steps at the model timestep, from start_date."""
    df = pd.read_pickle(hist_pkl)
    df = df[~((df.index.month == 2) & (df.index.day == 29))]
    res = (df[["wind_speed_10m", "shortwave_radiation"]]
           .resample(f"{interval_min}min").interpolate(method="linear"))
    start = pd.Timestamp(start_date, tz=res.index.tz)
    i = res.index.get_indexer([start], method="nearest")[0]
    if i + n_steps > len(res):
        raise ValueError("Requested window runs past the end of the historical record.")
    win = res.iloc[i:i + n_steps]
    return win["wind_speed_10m"].to_numpy(), win["shortwave_radiation"].to_numpy(), win.index[0]


def replay(sim, wind, solar_energy, whale, edges, bin_aware):
    """
    Deterministic policy rollout on fixed real weather. Mechanical transitions are assumed
    to succeed (so we see the full intended trajectory); battery depletion (soc<0) is
    deterministic and terminates. Returns the trajectory plus the exact cumulative
    mechanical-failure probability 1 - prod_t p_success_t along the flown path.
    """
    tl = sim.mdp.transition_logic
    H = len(wind)
    state = np.array([100.0, 0.0])
    energy = tl.soc_to_energy(state[0])
    socs, modes, actions = [state[0]], [int(state[1])], []
    log_surv = 0.0
    batt_fail_at = None
    for t in range(H):
        s2 = state[None, :]
        cur_bins = np.array([int(np.digitize(wind[t], edges[1:-1]))]) if bin_aware else None
        a = int(sim.choose_action_batch(
            s2, np.array([solar_energy[t]]), np.array([wind[t]]), np.array([whale[t]]), t,
            cur_bins=cur_bins)[0])
        actions.append(a)
        ec = tl._calculate_energy_consumption(s2, np.array([a]))
        nss, nse = tl._update_energy_and_state_continuous(energy, np.array([solar_energy[t]]), ec, np.array([a]))
        p_succ = float(tl.transition_model.compute_probability(np.array([wind[t]]), np.array([a]), s2)[0])
        log_surv += np.log(max(p_succ, 1e-12))
        cand = nss[0]
        if cand[1] == 2:                       # battery depletion (deterministic)
            state = np.array([-1.0, 2.0]); batt_fail_at = t + 1
            socs.append(state[0]); modes.append(2)
            break
        state, energy = cand, nse[0]
        socs.append(state[0]); modes.append(int(state[1]))
    return {"socs": np.array(socs), "modes": np.array(modes), "actions": np.array(actions),
            "batt_fail_at": batt_fail_at, "mech_fail_prob": 1.0 - np.exp(log_surv)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=14)
    ap.add_argument("--start", default="2020-07-15", help="historical window start date")
    ap.add_argument("--interval-min", type=int, default=15)
    args = ap.parse_args()
    H = int(args.days * 24 * 60 / args.interval_min)
    start_iso = f"{args.start}T00:00:00"

    if not os.path.exists(DATA_PKL):
        sys.exit(f"[error] missing {DATA_PKL}")
    if not os.path.exists(HIST_PKL):
        sys.exit(f"[error] missing historical data {HIST_PKL}")
    if not os.path.exists(CHAIN_PKL):
        print("[info] building wind-chain artifact ...")
        build_wind_chain_artifact(HIST_PKL, CHAIN_PKL, interval_minutes=15, n_bins=3)

    print(f"[info] solving i.i.d. and chain optimal policies (horizon={H}, {args.days} days) ...")
    sim_iid = make_optimal_sim(H, start_iso, chain=False)
    sim_chain = make_optimal_sim(H, start_iso, chain=True)
    edges = sim_chain.env_provider.wind_bin_edges

    # Real weather window (wind m/s, GHI W/m^2 -> solar energy J via the model's conversion).
    wind, ghi, win_start = load_historical_window(HIST_PKL, start_iso, H, args.interval_min)
    solar_energy = sim_chain.env_provider._energy_gain_from_solar(ghi)
    whale = np.array([sim_chain.env_provider.sample_whale_observation(t, 1)[0] for t in range(H)])
    print(f"[info] historical window starts {win_start}  (wind mean={wind.mean():.2f} m/s, "
          f"max={wind.max():.2f}, frac in high bin={np.mean(wind >= edges[-2]):.2f})")

    r_iid = replay(sim_iid, wind, solar_energy, whale, edges, bin_aware=False)
    r_ch = replay(sim_chain, wind, solar_energy, whale, edges, bin_aware=True)

    days = np.arange(H) / (24 * 60 / args.interval_min)
    L = min(len(r_iid["actions"]), len(r_ch["actions"]))
    n_diff = int(np.sum(r_iid["actions"][:L] != r_ch["actions"][:L]))

    def report(name, r):
        bf = "none" if r["batt_fail_at"] is None else f"t{r['batt_fail_at']} ({r['batt_fail_at']/(24*60/args.interval_min):.1f}d)"
        print(f"  {name:12s}: flight_hrs={r['actions'].sum()*args.interval_min/60:6.1f}  "
              f"battery_depletion={bf}  mech_fail_prob={100*r['mech_fail_prob']:5.1f}%")

    print(f"\nReal window {args.start} (+{args.days}d): {n_diff} differing actions of {L}")
    report("i.i.d.", r_iid)
    report("chain", r_ch)

    # ---- plot ----
    fig, ax = plt.subplots(3, 1, figsize=(13, 8), sharex=True, constrained_layout=True)
    ax[0].plot(days, wind, color="0.35", lw=0.7)
    for e in edges[1:-1]:
        ax[0].axhline(e, ls="--", color="0.6", lw=0.8)
    ax[0].fill_between(days, edges[-2], wind, where=wind >= edges[-2], color="tab:red", alpha=0.25,
                       label="high-wind bin")
    ax[0].set_ylabel("Wind [m/s]"); ax[0].legend(loc="upper right", fontsize=8)
    ax[0].set_title(f"Real historical weather, {win_start.date()} +{args.days}d  |  "
                    f"i.i.d.-optimal vs wind-persistence optimal")

    ax[1].step(days[:len(r_iid["socs"]) - 1], r_iid["socs"][:-1], where="post",
               color="tab:blue", label="i.i.d. policy")
    ax[1].step(days[:len(r_ch["socs"]) - 1], r_ch["socs"][:-1], where="post",
               color="tab:green", label="chain policy")
    for r, c in ((r_iid, "tab:blue"), (r_ch, "tab:green")):
        if r["batt_fail_at"]:
            ax[1].scatter(days[r["batt_fail_at"] - 1], 0, marker="x", s=70, color=c, zorder=5)
    ax[1].set_ylabel("State of charge [%]"); ax[1].legend(loc="lower left", fontsize=8)

    ax[2].step(days[:len(r_iid["actions"])], r_iid["actions"] + 0.02, where="post",
               color="tab:blue", label="i.i.d. action")
    ax[2].step(days[:len(r_ch["actions"])], r_ch["actions"] - 0.02, where="post",
               color="tab:green", label="chain action")
    ax[2].set_yticks([0, 1]); ax[2].set_yticklabels(["float/land", "fly"])
    ax[2].set_ylabel("Action"); ax[2].set_xlabel("Mission time [days]")
    ax[2].legend(loc="upper right", fontsize=8)

    out = os.path.join(REPO_ROOT, "policy_comparison_episode.png")
    fig.savefig(out, dpi=150)
    print(f"\nsaved: {out}")

    for f in os.listdir("."):
        if f.endswith(".npy") and "Wh_" in f:
            try:
                os.remove(f)
            except OSError:
                pass


if __name__ == "__main__":
    main()

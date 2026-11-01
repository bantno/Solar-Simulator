"""Verification that compare_chain_sweep.py's fast paths reproduce the original ones.

Two speed-ups in Scripts/compare_chain_sweep.py must not change any number:
  * read_episode_scalars splits a run's per-episode table by group once instead of
    filtering the whole table on every lookup;
  * _boot_ci draws bootstrap indices in blocks of resamples and evaluates the statistic
    row-wise, instead of one resample at a time.
This check runs the original implementations (copied below) side by side with the
current ones on synthetic data and requires exactly equal results: the same per-group
frames, the same bootstrap CIs for the same seed, and the generator left in the same
state afterwards (so later cells' CIs are unaffected too).

Run (any cwd) with the pvlib conda env:
    conda run -n pvlib python SolarSimulator/Tests/verify_compare_fast.py
"""
import os
import shutil
import sys
import tempfile

import numpy as np
import pandas as pd

TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
PKG_DIR = os.path.dirname(TESTS_DIR)
sys.path.insert(0, PKG_DIR)

from Scripts import compare_chain_sweep as ccs  # noqa: E402


def check(name, cond):
    status = "PASS" if cond else "FAIL"
    print(f"[{status}] {name}")
    if not cond:
        sys.exit(1)


# --- Original implementations (before the speed-ups) ------------------------------

def ref_read_episode_scalars(df, group):
    return df[df["group"] == group].set_index("episode_index").sort_index()


def ref_boot_ci(stat_fn, rng, n_boot, *arrays):
    n = len(arrays[0])
    stats = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        stats[b] = stat_fn(*[a[idx] for a in arrays])
    return float(np.percentile(stats, 2.5)), float(np.percentile(stats, 97.5))


def ref_paired_delta(iid_eps, chain_eps, n_boot, rng):
    out = {}
    joined = iid_eps.join(chain_eps, lsuffix="_iid", rsuffix="_chain", how="inner")
    arrays = {m: (joined[f"{m}_iid"].to_numpy(float), joined[f"{m}_chain"].to_numpy(float))
              for m in ("total_reward", "failure", "flight_hrs")}
    for metric, (a_iid, a_chain) in arrays.items():
        d = a_chain - a_iid
        lo, hi = ref_boot_ci(lambda x: x.mean(), rng, n_boot, d)
        out[f"d_{metric}"] = float(d.mean())
        out[f"d_{metric}_lo"], out[f"d_{metric}_hi"] = lo, hi
    r_iid, r_chain = arrays["total_reward"]
    lo, hi = ref_boot_ci(lambda a, b: ccs.cvar(b) - ccs.cvar(a), rng, n_boot, r_iid, r_chain)
    out["d_cvar10"] = ccs.cvar(r_chain) - ccs.cvar(r_iid)
    out["d_cvar10_lo"], out["d_cvar10_hi"] = lo, hi
    out["n_paired_episodes"] = len(joined)
    return out


# --- Synthetic run ------------------------------------------------------------------

def synthetic_scalars(n_groups, n_eps, seed=0):
    """Per-episode table shaped like a run's _episode_scalars.csv (shuffled rows)."""
    rng = np.random.default_rng(seed)
    frames = []
    for g in range(n_groups):
        frames.append(pd.DataFrame({
            "group": f"Threshold_{g}",
            "episode_index": np.arange(n_eps),
            "total_reward": rng.normal(50, 20, n_eps),
            "failure": rng.integers(0, 2, n_eps),
            "failure_step": rng.normal(1000, 300, n_eps),
            "flight_hrs": rng.normal(200, 50, n_eps),
        }))
    df = pd.concat(frames, ignore_index=True)
    return df.sample(frac=1.0, random_state=1).reset_index(drop=True)


def main():
    tmp = tempfile.mkdtemp(prefix="compare_fast_")
    try:
        df = synthetic_scalars(n_groups=6, n_eps=3001)
        df.to_csv(os.path.join(tmp, "_episode_scalars.csv"), index=False)
        on_disk = pd.read_csv(os.path.join(tmp, "_episode_scalars.csv"))

        # Grouped lookup == full-table filter, for every group and an unknown one.
        same = True
        for g in list(on_disk["group"].unique()) + ["no_such_group"]:
            fast = ccs.read_episode_scalars(tmp, g)
            ref = ref_read_episode_scalars(on_disk, g)
            same &= fast.equals(ref) and list(fast.columns) == list(ref.columns)
        check("grouped lookup equals the full-table filter for every group", same)
        check("unknown group returns an empty frame",
              len(ccs.read_episode_scalars(tmp, "no_such_group")) == 0)

        # Bootstrap: identical CIs and identical generator state, across chunk boundaries.
        a = ccs.read_episode_scalars(tmp, "Threshold_0")
        b = ccs.read_episode_scalars(tmp, "Threshold_1")
        for n_boot in (2000, 333, 1):
            rng_ref, rng_new = np.random.default_rng(12345), np.random.default_rng(12345)
            ref = ref_paired_delta(a, b, n_boot, rng_ref)
            new = ccs.paired_delta(a, b, n_boot, rng_new, paired=True)
            check(f"paired_delta identical to the one-at-a-time loop (n_boot={n_boot})",
                  ref == new)
            check(f"generator state identical afterwards (n_boot={n_boot})",
                  rng_ref.integers(0, 2**62) == rng_new.integers(0, 2**62))

        x = np.random.default_rng(3).normal(size=(5, 997))
        check("cvar_rows equals cvar per row",
              all(ccs.cvar_rows(x)[i] == ccs.cvar(x[i]) for i in range(len(x))))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("All checks passed.")


if __name__ == "__main__":
    main()

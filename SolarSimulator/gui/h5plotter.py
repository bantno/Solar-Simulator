import h5py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class HDF5RewardPlotter:
    """
    A class to load an HDF5 simulation file, extract thresholds and rewards
    from each episode, and plot:
      - Mean Total Reward for each (obs, wind) combo
      - Mean Failure Step for each (obs, wind) combo
      - Failure Percentage (%) for each (obs, wind) combo
    Also overlays the Optimal simulation stats on all plots.
    """
    def __init__(self, file_path):
        self.file_path = file_path
        self.file = None
        self._summary = None
        # placeholders for optimal stats
        self.opt_reward = None
        self.opt_failure_step = None
        self.opt_failure_pct = None

    def open_file(self):
        if self.file is None:
            self.file = h5py.File(self.file_path, 'r')

    def _load_summary(self):
        """
        Single-pass load of all simulation groups, aggregating metrics
        into a summary DataFrame, and caching optimal stats.
        """
        self.open_file()
        records = []
        for sim_group in self.file.keys():
            grp = self.file[sim_group]
            sim_type = grp.attrs.get('simulation_type', '')
            obs_t = grp.attrs.get('observation_threshold')
            wind_t = grp.attrs.get('wind_threshold')
            cap = grp.attrs.get('battery_capacity', grp.attrs.get('capacity', np.nan))
            horizon = grp.attrs.get('horizon', np.nan)

            # allow optimal group even if thresholds missing
            if (obs_t is None or wind_t is None) and 'optimal' not in sim_type.lower():
                continue
            # default missing thresholds for optimal to NaN
            if obs_t is None:
                obs_t = np.nan
            if wind_t is None:
                wind_t = np.nan

            # Accumulators
            rewards = []
            total_eps = 0
            fail_count = 0
            failure_steps = []

            episodes = grp.get('episodes', {})
            for ep in episodes.values():
                if 'total_reward' in ep:
                    rewards.append(ep['total_reward'][()])
                if 'failure' in ep and 'failure_step' in ep:
                    total_eps += 1
                    if bool(ep['failure'][()]):
                        fail_count += 1
                        failure_steps.append(ep['failure_step'][()])

            # skip groups with no data (except optimal)
            if not rewards and total_eps == 0 and 'optimal' not in sim_type.lower():
                continue

            mean_reward = np.mean(rewards) if rewards else np.nan
            failure_percentage = (fail_count / total_eps * 100) if total_eps else np.nan
            mean_failure_step = np.mean(failure_steps) if failure_steps else np.nan

            records.append({
                'sim_type': sim_type,
                'observation_threshold': obs_t,
                'wind_threshold': wind_t,
                'battery_capacity': cap,
                'horizon': horizon,
                'mean_reward': mean_reward,
                'failure_percentage': failure_percentage,
                'mean_failure_step': mean_failure_step
            })

        # build summary and cache optimal stats
        df = pd.DataFrame(records)
        self._summary = df
        opt_df = df[df['sim_type'].str.contains('optimal', case=False, na=False)]
        if not opt_df.empty:
            row = opt_df.iloc[0]
            self.opt_reward = row['mean_reward']
            self.opt_failure_step = row['mean_failure_step']
            self.opt_failure_pct = row['failure_percentage']

    def _get_summary(self):
        if self._summary is None:
            self._load_summary()
        return self._summary

    def plot_mean_by_thresholds(self):
        df = self._get_summary()
        df_main = df[~df['sim_type'].str.contains('optimal', case=False, na=False)]
        pivot = df_main.pivot(
            index='observation_threshold',
            columns='wind_threshold',
            values='mean_reward'
        )

        plt.figure()
        for w in pivot.columns:
            plt.plot(pivot.index, pivot[w], marker='x', label=f"Wind {w}")

        if self.opt_reward is not None:
            plt.axhline(
                self.opt_reward,
                linestyle='--',
                label=f"Optimal Mean Reward ({self.opt_reward:.3f})"
            )

        plt.xlabel("Observation Threshold")
        plt.ylabel("Mean Total Reward")
        plt.title("Mean Total Reward for each (Obs, Wind) Threshold Combination")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    def plot_mean_failure_step_by_thresholds(self):
        df = self._get_summary()
        df_main = df[~df['sim_type'].str.contains('optimal', case=False, na=False)]
        pivot = df_main.pivot(
            index='observation_threshold',
            columns='wind_threshold',
            values='mean_failure_step'
        )

        plt.figure()
        for w in pivot.columns:
            plt.plot(pivot.index, pivot[w], marker='x', label=f"Wind {w}")

        if self.opt_failure_step is not None:
            plt.axhline(
                self.opt_failure_step,
                linestyle='--',
                label=f"Optimal Mean Failure Step ({self.opt_failure_step:.2f})"
            )

        plt.xlabel("Observation Threshold")
        plt.ylabel("Mean Failure Step")
        plt.title("Mean Failure Step by Threshold Combination")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    def plot_failure_percentage_by_thresholds(self):
        df = self._get_summary()
        df_main = df[~df['sim_type'].str.contains('optimal', case=False, na=False)]
        pivot = df_main.pivot(
            index='observation_threshold',
            columns='wind_threshold',
            values='failure_percentage'
        )

        plt.figure()
        for w in pivot.columns:
            plt.plot(pivot.index, pivot[w], marker='x', label=f"Wind {w}")

        if self.opt_failure_pct is not None:
            plt.axhline(
                self.opt_failure_pct,
                linestyle='--',
                label=f"Optimal Failure % ({self.opt_failure_pct:.1f}%)"
            )

        plt.xlabel("Observation Threshold")
        plt.ylabel("Failure Percentage (%)")
        plt.title("Failure Percentage by Threshold Combination")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    def plot_reward_vs_capacity_by_thresholds(self):
        """
        Plot Mean Total Reward vs Battery Capacity:
          - One subplot per observation_threshold
          - One line per wind_threshold
          - Overlay optimal mean reward vs capacity
        """
        df = self._get_summary()
        df_main = df[~df['sim_type'].str.contains('optimal', case=False, na=False)]
        df_opt = df[df['sim_type'].str.contains('optimal', case=False, na=False)]

        obs_vals = sorted(df_main['observation_threshold'].dropna().unique())
        wind_vals = sorted(df_main['wind_threshold'].dropna().unique())

        n = len(obs_vals)
        cols = int(np.ceil(np.sqrt(n)))
        rows = int(np.ceil(n / cols))
        fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 4*rows), squeeze=False)

        for idx, obs in enumerate(obs_vals):
            ax = axes[idx//cols][idx%cols]
            subset = df_main[df_main['observation_threshold'] == obs]
            for w in wind_vals:
                series = subset[subset['wind_threshold'] == w]
                if series.empty:
                    continue
                series = series.sort_values('battery_capacity')
                ax.plot(series['battery_capacity'], series['mean_reward'], marker='o', label=f"Wind {w}")

            # overlay optimal
            if not df_opt.empty:
                opt_series = df_opt.dropna(subset=['battery_capacity', 'mean_reward'])
                opt_series = opt_series.sort_values('battery_capacity')
                ax.plot(opt_series['battery_capacity'], opt_series['mean_reward'], linestyle='--', marker='s', label='Optimal')

            ax.set_title(f"Obs Threshold = {obs}")
            ax.set_xlabel("Battery Capacity")
            ax.set_ylabel("Mean Total Reward")
            ax.grid(True)
            ax.legend()

        for idx in range(n, rows*cols):
            fig.delaxes(axes.flatten()[idx])

        plt.tight_layout()
        plt.show()

    def plot_reward_vs_horizon_by_thresholds(self):
        """
        Plot Mean Total Reward vs Horizon:
          - One subplot per observation_threshold
          - One line per wind_threshold
          - Overlay optimal mean reward vs horizon
        """
        df = self._get_summary()
        df_main = df[~df['sim_type'].str.contains('optimal', case=False, na=False)]
        df_opt = df[df['sim_type'].str.contains('optimal', case=False, na=False)]

        obs_vals = sorted(df_main['observation_threshold'].dropna().unique())
        wind_vals = sorted(df_main['wind_threshold'].dropna().unique())

        n = len(obs_vals)
        cols = int(np.ceil(np.sqrt(n)))
        rows = int(np.ceil(n / cols))
        fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 4*rows), squeeze=False)

        for idx, obs in enumerate(obs_vals):
            ax = axes[idx//cols][idx%cols]
            subset = df_main[df_main['observation_threshold'] == obs]
            for w in wind_vals:
                series = subset[subset['wind_threshold'] == w]
                if series.empty:
                    continue
                series = series.sort_values('horizon')
                ax.plot(series['horizon'], series['mean_reward'], marker='o', label=f"Wind {w}")

            # overlay optimal
            if not df_opt.empty:
                opt_series = df_opt.dropna(subset=['horizon', 'mean_reward'])
                opt_series = opt_series.sort_values('horizon')
                ax.plot(opt_series['horizon'], opt_series['mean_reward'], linestyle='--', marker='s', label='Optimal')

            ax.set_title(f"Obs Threshold = {obs}")
            ax.set_xlabel("Horizon")
            ax.set_ylabel("Mean Total Reward")
            ax.grid(True)
            ax.legend()

        for idx in range(n, rows*cols):
            fig.delaxes(axes.flatten()[idx])

        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    plotter = HDF5RewardPlotter(
        r"simulation_results\sim_5000_eps_20250519_101037.h5"
    )
    # plotter.plot_reward_vs_capacity_by_thresholds()
    # plotter.plot_reward_vs_horizon_by_thresholds()
    plotter.plot_reward_vs_capacity_by_thresholds()
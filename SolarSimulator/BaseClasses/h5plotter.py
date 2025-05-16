# import h5py
# import pandas as pd
# import matplotlib.pyplot as plt

# class HDF5RewardPlotter:
#     """
#     A class to load an HDF5 simulation file, extract thresholds and rewards
#     from each episode, and plot:
#       - Mean Total Reward for each (obs, wind) combo
#       - Mean Failure Step for each (obs, wind) combo
#       - Failure Percentage (%) for each (obs, wind) combo
#     Also overlays the Optimal simulation stats on all plots.
#     """
#     def __init__(self, file_path):
#         self.file_path = file_path
#         self.file = None

#     def open_file(self):
#         if self.file is None:
#             self.file = h5py.File(self.file_path, 'r')

#     def extract_mean_by_thresholds(self):
#         self.open_file()
#         records = []
#         for sim_group in self.file.keys():
#             grp = self.file[sim_group]
#             obs_t = grp.attrs.get('observation_threshold')
#             wind_t = grp.attrs.get('wind_threshold')
#             if obs_t is None or wind_t is None:
#                 continue
#             ep_parent = grp.get('episodes', {})
#             rewards = [ep['total_reward'][()] for ep in ep_parent.values() if 'total_reward' in ep]
#             if not rewards:
#                 continue
#             records.append({
#                 'observation_threshold': obs_t,
#                 'wind_threshold': wind_t,
#                 'mean_reward': sum(rewards) / len(rewards)
#             })
#         return pd.DataFrame(records).sort_values(['wind_threshold', 'observation_threshold'])

#     def extract_stats_by_thresholds(self):
#         self.open_file()
#         records = []
#         for sim_group in self.file.keys():
#             grp = self.file[sim_group]
#             obs_t = grp.attrs.get('observation_threshold')
#             wind_t = grp.attrs.get('wind_threshold')
#             if obs_t is None or wind_t is None:
#                 continue
#             ep_parent = grp.get('episodes', {})
#             total_eps = 0
#             fail_count = 0
#             failure_steps = []
#             for ep in ep_parent.values():
#                 if 'failure' in ep and 'failure_step' in ep:
#                     total_eps += 1
#                     if bool(ep['failure'][()]):
#                         fail_count += 1
#                         failure_steps.append(ep['failure_step'][()])
#             if total_eps == 0:
#                 continue
#             pct_fail = (fail_count / total_eps) * 100
#             mean_fail_step = (sum(failure_steps) / len(failure_steps)) if failure_steps else float('nan')
#             records.append({
#                 'observation_threshold': obs_t,
#                 'wind_threshold': wind_t,
#                 'failure_percentage': pct_fail,
#                 'mean_failure_step': mean_fail_step
#             })
#         return pd.DataFrame(records).sort_values(['wind_threshold', 'observation_threshold'])

#     def extract_optimal_stats(self):
#         """
#         Extract mean reward, mean failure step, and failure percentage for the Optimal simulation.
#         """
#         self.open_file()
#         for sim_group in self.file.keys():
#             grp = self.file[sim_group]
#             if 'Optimal' in grp.attrs.get('simulation_type', ''):
#                 ep_parent = grp.get('episodes', {})
#                 rewards = []
#                 total_eps = 0
#                 fail_count = 0
#                 failure_steps = []
#                 for ep in ep_parent.values():
#                     if 'total_reward' in ep:
#                         rewards.append(ep['total_reward'][()])
#                     if 'failure' in ep and 'failure_step' in ep:
#                         total_eps += 1
#                         if bool(ep['failure'][()]):
#                             fail_count += 1
#                             failure_steps.append(ep['failure_step'][()])
#                 mean_reward = sum(rewards) / len(rewards) if rewards else None
#                 mean_fail_step = sum(failure_steps) / len(failure_steps) if failure_steps else None
#                 pct_fail = (fail_count / total_eps) * 100 if total_eps else None
#                 return mean_reward, mean_fail_step, pct_fail
#         return None, None, None

#     def plot_mean_by_thresholds(self):
#         df = self.extract_mean_by_thresholds()
#         opt_reward, _, _ = self.extract_optimal_stats()
#         pivot = df.pivot(index='observation_threshold', columns='wind_threshold', values='mean_reward')

#         plt.figure()
#         for w in pivot.columns:
#             plt.plot(pivot.index, pivot[w], marker='x', label=f"Wind {w}")
#         if opt_reward is not None:
#             plt.axhline(opt_reward, linestyle='--', label=f"Optimal Mean Reward ({opt_reward:.3f})")
#         plt.xlabel("Observation Threshold")
#         plt.ylabel("Mean Total Reward")
#         plt.title("Mean Total Reward for each (Obs, Wind) Threshold Combination")
#         plt.legend()
#         plt.grid(True)
#         plt.tight_layout()
#         plt.show()

#     def plot_mean_failure_step_by_thresholds(self):
#         df = self.extract_stats_by_thresholds()
#         _, opt_fail_step, _ = self.extract_optimal_stats()
#         pivot = df.pivot(index='observation_threshold', columns='wind_threshold', values='mean_failure_step')

#         plt.figure()
#         for w in pivot.columns:
#             plt.plot(pivot.index, pivot[w], marker='x', label=f"Wind {w}")
#         if opt_fail_step is not None:
#             plt.axhline(opt_fail_step, linestyle='--', label=f"Optimal Mean Failure Step ({opt_fail_step:.2f})")
#         plt.xlabel("Observation Threshold")
#         plt.ylabel("Mean Failure Step")
#         plt.title("Mean Failure Step by Threshold Combination")
#         plt.legend()
#         plt.grid(True)
#         plt.tight_layout()
#         plt.show()

#     def plot_failure_percentage_by_thresholds(self):
#         df = self.extract_stats_by_thresholds()
#         _, _, opt_pct_fail = self.extract_optimal_stats()
#         pivot = df.pivot(index='observation_threshold', columns='wind_threshold', values='failure_percentage')

#         plt.figure()
#         for w in pivot.columns:
#             plt.plot(pivot.index, pivot[w], marker='x', label=f"Wind {w}")
#         if opt_pct_fail is not None:
#             plt.axhline(opt_pct_fail, linestyle='--', label=f"Optimal Failure % ({opt_pct_fail:.1f}%)")
#         plt.xlabel("Observation Threshold")
#         plt.ylabel("Failure Percentage (%)")
#         plt.title("Failure Percentage by Threshold Combination")
#         plt.legend()
#         plt.grid(True)
#         plt.tight_layout()
#         plt.show()




# if __name__ == "__main__":

#     # Example usage:
#     plotter = HDF5RewardPlotter(r"simulation_results\sim_5000_eps_20250512_232429.h5")

#     plotter.plot_mean_by_thresholds()                 # Mean reward vs thresholds (with optimal line)
#     plotter.plot_mean_failure_step_by_thresholds()    # Mean failure step
#     plotter.plot_failure_percentage_by_thresholds()   # Failure rate (%)
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

            # skip groups with no data (except optimal we still record NaNs)
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
                'mean_reward': mean_reward,
                'failure_percentage': failure_percentage,
                'mean_failure_step': mean_failure_step
            })

        # build summary and cache optimal stats
        df = pd.DataFrame(records)
        self._summary = df
        # find optimal group case-insensitively
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

    def plot_reward_histogram(self,
                              sim_type=None,
                              observation_threshold=None,
                              wind_threshold=None,
                              bins=20):
        """
        Plot a histogram of total_reward for each episode in a specified simulation.

        You can select by:
        - sim_type substring (case-insensitive)
        - observation_threshold value
        - wind_threshold value
        If multiple filters are provided, they are ANDed together.

        Parameters:
        - sim_type: (optional) substring to match simulation_type
        - observation_threshold: (optional) exact threshold to match
        - wind_threshold: (optional) exact threshold to match
        - bins: number of histogram bins
        """
        self.open_file()
        matches = []
        for grp in self.file.values():
            attrs = grp.attrs
            st = attrs.get('simulation_type', '')
            ot = attrs.get('observation_threshold')
            wt = attrs.get('wind_threshold')

            if sim_type and sim_type.lower() not in st.lower():
                continue
            if observation_threshold is not None and ot != observation_threshold:
                continue
            if wind_threshold is not None and wt != wind_threshold:
                continue
            matches.append(grp)

        if not matches:
            raise ValueError("No simulation group matches the given filters.")

        # choose first matched group
        grp = matches[0]
        # collect rewards
        rewards = []
        for ep in grp.get('episodes', {}).values():
            if 'total_reward' in ep:
                rewards.append(ep['total_reward'][()])
        if not rewards:
            raise ValueError("Matched simulation has no total_reward data.")

        plt.figure()
        plt.hist(rewards, bins=bins, edgecolor='black')
        plt.xlabel('Total Reward')
        plt.ylabel('Episode Count')
        title = f"Reward Distribution"
        if sim_type:
            title += f" for {sim_type} Policy"
        if observation_threshold is not None or wind_threshold is not None:
            title += f" (obs={observation_threshold}, wind={wind_threshold})"
        plt.title(title)
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
        # separate main vs optimal
        df_main = df[~df['sim_type'].str.contains('optimal', case=False, na=False)]
        df_opt = df[df['sim_type'].str.contains('optimal', case=False, na=False)]

        obs_vals = sorted(df_main['observation_threshold'].dropna().unique())
        wind_vals = sorted(df_main['wind_threshold'].dropna().unique())

        n = len(obs_vals)
        if n == 0:
            raise ValueError("No observation thresholds available for plotting.")

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

        # hide unused axes
        for idx in range(n, rows*cols):
            fig.delaxes(axes.flatten()[idx])

        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    # Example usage:
    plotter = HDF5RewardPlotter(
        r"simulation_results\sim_3000_eps_20250512_102741.h5"
    )
    # plotter.plot_mean_by_thresholds()                 # Mean reward vs thresholds (with optimal line)
    # plotter.plot_mean_failure_step_by_thresholds()    # Mean failure step (with optimal line)
    # plotter.plot_failure_percentage_by_thresholds()   # Failure rate (%) (with optimal line)
    # plotter.plot_reward_histogram('Optimal', bins=50)  # histogram of rewards for Optimal sim
    # plotter.plot_reward_histogram('Threshold',observation_threshold=0.1, wind_threshold=5.0, bins=50)
    plotter.plot_reward_vs_capacity_by_thresholds()   # Mean reward vs capacity (by obs/wind thresholds)


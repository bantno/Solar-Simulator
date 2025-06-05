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
        self.open_file()
        records = []
        for sim_group in self.file.keys():
            grp = self.file[sim_group]
            sim_type = grp.attrs.get('simulation_type', '')
            obs_t = grp.attrs.get('observation_threshold')
            wind_t = grp.attrs.get('wind_threshold')
            cap = grp.attrs.get('battery_capacity', grp.attrs.get('capacity', np.nan))
            location_id = grp.attrs.get('location_id', sim_group)

            # skip non-thresholded non-optimal runs
            if obs_t is None and 'optimal' not in sim_type.lower():
                continue
            obs_t = obs_t if obs_t is not None else np.nan
            wind_t = wind_t if wind_t is not None else np.nan

            rewards = []
            episodes = grp.get('episodes', {}) or {}
            for ep in episodes.values():
                if 'total_reward' in ep:
                    rewards.append(ep['total_reward'][()])

            if not rewards and 'optimal' not in sim_type.lower():
                continue

            records.append({
                'sim_type': sim_type,
                'observation_threshold': obs_t,
                'wind_threshold': wind_t,
                'battery_capacity': cap,
                'location_id': location_id,
                'mean_reward': np.mean(rewards) if rewards else np.nan
            })

        self._summary = pd.DataFrame(records)
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

    def plot_reward_vs_penalty(self):
        """
        Plot Mean Total Reward vs Failure Penalty:
          - Collapse across all other parameters
          - Overlay optimal policy reward vs penalty
        """
        df = self._get_summary()
        df_main = df[~df['sim_type'].str.contains('optimal', case=False, na=False)]
        df_opt  = df[df['sim_type'].str.contains('optimal', case=False, na=False)]

        if 'failure_penalty' not in df_main.columns:
            raise KeyError("No 'failure_penalty' column found in data")

        # average reward per penalty
        main_group = df_main.groupby('failure_penalty')['mean_reward'].mean()
        plt.figure()
        plt.plot(main_group.index, main_group.values, marker='o', label='Mean Reward')

        # overlay optimal
        if not df_opt.empty:
            opt_group = df_opt.groupby('failure_penalty')['mean_reward'].mean()
            plt.plot(opt_group.index, opt_group.values,
                     linestyle='--', marker='s', label='Optimal Reward')

        plt.xlabel("Failure Penalty")
        plt.ylabel("Mean Total Reward")
        plt.title("Mean Total Reward vs Failure Penalty")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()


        plt.tight_layout()
        plt.show()

    def plot_failure_percentage_by_penalty(self, subplots=True):
        """
        Plot failure percentage vs. observation threshold for each wind threshold,
        either as separate subplots per failure penalty (if subplots=True),
        or all penalty curves on a single plot (if subplots=False).
        """
        if subplots:
            self._plot_failure_percentage_by_penalty_subplots()
        else:
            self._plot_failure_percentage_by_penalty_single()

    def _plot_failure_percentage_by_penalty_subplots(self):
        df = self._get_summary()
        df_main = df[~df['sim_type'].str.contains('optimal', case=False, na=False)]
        df_opt  = df[df['sim_type'].str.contains('optimal', case=False, na=False)]
        fp_vals = sorted(df_main['failure_penalty'].dropna().unique())
        wind_vals = sorted(df_main['wind_threshold'].dropna().unique())
        n = len(fp_vals)
        cols = int(np.ceil(np.sqrt(n)))
        rows = int(np.ceil(n / cols))
        fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 4*rows), squeeze=False)

        opt_group = None
        if not df_opt.empty:
            opt_group = df_opt.groupby('failure_penalty')['failure_percentage'].mean()

        for idx, fp in enumerate(fp_vals):
            ax = axes[idx//cols][idx%cols]
            subset = df_main[df_main['failure_penalty'] == fp]
            pivot = subset.pivot(
                index='observation_threshold',
                columns='wind_threshold',
                values='failure_percentage'
            )
            for w in pivot.columns:
                ax.plot(pivot.index, pivot[w], marker='x', label=f"Wind {w}")
            if opt_group is not None and fp in opt_group.index:
                ax.axhline(opt_group[fp], linestyle='--', label=f"Optimal ({opt_group[fp]:.1f}%)")
            ax.set_title(f"Penalty = {fp}")
            ax.set_xlabel("Observation Threshold")
            ax.set_ylabel("Failure Percentage (%)")
            ax.grid(True)
            ax.legend()

        for idx in range(n, rows*cols):
            fig.delaxes(axes.flatten()[idx])

        plt.tight_layout()
        plt.show()

    def _plot_failure_percentage_by_penalty_single(self):
        df = self._get_summary()
        df_main = df[~df['sim_type'].str.contains('optimal', case=False, na=False)]
        df_opt  = df[df['sim_type'].str.contains('optimal', case=False, na=False)]

        # Identify unique threshold combinations
        combos = df_main[['observation_threshold', 'wind_threshold']].drop_duplicates()

        plt.figure()
        # Plot one line per threshold combo: failure_percentage vs failure_penalty
        for _, combo in combos.iterrows():
            obs = combo['observation_threshold']
            wind = combo['wind_threshold']
            subset = df_main[
                (df_main['observation_threshold'] == obs) &
                (df_main['wind_threshold'] == wind)
            ]
            if subset.empty:
                continue
            series = subset.sort_values('failure_penalty')
            plt.plot(
                series['failure_penalty'],
                series['failure_percentage'],
                marker='o',
                label=f"Obs {obs}, Wind {wind}"
            )

        # Overlay optimal policy series
        if not df_opt.empty:
            opt_series = df_opt.groupby('failure_penalty')['failure_percentage'].mean().reset_index()
            plt.plot(
                opt_series['failure_penalty'],
                opt_series['failure_percentage'],
                linestyle='--', marker='s',
                label='Optimal'
            )

        plt.xlabel("Failure Penalty")
        plt.ylabel("Failure Percentage (%)")
        plt.title("Failure Percentage vs Failure Penalty by Threshold Combination")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    def plot_failure_step_by_penalty(self, subplots=True):
        """
        Plot mean failure step vs. observation threshold (subplots),
        or vs. failure penalty for each threshold combo (single).
        """
        if subplots:
            self._plot_failure_step_by_penalty_subplots()
        else:
            self._plot_failure_step_by_penalty_single()

    def _plot_failure_step_by_penalty_subplots(self):
        df = self._get_summary()
        df_main = df[~df['sim_type'].str.contains('optimal', case=False, na=False)]
        df_opt  = df[df['sim_type'].str.contains('optimal', case=False, na=False)]
        fp_vals = sorted(df_main['failure_penalty'].dropna().unique())
        wind_vals = sorted(df_main['wind_threshold'].dropna().unique())
        n = len(fp_vals)
        cols = int(np.ceil(np.sqrt(n)))
        rows = int(np.ceil(n / cols))
        fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 4*rows), squeeze=False)

        opt_group = None
        if not df_opt.empty:
            opt_group = df_opt.groupby('failure_penalty')['mean_failure_step'].mean()

        for idx, fp in enumerate(fp_vals):
            ax = axes[idx//cols][idx%cols]
            subset = df_main[df_main['failure_penalty'] == fp]
            pivot = subset.pivot(
                index='observation_threshold',
                columns='wind_threshold',
                values='mean_failure_step'
            )
            for w in pivot.columns:
                ax.plot(pivot.index, pivot[w], marker='x', label=f"Wind {w}")
            if opt_group is not None and fp in opt_group.index:
                ax.axhline(opt_group[fp], linestyle='--', label=f"Optimal ({opt_group[fp]:.2f})")
            ax.set_title(f"Penalty = {fp}")
            ax.set_xlabel("Observation Threshold")
            ax.set_ylabel("Mean Failure Step")
            ax.grid(True)
            ax.legend()

        for idx in range(n, rows*cols):
            fig.delaxes(axes.flatten()[idx])

        plt.tight_layout()
        plt.show()

    def _plot_failure_step_by_penalty_single(self):
        df = self._get_summary()
        df_main = df[~df['sim_type'].str.contains('optimal', case=False, na=False)]
        df_opt  = df[df['sim_type'].str.contains('optimal', case=False, na=False)]

        combos = df_main[['observation_threshold', 'wind_threshold']].drop_duplicates()

        plt.figure()
        for _, combo in combos.iterrows():
            obs = combo['observation_threshold']
            wind = combo['wind_threshold']
            subset = df_main[
                (df_main['observation_threshold'] == obs) &
                (df_main['wind_threshold'] == wind)
            ]
            if subset.empty:
                continue
            series = subset.sort_values('failure_penalty')
            plt.plot(
                series['failure_penalty'],
                series['mean_failure_step'],
                marker='o',
                label=f"Obs {obs}, Wind {wind}"
            )

        if not df_opt.empty:
            opt_series = df_opt.groupby('failure_penalty')['mean_failure_step'].mean().reset_index()
            plt.plot(
                opt_series['failure_penalty'],
                opt_series['mean_failure_step'],
                linestyle='--', marker='s',
                label='Optimal'
            )

        plt.xlabel("Failure Penalty")
        plt.ylabel("Mean Failure Step")
        plt.title("Mean Failure Step vs Failure Penalty by Threshold Combination")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    def plot_optimal_reward_distribution_by_penalty(
        self,
        penalties=None,
        max_series=4,
        bins=50,
        subplots=False
    ):
        """
        Plot reward distribution histograms for optimal simulations across
        different failure penalties.

        Parameters:
        - penalties: list of specific penalty values to include (or None)
        - max_series: maximum number of penalties to plot
        - bins: number of histogram bins
        - subplots: if True, place each penalty on its own subplot with
          shared x-limits; if False, overlay histograms.
        """
        # gather data
        self.open_file()
        rewards_by_fp = {}
        for sim_group in self.file.keys():
            grp = self.file[sim_group]
            sim_type = grp.attrs.get('simulation_type', '')
            if 'optimal' not in sim_type.lower():
                continue
            fp = grp.attrs.get('failure_penalty', None)
            if fp is None:
                continue
            if penalties is not None and fp not in penalties:
                continue
            episodes = grp.get('episodes', {})
            for ep in episodes.values():
                if 'total_reward' in ep:
                    rewards_by_fp.setdefault(fp, []).append(ep['total_reward'][()])

        # select penalty series
        fps = sorted(rewards_by_fp.keys())
        fps = (fps if penalties else fps[:max_series])[:max_series]

        if subplots:
            self._plot_opt_reward_dist_subplots(rewards_by_fp, fps, bins)
        else:
            self._plot_opt_reward_dist_overlay(rewards_by_fp, fps, bins)

    def _plot_opt_reward_dist_overlay(self, rewards_by_fp, fps, bins):
        plt.figure()
        for fp in fps:
            data = rewards_by_fp.get(fp, [])
            if not data:
                continue
            plt.hist(data, bins=bins, alpha=0.5, label=f'Penalty {fp}')
        plt.xlabel('Total Reward')
        plt.ylabel('Episode Count')
        plt.title('Reward Distribution for Optimal Policy by Failure Penalty')
        plt.legend()
        plt.tight_layout()
        plt.show()

    def _plot_opt_reward_dist_subplots(self, rewards_by_fp, fps, bins):
        # compute global x-limits
        all_vals = [v for fp in fps for v in rewards_by_fp.get(fp, [])]
        if not all_vals:
            return
        xmin, xmax = min(all_vals), max(all_vals)

        # vertical stack: one column, one row per penalty
        n = len(fps)
        fig, axes = plt.subplots(n, 1, figsize=(6, 3 * n), sharex=True)
        # ensure axes is always a list
        if n == 1:
            axes = [axes]

        for idx, fp in enumerate(fps):
            ax = axes[idx]
            data = rewards_by_fp.get(fp, [])
            # opaque blue bars with black outline
            ax.hist(data, bins=bins, edgecolor='black')
            ax.set_xlim(xmin, xmax)
            ax.set_title(f'Penalty {fp}')
            ax.set_ylabel('Episode Count')
            ax.patch.set_alpha(0.3)        # make the bar background semi-transparent
            ax.grid(True)

        # label the bottom plot's x-axis only
        axes[-1].set_xlabel('Total Reward')

        plt.tight_layout()
        plt.show()

    def plot_optimal_failure_step_distribution_by_penalty(
        self,
        penalties=None,
        max_series=4,
        bins=50,
        subplots=False
    ):
        """
        Plot failure step distribution histograms for optimal simulations across
        different failure penalties.

        Parameters:
        - penalties: list of penalty values to include (or None)
        - max_series: maximum number of penalties to plot
        - bins: number of histogram bins
        - subplots: if True, separate subplots per penalty with shared x-limits;
                    if False, overlay histograms.
        """
        self.open_file()
        steps_by_fp = {}
        for sim_group in self.file.keys():
            grp = self.file[sim_group]
            if 'optimal' not in grp.attrs.get('simulation_type', '').lower():
                continue
            fp = grp.attrs.get('failure_penalty', None)
            if fp is None:
                continue
            if penalties is not None and fp not in penalties:
                continue
            episodes = grp.get('episodes', {})
            for ep in episodes.values():
                if 'failure_step' in ep:
                    steps_by_fp.setdefault(fp, []).append(ep['failure_step'][()])

        fps = sorted(steps_by_fp.keys())
        fps = (fps if penalties else fps[:max_series])[:max_series]
        if subplots:
            self._plot_opt_failure_step_subplots(steps_by_fp, fps, bins)
        else:
            self._plot_opt_failure_step_overlay(steps_by_fp, fps, bins)

    def _plot_opt_failure_step_overlay(self, steps_by_fp, fps, bins):
        plt.figure()
        for fp in fps:
            data = steps_by_fp.get(fp, [])
            if not data:
                continue
            plt.hist(data, bins=bins, alpha=0.5, label=f'Penalty {fp}')
        plt.xlabel('Failure Step')
        plt.ylabel('Episode Count')
        plt.title('Failure Step Distribution for Optimal Policy by Penalty')
        plt.legend()
        plt.tight_layout()
        plt.show()

    def _plot_opt_failure_step_subplots(self, steps_by_fp, fps, bins):
        # compute global x-limits
        all_vals = [v for fp in fps for v in steps_by_fp.get(fp, [])]
        if not all_vals:
            return
        xmin, xmax = min(all_vals), max(all_vals)
        # vertical stack
        n = len(fps)
        fig, axes = plt.subplots(n, 1, figsize=(6, 3*n), sharex=True)
        if n == 1:
            axes = [axes]
        for idx, fp in enumerate(fps):
            ax = axes[idx]
            data = steps_by_fp.get(fp, [])
            if data:
                ax.hist(data, bins=bins, edgecolor='black')
            ax.set_xlim(xmin, xmax)
            ax.set_title(f'Penalty {fp}')
            ax.set_ylabel('Episode Count')
            ax.grid(True)
        axes[-1].set_xlabel('Failure Step')
        plt.tight_layout()
        plt.show()

    def plot_reward_vs_location_by_thresholds(self):
        """
        Plot Mean Total Reward vs Latitude for each threshold combo,
        with a separate subplot per battery capacity arranged in two columns.
        """
        df = self._get_summary()
        df_main = df[~df['sim_type'].str.contains('optimal', case=False, na=False)].copy()
        df_opt = df[df['sim_type'].str.contains('optimal', case=False, na=False)].copy()

        # Extract latitude
        df_main['latitude'] = df_main['location_id'].str.extract(r'lat([\-\d\.]+)')[0].astype(float)
        if not df_opt.empty:
            df_opt['latitude'] = df_opt['location_id'].str.extract(r'lat([\-\d\.]+)')[0].astype(float)

        caps = sorted(df_main['battery_capacity'].unique())
        combos = df_main[['observation_threshold', 'wind_threshold']].drop_duplicates()

        # Arrange subplots in two columns
        n = len(caps)
        cols = 2
        rows = int(np.ceil(n / cols))
        fig, axes = plt.subplots(rows, cols, figsize=(8 * cols, 4 * rows), sharex=True, sharey=True)
        axes = axes.flatten()

        for ax, cap in zip(axes, caps):
            sub = df_main[df_main['battery_capacity'] == cap]
            for _, combo in combos.iterrows():
                obs, wind = combo
                sel = sub[(sub['observation_threshold'] == obs) & (sub['wind_threshold'] == wind)]
                if sel.empty:
                    continue
                sel = sel.sort_values('latitude')
                ax.plot(sel['latitude'], sel['mean_reward'], marker='o', label=f"Obs {obs}, Wind {wind}")
            if not df_opt.empty:
                sub_opt = df_opt[df_opt['battery_capacity'] == cap]
                if not sub_opt.empty:
                    opt_sel = sub_opt.groupby('latitude')['mean_reward'].mean().reset_index().sort_values('latitude')
                    ax.plot(opt_sel['latitude'], opt_sel['mean_reward'], linestyle='--', marker='s', label='Optimal')

            ax.set_title(f'Capacity = {cap}')
            ax.set_ylabel('Mean Total Reward')
            ax.grid(True)
            ax.legend()

        # Hide unused axes
        for ax in axes[len(caps):]:
            ax.axis('off')

        fig.supxlabel('Latitude')
        fig.supylabel('Mean Total Reward')
        fig.suptitle('Mean Total Reward vs Location by Battery Capacity and Thresholds', y=0.92)
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.show()

    def plot_reward_vs_capacity_by_location(self):
        """
        Plot Mean Total Reward vs Battery Capacity for each threshold combo,
        with a separate subplot per location (latitude) arranged in two columns.

        Each subplot shows only the optimal runs for the corresponding latitude.
        """
        df = self._get_summary()
        df_main = df[~df['sim_type'].str.contains('optimal', case=False, na=False)].copy()
        df_opt = df[df['sim_type'].str.contains('optimal', case=False, na=False)].copy()

        # Extract latitude
        df_main['latitude'] = df_main['location_id'].str.extract(r'lat([\-\d\.]+)')[0].astype(float)
        df_opt['latitude'] = df_opt['location_id'].str.extract(r'lat([\-\d\.]+)')[0].astype(float)

        lats = sorted(df_main['latitude'].unique())
        combos = df_main[['observation_threshold', 'wind_threshold']].drop_duplicates()

        n = len(lats)
        cols = 2
        rows = int(np.ceil(n / cols))
        fig, axes = plt.subplots(rows, cols, figsize=(8 * cols, 4 * rows), sharex=True, sharey=True)
        axes = axes.flatten()

        for ax, lat in zip(axes, lats):
            sub = df_main[df_main['latitude'] == lat]
            for _, combo in combos.iterrows():
                obs, wind = combo
                sel = sub[(sub['observation_threshold'] == obs) & (sub['wind_threshold'] == wind)]
                if sel.empty:
                    continue
                sel = sel.sort_values('battery_capacity')
                ax.plot(sel['battery_capacity'], sel['mean_reward'], marker='o', label=f"Obs {obs}, Wind {wind}")

            # Optimal only for this latitude
            sub_opt = df_opt[df_opt['latitude'] == lat]
            if not sub_opt.empty:
                opt_sel = sub_opt.groupby('battery_capacity')['mean_reward'].mean().reset_index().sort_values('battery_capacity')
                ax.plot(opt_sel['battery_capacity'], opt_sel['mean_reward'], linestyle='--', marker='s', label='Optimal')

            ax.set_title(f'Latitude = {lat}')
            ax.set_ylabel('Mean Total Reward')
            ax.grid(True)
            ax.legend()

        for ax in axes[len(lats):]:
            ax.axis('off')

        fig.supxlabel('Battery Capacity')
        fig.supylabel('Mean Total Reward')
        fig.suptitle('Mean Total Reward vs Battery Capacity by Location and Thresholds', y=0.92)
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.show()



if __name__ == "__main__":
    plotter = HDF5RewardPlotter(
        r"simulation_results\sim_3000_eps_20250521_124518.h5"
    )
    # plotter.plot_reward_vs_capacity_by_thresholds()
    # plotter.plot_reward_vs_horizon_by_thresholds()
    # plotter.plot_reward_vs_capacity_by_thresholds()
    # plotter.plot_failure_percentage_by_thresholds()
    # plotter.plot_reward_vs_penalty()
    # plotter.plot_failure_percentage_by_penalty(subplots=False)
    # plotter.plot_failure_step_by_penalty(subplots=False)
    # plotter.plot_optimal_reward_distribution_by_penalty(penalties=[0,5,10], bins = 50, subplots= True)
    # plotter.plot_optimal_failure_step_distribution_by_penalty(penalties=[0,5,10], bins=100, subplots=True)
    plotter.plot_reward_vs_location_by_thresholds()
    plotter.plot_reward_vs_capacity_by_location()
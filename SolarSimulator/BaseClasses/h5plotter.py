import h5py
import pandas as pd
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

    def open_file(self):
        if self.file is None:
            self.file = h5py.File(self.file_path, 'r')

    def extract_mean_by_thresholds(self):
        self.open_file()
        records = []
        for sim_group in self.file.keys():
            grp = self.file[sim_group]
            obs_t = grp.attrs.get('observation_threshold')
            wind_t = grp.attrs.get('wind_threshold')
            if obs_t is None or wind_t is None:
                continue
            ep_parent = grp.get('episodes', {})
            rewards = [ep['total_reward'][()] for ep in ep_parent.values() if 'total_reward' in ep]
            if not rewards:
                continue
            records.append({
                'observation_threshold': obs_t,
                'wind_threshold': wind_t,
                'mean_reward': sum(rewards) / len(rewards)
            })
        return pd.DataFrame(records).sort_values(['wind_threshold', 'observation_threshold'])

    def extract_stats_by_thresholds(self):
        self.open_file()
        records = []
        for sim_group in self.file.keys():
            grp = self.file[sim_group]
            obs_t = grp.attrs.get('observation_threshold')
            wind_t = grp.attrs.get('wind_threshold')
            if obs_t is None or wind_t is None:
                continue
            ep_parent = grp.get('episodes', {})
            total_eps = 0
            fail_count = 0
            failure_steps = []
            for ep in ep_parent.values():
                if 'failure' in ep and 'failure_step' in ep:
                    total_eps += 1
                    if bool(ep['failure'][()]):
                        fail_count += 1
                        failure_steps.append(ep['failure_step'][()])
            if total_eps == 0:
                continue
            pct_fail = (fail_count / total_eps) * 100
            mean_fail_step = (sum(failure_steps) / len(failure_steps)) if failure_steps else float('nan')
            records.append({
                'observation_threshold': obs_t,
                'wind_threshold': wind_t,
                'failure_percentage': pct_fail,
                'mean_failure_step': mean_fail_step
            })
        return pd.DataFrame(records).sort_values(['wind_threshold', 'observation_threshold'])

    def extract_optimal_stats(self):
        """
        Extract mean reward, mean failure step, and failure percentage for the Optimal simulation.
        """
        self.open_file()
        for sim_group in self.file.keys():
            grp = self.file[sim_group]
            if 'Optimal' in grp.attrs.get('simulation_type', ''):
                ep_parent = grp.get('episodes', {})
                rewards = []
                total_eps = 0
                fail_count = 0
                failure_steps = []
                for ep in ep_parent.values():
                    if 'total_reward' in ep:
                        rewards.append(ep['total_reward'][()])
                    if 'failure' in ep and 'failure_step' in ep:
                        total_eps += 1
                        if bool(ep['failure'][()]):
                            fail_count += 1
                            failure_steps.append(ep['failure_step'][()])
                mean_reward = sum(rewards) / len(rewards) if rewards else None
                mean_fail_step = sum(failure_steps) / len(failure_steps) if failure_steps else None
                pct_fail = (fail_count / total_eps) * 100 if total_eps else None
                return mean_reward, mean_fail_step, pct_fail
        return None, None, None

    def plot_mean_by_thresholds(self):
        df = self.extract_mean_by_thresholds()
        opt_reward, _, _ = self.extract_optimal_stats()
        pivot = df.pivot(index='observation_threshold', columns='wind_threshold', values='mean_reward')

        plt.figure()
        for w in pivot.columns:
            plt.plot(pivot.index, pivot[w], marker='x', label=f"Wind {w}")
        if opt_reward is not None:
            plt.axhline(opt_reward, linestyle='--', label=f"Optimal Mean Reward ({opt_reward:.3f})")
        plt.xlabel("Observation Threshold")
        plt.ylabel("Mean Total Reward")
        plt.title("Mean Total Reward for each (Obs, Wind) Threshold Combination")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    def plot_mean_failure_step_by_thresholds(self):
        df = self.extract_stats_by_thresholds()
        _, opt_fail_step, _ = self.extract_optimal_stats()
        pivot = df.pivot(index='observation_threshold', columns='wind_threshold', values='mean_failure_step')

        plt.figure()
        for w in pivot.columns:
            plt.plot(pivot.index, pivot[w], marker='x', label=f"Wind {w}")
        if opt_fail_step is not None:
            plt.axhline(opt_fail_step, linestyle='--', label=f"Optimal Mean Failure Step ({opt_fail_step:.2f})")
        plt.xlabel("Observation Threshold")
        plt.ylabel("Mean Failure Step")
        plt.title("Mean Failure Step by Threshold Combination")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    def plot_failure_percentage_by_thresholds(self):
        df = self.extract_stats_by_thresholds()
        _, _, opt_pct_fail = self.extract_optimal_stats()
        pivot = df.pivot(index='observation_threshold', columns='wind_threshold', values='failure_percentage')

        plt.figure()
        for w in pivot.columns:
            plt.plot(pivot.index, pivot[w], marker='x', label=f"Wind {w}")
        if opt_pct_fail is not None:
            plt.axhline(opt_pct_fail, linestyle='--', label=f"Optimal Failure % ({opt_pct_fail:.1f}%)")
        plt.xlabel("Observation Threshold")
        plt.ylabel("Failure Percentage (%)")
        plt.title("Failure Percentage by Threshold Combination")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()




if __name__ == "__main__":

    # Example usage:
    plotter = HDF5RewardPlotter("simulation_results\sim_5000_eps_20250423_223408.h5")

    plotter.plot_mean_by_thresholds()                 # Mean reward vs thresholds (with optimal line)
    plotter.plot_mean_failure_step_by_thresholds()    # Mean failure step
    plotter.plot_failure_percentage_by_thresholds()   # Failure rate (%)

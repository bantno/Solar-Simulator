import os
import re
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone

class SolarChargePlotter:
    def __init__(self, directory, start_date, time_step=None):
        """
        Initialize the SolarChargePlotter with directory, start date, and time step.
        
        Parameters:
        - directory: Path to the directory containing pickle files.
        - start_date: The starting date and time as a string (e.g., '2023-01-01 00:00:00').
        - time_step: Frequency string for the time step (e.g., 'H' for hourly, 'D' for daily).
        """
        self.directory = directory
        self.start_date = start_date
        self.time_step = time_step
        self.dt = int(self.time_step.replace("min", ""))

    def load_first_entry(self, file_path):
        """Load the first row of a DataFrame from the specified pickle file path."""
        df = pd.read_pickle(file_path)
        return df.head(1)

    def extract_parameters_from_filename(self, filename):
        """
        Extracts 'cap' and either 'p' or 'mdp_success_prob' from the filename.
        Returns a tuple with these values.
        """
        # Regex patterns for Greedy and MDP files
        match = re.match(r"(\w+)_Data_c(\d+)_(p|t)([\d.]+)_(\d+)min_(\d+-\d+)", filename)
        if match:
            algo, cap, param_type, param_value, dt, _ = match.groups()
            cap = int(cap)
            param_value = float(param_value)
            dt = int(dt)
            # return {
            #     "algo": algo,
            #     "cap": cap,
            #     "param_type": param_type,  # 'p' for optimal, 't' for threshold
            #     "param_value": param_value,
            #     "dt": dt
            # }
            return algo
        else:
            return "Unknown"

    def plot_data(self):
        """Plot the state of charge, solar history, and cumulative hours flown for the first entry in each file in the directory."""
        plt.figure(figsize=(15, 10))
        
        # Plot State of Charge
        plt.subplot(4, 1, 1)
        for file in os.listdir(self.directory):
            if file.endswith(".pkl"):
                file_path = os.path.join(self.directory, file)
                df = self.load_first_entry(file_path)
                
                # Extract parameters from filename
                label = self.extract_parameters_from_filename(file)

                # Extract state history and solar history from the first entry
                state_history = df['StateHistory'].values[0]  # State history list
                state_charge_levels = [state[0] for state in state_history]  # Extract charge levels
                
                # Generate datetime index
                time_index = pd.date_range(start=self.start_date, periods=len(state_charge_levels), freq=self.time_step)

                # Plot State of Charge
                plt.plot(time_index, state_charge_levels, label=label)

        plt.title('State of Charge Over Time')
        plt.xlabel('Datetime')
        plt.ylabel('Charge Level (%)')
        plt.legend()
        plt.grid(True)

        # Plot Cumulative Hours Flown
        plt.subplot(4, 1, 2)
        for file in os.listdir(self.directory):
            if file.endswith(".pkl"):
                file_path = os.path.join(self.directory, file)
                df = self.load_first_entry(file_path)

                # Extract parameters from filename
                label = self.extract_parameters_from_filename(file)

                # Extract cumulative hours from state history
                state_history = df['StateHistory'].values[0]
                states = [state[1] for state in state_history]  # Extract cumulative hours
                cumulative_hours = [0]
                for i in range(len(states)):
                    if states[i]=="flying":
                        cumulative_hours.append(cumulative_hours[-1]+self.dt/60.)
                    else:
                        cumulative_hours.append(cumulative_hours[-1])

                # Generate datetime index
                time_index = pd.date_range(start=self.start_date, periods=len(cumulative_hours), freq=self.time_step)

                # Plot Cumulative Hours Flown
                plt.plot(time_index, cumulative_hours, label=label)

        plt.title('Cumulative Hours Flown Over Time')
        plt.xlabel('Datetime')
        plt.ylabel('Cumulative Hours Flown')
        plt.legend()
        plt.grid(True)

        # Plot Solar History
        plt.subplot(4, 1, 3)
        for file in os.listdir(self.directory):
            if file.endswith(".pkl"):
                file_path = os.path.join(self.directory, file)
                df = self.load_first_entry(file_path)

                # Extract parameters from filename
                label = self.extract_parameters_from_filename(file)

                # Extract solar history from the first entry
                solar_history = df['SolarHistory'].values[0]
                expected_solar_history = df['ExpectedSolarHistory'].values[0]

                # Generate datetime index
                time_index = pd.date_range(start=self.start_date, periods=len(solar_history), freq=self.time_step)

                # Plot Solar History
                plt.plot(time_index, solar_history, label="Actual")
                plt.plot(time_index, expected_solar_history[:len(time_index)], label="Expected")
                break

        plt.title('Solar History Over Time')
        plt.xlabel('Datetime')
        plt.ylabel('Solar Power (W/$m^2$)')
        plt.legend()
        plt.grid(True)

        # Plot Solar History
        plt.subplot(4, 1, 4)
        for file in os.listdir(self.directory):
            if file.endswith(".pkl"):
                file_path = os.path.join(self.directory, file)
                df = self.load_first_entry(file_path)

                # Extract parameters from filename
                label = self.extract_parameters_from_filename(file)

                # Extract solar history from the first entry
                whale_history = df['WhaleHistory'].values[0]

                # Generate datetime index
                time_index = pd.date_range(start=self.start_date, periods=len(whale_history), freq=self.time_step)

                # Plot Solar History
                plt.plot(time_index, whale_history, label=label)
                break

        plt.title('Whale Surface Probability Over Time')
        plt.xlabel('Datetime')
        plt.ylabel('Probability')
        plt.grid(True)

        plt.tight_layout()
        plt.savefig(r"Figures\StatePlot\state_plot_"+f"{self.start_date.day_of_year}_{self.dt}.png")
        plt.show()

    def plot_reward_vs_threshold(self):
        """
        Plot reward vs threshold for all pickle files in the given directory.
        The function will extract the threshold from each file's name and plot the corresponding reward.
        """
        plt.figure(figsize=(8, 6))

        rewards = []
        thresholds = []
        # Iterate over all files in the directory
        for file_name in os.listdir(self.directory):
            if file_name.endswith(".pkl"):
                file_path = os.path.join(self.directory, file_name)

                # Extract the threshold from the filename using regex
                threshold_match = re.search(r't([\d\.]+)', file_name)  # Capture the value after 't'
                if threshold_match:
                    threshold_value = float(threshold_match.group(1))
                    # Load data from the pickle file
                    with open(file_path, 'rb') as f:
                        data = pd.read_pickle(f)

                    # Assuming the data has 'reward' as a key (adjust this if the structure is different)
                    thresholds.append(threshold_value)
                    rewards.append(data['Reward'].mean())
                else:
                    algo_match = re.search(r'([\w])_Data+', file_name)  # Capture the value after 't'
                    with open(file_path, 'rb') as f:
                        data = pd.read_pickle(f)

                    # Assuming the data has 'reward' as a key (adjust this if the structure is different)
                    optimal_reward = data['Reward'].mean()
                    print(optimal_reward)
                    


                # Plot reward vs threshold
        sorted_indices = np.argsort(thresholds)
        thresholds = np.array(thresholds)[sorted_indices]
        rewards = np.array(rewards)[sorted_indices]
        plt.plot(thresholds[1:],rewards[1:] , marker='o', linestyle='-',color='orange', label='Threshold')
        plt.axhline(y=optimal_reward, label='Optimal')
        plt.axhline(y=rewards[0], label='Greedy',color='red')




        plt.title('Reward vs Threshold')
        plt.xlabel('Threshold')
        plt.ylabel('Reward')
        plt.legend()
        plt.grid(True)
        plt.show()


if __name__ == '__main__':
    dt=10
    utc_offset = timezone(timedelta(hours=-6))
    start_date = pd.to_datetime(datetime(2019,6,2).replace(tzinfo=utc_offset))
    end_date = pd.to_datetime(datetime(2019,7,2).replace(tzinfo=utc_offset))
    plotter = SolarChargePlotter(r".",start_date=start_date,time_step=f"{dt}min")
    plotter.plot_data()
    # plotter.plot_reward_vs_threshold()
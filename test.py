import os
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import re

import os
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import re

import os
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import re

class Plotter:
    def __init__(self, directory):
        self.directory = directory
        self.files_data = []
        self.load_files()

    def load_files(self):
        # Get all .pkl files in the specified directory
        files = [f for f in os.listdir(self.directory) if f.endswith('.pkl')]
        self.files_data = []

        for file in files:
            # Extract information from the file name
            file_info = self.extract_file_info(file)
            
            # Load the .pkl file into a pandas dataframe
            file_path = os.path.join(self.directory, file)
            with open(file_path, 'rb') as f:
                df = pickle.load(f)
            
            # Store the dataframe with associated metadata
            self.files_data.append({
                'file_info': file_info,
                'df': df,
                'file_name': file
            })

    def extract_file_info(self, file_name):
        """
        Extract relevant metadata from the file name.
        File name format example:
        "Optimal_Data_c80_p0.75_10min_61-65_1000.pkl" or "Threshold_Data_c100_t0_10min_61-65_1000.pkl"
        
        We extract:
        - algorithm (Optimal_Data or Threshold_Data)
        - battery_capacity (c80)
        - success_probability (p0.75 or t0.75)
        - timestep (10min)
        - date_range (61-65)
        - sample_count (1000)
        """
        # Define the regex pattern for the filename structure
        pattern = re.compile(
            r"(?P<algorithm>[\w]+)_Data_"                  # Algorithm (e.g., Optimal_Data, Threshold_Data)
            r"c(?P<battery_capacity>\d+)"                  # Battery capacity
            r"_(?P<type>[pt])(?P<prob_or_threshold>\d+(\.\d+)?)"  # Probability (p) or Threshold (t)
            r"_(?P<timestep>\d+)min_"                      # Timestep in minutes
            r"(?P<date_range>\d+-\d+)_"
            r"(?P<sample_count>\d+)\.pkl"                  # Sample count
        )

        match = pattern.match(file_name)
        
        if match:
            algorithm = match.group('algorithm')
            battery_capacity = int(match.group('battery_capacity'))
            prob_or_threshold = float(match.group('prob_or_threshold'))
            timestep = int(match.group('timestep'))
            date_range = match.group('date_range')
            sample_count = int(match.group('sample_count'))
            
            # If the value after 'p' or 't' is 0, set the algorithm to "greedy"
            if prob_or_threshold == 0:
                algorithm = "greedy"
            
            # Return metadata as a dictionary
            return {
                'algorithm': algorithm,
                'battery_capacity': battery_capacity,
                'prob_or_threshold': prob_or_threshold,
                'timestep': timestep,
                'date_range': date_range,
                'sample_count': sample_count
            }
        else:
            raise ValueError

    def plot_battery_capacity_vs_mean_reward(self):
        """
        Plot battery capacity vs mean reward for each algorithm and probability/threshold.
        The data will be segmented by algorithm and probability/threshold.
        """
        # Prepare a dictionary to hold the data, segmented by algorithm and prob_or_threshold
        plot_data = {}

        # Loop over the loaded data and calculate mean Reward for each DataFrame
        for data in self.files_data:
            df = data['df']
            file_info = data['file_info']
            
            # Calculate mean Reward for the current DataFrame
            mean_reward = df['Reward'].mean()
            
            # Get battery capacity and algorithm info
            battery_capacity = file_info['battery_capacity']
            algorithm = file_info['algorithm']
            prob_or_threshold = file_info['prob_or_threshold']

            # Group by algorithm and prob_or_threshold
            key = (algorithm, prob_or_threshold)

            if key not in plot_data:
                plot_data[key] = {}

            # For each group, collect the mean rewards for each battery capacity
            if battery_capacity not in plot_data[key]:
                plot_data[key][battery_capacity] = []

            plot_data[key][battery_capacity].append(mean_reward)

        # Plotting
        plt.figure(figsize=(12, 8))

        # Loop through each (algorithm, prob_or_threshold) combination and plot
        for (algorithm, prob_or_threshold), battery_data in plot_data.items():
            # Prepare the data for plotting
            battery_capacities = sorted(battery_data.keys())  # Sorted battery capacities for x-axis
            mean_rewards = [sum(rewards) / len(rewards) for rewards in battery_data.values()]  # Average mean rewards for each battery capacity

            # Plot the data for this (algorithm, prob_or_threshold) combination
            plt.plot(battery_capacities, mean_rewards, marker='o', linestyle='-', label=f"{algorithm} (Threshold/Prob: {prob_or_threshold})")

        # Customize the plot
        plt.title("Battery Capacity vs Mean Reward (Segmented by Algorithm and Threshold/Probability)")
        plt.xlabel("Battery Capacity")
        plt.ylabel("Mean Reward")
        plt.legend(title="Algorithm and Threshold/Probability")
        plt.grid(True)
        plt.show()


    def get_file_metadata(self):
        # Return metadata for all files loaded
        return [data['file_info'] for data in self.files_data]



if __name__ == "__main__":
    # Instantiate the Plotter class with the target directory
    plotter = Plotter(".")

    # Plot the mean Reward for each file, segmented by algorithm and probability/threshold
    plotter.plot_battery_capacity_vs_mean_reward()


import os
import re
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone

class StateHistoryPlotter:
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

    def plot_data(self,save_dir):
        """Plot the state of charge, solar history, and cumulative hours flown for the first entry in each file in the directory."""

        plt.figure(figsize=(15, 10))

        # Extract battery capacity from the first file in the directory
        files = [file for file in os.listdir(self.directory) if file.endswith(".pkl")]
        if not files:
            raise ValueError("No data files found in the directory.")
        
        first_file = files[0]
        battery_capacity = self.extract_parameter_from_filename(first_file, r"c(\d+)", "Unknown Capacity") + " Ah"

        # Main Title with Battery Capacity
        plt.suptitle(f'Battery Capacity: {battery_capacity}', fontsize=16)

        # Plot State of Charge
        plt.subplot(4, 1, 1)
        for file in files:
            file_path = os.path.join(self.directory, file)
            df = self.load_first_entry(file_path)

            # Extract parameters from filename
            algorithm, label = self.parse_filename(file)

            # Extract state history and charge levels
            state_history = df['StateHistory'].values[0]
            state_charge_levels = [state[0] for state in state_history]

            # Generate datetime index
            time_index = pd.date_range(start=self.start_date, periods=len(state_charge_levels), freq=self.time_step)

            # Plot State of Charge
            plt.plot(time_index, state_charge_levels, label=f"{label}")

        plt.title('State of Charge Over Time')
        plt.xlabel('Datetime')
        plt.ylabel('Charge Level (%)')
        plt.legend()
        plt.grid(True)

        # Plot Cumulative Hours Flown
        plt.subplot(4, 1, 2)
        for file in files:
            file_path = os.path.join(self.directory, file)
            df = self.load_first_entry(file_path)

            # Extract parameters from filename
            algorithm, label = self.parse_filename(file)

            # Extract cumulative hours from state history
            state_history = df['StateHistory'].values[0]
            states = [state[1] for state in state_history]
            cumulative_hours = [0]
            for i in range(len(states)):
                if states[i] == 1:
                    cumulative_hours.append(cumulative_hours[-1] + self.dt / 60.0)
                else:
                    cumulative_hours.append(cumulative_hours[-1])

            # Generate datetime index
            time_index = pd.date_range(start=self.start_date, periods=len(cumulative_hours), freq=self.time_step)

            # Plot Cumulative Hours Flown
            plt.plot(time_index, cumulative_hours, label=f"{label}")

        plt.title('Cumulative Hours Flown Over Time')
        plt.xlabel('Datetime')
        plt.ylabel('Cumulative Hours Flown')
        plt.legend()
        plt.grid(True)

        # Plot Solar History
        plt.subplot(4, 1, 3)
        for file in files:
            file_path = os.path.join(self.directory, file)
            df = self.load_first_entry(file_path)

            # Extract parameters from filename
            algorithm, label = self.parse_filename(file)

            # Extract solar history
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

        # Plot Whale History
        plt.subplot(4, 1, 4)
        for file in files:
            file_path = os.path.join(self.directory, file)
            df = self.load_first_entry(file_path)

            # Extract parameters from filename
            algorithm, label = self.parse_filename(file)

            # Extract whale history
            whale_history = df['WhaleHistory'].values[0]

            # Generate datetime index
            time_index = pd.date_range(start=self.start_date, periods=len(whale_history), freq=self.time_step)

            # Plot Whale History
            plt.plot(time_index, whale_history, label=label)
            break

        plt.title('Whale Surface Probability Over Time')
        plt.xlabel('Datetime')
        plt.ylabel('Probability')
        plt.grid(True)

        plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust layout to fit title
        plt.savefig(save_dir + rf"\state_plot_{self.start_date.day_of_year}_{self.dt}.png")

    # Helper method for regex extraction
    def extract_parameter_from_filename(self, filename, pattern, default_value):
        """Extract a parameter from the filename using a regex pattern."""
        match = re.search(pattern, filename)
        if match:
            return "_".join(match.groups())
        return default_value

    def parse_filename(self, filename):
        """Parse the filename to extract algorithm and label."""
        import re

        # Regex patterns for each case
        optimal_pattern = r"Optimal_Data_c(\d+)_p([\d\.]+)_"
        threshold_pattern = r"Threshold_Data_c(\d+)_t([\d\.]+)_"
        greedy_pattern = r"Threshold_Data_c(\d+)_t0\.0_"

        # Extract parameters
        if re.search(optimal_pattern, filename):
            match = re.search(optimal_pattern, filename)
            algorithm = "Optimal"
            label = f"{algorithm}, p={match.group(2)}"
        elif re.search(threshold_pattern, filename):
            match = re.search(threshold_pattern, filename)
            threshold = float(match.group(2))
            algorithm = "Greedy" if threshold == 0.0 else "Threshold"
            label = f"{algorithm}, t={match.group(2)}"
        else:
            algorithm = "Unknown"
            label = "Unknown Parameters"

        return algorithm, label

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

class DataProcessor:
    def __init__(self, directory, show=False):
        """Initialize the processor with the directory containing pickle files."""
        self.directory = directory
        self.results = []  # Store results as a list of dictionaries
        self.show = show

    def process_files(self):
        """Read all pickle files and store their mean results."""
        for filename in os.listdir(self.directory):
            if filename.endswith(".pkl"):
                match = re.match(r"([\w\s]+)_Data_c(\d+)(?:_t([\d.]+))?(?:_p([\d.]+))?_(\d+)min_(\d+-\d+)_(\d+)(?:_lat(-?\d+))?", filename)
                if match:
                    algo, cap, threshold, prob, dt, date_range, runs, latitude = match.groups()
                    cap = int(cap)
                    dt = int(dt)
                    runs = int(runs)

                    # Convert threshold and probability to floats if they are found, otherwise set to None
                    threshold = float(threshold) if threshold is not None else None
                    prob = float(prob) if prob is not None else None
                    latitude = float(latitude) if latitude is not None else None

                    # Parse the date range
                    start_day, end_day = map(int, date_range.split("-"))

                    # Calculate the number of timesteps
                    num_timesteps = ((end_day - start_day + 1) * 24 * 60) // dt

                    # Calculate mean reward and failure step
                    mean_reward, mean_failure_step = self.calculate_mean_rewards_and_failures(
                        os.path.join(self.directory, filename)
                    )

                    # Store results in a dictionary
                    self.results.append({
                        "Algorithm": algo,
                        "Capacity": cap,
                        "Threshold": threshold,   # Will be None if threshold was not in the filename
                        "Timestep": dt,
                        "Probability": prob,      # Will be None if probability was not in the filename
                        "Latitude": latitude,
                        "StartDate": start_day,
                        "EndDate": end_day,
                        "NumTimesteps": num_timesteps,
                        "NumRuns": runs,
                        "MeanFailureStep": mean_failure_step,
                        "MeanReward": mean_reward
                    })
                else:
                    print(f"Filename {filename} does not match expected pattern.")

    def plot_optimal_battery_capacity(self,df,save_dir):
        """
        Plot the optimal battery capacity against mission duration for different latitudes using Matplotlib.
        
        Parameters:
        - df: DataFrame with columns 'Capacity', 'NumTimesteps', 'Timestep', 'Latitude', and 'MeanReward'.
        
        The function identifies the battery capacity with the highest mean reward for each mission duration and latitude,
        then plots mission duration on the X-axis, optimal battery capacity on the Y-axis, and different curves for each latitude.
        """
        # Step 1: Calculate mission duration in days
        df['MissionDurationDays'] = df['NumTimesteps'] * df['Timestep'] / (60 * 24)

        # Step 2: Identify the optimal battery capacity for each mission duration and latitude
        optimal_df = df.groupby(['MissionDurationDays', 'Latitude']).apply(
            lambda group: group.loc[group['MeanReward'].idxmax()]
        ).reset_index(drop=True)

        # Step 3: Plot the results using Matplotlib
        plt.figure(figsize=(10, 6))

        # Get unique latitudes
        latitudes = optimal_df['Latitude'].unique()

        # Plot each latitude separately
        for lat in latitudes:
            lat_data = optimal_df[optimal_df['Latitude'] == lat]
            plt.plot(lat_data['MissionDurationDays'], lat_data['Capacity'], label=f'Latitude {lat}', marker='o')

        # Customize the plot
        plt.title('Optimal Battery Capacity vs Mission Duration')
        plt.xlabel('Mission Duration (Days)')
        plt.ylabel('Optimal Battery Capacity (Ah)')
        plt.legend(title='Latitude',loc='upper left', bbox_to_anchor=(1.05, 1))
        plt.tight_layout()
        plt.grid(True)

        plt.savefig(save_dir + rf"\optimal_battery_plot.png")


    def calculate_mean_rewards_and_failures(self, filepath):
        """Calculate the mean reward and failure step for a given pickle file."""
        df = pd.read_pickle(filepath)

        if 'Reward' in df.columns and 'LastStep' in df.columns:
            mean_reward = df['Reward'].mean()
            mean_failure_step = round(df['LastStep'].mean())
            print(f"Number of runs in dataset {filepath}: {len(df)}")
            return mean_reward, mean_failure_step
        else:
            print(f"Missing columns in {filepath}")
            return None, None

    def get_results_df(self):
        """Convert the results list into a DataFrame."""
        return pd.DataFrame(self.results)

    def plot_by_algorithm_and_probability(self):
        """Plot data with series based on both algorithm and probability."""
        df = self.get_results_df()

        plt.figure(figsize=(12, 8))
        markers = ['o', 's', '^', 'D', 'P', 'X', '*']  # A list of markers for variety

        # Get unique probabilities and algorithms
        probabilities = df['Probability'].unique()
        algorithms = df['Algorithm'].unique()
        
        # Iterate over unique algorithms and probabilities
        for i, algo_name in enumerate(algorithms):
            for prob in probabilities:
                subset = df[(df['Algorithm'] == algo_name) & (df['Probability'] == prob)]
                if not subset.empty:
                    plt.scatter(
                        subset['Capacity'], subset['MeanReward'], 
                        marker=markers[i % len(markers)],  # Cycle through markers
                        label=f"{algo_name}, Prob={prob}"
                    )

        print(self.calculate_percent_improvement(df.sort_values('Capacity')))
        plt.title("Mean Reward vs Capacity for All Algorithms and Probabilities")
        plt.xlabel("Capacity")
        plt.ylabel("Mean Reward")
        plt.legend(title="Algorithm, Probability", loc='best')
        plt.grid(True)
        plt.tight_layout()  # Adjust layout to prevent overlap
        plt.show()

    def plot_all_data(self, save_dir):
        """
        Plot all data for each mission duration on separate plots, with series based on algorithm, threshold, and latitude.
        This includes plotting the 'Optimal', 'Threshold', and 'Charge Threshold' algorithms.

        Parameters:
            save_dir (str): The directory where the plots will be saved.
        """
        # Ensure the save directory exists
        os.makedirs(save_dir, exist_ok=True)

        df = self.get_results_df()
        df.to_csv("results.csv")
        print(df)

        # Get unique mission durations (StartDate-EndDate)
        mission_durations = df[['StartDate', 'EndDate']].drop_duplicates()

        for _, mission in mission_durations.iterrows():
            start_date, end_date = mission['StartDate'], mission['EndDate']
            mission_label = f"{start_date}-{end_date}"

            # Filter data for the current mission duration
            mission_df = df[(df['StartDate'] == start_date) & (df['EndDate'] == end_date)]

            # Get unique latitudes
            latitudes = mission_df['Latitude'].unique()

            for lat in latitudes:
                # Filter data for the current latitude
                lat_df = mission_df[mission_df['Latitude'] == lat]

                # Separate the optimal algorithm (Threshold = NaN)
                optimal_df = lat_df[lat_df['Threshold'].isna()]
                # Separate Charge Threshold algorithm
                charge_threshold_df = lat_df[lat_df['Algorithm'] == 'Charge Threshold']
                # Separate other Threshold algorithms
                threshold_df = lat_df[(lat_df['Algorithm'] == 'Threshold') & (lat_df['Threshold'].notna())]

                plt.figure(figsize=(12, 8))

                # Plot the optimal algorithm
                if not optimal_df.empty:
                    plt.scatter(
                        optimal_df['Capacity'], optimal_df['MeanReward'],
                        marker='X', color='black', s=100,
                        label="Optimal Algorithm"
                    )

                # Plot the Charge Threshold algorithm
                if not charge_threshold_df.empty:
                    plt.scatter(
                        charge_threshold_df['Capacity'], charge_threshold_df['MeanReward'],
                        marker='D', color='blue', s=60,
                        label="Charge Threshold"
                    )

                # Plot the Threshold algorithm grouped by Threshold value
                for threshold_value, subset in threshold_df.groupby('Threshold'):
                    plt.scatter(
                        subset['Capacity'], subset['MeanReward'],
                        label=f"Threshold, t={threshold_value}"
                    )

                # Plot customization
                plt.title(f"Mean Reward vs Capacity\nMission: {mission_label}, Latitude: {lat}")
                plt.xlabel("Capacity (Ah)")
                plt.ylabel("Mean Reward")
                plt.legend(title="Algorithm", loc='best')
                plt.grid(True)
                plt.tight_layout()  # Adjust layout to prevent overlap

                # Save the plot
                save_path = os.path.join(save_dir, f"mean_reward_{mission_label}_lat{lat}.png")
                plt.savefig(save_path)
                plt.close()

    def plot_combined_data(self, save_dir):
        """
        Plot all data for each mission duration on separate plots, with series based on latitude,
        algorithm, and threshold. This includes plotting the 'Optimal', 'Threshold', and 'Charge Threshold' algorithms.

        Parameters:
            save_dir (str): The directory where the plots will be saved.
        """
        import os
        import matplotlib.pyplot as plt

        # Ensure the save directory exists
        os.makedirs(save_dir, exist_ok=True)

        df = self.get_results_df()
        print(df)

        # Get unique mission durations (StartDate-EndDate)
        mission_durations = df[['StartDate', 'EndDate']].drop_duplicates()

        for _, mission in mission_durations.iterrows():
            start_date, end_date = mission['StartDate'], mission['EndDate']
            mission_label = f"{start_date}-{end_date}"

            # Filter data for the current mission duration
            mission_df = df[(df['StartDate'] == start_date) & (df['EndDate'] == end_date)]

            plt.figure(figsize=(12, 8))

            # Get unique latitudes
            latitudes = mission_df['Latitude'].unique()

            for lat in latitudes:
                lat_df = mission_df[mission_df['Latitude'] == lat]

                # Separate the optimal algorithm (Threshold = NaN)
                optimal_df = lat_df[lat_df['Threshold'].isna()]
                # Separate Charge Threshold algorithm
                charge_threshold_df = lat_df[lat_df['Algorithm'] == 'Charge Threshold']
                # Separate other Threshold algorithms
                threshold_df = lat_df[(lat_df['Algorithm'] == 'Threshold') & (lat_df['Threshold'].notna())]

                # Plot the optimal algorithm
                if not optimal_df.empty:
                    plt.scatter(
                        optimal_df['Capacity'], optimal_df['MeanReward'],
                        marker='X', s=100,
                        label=f"Optimal Algorithm (Lat {lat})"
                    )

                # Plot the Charge Threshold algorithm
                if not charge_threshold_df.empty:
                    plt.scatter(
                        charge_threshold_df['Capacity'], charge_threshold_df['MeanReward'],
                        marker='D', s=60,
                        label=f"Charge Threshold (Lat {lat})"
                    )

                # Plot the Threshold algorithm grouped by Threshold value
                for threshold_value, subset in threshold_df.groupby('Threshold'):
                    plt.scatter(
                        subset['Capacity'], subset['MeanReward'],
                        label=f"Threshold, t={threshold_value} (Lat {lat})"
                    )

            # Plot customization
            plt.title(f"Mean Reward vs Capacity\nMission: {mission_label}")
            plt.xlabel("Capacity (Ah)")
            plt.ylabel("Mean Reward")
            plt.legend(title="Algorithm & Latitude", loc='best')
            plt.grid(True)
            plt.tight_layout()  # Adjust layout to prevent overlap

            # Save the plot
            save_path = os.path.join(save_dir, f"mean_reward_{mission_label}.png")
            plt.savefig(save_path)
            plt.close()



    def plot_reward_vs_threshold(self, df, output_dir):
        """
        Create and save a separate plot of reward vs threshold for each battery capacity.

        Parameters:
        - df: DataFrame containing 'Capacity', 'Threshold', 'MeanReward', and 'Algorithm' columns.
        - output_dir: Directory where the plots will be saved.
        """
        # Ensure the output directory exists
        os.makedirs(output_dir, exist_ok=True)

        # Get the unique battery capacities
        capacities = df['Capacity'].unique()

        for capacity in capacities:
            plt.figure(figsize=(8, 6))

            # Filter data for the current capacity
            capacity_df = df[df['Capacity'] == capacity]

            # Filter out rows where 'Threshold' is not NaN (for Threshold-based algorithms)
            threshold_df = capacity_df[capacity_df['Threshold'].notna()]

            # Extract thresholds and rewards
            thresholds = threshold_df['Threshold'].values
            rewards = threshold_df['MeanReward'].values

            # Find the optimal reward (assuming 'Optimal' is in the 'Algorithm' column)
            optimal_df = capacity_df[capacity_df['Algorithm'] == 'Optimal']
            optimal_reward = optimal_df['MeanReward'].mean() if not optimal_df.empty else None

            # Find the greedy reward (assuming a threshold of 0.0 represents Greedy)
            greedy_df = threshold_df[threshold_df['Threshold'] == 0.0]
            greedy_reward = greedy_df['MeanReward'].mean() if not greedy_df.empty else None

            # Sort thresholds and rewards for plotting
            sorted_indices = np.argsort(thresholds)
            thresholds = thresholds[sorted_indices]
            rewards = rewards[sorted_indices]

            # Plot the threshold-based rewards
            plt.plot(thresholds, rewards, marker='o', linestyle='-', color='orange', label='Threshold')

            # Plot the optimal reward as a horizontal line if available
            if optimal_reward is not None:
                plt.axhline(y=optimal_reward, color='blue', linestyle='--', label='Optimal')

            # Plot the greedy reward as a horizontal line if available
            if greedy_reward is not None:
                plt.axhline(y=greedy_reward, color='red', linestyle='--', label='Greedy')

            # Customize the plot
            plt.title(f'Reward vs Threshold (Battery Capacity: {capacity} Ah)')
            plt.xlabel('Threshold')
            plt.ylabel('Mean Reward')
            plt.legend()
            plt.grid(True)

            # Save the plot to the specified directory
            output_path = os.path.join(output_dir, f'reward_vs_threshold_c{capacity}.png')
            plt.savefig(output_path)
            plt.close()

            print(f"Plot saved for battery capacity {capacity} Ah at: {output_path}")


    def plot_reward_histogram(self, directory, bins=50):
        """Plot histograms of Reward values from each file in a directory on separate subplots."""
        
        # Get a list of all .pkl files in the directory
        files = [f for f in os.listdir(directory) if f.endswith('.pkl')]
        
        # Set up the figure with the appropriate number of subplots
        fig, axes = plt.subplots(len(files), 1, figsize=(10, 4 * len(files)),sharex=True)
        fig.tight_layout(pad=3)

        # Ensure axes is always a list for consistent indexing, even with one file
        num_files = len(files)
        if num_files == 1:
            axes = [axes]
        elif num_files == 0:
            raise ValueError("No .pkl files found in the directory.")
        
        # Loop through each file and create a subplot
        for i, filename in enumerate(files):
            filepath = os.path.join(directory, filename)
            df = pd.read_pickle(filepath)

            # Regex to handle all cases (Optimal, Threshold, and Greedy)
            match = re.match(r"(\w+)_Data_c(\d+)_((p|t)([\d.]+))_(\d+)min_(\d+-\d+)", filename)
            if match:
                algo, cap, _, key, value, dt, days = match.groups()
                cap = int(cap)
                dt = int(dt)

                # Handle algorithm name for Greedy
                if algo == "Threshold" and key == "t" and float(value) == 0.0:
                    algo = "Greedy"
                    value = None  # Threshold is irrelevant for Greedy

                # Prepare title details
                details = f"Capacity: {cap} Ah, "
                details += f"Failure p={value}" if key == "p" else f"Threshold t={value}"
                title = f"{algo} ({details}, {dt} min steps)"

            else:
                title = f"Unmatched: {filename}"

            # Check if "Reward" column exists in the DataFrame
            if "Reward" in df.columns:
                ax = axes[i] if num_files > 1 else axes  # Single plot case
                ax.hist(df["Reward"], bins=bins, edgecolor="white")
                ax.set_title(title)
                ax.set_xlabel("Whales Spotted")
                ax.set_ylabel("Number of Cases")
                ax.set_xlim((0, 100))
                ax.tick_params(axis="x", which="both", labelbottom=True)
            else:
                print(f"No 'Reward' column found in {filename}. Skipping this file.")

        plt.subplots_adjust(hspace=0.5)  # Adjust space between subplots
        if self.show:
            plt.show()
        else:
            filename = "histogram.png"
            plt.savefig(r"Figures\Histogram")# + f"\{filename}")

    def plot_percent_improvement(self, df, save_dir):
        """
        Calculate and plot the percent improvement of the Optimal algorithm over Threshold
        and Charge Threshold algorithms for each mission duration.

        Parameters:
            df (DataFrame): The dataframe containing the results.
            save_dir (str): The directory where the plots will be saved.
        """
        # Ensure the save directory exists
        os.makedirs(save_dir, exist_ok=True)

        # Get unique mission durations (StartDate-EndDate)
        mission_durations = df[['StartDate', 'EndDate']].drop_duplicates()

        for _, mission in mission_durations.iterrows():
            start_date, end_date = mission['StartDate'], mission['EndDate']
            mission_label = f"{start_date}-{end_date}"

            # Filter data for the current mission duration
            mission_df = df[(df['StartDate'] == start_date) & (df['EndDate'] == end_date)]

            # Get Optimal algorithm results
            optimal_df = mission_df[mission_df['Algorithm'] == 'Optimal']

            # Get Threshold algorithm results
            threshold_df = mission_df[mission_df['Algorithm'] == 'Threshold']

            # Get Charge Threshold algorithm results
            charge_threshold_df = mission_df[mission_df['Algorithm'] == 'Charge Threshold']

            # Create lists to store results
            capacities = []
            improvements_threshold = []
            improvements_charge_threshold = []

            # Calculate percent improvement over Threshold and Charge Threshold
            for cap in mission_df['Capacity'].unique():
                # Get the optimal mean reward for the current capacity
                optimal_reward = optimal_df[optimal_df['Capacity'] == cap]['MeanReward'].max()

                # Get the threshold mean reward for the current capacity
                threshold_reward = threshold_df[threshold_df['Capacity'] == cap]['MeanReward'].max()

                # Get the charge threshold mean reward for the current capacity
                charge_threshold_reward = charge_threshold_df[charge_threshold_df['Capacity'] == cap]['MeanReward'].max()

                if not np.isnan(optimal_reward) and not np.isnan(threshold_reward):
                    improvement_threshold = ((optimal_reward - threshold_reward) / threshold_reward) * 100
                    improvements_threshold.append(improvement_threshold)
                else:
                    improvements_threshold.append(np.nan)

                if not np.isnan(optimal_reward) and not np.isnan(charge_threshold_reward):
                    improvement_charge_threshold = ((optimal_reward - charge_threshold_reward) / charge_threshold_reward) * 100
                    improvements_charge_threshold.append(improvement_charge_threshold)
                else:
                    improvements_charge_threshold.append(np.nan)

                capacities.append(cap)

            # Plot the results
            plt.figure(figsize=(10, 6))
            plt.scatter(capacities, improvements_threshold, marker='o', linestyle='-', label='Optimal vs Threshold', color='orange')
            plt.scatter(capacities, improvements_charge_threshold, marker='o', linestyle='-', label='Optimal vs Charge Threshold', color='blue')

            plt.title(f'Percent Improvement of Optimal Over Other Algorithms\nMission Duration: {mission_label}')
            plt.xlabel('Battery Capacity (Ah)')
            plt.ylabel('Percent Improvement (%)')
            plt.grid(True)
            plt.legend()

            # Save the plot
            plt.tight_layout()
            save_path = os.path.join(save_dir, f'percent_improvement_{mission_label}.png')
            plt.savefig(save_path)
            plt.close()




if __name__ == '__main__':
    dire = r"Results\Analysis"
    
    processor = DataProcessor(directory=dire)  # Use "." for the current directory
    processor.process_files()
    df = processor.get_results_df()
    processor.plot_all_data(r"Results\Analysis")
    # processor.plot_optimal_battery_capacity(df)
    # processor.plot_percent_improvement(df,"Figures")
    # processor.plot_reward_vs_threshold(df,r".")
    
    
    ## Histogram
    # processor.plot_reward_histogram(r"Figures\Histogram")

    # # # Plot States
    direct = r"Results\Analysis"
    utc_offset = timezone(timedelta(hours=0))
    start_date = pd.to_datetime(datetime(2024,1,1).replace(tzinfo=utc_offset))
    solar = StateHistoryPlotter(direct,start_date,"10min")
    solar.plot_data()

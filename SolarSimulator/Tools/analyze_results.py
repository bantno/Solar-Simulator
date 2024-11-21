import pandas as pd
import matplotlib.pyplot as plt
import os
import re

class PickleDataProcessor:
    def __init__(self, directory, show=False):
        """Initialize the processor with the directory containing pickle files."""
        self.directory = directory
        self.results = []  # Store results as a list of dictionaries
        self.show = show

    def process_files(self):
        """Read all pickle files and store their mean results."""
        for filename in os.listdir(self.directory):
            if filename.endswith(".pkl"):
                match = re.match(r"(\w+)_Data_c(\d+)(?:_t([\d.]+))?(?:_p([\d.]+))?_(\d+)min_(\d+-\d+)", filename)

                if match:
                    algo, cap, threshold, prob, dt, date_range = match.groups()
                    cap = int(cap)
                    dt = int(dt)
                    
                    # Convert threshold and probability to floats if they are found, otherwise set to None
                    threshold = float(threshold) if threshold is not None else None
                    prob = float(prob) if prob is not None else None

                    # Assuming calculate_mean_rewards_and_failures is defined elsewhere in your class
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
                        "MeanReward": mean_reward,
                        "MeanFailureStep": mean_failure_step
                    })
                else:
                    print(f"Filename {filename} does not match expected pattern.")

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


    def plot_all_data(self):
        """Plot all data on one plot with series based on algorithm and threshold."""
        df = self.get_results_df()
        print(df)

        # Separate the optimal algorithm (Threshold = NaN)
        optimal_df = df[df['Threshold'].isna()]
        other_df = df[df['Threshold'].notna()]

        plt.figure(figsize=(12, 8))
        # markers = ['o', 's', '^', 'D', 'P', 'X', '*']  # A list of markers for variety
        # colors = plt.cm.tab20.colors  # Use a colormap for consistent color variety

        # Plot the optimal algorithm
        if not optimal_df.empty:
            plt.scatter(
                optimal_df['Capacity'], optimal_df['MeanReward'], 
                # marker='X', color='black', s=100,  # Unique marker, size, and color
                label="Optimal Algorithm"
            )

        # Plot other algorithms grouped by Algorithm and Threshold
        for i, ((algo_name, threshold), subset) in enumerate(other_df.groupby(['Algorithm', 'Threshold'])):
            # marker = markers[i % len(markers)]  # Cycle through markers
            # color = colors[i % len(colors)]    # Cycle through colors

            plt.scatter(
                subset['Capacity'], subset['MeanReward'], 
                # marker=marker, color=color,  # Assign unique marker and color
                label=f"{algo_name}, Threshold={threshold}"
            )

        plt.title("Mean Reward vs Capacity for All Algorithms")
        plt.xlabel("Capacity")
        plt.ylabel("Mean Reward")
        plt.legend(title="Algorithm", loc='best')
        plt.grid(True)
        plt.tight_layout()  # Adjust layout to prevent overlap
        plt.show()
    
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
                ax.set_xlim((0, 175))
                ax.tick_params(axis="x", which="both", labelbottom=True)
            else:
                print(f"No 'Reward' column found in {filename}. Skipping this file.")

        plt.subplots_adjust(hspace=0.5)  # Adjust space between subplots
        if self.show:
            plt.show()
        else:
            filename = "histogram.png"
            plt.savefig(r"Figures\Histogram")# + f"\{filename}")

    def calculate_percent_improvement(self,df):
        # Separate data for Optimal and Threshold algorithms
        optimal_df = df[df['Algorithm'] == 'Optimal'].set_index('Capacity')
        threshold_df = df[df['Algorithm'] == 'Threshold'].set_index('Capacity')

        # Align dataframes by Capacity to ensure we calculate the difference on matching capacities
        merged_df = optimal_df[['MeanReward']].join(threshold_df[['MeanReward']], lsuffix='_optimal', rsuffix='_threshold')

        # Calculate percent improvement
        merged_df['Percent Improvement'] = ((merged_df['MeanReward_optimal'] - merged_df['MeanReward_threshold']) / merged_df['MeanReward_threshold']) * 100
        
        # Reset index for better readability if needed
        return merged_df[['Percent Improvement']]

# Example usage
if __name__ == "__main__":
    dire = r"."
    processor = PickleDataProcessor(directory=dire)  # Use "." for the current directory
    # processor.plot_reward_histogram(directory=dire,bins=30)
    processor.process_files()
    df = processor.get_results_df()
    processor.calculate_percent_improvement(df).to_csv("test.csv")
    # processor.plot_all_data()
    processor.plot_reward_histogram(directory=dire,)
    # print(processor.results)
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
        """Plot all data on one plot with series based on algorithm only."""
        df = self.get_results_df()

        plt.figure(figsize=(10, 7))
        markers = ['o', 's', '^', 'D', 'P', 'X', '*']  # A list of markers for variety

        # Iterate over unique algorithms
        for i, (algo_name, subset) in enumerate(df.groupby('Algorithm')):
            plt.scatter(
                subset['Capacity'], subset['MeanReward'], 
                marker=markers[i % len(markers)],  # Cycle through markers
                label=f"{algo_name}"
            )
        print(self.calculate_percent_improvement(df.sort_values('Capacity')))
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
        if len(files) == 1:
            axes = [axes]
        
        # Loop through each file and create a subplot
        for i, filename in enumerate(files):
            filepath = os.path.join(directory, filename)
            df = pd.read_pickle(filepath)
            # match = re.match(r"(\w+)_Data_c(\d+)_p([\d.]+)_(\d+)min\.pkl", filename)
            match = re.match(r"(\w+)_Data_c(\d+)_p([\d.]+)_(\d+)min_(\d+-\d+)", filename)

            if match:
                algo, cap, prob, dt,_ = match.groups()
                cap = int(cap)
                prob = float(prob)
                dt = int(dt)

            # Check if "Reward" column exists in the file
            if "Reward" in df.columns:
                ax = axes[i]
                ax.hist(df["Reward"], bins=bins,edgecolor='white')
                ax.set_title(f"{algo}")
                ax.set_xlabel("Whales Spotted")
                ax.set_ylabel("Number of Cases")
                ax.set_xlim((0, 60))
                ax.tick_params(axis='x', which='both', labelbottom=True)
            else:
                print(f"No 'Reward' column found in {filename}. Skipping this file.")

        plt.subplots_adjust(hspace=0.5)  # Adjust space between subplots
        if self.show:
            plt.show()
        else:
            filename = "histogram.png"
            plt.savefig(r"Figures\Histogram" + f"\{filename}")

    def calculate_percent_improvement(self,df):
        # Separate data for Optimal and Threshold algorithms
        optimal_df = df[df['Algorithm'] == 'Optimal'].set_index('Capacity')
        threshold_df = df[df['Algorithm'] == 'Threshold'].set_index('Capacity')

        # Align dataframes by Capacity to ensure we calculate the difference on matching capacities
        merged_df = optimal_df[['MeanReward']].join(threshold_df[['MeanReward']], lsuffix='_optimal', rsuffix='_threshold')

        # Calculate percent improvement
        merged_df['Percent Improvement'] = ((merged_df['MeanReward_optimal'] - merged_df['MeanReward_threshold']) / merged_df['MeanReward_threshold']) * 100

        # Reset index for better readability if needed
        return merged_df.reset_index()[['Capacity', 'Percent Improvement']]

# Example usage
if __name__ == "__main__":
    dire = r"Results\11-5\Jan2-Jun2\1000"
    processor = PickleDataProcessor(directory=dire)  # Use "." for the current directory
    processor.process_files()

    results_df = processor.get_results_df()
    # results_df.to_csv("Run.csv")
    print(results_df)  # Display the DataFrame

    processor.plot_by_algorithm_and_probability()
    # processor.plot_all_data()
    # processor.plot_reward_histogram(directory=dire,bins=30)
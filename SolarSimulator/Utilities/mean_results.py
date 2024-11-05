import pandas as pd
import matplotlib.pyplot as plt
import os
import re

class PickleDataProcessor:
    def __init__(self, directory):
        """Initialize the processor with the directory containing pickle files."""
        self.directory = directory
        self.results = []  # Store results as a list of dictionaries

    def process_files(self):
        """Read all pickle files and store their mean results."""
        for filename in os.listdir(self.directory):
            if filename.endswith(".pkl"):
<<<<<<< HEAD
                match = re.match(r"(\w+)_Data_c(\d+)_p([\d.]+)_(\d+)min\.pkl", filename)
                if match:
                    algo, cap, prob, dt = match.groups()
                    cap = int(cap)
                    prob = float(prob)
                    dt = int(dt)
=======
                match = re.match(r"(\w+)_Data_c(\d+)_p([\d.]+)\.pkl", filename)
                if match:
                    algo, cap, prob = match.groups()
                    cap = int(cap)
                    prob = float(prob)
>>>>>>> 255344c5e08cfb4d772612923fec16f8e17059ff

                    mean_reward, mean_failure_step = self.calculate_mean_rewards_and_failures(
                        os.path.join(self.directory, filename)
                    )

                    # Store results in a dictionary
                    self.results.append({
                        "Algorithm": algo,
                        "Capacity": cap,
<<<<<<< HEAD
                        "Timestep": dt,
=======
>>>>>>> 255344c5e08cfb4d772612923fec16f8e17059ff
                        "Probability": prob,
                        "MeanReward": mean_reward,
                        "MeanFailureStep": mean_failure_step
                    })

    def calculate_mean_rewards_and_failures(self, filepath):
        """Calculate the mean reward and failure step for a given pickle file."""
        df = pd.read_pickle(filepath)

        if 'Reward' in df.columns and 'LastStep' in df.columns:
            mean_reward = df['Reward'].mean()
            mean_failure_step = round(df['LastStep'].mean())
            return mean_reward, mean_failure_step
        else:
            print(f"Missing columns in {filepath}")
            return None, None

    def get_results_df(self):
        """Convert the results list into a DataFrame."""
        return pd.DataFrame(self.results)

    def plot_results_by_probability(self):
        """Plot Greedy and MDP datapoints with Capacity on the x-axis for each probability."""
        df = self.get_results_df()

        # Plot for each unique probability
        for prob in df['Probability'].unique():
            subset = df[df['Probability'] == prob]

            plt.figure(figsize=(8, 6))
            for algo in subset['Algorithm'].unique():
                algo_data = subset[subset['Algorithm'] == algo]
                plt.scatter(
                    algo_data['Capacity'], algo_data['MeanReward'], marker='o', label=algo
                )

            plt.title(f"Mean Reward vs Capacity (Probability={prob})")
            plt.xlabel("Capacity")
            plt.ylabel("Mean Reward")
            plt.legend(title="Algorithm")
            plt.grid(True)
            plt.show()
    
    def plot_all_data(self):
        """Plot all data on one plot with series based on algorithm and probability."""
        df = self.get_results_df()

        plt.figure(figsize=(10, 7))
        markers = ['o', 's', '^', 'D', 'P', 'X', '*']  # A list of markers for variety

        # Iterate over unique (Algorithm, Probability) combinations
        for i, (algo, prob) in enumerate(df.groupby(['Algorithm', 'Probability'])):
            algo_name, prob_value = algo
            subset = prob

            plt.scatter(
                subset['Capacity'], subset['MeanReward'], 
                marker=markers[i % len(markers)],  # Cycle through markers
                label=f"{algo_name} (p={prob_value})"
            )

        plt.title("Mean Reward vs Capacity for All Algorithms and Probabilities")
        plt.xlabel("Capacity")
        plt.ylabel("Mean Reward")
        plt.legend(title="Algorithm (Probability)", loc='best')
        plt.grid(True)
        plt.tight_layout()  # Adjust layout to prevent overlap
        plt.show()

# Example usage
if __name__ == "__main__":
<<<<<<< HEAD
    processor = PickleDataProcessor(directory=r".")  # Use "." for the current directory
    # processor = PickleDataProcessor(directory=r"Results\11-3\Jan2-Jun2-1000")  # Use "." for the current directory
=======
    # processor = PickleDataProcessor(directory=r"Results\11-3\Jan2-Jun2-1000")  # Use "." for the current directory
    processor = PickleDataProcessor(directory=r".")  # Use "." for the current directory
>>>>>>> 255344c5e08cfb4d772612923fec16f8e17059ff
    processor.process_files()

    results_df = processor.get_results_df()
    results_df.to_csv("Run.csv")
    print(results_df)  # Display the DataFrame

    # processor.plot_results_by_probability()
    processor.plot_all_data()

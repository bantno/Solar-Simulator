import pandas as pd
import matplotlib.pyplot as plt
import gc
import os
import re
from datetime import datetime, timedelta, timezone
from tqdm import tqdm
from multiprocessing import Pool

from seaplane_base import Seaplane
from simulation_base import Simulation
from plotting_base import SolarChargePlotter

class SolarPlaneSimulation:
    def __init__(self, lat=25, lon=-90, tz="Etc/GMT-5", pdc0=0, gamma=-0.0047,
                 capacity_ah=50.0, voltage=22.2, Cdtot=0.0, Cd0=0.02584, S=0.653,
                 af_mass=8.8, cruise_speed=20.0, rho=1.19, N_PROP=0.82, N_ESC=0.9,
                 start_date="2019-07-01", end_date="2019-08-02", dt=30,
                 num_runs=10000, visualize=False, save_dir=".", show=False):
        
        # Define plane parameters
        self.lat = lat
        self.lon = lon
        self.tz = tz
        self.capacity_ah = capacity_ah
        self.voltage = voltage
        self.S = S
        self.af_mass = af_mass
        self.cruise_speed = cruise_speed
        self.rho = rho
        self.Cd0 = Cd0 * 1.5
        self.Cdtot = Cdtot
        self.n_tot = N_PROP * N_ESC
        self.show = show

        # Define simulation parameters
        self.dt = dt
        self.num_runs = num_runs
        self.visualize = visualize
        self.save_dir = save_dir

        # Time settings
        utc_offset = timezone(timedelta(hours=-5))
        self.start_date = pd.to_datetime(datetime.strptime(start_date, "%Y-%m-%d").replace(tzinfo=utc_offset))
        self.end_date = pd.to_datetime(datetime.strptime(end_date, "%Y-%m-%d").replace(tzinfo=utc_offset))

        # Initialize the plane
        self.plane = Seaplane(
            lat=self.lat, lon=self.lon, tz=self.tz, pdc0=pdc0, gamma=gamma,
            cd0=self.Cd0, cs=True, tracking=False, cdtot=self.Cdtot,
            n_tot=self.n_tot, S=self.S, af_mass=self.af_mass,
            voltage=self.voltage, capacity=self.capacity_ah
        )

        # Initialize the simulation
        self.simulation = Simulation(self.plane, self.lat, self.lon, self.tz, save_history=self.visualize)
        self.results = []  # To store processed results

    def run(self, capacities=[50], thresholds=[0.1], mdp_probs=[0.9], success_prob=1.0):
        for cap in tqdm(capacities, desc="Processing capacities"):
            self.simulation.plane.capacity = cap

            for threshold in tqdm(thresholds, desc=f"Processing thresholds for cap={cap}", leave=False):
                algo = 'Threshold'
                times, data = self.simulation.run_simulation(
                    self.start_date, self.end_date, self.dt, algo=algo,
                    mdp_success_prob=0.9, true_success_prob=success_prob,
                    runs=self.num_runs, threshold=threshold
                )
                data.to_pickle(f"{self.save_dir}/{algo}_Data_c{cap}_t{threshold}_{self.dt}min_{self.start_date.day_of_year}-{self.end_date.day_of_year}_{self.num_runs}.pkl")
                del data
                gc.collect()

            for mdp_success_prob in tqdm(mdp_probs, desc=f"Processing probabilities for cap={cap}", leave=False):
                algo = 'Optimal'
                times, data = self.simulation.run_simulation(
                    self.start_date, self.end_date, self.dt, algo=algo,
                    mdp_success_prob=mdp_success_prob, true_success_prob=success_prob,
                    runs=self.num_runs
                )
                data.to_pickle(f"{self.save_dir}/{algo}_Data_c{cap}_p{mdp_success_prob}_{self.dt}min_{self.start_date.day_of_year}-{self.end_date.day_of_year}_{self.num_runs}.pkl")
                del data
                gc.collect()

        if self.visualize:
            plotter = SolarChargePlotter(self.save_dir, start_date=self.start_date, time_step=f"{self.dt}min")
            plotter.plot_data()

    def process_files(self):
        """Read all pickle files and store their mean results."""
        for filename in os.listdir(self.save_dir):
            if filename.endswith(".pkl"):
                match = re.match(r"(\w+)_Data_c(\d+)(?:_t([\d.]+))?(?:_p([\d.]+))?_(\d+)min_(\d+-\d+)", filename)
                if match:
                    algo, cap, threshold, prob, dt, date_range = match.groups()
                    cap = int(cap)
                    dt = int(dt)
                    threshold = float(threshold) if threshold is not None else None
                    prob = float(prob) if prob is not None else None

                    mean_reward, mean_failure_step = self.calculate_mean_rewards_and_failures(
                        os.path.join(self.save_dir, filename)
                    )

                    self.results.append({
                        "Algorithm": algo,
                        "Capacity": cap,
                        "Threshold": threshold,
                        "Timestep": dt,
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
        return None, None

    def get_results_df(self):
        """Convert the results list into a DataFrame."""
        return pd.DataFrame(self.results)

    def plot_by_algorithm_and_probability(self):
        """Plot data with series based on both algorithm and probability."""
        df = self.get_results_df()
        plt.figure(figsize=(12, 8))
        markers = ['o', 's', '^', 'D', 'P', 'X', '*']

        probabilities = df['Probability'].unique()
        algorithms = df['Algorithm'].unique()

        for i, algo_name in enumerate(algorithms):
            for prob in probabilities:
                subset = df[(df['Algorithm'] == algo_name) & (df['Probability'] == prob)]
                if not subset.empty:
                    plt.scatter(
                        subset['Capacity'], subset['MeanReward'], 
                        marker=markers[i % len(markers)], 
                        label=f"{algo_name}, Prob={prob}"
                    )

        plt.title("Mean Reward vs Capacity for All Algorithms and Probabilities")
        plt.xlabel("Capacity")
        plt.ylabel("Mean Reward")
        plt.legend(title="Algorithm, Probability", loc='best')
        plt.grid(True)
        plt.tight_layout()

        if self.show:
            plt.show()
        else:
            plot_path = os.path.join(self.save_dir, "Plot_By_Algorithm_Probability.png")
            plt.savefig(plot_path)
            plt.close()

    def plot_all_data(self):
        """Plot all data on one plot with series based on algorithm only."""
        df = self.get_results_df()
        plt.figure(figsize=(10, 7))
        markers = ['o', 's', '^', 'D', 'P', 'X', '*']

        for i, (algo_name, subset) in enumerate(df.groupby('Algorithm')):
            plt.scatter(
                subset['Capacity'], subset['MeanReward'], 
                marker=markers[i % len(markers)], 
                label=f"{algo_name}"
            )

        plt.title("Mean Reward vs Capacity for All Algorithms")
        plt.xlabel("Capacity")
        plt.ylabel("Mean Reward")
        plt.legend(title="Algorithm", loc='best')
        plt.grid(True)
        plt.tight_layout()

        if self.show:
            plt.show()
        else:
            plot_path = os.path.join(self.save_dir, "Plot_All_Data.png")
            plt.savefig(plot_path)
            plt.close()

    def plot_reward_histogram(self, bins=50):
        """Plot histograms of Reward values from each file in the directory."""
        files = [f for f in os.listdir(self.save_dir) if f.endswith('.pkl')]
        fig, axes = plt.subplots(len(files), 1, figsize=(10, 4 * len(files)), sharex=True)
        fig.tight_layout(pad=3)

        if len(files) == 1:
            axes = [axes]

        for i, filename in enumerate(files):
            filepath = os.path.join(self.save_dir, filename)
            df = pd.read_pickle(filepath)
            match = re.match(r"(\w+)_Data_c(\d+)_p([\d.]+)_(\d+)min_(\d+-\d+)", filename)
            if match:
                algo, cap, prob, dt, _ = match.groups()
                cap = int(cap)
                prob = float(prob)
                dt = int(dt)

            if "Reward" in df.columns:
                ax = axes[i]
                ax.hist(df["Reward"], bins=bins, edgecolor='white')
                ax.set_title(f"{algo}")
                ax.set_xlabel("Reward")
                ax.set_ylabel("Frequency")
                ax.set_xlim((0, 60))

        plt.subplots_adjust(hspace=0.5)
        if self.show:
            plt.show()
        else:
            plot_path = os.path.join(self.save_dir, "Reward_Histogram.png")
            plt.savefig(plot_path)
            plt.close()


    def calculate_percent_improvement(self):
        df = self.get_results_df().sort_values('Capacity')
        optimal_df = df[df['Algorithm'] == 'Optimal'].set_index('Capacity')
        threshold_df = df[df['Algorithm'] == 'Threshold'].set_index('Capacity')
        merged_df = optimal_df[['MeanReward']].join(threshold_df[['MeanReward']], lsuffix='_optimal', rsuffix='_threshold')
        merged_df['Percent Improvement'] = ((merged_df['MeanReward_optimal'] - merged_df['MeanReward_threshold']) / merged_df['MeanReward_threshold']) * 100
        return merged_df[['Percent Improvement']]
    
    def save_results_with_metadata(self, results, file_prefix="Simulation"):
        """
        Save results to a dated folder with filenames indicating key simulation parameters.

        Args:
            results (pd.DataFrame): DataFrame containing simulation results to save.
            file_prefix (str): Prefix for the saved file names.
        """
        # Create a dated results directory
        date_folder = datetime.now().strftime("%Y-%m-%d-%h-%m")
        results_dir = os.path.join(self.save_dir, date_folder)
        os.makedirs(results_dir, exist_ok=True)

        # Generate filename with metadata
        metadata = f"lat{self.lat}_lon{self.lon}_cap{self.capacity_ah}_dt{self.dt}"
        filename = f"{file_prefix}_{metadata}.csv"

        # Save results to the directory
        results_path = os.path.join(results_dir, filename)
        results.to_csv(results_path, index=False)

        print(f"Results saved to: {results_path}")

        # Optionally, save visualizations in the same directory
        plot_filename = f"{file_prefix}_{metadata}_plot.png"
        plot_path = os.path.join(results_dir, plot_filename)
        
        if self.visualize:
            self.plot_all_data()  # Ensure a method like plot_all_data saves plots
            plt.savefig(plot_path)
            print(f"Plot saved to: {plot_path}")

    def _run_simulation(self, args):
        """Helper function to execute a single simulation run."""
        cap, algo, threshold, mdp_success_prob, success_prob = args
        self.simulation.plane.capacity = cap
        
        if algo == "Threshold":
            times, data = self.simulation.run_simulation(
                self.start_date, self.end_date, self.dt, algo=algo,
                mdp_success_prob=0.9, true_success_prob=success_prob,
                runs=self.num_runs, threshold=threshold
            )
            filename = f"{self.save_dir}/{algo}_Data_c{cap}_t{threshold}_{self.dt}min_{self.start_date.day_of_year}-{self.end_date.day_of_year}_{self.num_runs}.pkl"
        elif algo == "Optimal":
            times, data = self.simulation.run_simulation(
                self.start_date, self.end_date, self.dt, algo=algo,
                mdp_success_prob=mdp_success_prob, true_success_prob=success_prob,
                runs=self.num_runs
            )
            filename = f"{self.save_dir}/{algo}_Data_c{cap}_p{mdp_success_prob}_{self.dt}min_{self.start_date.day_of_year}-{self.end_date.day_of_year}_{self.num_runs}.pkl"
        else:
            return None

        data.to_pickle(filename)
        del data
        gc.collect()

        return filename

    def run(self, capacities=[50], thresholds=[0.1], mdp_probs=[0.9], success_prob=1.0):
        tasks = []
        for cap in capacities:
            for threshold in thresholds:
                tasks.append((cap, "Threshold", threshold, None, success_prob))
            for mdp_prob in mdp_probs:
                tasks.append((cap, "Optimal", None, mdp_prob, success_prob))

        # Use multiprocessing to execute tasks in parallel
        with Pool(processes=6) as pool:
            list(tqdm(pool.imap(self._run_simulation, tasks), total=len(tasks), desc="Running simulations"))


# Example usage
if __name__ == "__main__":
    # Initialize the SolarPlaneSimulation with relevant parameters
    simulation = SolarPlaneSimulation(
        lat=25, lon=-90, tz="Etc/GMT-5",  # Location parameters
        start_date="2019-01-01",          # Simulation start date
        end_date="2019-07-01",            # Simulation end date
        dt=30,                            # Time step in minutes
        num_runs=1000,                     # Number of simulation runs
        visualize=True,                   # Enable visualization
        save_dir=r"Results\6monthRun-0.9fail", # Directory to save results
        show=True                         # Suppress immediate plot display
    )

    # Define simulation parameters
    capacities = [10, 20, 30, 40, 50, 60, 70, 80]  # Battery capacities in Amp-hours
    thresholds = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3] # Threshold values for 'Threshold' algorithm
    mdp_probs = [0.9]                              # MDP success probabilities for 'Optimal' algorithm
    success_prob = 0.9                             # True success probability

    # Run the simulation
    simulation.run(capacities=capacities, thresholds=thresholds, mdp_probs=mdp_probs, success_prob=success_prob)

    # Process files to extract and organize results
    simulation.process_files()

    # Generate results DataFrame
    results_df = simulation.get_results_df()
    print("Simulation Results:")
    print(results_df.head())  # Display the first few rows of results

    # Save results and visualizations with descriptive filenames and folders
    simulation.save_results_with_metadata(results_df, file_prefix="SolarPlaneSim")

    # Generate and save additional visualizations
    simulation.plot_all_data()                           # Plot all data by algorithm
    simulation.plot_reward_histogram(bins=30)           # Plot histogram of rewards

    # Calculate and display percent improvement
    improvements = simulation.calculate_percent_improvement()
    print("Percent Improvement in Mean Reward:")
    print(improvements)

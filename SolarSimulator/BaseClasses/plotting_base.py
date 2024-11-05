import os
import re
import matplotlib.pyplot as plt
import pandas as pd

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
        greedy_pattern = r"Greedy_Data_c(\d+)_p(0\.\d+)"
        mdp_pattern = r"MDP_Data_c(\d+)_p(\d+\.\d+)"
        
        if match := re.search(greedy_pattern, filename):
            cap, prob = match.groups()
            return f"Greedy, cap={cap}, p={prob}"
        elif match := re.search(mdp_pattern, filename):
            cap, prob = match.groups()
            return f"MDP, cap={cap}, p={prob}"
        else:
            return "Unknown"

    def plot_data(self):
        """Plot the state of charge and solar history for the first entry in each file in the directory."""
        plt.figure(figsize=(10, 10))
        
        # Plot State of Charge
        plt.subplot(2, 1, 1)
        for file in os.listdir(self.directory):
            if file.endswith(".pkl"):
                file_path = os.path.join(self.directory, file)
                df = self.load_first_entry(file_path)
                
                # Extract parameters from filename
                label = self.extract_parameters_from_filename(file)

                # Extract state history and solar history from the first entry
                state_history = df['StateHistory'].values[0]  # Convert string to list of tuples
                state_charge_levels = [state[0] for state in state_history]  # Extract charge levels
                solar_history = df['SolarHistory'].values[0]  # Convert string to list

                # Generate datetime index
                time_index = pd.date_range(start=self.start_date, periods=len(state_charge_levels), freq=self.time_step)

                # Plot State of Charge
                plt.plot(time_index, state_charge_levels, label=label)

        plt.title('State of Charge Over Time')
        plt.xlabel('Datetime')
        plt.ylabel('Charge Level (%)')
        plt.legend()
        plt.grid(True)

        # Plot Solar History
        plt.subplot(2, 1, 2)
        for file in os.listdir(self.directory):
            if file.endswith(".pkl"):
                file_path = os.path.join(self.directory, file)
                df = self.load_first_entry(file_path)

                # Extract parameters from filename
                label = self.extract_parameters_from_filename(file)

                # Extract solar history from the first entry
                solar_history = df['SolarHistory'].values[0]  # Convert string to list

                # Generate datetime index
                time_index = pd.date_range(start=self.start_date, periods=len(solar_history), freq=self.time_step)

                # Plot Solar History
                plt.plot(time_index, solar_history, label=label)

        plt.title('Solar History Over Time')
        plt.xlabel('Datetime')
        plt.ylabel('Solar Power (W/$m^2$)')
        plt.legend()
        plt.grid(True)

        plt.tight_layout()
        plt.show()

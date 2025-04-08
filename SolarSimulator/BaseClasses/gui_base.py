#!/usr/bin/env python3
"""
simulation_gui.py

A GUI to configure and run energy simulations with enhanced aesthetics and extended
functionality for setting threshold parameters for the threshold-based simulation.
User-configurable parameters include:
  - Results storage directory (simulation results are saved as compressed .npz files)
  - Latitude and longitude for fetching environment data (used to construct the file name)
  - Transition model (e.g., "moderate", "linear", etc.)
  - Simulation types (all simulation types are selectable)
  - Threshold parameters (Observation Threshold and Wind Threshold) that are enabled when
    the "Observation Threshold" simulation is selected
  - Horizon (number of time steps per episode)
  - Episodes per simulation run
  - Option to use multiprocessing

When “Run Simulation” is clicked, the GUI loads weather distributions from a pickle file
(assumed naming convention: data_expected_lat{lat}_lon{lon}_15min.pkl under Data/EXPECTED_DATA),
creates the environment provider and MDP (via a seaplane object for power parameters),
builds a backward induction solver, instantiates simulation objects per the selections (with threshold
parameters applied if needed), and then hands them off to the SimulationRunManager to run and store
the simulations.
"""

import os
import threading
import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter import ttk

import numpy as np
import pandas as pd

# Import from your code base – adjust these if your package structure differs.
from BaseClasses.environment_provider_base import StochasticWindSolarEnvironmentProvider
from BaseClasses.mdp_base import stochasticMDP
from BaseClasses.backward_induction_base import mdpBackwardSolver
from BaseClasses.seaplane_base import Seaplane
from BaseClasses.whale_base import WhaleRewardSeriesFactory
from BaseClasses.simulation_base import (OptimalAnalyticalPolicySimulation,
                                         OptimalContinuousAnalyticalPolicySimulation,
                                         ObservationThresholdSimulation,ObservationThresholdContinuousSimulation)
from BaseClasses.simulation_run_manager import SimulationRunManager

# Fixed simulation parameters
BATTERY_CAPACITY = 400.0    # in Wh
DELTA_T = 15                # time step duration (minutes, or as required)
INITIAL_STATE = np.array([100.0, 0])  # [State-of-charge, mode]

# Mapping of simulation type names to their instantiation lambdas.
# The "Observation Threshold" simulation lambda now accepts observation_threshold and wind_threshold values.
def create_simulations(mdp, solver, horizon, env_provider, observation_threshold=0.5, wind_threshold=10):
    sim_map = {
        "Optimal Energy Analytical": lambda: OptimalContinuousAnalyticalPolicySimulation(solver, horizon, INITIAL_STATE, env_provider),
        "Optimal State Analytical": lambda: OptimalAnalyticalPolicySimulation(solver, horizon, INITIAL_STATE, env_provider),
        "Observation State Threshold": lambda: ObservationThresholdSimulation(
            mdp, horizon, INITIAL_STATE,
            observation_threshold=observation_threshold,
            wind_threshold=wind_threshold,
            env_provider=env_provider
        ),
        "Observation Energy Threshold": lambda: ObservationThresholdContinuousSimulation(
            mdp, horizon, INITIAL_STATE,
            observation_threshold=observation_threshold,
            wind_threshold=wind_threshold,
            env_provider=env_provider
        ),
    }
    return sim_map

class SimulationGUI:
    def __init__(self, master):
        self.master = master
        master.title("Simulation Run Manager")
        master.geometry("750x650")

        # Set up ttk style
        self.style = ttk.Style()
        self.style.theme_use('clam')
        self.style.configure("TLabel", font=("Helvetica", 10))
        self.style.configure("TButton", font=("Helvetica", 10))
        self.style.configure("TEntry", font=("Helvetica", 10))
        self.style.configure("TCombobox", font=("Helvetica", 10))

        # Main container frame with padding
        self.container = ttk.Frame(master, padding="10 10 10 10")
        self.container.grid(row=0, column=0, sticky="nsew")
        master.columnconfigure(0, weight=1)
        master.rowconfigure(0, weight=1)

        # Header label
        header = ttk.Label(self.container, text="Simulation Configuration", font=("Helvetica", 16, "bold"))
        header.grid(row=0, column=0, columnspan=3, pady=(0, 15))

        # Storage Directory Frame
        dir_frame = ttk.Frame(self.container)
        dir_frame.grid(row=1, column=0, columnspan=3, sticky="ew", pady=5)
        ttk.Label(dir_frame, text="Results Storage Directory:").grid(row=0, column=0, sticky="w")
        self.dir_var = tk.StringVar()
        self.dir_entry = ttk.Entry(dir_frame, textvariable=self.dir_var, width=50)
        self.dir_entry.grid(row=0, column=1, padx=5)
        ttk.Button(dir_frame, text="Browse…", command=self.browse_directory).grid(row=0, column=2)

        # Coordinate Frame (Latitude and Longitude)
        coord_frame = ttk.Frame(self.container)
        coord_frame.grid(row=2, column=0, columnspan=3, sticky="ew", pady=5)
        ttk.Label(coord_frame, text="Latitude:").grid(row=0, column=0, sticky="w")
        self.lat_var = tk.StringVar(value="30")
        ttk.Entry(coord_frame, textvariable=self.lat_var, width=20).grid(row=0, column=1, padx=5)
        ttk.Label(coord_frame, text="Longitude:").grid(row=0, column=2, sticky="w")
        self.lon_var = tk.StringVar(value="-90")
        ttk.Entry(coord_frame, textvariable=self.lon_var, width=20).grid(row=0, column=3, padx=5)

        # Transition Model Selection
        ttk.Label(self.container, text="Transition Model:").grid(row=3, column=0, sticky="w", pady=5)
        self.transition_model_var = tk.StringVar(value="moderate")
        transition_options = ["moderate", "linear", "optimistic", "nowind", "nofail"]
        ttk.Combobox(self.container, textvariable=self.transition_model_var, values=transition_options, state="readonly", width=18).grid(row=3, column=1, sticky="w", pady=5)

        # Simulation Types Frame (Listbox with scrollbar)
        sim_frame = ttk.Frame(self.container)
        sim_frame.grid(row=4, column=0, columnspan=3, sticky="ew", pady=5)
        ttk.Label(sim_frame, text="Simulation Types:").grid(row=0, column=0, sticky="nw")
        self.sim_listbox = tk.Listbox(sim_frame, selectmode=tk.MULTIPLE, height=6, exportselection=False, font=("Helvetica", 10))
        simulation_type_options = [
            "Optimal Continuous Analytical", "Optimal Analytical",
            "Always Fly", "Always Float",
            "Observation Threshold", "Deterministic Optimal"
        ]
        for option in simulation_type_options:
            self.sim_listbox.insert(tk.END, option)
        self.sim_listbox.grid(row=0, column=1, padx=5)
        sim_scrollbar = ttk.Scrollbar(sim_frame, orient="vertical", command=self.sim_listbox.yview)
        sim_scrollbar.grid(row=0, column=2, sticky="ns")
        self.sim_listbox.config(yscrollcommand=sim_scrollbar.set)
        # Bind listbox selection event to update threshold fields.
        self.sim_listbox.bind("<<ListboxSelect>>", self.update_threshold_fields)

        # Threshold Parameters Frame (for Observation Threshold simulation)
        self.thresh_frame = ttk.Frame(self.container)
        self.thresh_frame.grid(row=5, column=0, columnspan=3, sticky="ew", pady=5)
        ttk.Label(self.thresh_frame, text="Observation Threshold:").grid(row=0, column=0, sticky="w")
        self.obs_threshold_var = tk.StringVar(value="0.5")
        self.obs_threshold_entry = ttk.Entry(self.thresh_frame, textvariable=self.obs_threshold_var, width=10)
        self.obs_threshold_entry.grid(row=0, column=1, padx=5)
        ttk.Label(self.thresh_frame, text="Wind Threshold:").grid(row=0, column=2, sticky="w", padx=(20,0))
        self.wind_threshold_var = tk.StringVar(value="10")
        self.wind_threshold_entry = ttk.Entry(self.thresh_frame, textvariable=self.wind_threshold_var, width=10)
        self.wind_threshold_entry.grid(row=0, column=3, padx=5)
        # Initially disable threshold fields until "Observation Threshold" is selected.
        self.set_threshold_fields_state("disabled")

        # Horizon and Episodes Frame
        param_frame = ttk.Frame(self.container)
        param_frame.grid(row=6, column=0, columnspan=3, sticky="ew", pady=5)
        ttk.Label(param_frame, text="Horizon (time steps):").grid(row=0, column=0, sticky="w")
        self.horizon_var = tk.StringVar(value="1000")
        ttk.Entry(param_frame, textvariable=self.horizon_var, width=15).grid(row=0, column=1, padx=5)
        ttk.Label(param_frame, text="Episodes per Simulation:").grid(row=0, column=2, sticky="w", padx=(20,0))
        self.episodes_var = tk.StringVar(value="100")
        ttk.Entry(param_frame, textvariable=self.episodes_var, width=15).grid(row=0, column=3, padx=5)

        # Multiprocessing Option
        self.multiproc_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(self.container, text="Use Multiprocessing", variable=self.multiproc_var).grid(row=7, column=0, columnspan=3, sticky="w", pady=5)

        # Run Simulation Button
        run_button = ttk.Button(self.container, text="Run Simulation", command=self.run_simulation_thread)
        run_button.grid(row=8, column=0, columnspan=3, pady=15)

        # Status Text Area with a Label
        ttk.Label(self.container, text="Status Log:").grid(row=9, column=0, sticky="w")
        self.status_text = tk.Text(self.container, width=70, height=10, state="disabled", font=("Helvetica", 10))
        self.status_text.grid(row=10, column=0, columnspan=3, pady=5)

    def set_threshold_fields_state(self, state):
        """Enable or disable the threshold entries based on provided state."""
        self.obs_threshold_entry.config(state=state)
        self.wind_threshold_entry.config(state=state)

    def update_threshold_fields(self, event=None):
        """Enable threshold fields if 'Observation Threshold' is among selections; otherwise disable them."""
        selected_indices = self.sim_listbox.curselection()
        selected = [self.sim_listbox.get(i) for i in selected_indices]
        if "Observation Threshold" in selected:
            self.set_threshold_fields_state("normal")
        else:
            self.set_threshold_fields_state("disabled")

    def browse_directory(self):
        directory = filedialog.askdirectory()
        if directory:
            self.dir_var.set(directory)

    def log_status(self, message):
        self.status_text.config(state="normal")
        self.status_text.insert(tk.END, message + "\n")
        self.status_text.see(tk.END)
        self.status_text.config(state="disabled")
        self.master.update()

    def run_simulation_thread(self):
        # Run simulation in a separate thread to keep the GUI responsive.
        thread = threading.Thread(target=self.run_simulation)
        thread.start()

    def run_simulation(self):
        try:
            self.log_status("Starting simulation run...")
            # Retrieve parameters from GUI
            results_dir = self.dir_var.get().strip()
            if not results_dir:
                messagebox.showerror("Error", "Please select a storage directory.")
                return

            try:
                lat = float(self.lat_var.get().strip())
                lon = float(self.lon_var.get().strip())
            except ValueError:
                messagebox.showerror("Error", "Latitude and Longitude must be numeric.")
                return

            transition_model = self.transition_model_var.get().strip()
            try:
                horizon = int(self.horizon_var.get().strip())
                episodes_per_sim = int(self.episodes_var.get().strip())
            except ValueError:
                messagebox.showerror("Error", "Horizon and Episodes must be valid integers.")
                return

            use_multiproc = self.multiproc_var.get()
            selected_indices = self.sim_listbox.curselection()
            if not selected_indices:
                messagebox.showerror("Error", "Please select at least one simulation type.")
                return
            simulation_type_options = [self.sim_listbox.get(i) for i in selected_indices]

            # Read threshold parameters from GUI (even if not enabled, default values will be used)
            try:
                observation_threshold = float(self.obs_threshold_var.get().strip())
                wind_threshold = float(self.wind_threshold_var.get().strip())
            except ValueError:
                messagebox.showerror("Error", "Threshold parameters must be numeric.")
                return

            self.log_status(f"Results directory: {results_dir}")
            self.log_status(f"Latitude: {lat}, Longitude: {lon}")
            self.log_status(f"Transition Model: {transition_model}")
            self.log_status(f"Simulation Types: {', '.join(simulation_type_options)}")
            self.log_status(f"Horizon: {horizon} steps; Episodes: {episodes_per_sim}")
            self.log_status(f"Multiprocessing: {use_multiproc}")

            # Load environment data from pickle.
            data_filename = f"data_expected_lat{lat}_lon{lon}_15min.pkl"
            data_path = os.path.join("Data", "EXPECTED_DATA", data_filename)
            self.log_status(f"Loading environment data from {data_path} ...")
            data = pd.read_pickle(data_path)
            # Build distribution arrays (truncate to chosen horizon)
            wind_shape = data['weibull_k'].values[:horizon]
            wind_scale = data['weibull_scale'].values[:horizon]
            solar_alpha = data['beta_alpha'].values[:horizon]
            solar_beta = data['beta_beta'].values[:horizon]
            wind_distributions = np.column_stack((wind_shape, wind_scale))
            solar_distributions = np.column_stack((solar_alpha, solar_beta))
            whale_reward_series = WhaleRewardSeriesFactory.create_series("real", horizon)
            self.log_status("Environment data loaded successfully.")

            # Create environment provider.
            env_provider = StochasticWindSolarEnvironmentProvider(
                solar_distributions=solar_distributions,
                wind_distributions=wind_distributions,
                whale_reward_series=whale_reward_series,
                delta_t=DELTA_T
            )

            # Create seaplane to obtain power parameters.
            seaplane = Seaplane(lat, lon, "none", capacity=BATTERY_CAPACITY / 22.2)
            power_params = seaplane.get_mdp_power_params()

            # Instantiate the MDP.
            mdp = stochasticMDP(
                battery_capacity_wh=BATTERY_CAPACITY,
                idle_power=power_params["idle_power"],
                cruise_power=power_params["cruise_power"],
                takeoff_power=power_params["takeoff_power"],
                failure_penalty=15,
                delta_t=DELTA_T,
                gamma=1.0,
                transition_model_name=transition_model,
                soc_increment=1.0,
                env_provider=env_provider
            )

            # Build the backward induction solver.
            solver = mdpBackwardSolver(mdp, horizon)

            # Create simulation instances.
            sim_map = create_simulations(mdp, solver, horizon, env_provider,
                                         observation_threshold=observation_threshold,
                                         wind_threshold=wind_threshold)
            simulation_list = []
            for sim_type in simulation_type_options:
                if sim_type in sim_map:
                    simulation_instance = sim_map[sim_type]()
                    simulation_list.append(simulation_instance)
                    self.log_status(f"Created instance for: {sim_type}")
                else:
                    self.log_status(f"Warning: Unknown simulation type: {sim_type}")

            if not simulation_list:
                messagebox.showerror("Error", "No valid simulation types selected.")
                return

            # Create the simulation run manager.
            manager = SimulationRunManager(episodes_per_simulation=episodes_per_sim, storage_dir=results_dir)
            self.log_status("Running simulations ... (this may take a while)")
            manager.run_simulations(simulation_list, use_multiprocessing=use_multiproc)
            self.log_status("Simulations completed and stored successfully.")
            messagebox.showinfo("Success", "Simulation run completed!")

        except Exception as e:
            messagebox.showerror("Error", f"An error occurred:\n{str(e)}")
            self.log_status(f"Error: {str(e)}")

def main():
    root = tk.Tk()
    app = SimulationGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()

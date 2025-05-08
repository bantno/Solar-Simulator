# yaml_simulation_runner.py
import yaml
import multiprocessing
import numpy as np
import pandas as pd
from BaseClasses.seaplane_base import Seaplane
from BaseClasses.environment_provider_base import StochasticWindSolarEnvironmentProvider
from BaseClasses.mdp_base import stochasticMDP
from BaseClasses.backward_induction_base import mdpBackwardSolver
from BaseClasses.simulation_base import (
    ObservationThresholdContinuousSimulation,
    OptimalContinuousAnalyticalPolicySimulation,
    UnifiedThresholdContinuousSimulation,
)
from BaseClasses.simulation_run_manager import SimulationRunManager
from BaseClasses.whale_base import WhaleRewardSeriesFactory


class SimulationFactory:
    def __init__(self, config):
        self.config    = config
        self.horizon   = config["horizon"]
        self.delta_t   = config.get("delta_t", 15)
        self.data_path = config["data_path"]
        self.start_dt  = pd.to_datetime(config["start_datetime"])
        self.episodes = config.get("episodes", 3000)
        self.transition_model = config.get("transition_model", "moderate")
        self.solar_model = config.get("solar_panel_model", "constant")

        self.wind_distributions = None
        self.solar_distributions = None
        self.whale_rewards = None

        self.env_provider = None
        self.base_mdp = None

        self._load_environment()
        self._initialize_mdp()

    def _load_environment(self):
        import numpy as np

        # 1) load the full DataFrame
        data = pd.read_pickle(self.data_path)

        # 2) pull out the user’s start time components
        md = self.start_dt.month
        dd = self.start_dt.day
        hh = self.start_dt.hour
        mm = self.start_dt.minute

        # 3) find the first row matching that time‑of‑year
        mask = (
            (data["month"]  == md) &
            (data["day"]    == dd) &
            (data["hour"]   == hh) &
            (data["minute"] == mm)
        )
        if not mask.any():
            raise ValueError(
                f"No rows in your data match {md}/{dd} {hh:02d}:{mm:02d}"
            )
        start_idx = mask.idxmax()

        # 4) grab exactly `horizon` rows, wrapping around the end of year if needed
        n = len(data)
        idxs = (np.arange(start_idx, start_idx + self.horizon) % n)
        window = data.iloc[idxs].reset_index(drop=True)

        # 5) extract distributions from that window
        wind_shape  = window["weibull_k"].values
        wind_scale  = window["weibull_scale"].values
        solar_alpha = window["beta_alpha"].values
        solar_beta  = window["beta_beta"].values
        whale_type  = self.config.get("whale_series", "real")

        self.wind_distributions  = np.column_stack((wind_shape,  wind_scale))
        self.solar_distributions = np.column_stack((solar_alpha, solar_beta))
        self.whale_rewards       = WhaleRewardSeriesFactory.create_series(whale_type, self.horizon)

        # 6) rebuild your provider with those exact slices
        self.env_provider = StochasticWindSolarEnvironmentProvider(
            solar_distributions=self.solar_distributions,
            wind_distributions=self.wind_distributions,
            whale_reward_series=self.whale_rewards,
            delta_t_min=self.delta_t,
            solar_panel_model=self.solar_model,
        )

        

    def _initialize_mdp(self):
        self.power_params = self._get_power_params()

    def _get_power_params(self):
        plane = Seaplane(
            lat=30, lon=-90, tz="none",
            capacity=self.config["battery_capacities"][0] / 22.2
        )
        return plane.get_mdp_power_params()

    def build_mdp(self, cap):
        return stochasticMDP(
            battery_capacity_wh=cap,
            idle_power=self.power_params["idle_power"],
            cruise_power=self.power_params["cruise_power"],
            takeoff_power=self.power_params["takeoff_power"],
            failure_penalty=15,
            delta_t=15,
            gamma=1.0,
            transition_model_name=self.transition_model,
            soc_increment=1.0,
            env_provider=self.env_provider,
        )
    
    def create_simulation(self, sim_type, cap, threshold=None, wind_threshold=None, save_states=False,full_history_episodes=None):
        """
        Create a simulation instance, optionally saving full state history
        for the first N episodes, then only summary info thereafter.

        Parameters:
        - save_states (bool): if True, saves full history for all episodes.
        - full_history_episodes (int, optional): if set, saves full history
          only for episodes with index < full_history_episodes, summary thereafter.
        """
        mdp = self.build_mdp(cap)
        initial_state = np.array([100.0, 0])
        datetime_str = self.start_dt.strftime("%Y-%m-%d %H:%M:%S")
        if sim_type == "threshold":
            return UnifiedThresholdContinuousSimulation(
                mdp,
                self.horizon,
                initial_state,
                observation_threshold=threshold,
                wind_threshold=wind_threshold,
                start_datetime=datetime_str,
                env_provider=self.env_provider,
                save_history=save_states,
                full_history_episodes=full_history_episodes
            )
        elif sim_type == "optimal":
            solver = mdpBackwardSolver(mdp, self.horizon)
            return OptimalContinuousAnalyticalPolicySimulation(
                solver,
                self.horizon,
                initial_state,
                start_datetime=datetime_str,
                env_provider=self.env_provider,
                save_history=save_states,
                full_history_episodes=full_history_episodes
            )
        else:
            raise ValueError(f"Unknown simulation type: {sim_type}")


class YAMLSimulationRunner:
    def __init__(self, config_path):
        with open(config_path, 'r') as file:
            self.config = yaml.safe_load(file)
        self.factory = SimulationFactory(self.config)

    def _build_parameter_list(self):
        params = []
        for cap in self.config["battery_capacities"]:
            for th in self.config["threshold_values"]:
                for wth in self.config["wind_thresholds"]:
                    params.append(("threshold", cap, th, wth))
            params.append(("optimal", cap, None, None))
        return params

    def run(self, use_multiprocessing=False):
        param_list = self._build_parameter_list()

        def _create(args):
            return self.factory.create_simulation(*args)

        if use_multiprocessing:
            with multiprocessing.Pool() as pool:
                simulations = pool.map(_create, param_list)
        else:
            simulations = list(map(_create, param_list))

        print(f"Created {len(simulations)} simulation objects.")
        manager = SimulationRunManager(
            episodes_per_simulation=self.config.get("episodes", 3000),
            storage_dir="." # TODO: Enable setting of storage directory in config file.
        )
        manager.run_simulations(simulations, use_multiprocessing=use_multiprocessing)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", type=str, default="threshold_test.yaml", help="Path to config YAML")
    args = parser.parse_args()

    runner = YAMLSimulationRunner(args.config)
    runner.run(use_multiprocessing=False)

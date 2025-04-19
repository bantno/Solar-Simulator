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
)
from BaseClasses.simulation_run_manager import SimulationRunManager
from BaseClasses.whale_base import WhaleRewardSeriesFactory


class SimulationFactory:
    def __init__(self, config):
        self.config = config
        self.horizon = config["horizon"]
        self.data_path = config["data_path"]
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
        data = pd.read_pickle(self.data_path)
        wind_shape = data['weibull_k'].values[:self.horizon]
        wind_scale = data['weibull_scale'].values[:self.horizon]
        solar_alpha = data['beta_alpha'].values[:self.horizon]
        solar_beta = data['beta_beta'].values[:self.horizon]
        whale_type = self.config.get("whale_series", "real")

        self.wind_distributions = np.column_stack((wind_shape, wind_scale))
        self.solar_distributions = np.column_stack((solar_alpha, solar_beta))
        self.whale_rewards = WhaleRewardSeriesFactory.create_series(whale_type, self.horizon)

        self.env_provider = StochasticWindSolarEnvironmentProvider(
            solar_distributions=self.solar_distributions,
            wind_distributions=self.wind_distributions,
            whale_reward_series=self.whale_rewards,
            delta_t_min=15,
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


    def create_simulation(self, sim_type, cap, threshold=None, wind_threshold=None):
        mdp = self.build_mdp(cap)
        initial_state = np.array([100.0, 0])
        if sim_type == "threshold":
            return ObservationThresholdContinuousSimulation(
                mdp, self.horizon, initial_state,
                observation_threshold=threshold,
                wind_threshold=wind_threshold,
                env_provider=self.env_provider
            )
        elif sim_type == "optimal":
            solver = mdpBackwardSolver(mdp, self.horizon)
            return OptimalContinuousAnalyticalPolicySimulation(
                solver, self.horizon, initial_state, env_provider=self.env_provider
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
    parser.add_argument("-c", "--config", type=str, default="quick_test.yaml", help="Path to config YAML")
    args = parser.parse_args()

    runner = YAMLSimulationRunner(args.config)
    runner.run(use_multiprocessing=False)

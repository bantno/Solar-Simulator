import argparse
import multiprocessing
from typing import List, Dict, Tuple, Optional
from itertools import product
import os


import yaml
import numpy as np
import pandas as pd

from BaseClasses.seaplane_base import Seaplane
from BaseClasses.environment_provider_base import StochasticWindSolarEnvironmentProvider
from BaseClasses.mdp_base import stochasticMDP
from BaseClasses.backward_induction_base import mdpAnalyticalBackwardSolver
from BaseClasses.simulation_base import (
    OptimalContinuousAnalyticalPolicySimulation,
    UnifiedThresholdContinuousSimulation,
)
from BaseClasses.simulation_run_manager import SimulationRunManager
from BaseClasses.whale_base import WhaleRewardSeriesFactory


class EnvironmentLoader:
    """
    Helper to load and prepare environmental data for a given location.
    """
    def __init__(
        self,
        data_path: str,
        start_dt: pd.Timestamp,
        horizon: int,
        delta_t: int,
        solar_model: str,
        whale_type: str,
    ):
        self.data_path = data_path
        self.start_dt = start_dt
        self.horizon = horizon
        self.delta_t = delta_t
        self.solar_model = solar_model
        self.whale_type = whale_type

    def _find_start_index(self, df: pd.DataFrame) -> int:
        mask = (
            (df["month"] == self.start_dt.month) &
            (df["day"] == self.start_dt.day) &
            (df["hour"] == self.start_dt.hour) &
            (df["minute"] == self.start_dt.minute)
        )
        if not mask.any():
            raise ValueError(f"No data for start time {self.start_dt}")
        return mask.idxmax()

    def _slice_window(self, df: pd.DataFrame, start_idx: int) -> pd.DataFrame:
        total = len(df)
        idxs = np.arange(start_idx, start_idx + self.horizon) % total
        return df.iloc[idxs].reset_index(drop=True)

    def _extract_distributions(
        self, window: pd.DataFrame
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        wind_k = window["weibull_k"].values
        wind_scale = window["weibull_scale"].values
        solar_a = window["beta_alpha"].values
        solar_b = window["beta_beta"].values
        wind_dist = np.column_stack(tup=(wind_k, wind_scale))
        solar_dist = np.column_stack(tup=(solar_a, solar_b))
        whale_series = WhaleRewardSeriesFactory.create_series(
            self.whale_type, self.horizon
        )
        return wind_dist, solar_dist, whale_series

    def _build_env_provider(
        self,
        wind_dist: np.ndarray,
        solar_dist: np.ndarray,
        whale_series: np.ndarray,
    ) -> StochasticWindSolarEnvironmentProvider:
        return StochasticWindSolarEnvironmentProvider(
            solar_distributions=solar_dist,
            wind_distributions=wind_dist,
            whale_reward_series=whale_series,
            delta_t_min=self.delta_t,
            solar_panel_model=self.solar_model,
        )

    def load(self) -> StochasticWindSolarEnvironmentProvider:
        df = pd.read_pickle(self.data_path)
        start_idx = self._find_start_index(df)
        window = self._slice_window(df, start_idx)
        wind_dist, solar_dist, whale_series = self._extract_distributions(window)
        return self._build_env_provider(wind_dist, solar_dist, whale_series)


class SimulationFactory:
    def __init__(
        self,
        config: Dict,
        location: Dict,
        horizon: int,
        failure_penalty: float,
        config_name: Optional[str] = None,
    ):
        """
        Factory to set up MDP parameters and simulations
        for given config, location, horizon, and penalty.
        """
        self.config_name = config_name
        self.config = config
        self.location = location
        self.horizon = horizon
        self.failure_penalty = failure_penalty
        self.delta_t = config.get("delta_t", 15)
        self.start_dt = pd.to_datetime(config["start_datetime"])
        self.transition_model = config.get("transition_model", "moderate")
        self.solar_model = config.get("solar_panel_model", "constant")
        self.whale_type = config.get("whale_series", "real")
        # self.soc_increment = config.get("soc_increment", 1.0)
        self.energy_increment_wh = config.get("energy_increment_wh", None)

        loader = EnvironmentLoader(
            data_path=location["data_path"],
            start_dt=self.start_dt,
            horizon=self.horizon,
            delta_t=self.delta_t,
            solar_model=self.solar_model,
            whale_type=self.whale_type,
        )
        self.env_provider = loader.load()

    def _compute_power_params(self, capacity_wh: float) -> Dict[str, float]:
        plane = Seaplane(
            lat=self.location.get("latitude", 0.0),
            lon=self.location.get("longitude", 0.0),
            tz=self.location.get("timezone", "UTC"),
            capacity=capacity_wh / 22.2,
        )
        return plane.get_mdp_power_params()

    def build_mdp(self, capacity_wh: float) -> stochasticMDP:
        params = self._compute_power_params(capacity_wh)
        
        # Convert absolute Wh‐step to percent, if requested
        if self.energy_increment_wh is not None:
            soc_inc = (self.energy_increment_wh / capacity_wh) * 100.0
        # else:
        #     soc_inc = self.soc_increment

        return stochasticMDP(
            battery_capacity_wh=capacity_wh,
            idle_power=params["idle_power"],
            cruise_power=params["cruise_power"],
            takeoff_power=params["takeoff_power"],
            landing_power=params["landing_power"],
            failure_penalty=self.failure_penalty,
            delta_t=self.delta_t,
            gamma=1.0,
            transition_model_name=self.transition_model,
            soc_increment=soc_inc,
            env_provider=self.env_provider,
        )

    def create_simulation(
        self,
        sim_type: str,
        cap: float,
        threshold: Optional[float] = None,
        wind_threshold: Optional[float] = None,
        save_states: bool = False,
        full_history_episodes: Optional[int] = None,
    ):
        
        state0 = np.array([100.0, 0])
        start_str = self.start_dt.strftime("%Y-%m-%d %H:%M:%S")
        mdp = self.build_mdp(cap)

        if sim_type == "threshold":
            sim = UnifiedThresholdContinuousSimulation(
                    mdp, self.horizon, state0,
                    observation_threshold=threshold,
                    wind_threshold=wind_threshold,
                    start_datetime=start_str,
                    env_provider=self.env_provider,
                    save_history=save_states,
                    full_history_episodes=full_history_episodes,
                )
            sim.location = self.location
            sim.failure_penalty = self.failure_penalty
            if self.energy_increment_wh is not None:
                soc_inc = (self.energy_increment_wh / cap) * 100.0
                sim.soc_increment = soc_inc
            return sim

        if sim_type == "optimal":
            # solver = mdpBackwardSolver(mdp, self.horizon)
            solver = mdpAnalyticalBackwardSolver(mdp,self.horizon, sim_name_prefix=self.config_name)
            solver.set_start_date(start_str)
            # solver.set_location(self.location)
            sim = OptimalContinuousAnalyticalPolicySimulation(
                solver, self.horizon, state0,
                start_datetime=start_str,
                env_provider=self.env_provider,
                save_history=save_states,
                full_history_episodes=full_history_episodes,
            )
            sim.location = self.location
            sim.failure_penalty = self.failure_penalty
            if self.energy_increment_wh is not None:
                soc_inc = (self.energy_increment_wh / cap) * 100.0
                sim.soc_increment = soc_inc
            return sim
        raise ValueError(f"Unknown simulation type: {sim_type}")


class YAMLSimulationRunner:
    def __init__(self, config_path: str):
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)
        self.start_datetimes = self.config.get("start_datetimes",
                                       [ self.config.get("start_datetime") ])
        self.config_basename = os.path.splitext(os.path.basename(config_path))[0]
        # Hyperparameter lists with fallbacks
        #  — allow multiple horizons
        self.horizons = self.config.get("horizons") or [self.config.get("horizon")]
        #  — allow multiple failure penalties (or fall back to a single one)
        self.failure_penalties = (
            self.config.get("failure_penalties")
            or [ self.config.get("failure_penalty", 15) ]
        )

        if "locations" in self.config:
            self.locations = self.config["locations"]
        else:
            # Fallback to single data_path
            self.locations = [{
                "data_path": self.config["data_path"],
                "latitude": self.config.get("latitude", 0.0),
                "longitude": self.config.get("longitude", 0.0),
                "timezone": self.config.get("timezone", "UTC"),
            }],

    def _build_param_list(self) -> List[Tuple]:
        """
        Build a list of parameters for each simulation type.
        Each entry is a tuple of (factory, sim_type, cap, threshold, wind_threshold).
        """

        thresholds      = self.config.get("threshold_values", [])
        wind_thresholds = self.config.get("wind_thresholds", [])
        params: List[Tuple] = []
        for start_dt in self.start_datetimes:
            # temporarily override the single start_datetime
            self.config["start_datetime"] = start_dt
            # 1) Threshold-based simulations, now including each failure_penalty:
            for loc, H, fp, cap, th, wth in product(
                self.locations,
                self.horizons,
                self.failure_penalties,
                self.config["battery_capacities"],
                thresholds,
                wind_thresholds,
            ):
                factory = SimulationFactory(self.config, loc, H, fp)
                params.append((factory, "threshold", cap, th, wth))

            # 2) Optimal-policy simulations, also over each failure_penalty:
            for loc, H, fp, cap in product(
                self.locations,
                self.horizons,
                self.failure_penalties,
                self.config["battery_capacities"],
            ):
                factory = SimulationFactory(self.config, loc, H, fp,config_name=self.config_basename)
                params.append((factory, "optimal", cap, None, None))

        return params


    def run(self, use_multiprocessing: bool = False) -> None:
        param_list = self._build_param_list()

        def create_task(args):
            factory, sim_type, cap, th, wth = args
            return factory.create_simulation(
                sim_type=sim_type,
                cap=cap,
                threshold=th,
                wind_threshold=wth,
                save_states=self.config.get("save_states", False),
                full_history_episodes=self.config.get("full_history_episodes"),
            )

        if use_multiprocessing:
            with multiprocessing.Pool() as pool:
                sims = pool.map(create_task, param_list)
        else:
            sims = [create_task(p) for p in param_list]

        print(f"Created {len(sims)} simulation objects.")
        manager = SimulationRunManager(
            episodes_per_simulation=self.config.get("episodes", 3000),
            storage_dir=self.config.get("storage_dir", "."),
            sim_name_prefix=self.config_basename
        )
        manager.run_simulations(sims, use_multiprocessing=use_multiprocessing)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run simulations from a YAML config."
    )
    parser.add_argument(
        "-c", "--config", default="config.yaml",
        help="YAML config file path"
    )
    parser.add_argument(
        "-p", "--parallel", action="store_true",
        help="Enable multiprocessing"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    runner = YAMLSimulationRunner(args.config)
    runner.run(use_multiprocessing=args.parallel)


if __name__ == "__main__":
    main()

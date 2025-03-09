import numpy as np
from abc import ABC, abstractmethod
from BaseClasses.transition_model_base import DeterministicTransitionLogic, ProbabilityModelFactory
from BaseClasses.environment_provider_base import AbstractEnvironmentProvider, DeterministicEnvironmentProvider

class AbstractMDP(ABC):
    def __init__(self,
                 battery_capacity_wh,
                 idle_power,
                 cruise_power,
                 takeoff_power,
                 solar_rate_series: np.ndarray,
                 wind_series: np.ndarray,
                 whale_reward_series: np.ndarray,
                 failure_penalty,
                 delta_t,
                 gamma,
                 transition_model_name: str,
                 soc_increment: float,
                 env_provider: AbstractEnvironmentProvider = None):
        """
        Initialize the MDP with time series inputs.
        """
        self.battery_capacity_wh = battery_capacity_wh
        self.battery_capacity_joules = battery_capacity_wh * 3600
        self.idle_power = idle_power
        self.cruise_power = cruise_power
        self.takeoff_power = takeoff_power
        self.failure_penalty = failure_penalty
        self.delta_t = delta_t
        self.gamma = gamma
        self.soc_increment = soc_increment

        if env_provider is None:
            env_provider = DeterministicEnvironmentProvider(solar_rate_series, wind_series, whale_reward_series, delta_t)
        self.env_provider = env_provider

        self.solar_rate_series = solar_rate_series
        self.wind_speed_series = wind_series
        self.whale_reward_series = whale_reward_series

        self.transition_model = ProbabilityModelFactory.select_probability_model(transition_model_name)
        self.actions = [0, 1]

    def _get_states(self):
        soc = np.arange(0, 100 + self.soc_increment, float(self.soc_increment))
        modes = np.array([0, 1])
        soc_grid, mode_grid = np.meshgrid(soc, modes)
        states = np.column_stack((soc_grid.ravel(), mode_grid.ravel()))
        states = np.row_stack((states, np.array([-1.0, 2])))
        return states

    def _ensure_vectorized_input(self, x, name="input"):
        if not isinstance(x, np.ndarray):
            raise TypeError(f"{name} must be a numpy array, got {type(x)}.")

    def sample_sunlight(self, t: int, n: int) -> np.ndarray:
        return self.env_provider.sample_sunlight(t, n)
    
    def sample_wind_speed(self, t: int, n: int) -> np.ndarray:
        return self.env_provider.sample_wind_speed(t, n)
    
    @abstractmethod
    def transition(self, states: np.ndarray, actions: np.ndarray, t: int) -> np.ndarray:
        pass

    @abstractmethod
    def reward(self, states: np.ndarray, actions: np.ndarray, next_states: np.ndarray, t: int) -> np.ndarray:
        pass

    def step(self, states: np.ndarray, actions: np.ndarray, t: int):
        self._ensure_vectorized_input(states, "states")
        self._ensure_vectorized_input(actions, "actions")
        next_states = self.transition(states, actions, t)
        rewards = self.reward(states, actions, next_states, t)
        return next_states, rewards

class DeterministicMDP(AbstractMDP):
    def __init__(self, battery_capacity_wh, idle_power, cruise_power, takeoff_power,
                 solar_rate_series: np.ndarray, wind_series: np.ndarray, whale_reward_series: np.ndarray,
                 failure_penalty, delta_t, gamma, transition_model_name: str, soc_increment: float,
                 env_provider: AbstractEnvironmentProvider = None):
        super().__init__(battery_capacity_wh, idle_power, cruise_power, takeoff_power,
                         solar_rate_series, wind_series, whale_reward_series,
                         failure_penalty, delta_t, gamma, transition_model_name, soc_increment, env_provider)
        self.transition_logic = DeterministicTransitionLogic(
            battery_capacity_joules=self.battery_capacity_joules,
            soc_increment=self.soc_increment,
            idle_power=self.idle_power,
            cruise_power=self.cruise_power,
            takeoff_power=self.takeoff_power,
            delta_t=self.delta_t,
            solar_rate_series=self.solar_rate_series,
            wind_series=self.wind_speed_series,
            transition_model=self.transition_model,
            env_provider=self.env_provider
        )

    def transition(self, states: np.ndarray, actions: np.ndarray, t: int) -> np.ndarray:
        return self.transition_logic.transition(states, actions, t)

    def reward(self, states: np.ndarray, actions: np.ndarray, next_states: np.ndarray, t: int) -> np.ndarray:
        whale_reward = np.where(actions == 1, self.whale_reward_series[t], 0.0)
        failure_penalty = np.where(next_states[:, 1] == 2, self.failure_penalty, 0.0)
        rewards = whale_reward - failure_penalty
        return rewards

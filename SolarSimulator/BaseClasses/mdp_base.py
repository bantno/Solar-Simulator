import numpy as np
from abc import ABC, abstractmethod
from BaseClasses.transition_model_base import DeterministicTransitionLogic, StochasticTransitionLogic, ProbabilityModelFactory, AbstractTransitionLogic
from BaseClasses.environment_provider_base import AbstractEnvironmentProvider, DeterministicEnvironmentProvider

class AbstractMDP(ABC):
    def __init__(self,
                 battery_capacity_wh,
                 idle_power,
                 cruise_power,
                 takeoff_power,
                 failure_penalty,
                 delta_t,
                 gamma,
                 transition_model_name: str,
                 soc_increment: float,
                 env_provider: AbstractEnvironmentProvider):
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
        self.env_provider = env_provider
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
    
    def get_obs(self,stage:int):
        return self.env_provider.sample_whale_observation(stage)

# class deterministicMDP(AbstractMDP):
#     def __init__(self, battery_capacity_wh, idle_power, cruise_power, takeoff_power,
#                  failure_penalty, delta_t, gamma, transition_model_name: str, soc_increment: float,
#                  env_provider: AbstractEnvironmentProvider):
#         super().__init__(battery_capacity_wh, idle_power, cruise_power, takeoff_power,
#                          failure_penalty, delta_t, gamma, transition_model_name, soc_increment, env_provider)
#         self.transition_logic = DeterministicTransitionLogic(
#             battery_capacity_joules=self.battery_capacity_joules,
#             soc_increment=self.soc_increment,
#             idle_power=self.idle_power,
#             cruise_power=self.cruise_power,
#             takeoff_power=self.takeoff_power,
#             delta_t=self.delta_t,
#             transition_model=self.transition_model,
#             env_provider=self.env_provider
#         )

#     def transition(self, states: np.ndarray, actions: np.ndarray, t: int) -> np.ndarray:
#         return self.transition_logic.transition(states, actions, t)

#     def reward(self, states: np.ndarray, actions: np.ndarray, next_states: np.ndarray, t: int) -> np.ndarray:
#         whale_reward = np.where(actions == 1, self.env_provider.sample_whale_observation(t,len(actions)), 0.0)
#         failure_penalty = np.where(next_states[:, 1] == 2, self.failure_penalty, 0.0)
#         rewards = whale_reward - failure_penalty
#         return rewards

class stochasticMDP(AbstractMDP):
    def __init__(self, battery_capacity_wh, idle_power, cruise_power, takeoff_power,
                 failure_penalty, delta_t, gamma, transition_model_name: str, soc_increment: float,
                 env_provider: AbstractEnvironmentProvider):
        super().__init__(battery_capacity_wh, idle_power, cruise_power, takeoff_power,
                         failure_penalty, delta_t, gamma, transition_model_name, soc_increment, env_provider)
        self.transition_logic = StochasticTransitionLogic(
            battery_capacity_joules=self.battery_capacity_joules,
            soc_increment=self.soc_increment,
            idle_power=self.idle_power,
            cruise_power=self.cruise_power,
            takeoff_power=self.takeoff_power,
            delta_t=self.delta_t,
            transition_model=self.transition_model,
            env_provider=self.env_provider
        )

    def transition(self, states: np.ndarray, actions: np.ndarray, t: int) -> np.ndarray:
        """Apply the transition logic to the provided set of states and actions."""
        return self.transition_logic.transition(states, actions, t)

    def reward(self, states: np.ndarray, actions: np.ndarray, next_states: np.ndarray, t: int) -> np.ndarray:
        """Calculate the reward for taking the given actions in the provided states."""
        # TODO: Determine if the sampling of the reward here is appropriate.
        if isinstance(actions,int):
            length = 1
        else:
            length = len(actions)
        samples = self.env_provider.sample_whale_observation(t, length)
        rewards = actions * samples
        fail_mask = next_states[:, 1] == 2
        rewards[fail_mask] -= self.failure_penalty
        return rewards


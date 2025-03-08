import numpy as np
from abc import ABC, abstractmethod
from BaseClasses.transition_model_base import DeterministicTransitionLogic, ProbabilityModelFactory

class AbstractMDP(ABC):
    def __init__(self,
                 battery_capacity_wh,
                 idle_power,
                 cruise_power,
                 takeoff_power,
                 solar_rate_series: np.ndarray,
                 wind_series:np.ndarray,
                 whale_reward_series: np.ndarray,
                 failure_penalty,
                 delta_t,
                 gamma,
                 transition_model_name: str,
                 soc_increment: float
                 ):
        """
        Initialize the MDP with time series inputs.
        
        Parameters:
            idle_power: Energy consumption when moored (per time step).
            cruise_power: Energy consumption when flying (per time step).
            takeoff_power: Additional energy consumption when taking off.
            solar_rate_series: np.ndarray of shape (T,) representing the available solar power at each time step.
            whale_reward_series: np.ndarray of shape (T,) representing the whale reward for flying at each time step.
            failure_penalty: Penalty applied when the plane becomes broken.
            delta_t: Duration of each time step.
            gamma: Discount factor.
        """
        self.battery_capacity_wh = battery_capacity_wh
        self.battery_capacity_joules = battery_capacity_wh * 3600
        self.idle_power = idle_power
        self.cruise_power = cruise_power
        self.takeoff_power = takeoff_power
        self.solar_rate_series = solar_rate_series
        self.wind_speed_series = wind_series
        self.whale_reward_series = whale_reward_series
        self.failure_penalty = failure_penalty
        self.delta_t = delta_t
        self.gamma = gamma
        self.transition_model = ProbabilityModelFactory.select_probability_model(transition_model_name)
        self.soc_increment = soc_increment
        
        # Define available actions: 0 -> Moor, 1 -> Fly.
        self.actions = [0, 1]

    def _get_states(self):
        """
        Returns array of all possible states as np.array([SoC, mode]),
        sorted by mode (ascending) and then by SoC (ascending).
        """
        # Create the SoC values as floats.
        soc = np.arange(0, 100 + self.soc_increment, float(self.soc_increment))
        
        # Define the modes array. (Or you could pass this in.)
        modes = np.array([0, 1])
        
        # Create a meshgrid so that each mode pairs with every SoC.
        soc_grid, mode_grid = np.meshgrid(soc, modes)
        
        # Combine the grids into a 2D array where each row is [SoC, mode].
        states = np.column_stack((soc_grid.ravel(), mode_grid.ravel()))

        states = np.row_stack((states,np.array([-1.0,2])))
        return states


    def _ensure_vectorized_input(self, x, name="input"):
        """
        Helper method to assert that the input is a numpy array.
        """
        if not isinstance(x, np.ndarray):
            raise TypeError(f"{name} must be a numpy array, got {type(x)}.")

    def sample_sunlight(self, t: int, n: int) -> np.ndarray:
        """
        Return an array of solar energy values for n states at time step t.
        For a deterministic implementation, simply return a fixed value based on the solar_rate_series.
        
        Parameters:
            t: Time step index.
            n: Number of states.
            
        Returns:
            A numpy array of shape (n,) with the solar energy value at time t.
        """
        if t >= len(self.solar_rate_series):
            raise IndexError("Time index t exceeds the length of solar_rate_series.")
        return np.full((n,), self.solar_rate_series[t] * self.delta_t)
    
    def sample_wind_speed(self, t: int, n: int) -> np.ndarray:
        """
        Return an array of wind speed values for n states at time step t.
        For a deterministic implementation, simply return a fixed value based on the wind_speed_series.

        Parameters:
            t: Time step index.
            n: Number of states.
            
        Returns:
            A numpy array of shape (n,) with the wind speed value at time t.
        """
        if t >= len(self.wind_speed_series):
            raise IndexError("Time index t exceeds the length of wind_speed_series.")
        return np.full((n,), self.wind_speed_series[t])
    
    @abstractmethod
    def transition(self, states: np.ndarray, actions: np.ndarray, t: int) -> np.ndarray:
        """
        Vectorized transition function at time step t.
        
        Given an array of current states and corresponding actions,
        compute and return an array of next states.
        
        Parameters:
            states: np.ndarray of shape (n, state_dim) where each row is a state (e.g., (SOC, mode)).
            actions: np.ndarray of shape (n,) representing the action for each state.
            t: The current time step index to use for time-dependent parameters.
            
        Returns:
            next_states: np.ndarray of shape (n, state_dim).
        """
        pass

    @abstractmethod
    def reward(self, states: np.ndarray, actions: np.ndarray, next_states: np.ndarray, t: int) -> np.ndarray:
        """
        Vectorized reward function at time step t.
        
        Computes the reward for each transition using:
          - The whale reward (or probability) from whale_reward_series if the plane is flying.
          - A failure penalty if the plane transitions to the broken state.
          
        Parameters:
            states: np.ndarray of shape (n, state_dim).
            actions: np.ndarray of shape (n,).
            next_states: np.ndarray of shape (n, state_dim).
            t: The current time step index.
            
        Returns:
            rewards: np.ndarray of shape (n,).
        """
        pass

    def step(self, states: np.ndarray, actions: np.ndarray, t: int):
        """
        Execute one vectorized time step of the MDP.
        
        Parameters:
            states: np.ndarray of shape (n, state_dim).
            actions: np.ndarray of shape (n,).
            t: The current time step index.
            
        Returns:
            next_states: np.ndarray of shape (n, state_dim).
            rewards: np.ndarray of shape (n,).
        """
        self._ensure_vectorized_input(states, "states")
        self._ensure_vectorized_input(actions, "actions")
        next_states = self.transition(states, actions, t)
        rewards = self.reward(states, actions, next_states, t)
        return next_states, rewards
    
class DeterministicMDP(AbstractMDP):
    def __init__(self, battery_capacity_wh, idle_power, cruise_power, takeoff_power,
                 solar_rate_series: np.ndarray, wind_series: np.ndarray, whale_reward_series: np.ndarray,
                 failure_penalty, delta_t, gamma, transition_model_name: str, soc_increment: float):
        super().__init__(battery_capacity_wh, idle_power, cruise_power, takeoff_power,
                         solar_rate_series, wind_series, whale_reward_series,
                         failure_penalty, delta_t, gamma, transition_model_name, soc_increment)
        # Instantiate the probability model using the factory.
        probability_model = ProbabilityModelFactory.select_probability_model(transition_model_name)
        # Instantiate the transition logic using the new abstract-based implementation.
        self.transition_logic = DeterministicTransitionLogic(
            battery_capacity_joules=self.battery_capacity_joules,
            soc_increment=self.soc_increment,
            idle_power=self.idle_power,
            cruise_power=self.cruise_power,
            takeoff_power=self.takeoff_power,
            delta_t=self.delta_t,
            solar_rate_series=self.solar_rate_series,
            wind_series=self.wind_speed_series,
            transition_model=probability_model
        )

    def transition(self, states: np.ndarray, actions: np.ndarray, t: int) -> np.ndarray:
        """
        Vectorized transition function at time step t.
        
        Given an array of current states and corresponding actions,
        compute and return an array of next states.
        
        Parameters:
            states: np.ndarray of shape (n, state_dim) where each row is a state (e.g., (SOC, mode)).
            actions: np.ndarray of shape (n,) representing the action for each state.
            t: The current time step index to use for time-dependent parameters.
            
        Returns:
            next_states: np.ndarray of shape (n, state_dim).
        """
        return self.transition_logic.transition(states, actions, t)


    def reward(self, states: np.ndarray, actions: np.ndarray, next_states: np.ndarray, t: int) -> np.ndarray:
        """
        Vectorized reward function at time step t.
        
        Computes the reward for each transition using:
          - The whale reward (or probability) from whale_reward_series if the plane is flying.
          - A failure penalty if the plane transitions to the broken state.
          
        Parameters:
            states: np.ndarray of shape (n, state_dim).
            actions: np.ndarray of shape (n,).
            next_states: np.ndarray of shape (n, state_dim).
            t: The current time step index.
            
        Returns:
            rewards: np.ndarray of shape (n,).
        """
        # Compute the reward components.
        whale_reward = np.where(actions == 1, self.whale_reward_series[t], 0.0)
        failure_penalty = np.where(next_states[:, 1] == 2, self.failure_penalty, 0.0)
        rewards = whale_reward - failure_penalty
        return rewards
import numpy as np
from abc import ABC, abstractmethod
from BaseClasses.transition_model_base import ActionSuccessProbabilityModel, ProbabilityModelFactory

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
    
    def soc_to_energy(self, soc: np.ndarray) -> np.ndarray:
        """
        Convert state of charge (SOC) to energy in Joules
        """
        # Convert percentage to a fraction (0 to 1) and multiply by the total energy capacity.
        return (soc / 100.0) * self.battery_capacity_joules
    
    def energy_to_soc(self, next_energy: np.ndarray) -> np.ndarray:
        """
        Convert energy in Joules to state of charge (SOC) percentage,
        flooring the result to the nearest soc_increment.

        Parameters:
            next_energy (np.ndarray): Array of energy values in Joules.
            
        Returns:
            np.ndarray: SOC values floored to the nearest soc_increment.
        """
        # Compute the raw SOC as a percentage.
        raw_soc = (next_energy / self.battery_capacity_joules) * 100.0

        # Floor to the nearest soc_increment
        floored_soc = np.floor(raw_soc / self.soc_increment) * self.soc_increment

        return floored_soc

    def min_to_seconds(self, minutes: float) -> float:
        """
        Convert minutes to seconds.
        """
        return minutes * 60.

    def sample_reward_component(self, states: np.ndarray, actions: np.ndarray, next_states: np.ndarray, t: int) -> np.ndarray:
        """
        Computes the reward components in a vectorized manner.
        
        For instance, if a plane is flying (mode == 1), it earns the whale reward from whale_reward_series for time step t.
        If a transition leads to a broken state (mode == 2), a failure penalty is applied.
        
        Parameters:
            states: np.ndarray of shape (n, state_dim).
            actions: np.ndarray of shape (n,).
            next_states: np.ndarray of shape (n, state_dim).
            t: The current time step index.
            
        Returns:
            A numpy array of shape (n,) with the computed reward components.
        """
        # For flying states (mode == 1), add the whale reward for time t.
        whale_component = np.where(actions[:, 1] == 1, self.whale_reward_series[t], 0.0)
        # If the plane is broken in the next state (mode == 2), subtract the failure penalty.
        failure_component = np.where(next_states[:, 1] == 2, self.failure_penalty, 0.0)
        return whale_component - failure_component

    def _vectorized_whale_spotted(self, states: np.ndarray, actions: np.ndarray, next_states: np.ndarray, t: int) -> np.ndarray:
        """
        Vectorized helper for determining if a whale is spotted.
        For now, this is deterministic and may simply return True for flying states.
        Override this method if you wish to introduce stochasticity (e.g., sampling from a probability distribution).
        
        Parameters:
            states: np.ndarray of shape (n, state_dim).
            actions: np.ndarray of shape (n,).
            next_states: np.ndarray of shape (n, state_dim).
            t: The current time step index.
            
        Returns:
            A boolean np.ndarray of shape (n,) indicating whether a whale was spotted.
        """
        # Deterministically assume a whale is spotted if the plane is flying.
        return np.full((states.shape[0],), True)
    

class DeterministicMDP(AbstractMDP):
    def __init__(self, battery_capacity_wh, idle_power, cruise_power, takeoff_power,
                solar_rate_series: np.ndarray, wind_series:np.ndarray, whale_reward_series: np.ndarray,
                failure_penalty, delta_t, gamma, transition_model_name: str, soc_increment: float):
        
        super().__init__(battery_capacity_wh, idle_power, cruise_power, takeoff_power,
                 solar_rate_series, wind_series, whale_reward_series,
                 failure_penalty, delta_t, gamma, transition_model_name, soc_increment)

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
        # Compute the energy consumption for each state-action pair.
        moored_float_energy = self.idle_power * self.min_to_seconds(self.delta_t)
        takeoff_energy = (self.cruise_power + self.takeoff_power) * self.min_to_seconds(self.delta_t)
        land_energy = self.cruise_power * self.min_to_seconds(self.delta_t)/4
        continue_flight_energy = self.cruise_power * self.min_to_seconds(self.delta_t)
        
        # Need to consider case where mode in state is 2
        energy_lookup = np.array([
            [moored_float_energy, takeoff_energy],
            [land_energy, continue_flight_energy],
            [0, 0]  # Dummy values for mode 2; these won't affect final result.
        ])
        
        energy_consumption = energy_lookup[states[:,1].astype(int),actions]
        
        # Compute the energy gain for each state.
        energy_gain = self.sample_sunlight(t, states.shape[0])

        # Compute the current energy from state of charge (SOC).
        current_energy = self.soc_to_energy(states[:,0])

        # Compute the next energy and state of charge based on energy conservation.
        next_energy = current_energy + energy_gain - energy_consumption
        
        # Compute the next states if there is no failure.
        next_soc = self.energy_to_soc(next_energy)
        next_mode = np.where(next_soc <= 0, 2, np.where(actions == 0, 0, 1))
        next_state = np.column_stack((next_soc, next_mode))


        wind_speeds = self.sample_wind_speed(t, states.shape[0])
        # Compute the probability of transitioning to the next state successfully for each state.
        success_probabilities = self.transition_model.compute_probability(wind_speeds,actions, states)

        # Determine the next state based on the success probabilities.
        false_states = np.tile(np.array([-1., 2]), (states.shape[0], 1))
        next_states = np.where(np.random.rand(states.shape[0])[:, np.newaxis] < success_probabilities[:, np.newaxis],
                            next_state,
                            false_states)
        # Ensure all failed states have soc of -1
        mode2_mask = next_states[:, 1] == 2
        next_states[mode2_mask, 0] = -1.
        
        return next_states


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
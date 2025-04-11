from abc import ABC, abstractmethod
import numpy as np
from BaseClasses.environment_provider_base import AbstractEnvironmentProvider
from tqdm import tqdm


####################################################################################
# Abstract Simulation Classes
####################################################################################

class AbstractSimulation(ABC):
    """
    Abstract base class for simulating decision-making policies in an MDP.

    The simulation loop now relies on an environment provider to supply environmental
    data (solar, wind, whale observation) rather than passing these arrays explicitly.
    """
    # TODO: Add way to save simulation environment data when save_history is set to True.
    def __init__(self, mdp, horizon: int, initial_state: np.ndarray, env_provider: AbstractEnvironmentProvider = None, save_history = False):
        """
        Parameters:
            mdp: An instance of a class that implements the MDP.
            horizon: Total number of simulation time steps.
            initial_state: The starting state.
            env_provider: Provides environmental samples. If not provided, the simulation
                          will attempt to use the MDP’s own provider.
        """
        self.mdp = mdp
        self.horizon = horizon
        self.initial_state = initial_state
        self.save_history = save_history
        if env_provider is None and hasattr(mdp, 'env_provider'):
            self.env_provider = mdp.env_provider
        else:
            self.env_provider = env_provider

    @abstractmethod
    def choose_action(self, **kwargs) -> int:
        """
        Select an action given the current state and time step.
        """
        pass

    def simulate_episode(self):
        """
        Simulate a single episode using the policy and environment provider,
        using NumPy arrays for efficiency.
        """
        # Convert the initial state to a NumPy array and determine its shape.
        state = np.asarray(self.initial_state)
        state_shape = state.shape

        # Preallocate arrays.
        trajectory = np.empty((self.horizon + 1, *state_shape), dtype=state.dtype)
        trajectory[0] = state
        actions = np.empty(self.horizon)
        rewards = np.empty(self.horizon)
        solar_samples = np.empty(self.horizon)
        wind_samples = np.empty(self.horizon)
        whale_samples = np.empty(self.horizon)

        for t in range(self.horizon):
            # Sample environmental data
            solar_sample = self.env_provider.sample_sunlight(t, 1)[0]
            wind_sample = self.env_provider.sample_wind_speed(t, 1)[0]
            whale_observation = self.env_provider.sample_whale_observation(t, 1)[0]

            # Choose action using the current state and environmental samples
            action = self.choose_action(
                state=state,
                solar_sample_j=solar_sample,
                wind_sample_ms=wind_sample,
                whale_observation=whale_observation,
                t=t
            )
            actions[t] = action

            # Take a step in the environment
            next_state, reward = self.step(state[np.newaxis,:], action, solar_sample, wind_sample, whale_observation, t)
            state = next_state[0]
            trajectory[t + 1] = state
            rewards[t] = reward[0]
            solar_samples[t] = solar_sample
            wind_samples[t] = wind_sample
            whale_samples[t] = whale_observation

            # Check for failure condition.
            if state[1] == 2:
                # print("Failure at time step", t)
                # print("State:", state)
                # print("Trajectory:", trajectory[t])  # prints the state before the failure state
                # Return only the slices corresponding to completed steps.
                return (trajectory[:t + 2],
                        actions[:t + 1],
                        rewards[:t + 1],
                        solar_samples[:t + 1],
                        wind_samples[:t + 1],
                        whale_samples[:t + 1])

        # If no failure, return the full arrays.
        return trajectory, actions, rewards, solar_samples, wind_samples, whale_samples

    
    def step(self,state,action,solar_sample,wind_sample,whale_observation,t):
        """
        Progress the simulation by one time step based on the vehicle state at the beginning of the time step,
        the selected action, and the sampled environmental data.

        Parameters:
            state (np.ndarray): Current state of the vehicle.
            action (int): Action taken by the vehicle.
            solar_sample (float): Sampled collected solar energy.
            wind_sample (float): Sampled wind speed in meters per second.
            whale_observation (float): Sampled whale observation probability.

        Returns:
            next_state (np.ndarray): The next state of the vehicle after the transition is applied.
            reward (float): The reward received after the transition is executed.
        """

        next_state = self.mdp.transition_logic.transition_with_wind_and_energy(state, action, solar_sample, wind_sample)
        reward = self.mdp.reward(state[np.newaxis,:], np.full((1,),action), next_state, t)
        return next_state, reward

    def simulate_multiple_episodes(self, num_episodes: int):
        """
        Generate simulation episodes one-by-one.
        Each episode is yielded as a dictionary containing its trajectory, actions, rewards,
        and a metadata dictionary with the episode index.
        """
        for episode_index in tqdm(range(num_episodes)):
            self.env_provider.reset(episode_index)
            traj, acts, rews, solar, wind, whale = self.simulate_episode()
            if self.save_history:
                episode_data = {
                    'trajectory': traj,
                    'actions': acts,
                    'rewards': rews,
                    'solar_series': solar,
                    'wind_series': wind,
                    'whale_series': whale,
                    'metadata': {'episode_index': episode_index},
                    'total_reward': sum(rews),
                }
            else:
                episode_data = {
                    'metadata': {'episode_index': episode_index},
                    'total_reward': sum(rews),
                }
            yield episode_data

class AbstractContinuousEnergySimulation(ABC):
    """
    Abstract base class for simulating decision-making policies in an MDP.

    The simulation loop now relies on an environment provider to supply environmental
    data (solar, wind, whale observation) rather than passing these arrays explicitly.
    """
    
    def __init__(self, mdp, horizon: int, initial_state: np.ndarray, env_provider: AbstractEnvironmentProvider = None, save_history = True):
        """
        Parameters:
            mdp: An instance of a class that implements the MDP.
            horizon: Total number of simulation time steps.
            initial_state: The starting state.
            env_provider: Provides environmental samples. If not provided, the simulation
                          will attempt to use the MDP’s own provider.
        """
        self.mdp = mdp
        self.horizon = horizon
        self.initial_state = initial_state
        self.save_history = save_history
        if env_provider is None and hasattr(mdp, 'env_provider'):
            self.env_provider = mdp.env_provider
        else:
            self.env_provider = env_provider

    @abstractmethod
    def choose_action(self, **kwargs) -> int:
        """
        Select an action given the current state and time step.
        """

    def simulate_episode(self):
        """
        Simulate a single episode using the policy and environment provider.
        """
        state = self.initial_state
        energy = self.mdp.transition_logic.soc_to_energy(state[0])
        energies = [energy]
        trajectory = [state]
        actions = []
        rewards = []
        solar_samples = []
        wind_samples = []
        whale_samples = []
        for t in range(self.horizon):
            # Sample environmental data
            solar_sample = self.env_provider.sample_sunlight(t, 1)[0]
            wind_sample = self.env_provider.sample_wind_speed(t, 1)[0]
            whale_observation = self.env_provider.sample_whale_observation(t, 1)[0]
            action = self.choose_action(state=state,
                                        solar_sample_w=solar_sample,
                                        wind_sample_ms=wind_sample,
                                        whale_observation=whale_observation,
                                        t=t)
            actions.append(action)
            next_state, reward, next_energy = self.step(energy, state[np.newaxis, :], np.full((1,),action), solar_sample, wind_sample,t)
            state = next_state[0]
            energy = next_energy
            trajectory.append(state)
            rewards.append(reward[0])
            energies.append(energy)
            solar_samples.append(solar_sample)
            wind_samples.append(wind_sample)
            whale_samples.append(whale_observation)
            if state[1] == 2:
                break
        return trajectory, actions, rewards, solar_samples, wind_samples, whale_samples, energies
    
    def step(self, energy, states, actions, energy_gain, wind_sample,t):
        # The samples for solar, wind, and whale need to be exposed in this function. Right now these are sampled in the transition function, but it needs to be in the simulation loop.
        next_states, next_energy = self.mdp.transition_logic.transition_continuous_energy_with_wind_and_energy(energy, states, actions, wind_sample, energy_gain)
        rewards = self.mdp.reward(states, actions, next_states, t)
        return next_states, rewards, next_energy


    def simulate_multiple_episodes(self, num_episodes: int):
        """
        Generate simulation episodes one-by-one.
        Each episode is yielded as a dictionary containing its trajectory, actions, rewards,
        and a metadata dictionary with the episode index.
        """
        for episode_index in tqdm(range(num_episodes)):
            self.env_provider.reset(episode_index)
            traj, acts, rews, solar, wind, whale, energies = self.simulate_episode()
            if self.save_history:
                episode_data = {
                    'trajectory': traj,
                    'actions': acts,
                    'rewards': rews,
                    'solar_series': solar,
                    'wind_series': wind,
                    'whale_series': whale,
                    'energy_series': energies,
                    'metadata': {'episode_index': episode_index},
                    'total_reward': sum(rews),
                }
                # print(episode_data['energy_series'])
            else:
                episode_data = {
                    'metadata': {'episode_index': episode_index},
                    'total_reward': sum(rews),
                }
            yield episode_data

####################################################################################
# Simple Simulation Classes
####################################################################################

class AlwaysFlySimulation(AbstractSimulation):
    def choose_action(self, **kwargs) -> int:
        return 1

class AlwaysFloatSimulation(AbstractSimulation):
    def choose_action(self, **kwargs) -> int:
        return 0
    

####################################################################################
# State based transition simulation classes
####################################################################################

class ObservationThresholdSimulation(AbstractSimulation):
    def __init__(self, mdp, horizon: int, initial_state: np.ndarray, observation_threshold: float, wind_threshold: float, env_provider=None):
        super().__init__(mdp, horizon, initial_state, env_provider)
        self.observation_threshold = observation_threshold
        self.wind_threshold = wind_threshold
        self.low_battery_threshold = 30.

    def choose_action(self, state, solar_sample_w, wind_sample_ms, whale_observation, t) -> int:
        action = 0
        is_wind_acceptable = wind_sample_ms < self.wind_threshold
        is_observation_sufficient = whale_observation > self.observation_threshold
        is_battery_sufficient = state[0] > self.low_battery_threshold
        if is_wind_acceptable and is_observation_sufficient and is_battery_sufficient:
            action = 1
        return action

class DeterministicOptimalSimulation(AbstractSimulation):
    def __init__(self, mdp_solver, horizon: int, initial_state: np.ndarray, env_provider=None):
        super().__init__(mdp_solver.mdp, horizon, initial_state, env_provider)
        mdp_solver.solve()
        self.mdp_solver = mdp_solver

    def choose_action(self, state, solar_sample_w, wind_sample_ms, whale_observation, t) -> int:
        value_list = [-10000, -10000]
        for action in [0, 1]:
            next_state, reward = self.mdp.step(np.array([state]), np.array([action]), t)
            value = self.mdp_solver.value_function(t, reward, next_state)
            value_list[action] = value
        return int(np.argmax(value_list))

class OptimalPolicySimulation(AbstractSimulation):
    """
    Simulation class that selects the optimal action by evaluating the Bellman value 
    for each action (0 or 1) using a backward induction solver.
    
    This class fits into the same framework as AlwaysFlySimulation and AlwaysFloatSimulation.
    """

    def __init__(self, mdp_solver, horizon: int, initial_state: np.ndarray, env_provider=None):
        """
        Parameters:
            mdp_solver: A backward induction solver that has computed the value function.
                        It must have attributes `mdp` and a method `value_function(t, reward, next_state)`.
            horizon (int): Total number of simulation time steps.
            initial_state (np.ndarray): Starting state (e.g. [SoC, mode]).
            env_provider: (Optional) An environment provider to sample solar, wind, and whale data.
        """
        # Initialize the simulation using the MDP from the solver.
        super().__init__(mdp_solver.mdp, horizon, initial_state, env_provider)
        # Pre-solve the MDP (compute the value function using backward induction).
        mdp_solver.solve()
        self.mdp_solver = mdp_solver

    def choose_action(self, state, solar_sample_j, wind_sample_ms, whale_observation, t) -> int:
        """
        For the current state and time t, simulate both possible actions (0 and 1)
        using the MDP's step function, then evaluate the Bellman value (reward + γ * future value)
        via the solver's value_function. The action with the highest value is returned.
        """
        # TODO: Ensure that this uses the samples provided to the method.
        value_list = [-np.inf, -np.inf]
        N=10000
        states = np.full((N,2),state)
        solar_samples = np.full((N,),solar_sample_j)
        wind_samples = np.full((N,),wind_sample_ms)
        
        for action in [0, 1]:
            # Roll forward one time step with the candidate action.
            actions = np.full((N,),action)
            next_states = self.mdp.transition_logic.transition_with_wind_and_energy(
                states,
                actions,
                solar_samples,
                wind_samples,
                )
            
            rewards = self.mdp.reward(states, actions, next_states, t)
            # Compute the value using the backward induction solver's value function.
            value = self.mdp_solver.value_function(t, rewards, next_states)
            value_list[action] = value
        # Return the action that yields the highest value.
        return int(np.argmax(value_list))
    
class OptimalAnalyticalPolicySimulation(AbstractSimulation):
    """
    Simulation class that selects the optimal action by evaluating the Bellman value 
    for each action (0 or 1) using a backward induction solver.
    """

    def __init__(self, mdp_solver, horizon: int, initial_state: np.ndarray, env_provider=None):
        # Initialize the simulation using the MDP from the solver.
        super().__init__(mdp_solver.mdp, horizon, initial_state, env_provider)
        # Pre-solve the MDP (compute the value function using backward induction).
        mdp_solver.solve()
        self.mdp_solver = mdp_solver

    def choose_action(self, state, solar_sample_j, wind_sample_ms, whale_observation, t) -> int:
        """
        For the current state and time t, for each candidate action (0 or 1) we compute
        the success probability and then use a two-outcome integration:
        - With probability p_success, the transition is successful.
        - With probability (1 - p_success), the transition fails (yielding a failure state).
        The expected value is the weighted average of these two outcomes.
        """

        # List to hold the integrated value for each candidate action.
        value_list = [None, None]
        # Convert state to a numpy array (if not already) and compute current energy.
        current_energy = self.mdp.transition_logic.soc_to_energy(np.array([state[0]]))[0]
        # Define the failure state (typically a state indicating a crash/failure).
        failure_state = np.array([[-1.0, 2]])

        for action in [0, 1]:
            # Compute the success probability for this action.
            # (Assuming the transition model returns an array; take the first element.)
            p_success = self.mdp.transition_logic.transition_model.compute_probability(
                np.array([wind_sample_ms]),
                np.array([action]),
                np.array([state])
            )[0]

            # --- Compute the "successful" outcome ---
            # Prepare inputs as one-sample arrays.
            states_arr = np.array([state])
            actions_arr = np.array([action])
            # Use the energy gain (solar_sample_w) as given.
            energy_gain = np.array([solar_sample_j])
            # Calculate energy consumption for the given state and action.
            energy_consumption = self.mdp.transition_logic._calculate_energy_consumption(states_arr, actions_arr)
            # Determine the next state (and energy) if the transition succeeds.
            next_state_success = self.mdp.transition_logic._update_energy_and_state(
                states_arr,
                energy_gain,
                energy_consumption,
                actions_arr
            )
            # Compute rewards for the successful outcome.
            reward_success = self.mdp.reward(states_arr, actions_arr, next_state_success, t)[0]
            # Evaluate the value function for the successful outcome.
            value_success = self.mdp_solver.value_function(t, np.array([reward_success]), next_state_success)

            # --- Compute the "failure" outcome ---
            # Here, we define the failure state's reward and value.
            reward_failure = self.mdp.reward(states_arr, actions_arr, failure_state, t)[0]

            # --- Combine outcomes according to the success probability ---
            expected_value = p_success * value_success + (1.0 - p_success) * reward_failure
            value_list[action] = expected_value

        # Return the action with the highest expected value.
        return int(np.argmax(value_list))

####################################################################################
# Continuous Energy Simulation Classes
####################################################################################
class OptimalContinuousAnalyticalPolicySimulation(AbstractContinuousEnergySimulation):
    """
    Simulation class that selects the optimal action by evaluating the Bellman value 
    for each action (0 or 1) using a backward induction solver.
    """

    def __init__(self, mdp_solver, horizon: int, initial_state: np.ndarray, env_provider=None):
        # Initialize the simulation using the MDP from the solver.
        super().__init__(mdp_solver.mdp, horizon, initial_state, env_provider)
        # Pre-solve the MDP (compute the value function using backward induction).
        # mdp_solver.solve()
        self.mdp_solver = mdp_solver

    def choose_action(self, state, solar_sample_w, wind_sample_ms, whale_observation, t) -> int:
        """
        For the current state and time t, for each candidate action (0 or 1) we compute
        the success probability and then use a two-outcome integration:
        - With probability p_success, the transition is successful.
        - With probability (1 - p_success), the transition fails (yielding a failure state).
        The expected value is the weighted average of these two outcomes.
        """

        # List to hold the integrated value for each candidate action.
        value_list = [None, None]
        # Convert state to a numpy array (if not already) and compute current energy.
        current_energy = self.mdp.transition_logic.soc_to_energy(np.array([state[0]]))[0]
        # Define the failure state (typically a state indicating a crash/failure).
        failure_state = np.array([[-1.0, 2]])

        for action in [0, 1]:
            # Compute the success probability for this action.
            # (Assuming the transition model returns an array; take the first element.)
            p_success = self.mdp.transition_logic.transition_model.compute_probability(
                np.array([wind_sample_ms]),
                np.array([action]),
                np.array([state])
            )[0]

            # --- Compute the "successful" outcome ---
            # Prepare inputs as one-sample arrays.
            states_arr = np.array([state])
            actions_arr = np.array([action])
            # Use the energy gain (solar_sample_w) as given.
            energy_gain = np.array([solar_sample_w])
            # Calculate energy consumption for the given state and action.
            energy_consumption = self.mdp.transition_logic._calculate_energy_consumption(states_arr, actions_arr)
            # Determine the next state (and energy) if the transition succeeds.
            next_state_success, _ = self.mdp.transition_logic._update_energy_and_state_continuous(
                current_energy,
                energy_gain,
                energy_consumption,
                actions_arr
            )
            # Compute rewards for the successful outcome.
            reward_success = self.mdp.reward(states_arr, actions_arr, next_state_success, t)[0]
            # Evaluate the value function for the successful outcome.
            value_success = self.mdp_solver.value_function(t, np.array([reward_success]), next_state_success)

            # --- Compute the "failure" outcome ---
            # Here, we define the failure state's reward and value.
            reward_failure = self.mdp.reward(states_arr, actions_arr, failure_state, t)[0]

            # --- Combine outcomes according to the success probability ---
            expected_value = p_success * value_success + (1.0 - p_success) * reward_failure
            value_list[action] = expected_value

        # Return the action with the highest expected value.
        return int(np.argmax(value_list))

class ObservationThresholdContinuousSimulation(AbstractContinuousEnergySimulation):
    def __init__(self, mdp, horizon: int, initial_state: np.ndarray, observation_threshold: float, wind_threshold: float, env_provider=None):
        super().__init__(mdp, horizon, initial_state, env_provider)
        self.observation_threshold = observation_threshold
        self.wind_threshold = wind_threshold
        self.low_battery_threshold = 30.

    def choose_action(self, state, solar_sample_w, wind_sample_ms, whale_observation, t) -> int:
        action = 0
        is_wind_acceptable = wind_sample_ms < self.wind_threshold
        is_observation_sufficient = whale_observation > self.observation_threshold
        is_battery_sufficient = state[0] > self.low_battery_threshold
        if is_wind_acceptable and is_observation_sufficient and is_battery_sufficient:
            action = 1
        return action
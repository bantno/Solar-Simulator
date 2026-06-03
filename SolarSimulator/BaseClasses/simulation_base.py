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
        Yield episode data; full history for first full_history_episodes,
        then only summary for the remainder.
        """
        for episode_index in tqdm(range(num_episodes)):
            self.env_provider.reset(episode_index)
            traj, acts, rews, solar, wind, whale = self.simulate_episode()

            # Determine if this episode should save full history
            if self.save_history or (
               self.full_history_episodes is not None 
               and episode_index < self.full_history_episodes):
                episode_data = {
                    'trajectory': traj,
                    'actions': acts,
                    'rewards': rews,
                    'solar_series': solar,
                    'wind_series': wind,
                    'whale_series': whale,
                    'metadata': {'episode_index': episode_index},
                    'total_reward': float(sum(rews)),
                    'flight_hrs': float(sum(acts)/4.), # TODO: Make this not hardcoded for 15min time step
                }
            else:
                # summary only
                episode_data = {
                    'metadata': {'episode_index': episode_index},
                    'failure': bool(traj[-1][1] == 2),
                    'failure_step': len(traj) - 1 if traj[-1][1] == 2 else self.horizon,
                    'total_reward': float(sum(rews)),
                    'flight_hrs': float(sum(acts)/4.), # TODO: Make this not hardcoded for 15min time step

                }

            yield episode_data

class AbstractContinuousEnergySimulation(ABC):
    """
    Abstract base class for simulating decision-making policies in an MDP.

    The simulation loop now relies on an environment provider to supply environmental
    data (solar, wind, whale observation) rather than passing these arrays explicitly.
    """
    
    def __init__(self,
                mdp,
                horizon: int,
                initial_state: np.ndarray,
                start_datetime,
                env_provider: AbstractEnvironmentProvider = None,
                save_history=False,
                full_history_episodes=None):
        self.mdp = mdp
        self.horizon = horizon
        self.initial_state = initial_state
        self.start_datetime = start_datetime
        self.env_provider = env_provider or mdp.env_provider
        self.save_history = save_history
        self.full_history_episodes = full_history_episodes

    @abstractmethod
    def choose_action(self, **kwargs) -> int:
        """
        Select an action given the current state and time step.
        """
        raise NotImplementedError("Subclasses must implement this method.")

    def simulate_episode(self):
        """
        Simulate a single episode using the policy and environment provider.
        Pre-allocates fixed-size arrays and truncates them at the first failure.
        """
        max_steps = self.horizon
        # assume state is a 1D array of length S
        state = self.initial_state
        S = state.shape[0]
        
        # pre-allocate
        trajectory      = np.zeros((max_steps+1, S))
        energies        = np.zeros(max_steps+1)
        actions         = np.zeros(max_steps, dtype=int)
        rewards         = np.zeros(max_steps)
        solar_samples   = np.zeros(max_steps)
        wind_samples    = np.zeros(max_steps)
        whale_samples   = np.zeros(max_steps)
        
        # initialize at t=0
        energy = self.mdp.transition_logic.soc_to_energy(state[0])
        trajectory[0] = state
        energies[0]   = energy
        rewards[0]    = 0.0
        actions[0]    = 0
        solar_samples[0] = 0.0
        wind_samples[0]  = 0.0
        whale_samples[0] = 0.0
        
        # run
        for t in range(max_steps):
            # sample environment
            solar = self.env_provider.sample_sunlight(t, 1)[0]
            wind  = self.env_provider.sample_wind_speed(t, 1)[0]
            whale = self.env_provider.sample_whale_observation(t, 1)[0]
            
            # choose and record action
            a = self.choose_action(
                state=state,
                solar_sample_w=solar,
                wind_sample_ms=wind,
                whale_observation=whale,
                t=t
            )
            actions[t]       = a
            solar_samples[t] = solar
            wind_samples[t]  = wind
            whale_samples[t] = whale
            
            # step
            (next_state_arr, reward_arr, next_energy_arr) = self.step(
                energy,
                state[np.newaxis, :],
                np.full((1,), a),
                solar, wind, t
            )
            state  = next_state_arr[0]
            reward = reward_arr[0]
            energy = next_energy_arr
            
            # record
            trajectory[t+1] = state
            rewards[t]      = reward
            energies[t+1]   = energy
            
            # check for failure (mode==2)
            if state[1] == 2:
                last_idx = t + 1
                if next_energy_arr < 0:
                    failure_type = 2 # Battery Depletion Failure
                else:
                    failure_type = 1 # Crash Failure
                break
        else:
            # no failure in full horizon
            last_idx = max_steps
            failure_type = 0 # No failure
        
        # slice out only the filled portion
        return (
            trajectory[:last_idx+1],     # states from t=0 to last_idx
            actions[:last_idx],          # actions t=0 … last_idx-1
            rewards[:last_idx],          # rewards t=0 … last_idx-1
            solar_samples[:last_idx],
            wind_samples[:last_idx],
            whale_samples[:last_idx],
            energies[:last_idx+1],       # energies from t=0 to last_idx
            failure_type,
        )
    
    def step(self, energy, states, actions, energy_gain, wind_sample,t):
        next_states, next_energy = self.mdp.transition_logic.transition_continuous_energy_with_wind_and_energy(energy, states, actions, wind_sample, energy_gain)
        rewards = self.mdp.reward(states, actions, next_states, t)
        return next_states, rewards, next_energy

    def choose_action_batch(self, state, solar, wind, whale, t, cur_bins=None) -> np.ndarray:
        """
        Vectorized policy: choose an action for every episode (row of ``state``) at once.
        Subclasses must override. ``state`` is (n,2); solar/wind/whale are (n,).
        ``cur_bins`` is the current wind-bin per lane (only used by the optimal policy when
        the wind Markov chain is active; ignored otherwise).
        Returns an (n,) int array of actions.
        """
        raise NotImplementedError("Subclasses must implement choose_action_batch.")

    def step_batch(self, energy, states, actions, solar, wind, t):
        """Batched single-step transition + reward for n episodes at once."""
        next_states, next_energy = self.mdp.transition_logic.transition_continuous_energy_with_wind_and_energy(
            energy, states, actions, wind, solar
        )
        rewards = self.mdp.reward(states, actions, next_states, t)
        return next_states, rewards, next_energy

    def simulate_episode_batch(self, n: int):
        """
        Simulate n episodes simultaneously as vectorized (n,...) batches.

        Episodes are independent Monte Carlo rollouts; running them together replaces
        the per-episode Python loop with one vectorized pass per time step. Failed
        episodes (mode==2) are frozen via a done mask so later steps don't mutate
        them. Full per-step history is retained only for the first K episodes
        (K = n if save_history else full_history_episodes), keeping memory bounded.

        Returns a dict of per-episode results (see ''simulate_multiple_episodes'').
        """
        max_steps = self.horizon
        S = self.initial_state.shape[0]

        state = np.tile(np.asarray(self.initial_state, dtype=float), (n, 1))   # (n, S)
        energy = np.full(n, self.mdp.transition_logic.soc_to_energy(self.initial_state[0]))

        done = np.zeros(n, dtype=bool)
        failure_type = np.zeros(n, dtype=int)
        failure_step = np.full(n, max_steps, dtype=int)
        total_reward = np.zeros(n)
        action_sum = np.zeros(n)

        # Full-history columns to retain
        if self.save_history:
            K = n
        else:
            K = min(self.full_history_episodes or 0, n)
        if K > 0:
            traj_hist   = np.zeros((max_steps + 1, K, S))
            act_hist    = np.zeros((max_steps, K), dtype=int)
            rew_hist    = np.zeros((max_steps, K))
            solar_hist  = np.zeros((max_steps, K))
            wind_hist   = np.zeros((max_steps, K))
            whale_hist  = np.zeros((max_steps, K))
            energy_hist = np.zeros((max_steps + 1, K))
            traj_hist[0]   = state[:K]
            energy_hist[0] = energy[:K]

        for t in range(max_steps):
            active_idx = np.nonzero(~done)[0]
            if active_idx.size == 0:
                break

            # Sample the full population so episode i always sees draw i at step t,
            # independent of which other lanes have already failed; then operate only
            # on the still-active lanes (the transition model rejects broken states).
            solar = self.env_provider.sample_sunlight(t, n)[active_idx]
            wind  = self.env_provider.sample_wind_speed(t, n)[active_idx]
            whale = self.env_provider.sample_whale_observation(t, n)[active_idx]

            # Current wind bin per active lane (only when the Markov chain is active).
            cur_bins = None
            if getattr(self.env_provider, "use_wind_chain", False):
                cur_bins = self.env_provider.last_wind_bins[active_idx]

            s = state[active_idx]
            e = energy[active_idx]
            a = self.choose_action_batch(s, solar, wind, whale, t, cur_bins=cur_bins).astype(int)

            next_state, reward, next_energy = self.step_batch(e, s, a, solar, wind, t)

            total_reward[active_idx] += reward
            action_sum[active_idx]   += a

            newly_local = next_state[:, 1] == 2
            newly_global = active_idx[newly_local]
            failure_step[newly_global] = t + 1
            failure_type[newly_global] = np.where(next_energy[newly_local] < 0, 2, 1)

            if K > 0:
                in_hist = np.nonzero(active_idx < K)[0]
                if in_hist.size:
                    cols = active_idx[in_hist]
                    act_hist[t, cols]        = a[in_hist]
                    rew_hist[t, cols]        = reward[in_hist]
                    solar_hist[t, cols]      = solar[in_hist]
                    wind_hist[t, cols]       = wind[in_hist]
                    whale_hist[t, cols]      = whale[in_hist]
                    traj_hist[t + 1, cols]   = next_state[in_hist]
                    energy_hist[t + 1, cols] = next_energy[in_hist]

            state[active_idx]  = next_state
            energy[active_idx] = next_energy
            done[newly_global] = True

        results = {
            'n': n,
            'done': done,
            'failure_type': failure_type,
            'failure_step': failure_step,
            'total_reward': total_reward,
            'action_sum': action_sum,
            'K': K,
        }
        if K > 0:
            results.update({
                'traj_hist': traj_hist, 'act_hist': act_hist, 'rew_hist': rew_hist,
                'solar_hist': solar_hist, 'wind_hist': wind_hist, 'whale_hist': whale_hist,
                'energy_hist': energy_hist,
            })
        return results

    def simulate_multiple_episodes(self, num_episodes: int):
        """
        Yield episode data; full history for first full_history_episodes,
        then only summary for the remainder.

        All episodes are simulated in a single vectorized batch (see
        ``simulate_episode_batch``). The environment RNG is seeded once for the whole
        batch, so individual episodes are not re-seeded per index as before; results
        remain statistically equivalent i.i.d. Monte Carlo rollouts.
        """
        self.env_provider.reset(0)
        res = self.simulate_episode_batch(num_episodes)
        K = res['K']

        for episode_index in range(num_episodes):
            failed = bool(res['done'][episode_index])
            last_idx = int(res['failure_step'][episode_index])
            total_reward = float(res['total_reward'][episode_index])
            flight_hrs = float(res['action_sum'][episode_index]) / 4  # TODO: hardcoded 15min step
            failure_type = int(res['failure_type'][episode_index])

            if K > 0 and episode_index < K:
                j = episode_index
                episode_data = {
                    'trajectory':   res['traj_hist'][:last_idx + 1, j],
                    'actions':      res['act_hist'][:last_idx, j],
                    'rewards':      res['rew_hist'][:last_idx, j],
                    'solar_series': res['solar_hist'][:last_idx, j],
                    'wind_series':  res['wind_hist'][:last_idx, j],
                    'whale_series': res['whale_hist'][:last_idx, j],
                    'energy_series': res['energy_hist'][:last_idx + 1, j],
                    'metadata': {'episode_index': episode_index},
                    'total_reward': total_reward,
                    'flight_hrs': flight_hrs,
                    'failure': failed,
                    'failure_step': last_idx,
                    'failure_type': failure_type,
                }
            else:
                episode_data = {
                    'metadata': {'episode_index': episode_index},
                    'failure': failed,
                    'failure_step': last_idx,
                    'total_reward': total_reward,
                    'flight_hrs': flight_hrs,
                    'failure_type': failure_type,
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
# Continuous Energy Simulation Classes
####################################################################################
class OptimalContinuousAnalyticalPolicySimulation(AbstractContinuousEnergySimulation):
    """
    Simulation class that selects the optimal action by evaluating the Bellman value 
    for each action (0 or 1) using a backward induction solver.
    """

    def __init__(self, mdp_solver, horizon: int, initial_state: np.ndarray,
                 start_datetime, env_provider=None, save_history=False,full_history_episodes=None):
        # Pre-solve MDP
        mdp_solver.solve()
        super().__init__(
            mdp_solver.mdp,
            horizon,
            initial_state,
            start_datetime,
            env_provider,
            save_history=save_history,
            full_history_episodes=full_history_episodes,
        )
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

    def choose_action_batch(self, state, solar, wind, whale, t, cur_bins=None) -> np.ndarray:
        """
        Vectorized optimal policy: for every episode and both candidate actions,
        compute the two-outcome expected value and return the argmax action.

        This mirrors ``choose_action`` but evaluates all n episodes at once and uses
        the solver's O(n) ``value_function_batch`` instead of the per-call state scan.
        When the wind Markov chain is active, the future value is taken over the next
        wind bin via the stage transition matrix conditioned on ``cur_bins``.
        """
        n = state.shape[0]
        current_energy = self.mdp.transition_logic.soc_to_energy(state[:, 0])   # (n,)
        failure_state = np.tile(np.array([-1.0, 2]), (n, 1))                    # (n,2)
        values = np.empty((n, 2))

        # Stage transition matrix for the chain (None -> i.i.d. lookup).
        P = self.env_provider.get_wind_transition(t) if cur_bins is not None else None

        for action in (0, 1):
            actions_arr = np.full(n, action, dtype=int)
            p_success = self.mdp.transition_logic.transition_model.compute_probability(
                wind, actions_arr, state
            )                                                                  # (n,)
            energy_consumption = self.mdp.transition_logic._calculate_energy_consumption(
                state, actions_arr
            )
            next_state_success, _ = self.mdp.transition_logic._update_energy_and_state_continuous(
                current_energy, solar, energy_consumption, actions_arr
            )
            reward_success = self.mdp.reward(state, actions_arr, next_state_success, t)
            value_success = self.mdp_solver.value_function_batch(
                t, reward_success, next_state_success, cur_bins=cur_bins, P=P
            )
            reward_failure = self.mdp.reward(state, actions_arr, failure_state, t)
            values[:, action] = p_success * value_success + (1.0 - p_success) * reward_failure

        return np.argmax(values, axis=1).astype(int)

class UnifiedThresholdContinuousSimulation(AbstractContinuousEnergySimulation):
    def __init__(self, mdp, horizon: int, initial_state: np.ndarray,
                 observation_threshold: float, wind_threshold: float,
                 start_datetime, env_provider=None, save_history=False, full_history_episodes=None):
        super().__init__(
            mdp,
            horizon,
            initial_state,
            start_datetime,
            env_provider,
            save_history=save_history,
            full_history_episodes=full_history_episodes)
        
        self.observation_threshold = observation_threshold
        self.wind_threshold = wind_threshold
        self.low_battery_threshold = self.calculate_low_battery_threshold(mdp)

    def calculate_low_battery_threshold(self, mdp):
        """
        Calculate the low battery threshold based on the MDP's transition logic.
        This is a placeholder for a more complex calculation if needed.
        """
        timestep = 15*60 # 15 minutes in seconds TODO: make this not hardcoded
        cruise_power = mdp.cruise_power
        landing_power = mdp.landing_power
        total_reserve_energy = (cruise_power + landing_power) * timestep
        cutoff_soc = total_reserve_energy / mdp.transition_logic.battery_capacity_joules * 100  # Convert to percentage
        return cutoff_soc

    def choose_action(self, state, solar_sample_w, wind_sample_ms, whale_observation, t) -> int:
        
        # Behavior when current state is 0 (floating)
        # Choose to takeoff only if wind is acceptable, observation is sufficient, and battery is sufficient.
        if state[1] == 0:
            action = 0
            is_wind_acceptable = wind_sample_ms < self.wind_threshold
            is_observation_sufficient = whale_observation > self.observation_threshold
            is_battery_sufficient = state[0] > self.low_battery_threshold and state[0] > 95
            if is_wind_acceptable and is_observation_sufficient and is_battery_sufficient:
                action = 1

        # Behavior when current state is 1 (flying)
        # Choose to continue flying as long as possible. Choose to land if battery is low, if whale observation chance is low, or if wind is very low.
        elif state[1] == 1:
            is_batt_low      = state[0] < self.low_battery_threshold
            is_obs_low       = whale_observation < self.observation_threshold
            is_wind_landable    = wind_sample_ms <= self.wind_threshold-3
            if is_batt_low or is_obs_low or is_wind_landable:
                action = 0  # land
            else:
                action = 1  # continue flying

        else:
            raise ValueError("Invalid state: {}".format(state))

        return action

    def choose_action_batch(self, state, solar, wind, whale, t, cur_bins=None) -> np.ndarray:
        """Vectorized form of ``choose_action`` over n episodes (wind bin unused)."""
        soc = state[:, 0]
        mode = state[:, 1]
        action = np.zeros(state.shape[0], dtype=int)

        # mode 0 (floating): take off only if wind, observation, and battery allow
        m0 = mode == 0
        takeoff = (
            m0
            & (wind < self.wind_threshold)
            & (whale > self.observation_threshold)
            & (soc > self.low_battery_threshold)
            & (soc > 95)
        )
        action[takeoff] = 1

        # mode 1 (flying): keep flying unless battery low, observation low, or wind low
        m1 = mode == 1
        land = m1 & (
            (soc < self.low_battery_threshold)
            | (whale < self.observation_threshold)
            | (wind <= self.wind_threshold - 3)
        )
        action[m1 & ~land] = 1
        # mode 2 (broken) lanes stay 0; they are frozen by the done-mask anyway

        return action

class ObservationThresholdContinuousSimulation(AbstractContinuousEnergySimulation):
    def __init__(self, mdp, horizon: int, initial_state: np.ndarray,
                 observation_threshold: float, wind_threshold: float,
                 start_datetime, env_provider=None, save_history=False, full_history_episodes=None):
        super().__init__(
            mdp,
            horizon,
            initial_state,
            start_datetime,
            env_provider,
            save_history=save_history,
            full_history_episodes=full_history_episodes)
        
        self.observation_threshold = observation_threshold
        self.wind_threshold = wind_threshold
        self.low_battery_threshold = 15.

    def choose_action(self, state, solar_sample_w, wind_sample_ms, whale_observation, t) -> int:
        action = 0
        is_wind_acceptable = wind_sample_ms < self.wind_threshold
        is_observation_sufficient = whale_observation > self.observation_threshold
        is_battery_sufficient = state[0] > self.low_battery_threshold
        if is_wind_acceptable and is_observation_sufficient and is_battery_sufficient:
            action = 1
        return action

    def choose_action_batch(self, state, solar, wind, whale, t, cur_bins=None) -> np.ndarray:
        """Vectorized form of ``choose_action`` over n episodes (wind bin unused)."""
        soc = state[:, 0]
        action = np.zeros(state.shape[0], dtype=int)
        fly = (
            (wind < self.wind_threshold)
            & (whale > self.observation_threshold)
            & (soc > self.low_battery_threshold)
        )
        action[fly] = 1
        return action
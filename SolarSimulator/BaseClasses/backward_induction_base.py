import numpy as np
from typing import Callable, Union
from tqdm import tqdm
from BaseClasses.plotting_utils_base import PlottingUtils
from BaseClasses.mdp_base import AbstractMDP, stochasticMDP
from scipy.stats import beta, weibull_min
from scipy.special import betainc

from numpy.polynomial.legendre import leggauss


class mdpBackwardSolver:
    """
    Backward induction solver for a given MDP using Monte Carlo sampling.

    This class approximates the value function by sampling transitions
    for each state and both actions at each stage, then performs
    backward induction over a finite horizon. The resulting value table
    is plotted as an interactive surface.

    Attributes:
        mdp (AbstractMDP): The Markov Decision Process to solve.
        horizon (int): Number of time steps in the planning horizon.
        states (np.ndarray): Array of all possible states from the MDP.
        _GAMMA (float): Discount factor (currently fixed to 1.0).
        future_value_table (np.ndarray): Table of expected future values,
            shape (num_states, horizon).
    """

    def __init__(self, mdp: AbstractMDP, horizon: int):
        """
        Initialize the backward solver.

        Args:
            mdp (AbstractMDP): MDP instance providing `.step()` for transitions and rewards.
            horizon (int): Number of time steps over which to plan.
        """
        self.mdp = mdp
        self.horizon = horizon
        self.states = self.mdp._get_states()
        self._GAMMA = 1.0
        self.future_value_table = self._initialize_future_value_table()

    def _initialize_future_value_table(self) -> np.ndarray:
        """
        Create an empty future value table.

        Returns:
            np.ndarray: Zero-initialized array of shape (num_states, horizon).
        """
        num_states = self.states.shape[0]
        T = self.horizon
        return np.zeros((num_states, T))

    def solve(self) -> None:
        """
        Perform backward induction via Monte Carlo sampling.

        For each stage from horizon-1 down to 0, and for each non-terminal state:
        1. Sample a batch of next states and rewards for both actions.
        2. Estimate immediate rewards (at the last stage) or use
           `value_function` to include future values.
        3. Store the maximal value across the two actions.
        Finally, plot the completed future value table as a 3D surface.
        """
        NUM_STATE_SAMPLES = 5000
        for stage in tqdm(range(self.horizon - 1, -1, -1)):
            for i, state in enumerate(self.states[:-1, :]):
                # Prepare sample states and actions
                states = np.full((NUM_STATE_SAMPLES, 2), state)
                actions = np.zeros(NUM_STATE_SAMPLES, dtype=int)
                actions[NUM_STATE_SAMPLES // 2:] = 1

                # Step MDP to get next states and rewards
                next_states, rewards = self.mdp.step(states, actions, stage)

                # Split samples by action
                half = NUM_STATE_SAMPLES // 2
                float_rewards = rewards[:half]
                float_next = next_states[:half]
                fly_rewards = rewards[half:]
                fly_next = next_states[half:]

                # Compute value estimate
                if stage == self.horizon - 1:
                    value = max(float_rewards.mean(), fly_rewards.mean())
                else:
                    float_val = self.value_function(stage, float_rewards, float_next)
                    fly_val = self.value_function(stage, fly_rewards, fly_next)
                    value = max(float_val, fly_val)

                self.future_value_table[i, stage] = value
        filename = f"future_value_table_{self.mdp.battery_capacity_wh}Wh_{self.horizon}h_{self.mdp.failure_penalty}p.npy"
        np.save(filename, self.future_value_table)
        print("Value function table saved to:", filename)
        # PlottingUtils.plot_surface_plotly(self.future_value_table, self.mdp.battery_capacity_wh, filename)

    def value_function(self, stage: int, rewards: np.ndarray, next_states: np.ndarray) -> float:
        """
        Estimate the expected value for a batch of transitions at a given stage.

        Args:
            stage (int): Current time step.
            rewards (np.ndarray): 1D array of immediate rewards from the transitions.
            next_states (np.ndarray): 2D array of resulting states.

        Returns:
            float: The estimated expected return = E[reward + γ·V(next_state)].
        """
        # Group identical next_states
        unique_states, inv_idx, counts = np.unique(
            next_states, axis=0, return_inverse=True, return_counts=True
        )
        total = len(rewards)
        next_stage = stage + 1

        # Lookup future values or zero if beyond horizon
        if next_stage < self.horizon:
            future_vals = self.lookup_future_values(
                unique_states,
                np.full(unique_states.shape[0], next_stage, dtype=int)
            )
        else:
            future_vals = np.zeros_like(rewards)

        # Compute weighted average of (reward + γ·future_value)
        value = 0.0
        for idx, _ in enumerate(unique_states):
            mask = (inv_idx == idx)
            p = counts[idx] / total
            avg_r = rewards[mask].mean()
            value += p * (avg_r + self._GAMMA * future_vals[idx])

        return value

    def lookup_future_values(self, states: np.ndarray, stages: np.ndarray) -> np.ndarray:
        """
        Retrieve precomputed future values for given states and stages.
        Any state with mode==2 is treated as the broken state [-1.0, 2].

        Args:
            states (np.ndarray): Array of states to look up (shape: [n, state_dim]).
            stages (np.ndarray): Array of time indices corresponding to each state (shape: [n]).

        Returns:
            np.ndarray: Array of future values for each (state, stage) pair.

        Raises:
            ValueError: If any non-broken state isn’t in the solver’s state table.
        """
        # 1) Locate the broken‐state index
        is_broken_state = (self.states[:, 1] == 2) & (self.states[:, 0] == -1.0)
        try:
            broken_idx = np.nonzero(is_broken_state)[0][0]
        except IndexError:
            raise ValueError("Broken state [-1.0, 2] not found in state table.")

        n = states.shape[0]
        state_idxs = np.empty(n, dtype=int)

        # 2) Map all mode==2 inputs to the broken index
        broken_mask = (states[:, 1] == 2)
        state_idxs[broken_mask] = broken_idx

        # 3) Handle the remaining (mode 0 or 1) inputs
        normal_idxs = np.nonzero(~broken_mask)[0]
        if normal_idxs.size > 0:
            norm_states = states[normal_idxs]
            # exact match on soc and mode
            mask = np.all(self.states[None, :, :] == norm_states[:, None, :], axis=2)

            # check that every normal state was found
            if not np.all(mask.any(axis=1)):
                missing = normal_idxs[~mask.any(axis=1)]
                bad_states = states[missing]
                raise ValueError(f"States {bad_states.tolist()} at indices {missing.tolist()} not found.")

            # pick the first matching index for each
            matched_idxs = np.argmax(mask, axis=1)
            state_idxs[normal_idxs] = matched_idxs

        # 4) Finally, pull from the table
        return self.future_value_table[state_idxs, stages]


class mdpAnalyticalBackwardSolver:
    """
    Analytically solves a finite-horizon stochastic MDP via backward induction.

    Integrates over solar and wind distributions to compute expected action values
    and fills a future-value table, which can then be visualized.
    """

    def __init__(self, mdp, horizon: int):
        self.mdp = mdp
        self.horizon = horizon
        self.states = self.mdp._get_states()
        self._GAMMA = 1.0
        self.future_value_table = self._initialize_future_value_table()
        self.G_MAX = self.mdp.env_provider._energy_gain_from_solar(1.)
        soc_values = self.states[:-1, 0]  # exclude the 'broken' terminal state
        self._soc_grid = np.array(sorted(set(np.round(soc_values, 6))))
        self._num_modes = 2  # assume modes 0 and 1, mode==2 is broken

    def _initialize_future_value_table(self) -> np.ndarray:
        num_states = self.states.shape[0]
        return np.zeros((num_states, self.horizon))

    def _value(
        self,
        state: np.ndarray, # shape (n,2): [SoC, mode]
        a: np.ndarray,     # shape (n,): 0 or 1
        t: int,
    ):
        """
        Compute the value of a state-action pair at a given time step.
        """
        p_f_total = self._compute_failure_probability(state,a,t)
        observation_k = self.mdp.get_obs(t)
        expected_one_stage_reward = self.expected_reward(a,observation_k,self.mdp.failure_penalty,p_f_total)
        expected_future_value = self.expected_future_value(state,a,t,p_f_total)
        value = expected_one_stage_reward + self._GAMMA*expected_future_value
        return value
    
    def _compute_failure_probability(
        self,
        states: np.ndarray,    # shape (n,2): [SoC, mode]
        actions: np.ndarray,   # shape (n,): 0 or 1
        t: int
    ) -> np.ndarray:
        """
        Compute overall failure probability p_fail = p_B + p_M - p_B * p_M
        for each (state, action) pair by delegating to helper methods.
        """
        p_B = self._battery_failure_probability(states, actions, t)
        p_M = self._mechanical_failure_probability(states, actions, t)
        return self._combine_failure_probabilities(p_B, p_M)

    def _battery_failure_probability(
        self,
        states: np.ndarray,
        actions: np.ndarray,
        t: int
    ) -> np.ndarray:
        """
        Step 1: Compute p_B = P(battery failure) using a Beta CDF of energy deficit.
        """
        tl = self.mdp.transition_logic
        env = tl.env_provider
        alpha = env.get_solar_alpha(t)
        beta_param = env.get_solar_beta(t)
        scale = self.G_MAX
        solar_dist = beta(a=alpha, b=beta_param, loc=0.0, scale=scale)
        stored = tl.soc_to_energy(states[0,0])
        required = tl._calculate_energy_consumption(states, actions)
        deficit = required - stored
        return solar_dist.cdf(deficit)

    def _mechanical_failure_probability(
        self,
        states: np.ndarray,
        actions: np.ndarray,
        t: int
    ) -> np.ndarray:
        """
        Step 2: Compute p_M via Weibull-sigmoid integration over wind.
        """
        actions = np.array(actions)
        tl = self.mdp.transition_logic
        env = tl.env_provider
        shape_param = env.get_wind_shape(t)
        scale_param = env.get_wind_scale(t)
        wind_dist = weibull_min(c=shape_param, loc=0.0, scale=scale_param)
        grid_size = 400
        w_max = wind_dist.ppf(0.999999999999999)
        w = np.linspace(0.0, w_max, grid_size)
        pdf = wind_dist.pdf(w)
        n = states.shape[0]
        w_mat = np.repeat(w[:, None], n, axis=1).ravel()
        action_mat = np.repeat(actions[None, :], grid_size, axis=0).ravel()
        state_mat = np.repeat(states[None, :, :], grid_size, axis=0).reshape(-1, states.shape[1])
        succ = tl.transition_model.compute_probability(w_mat, action_mat, state_mat)
        succ = succ.reshape(grid_size, n)
        fail_cond = 1.0 - succ
        return np.trapz(fail_cond * pdf[:, None], w, axis=0)

    def _combine_failure_probabilities(
        self,
        p_B: np.ndarray,
        p_M: np.ndarray
    ) -> np.ndarray:
        """
        Step 3: Combine independent failure probabilities:
        p_fail = p_B + p_M - (p_B * p_M)
        """
        return p_B + p_M - (p_B * p_M)
    
    def solve(self) -> None:
        """
        Perform backward induction, filling future_value_table.
        """
        states = self.states
        action_list = [np.array(0)[np.newaxis],np.array(1)[np.newaxis]]
        values = np.zeros(2)
        for t in tqdm(range(self.horizon-2, -1, -1)):
            for i, s in enumerate(states[:-1]):
                for a in action_list:
                    values[a] = self._value(s[np.newaxis],a,t)
 
                self.future_value_table[i, t] = max(values)
        filename = f"future_value_table_{self.mdp.battery_capacity_wh}Wh_{self.horizon}h_{self.mdp.failure_penalty}p.npy"
        np.save(filename, self.future_value_table)
        print("Value function table saved to:", filename)

    def lookup_future_values(self, states: np.ndarray, stages: np.ndarray) -> np.ndarray:
        """
        Vectorized retrieval of future values for given states and stages.
        Any state with mode==2 is treated as the broken state.
        """
        # find broken-state index
        broken_mask = (self.states[:, 0] == -1.0) & (self.states[:, 1] == 2)
        broken_idx = np.flatnonzero(broken_mask)[0]

        # unpack query arrays
        soc_vals = states[:, 0]
        modes    = states[:, 1].astype(int)

        # find bin indices for each SoC
        bin_idxs = np.searchsorted(self._soc_grid, soc_vals, side='right') - 1
        bin_idxs = np.clip(bin_idxs, 0, len(self._soc_grid) - 1)

        # flat index = bin_idx * num_modes + mode
        state_idxs = bin_idxs * self._num_modes + modes

        # override broken-mode entries
        state_idxs[modes == 2] = broken_idx

        return self.future_value_table[state_idxs, stages]

    @staticmethod
    def expected_reward(a_k,O_k,penalty,p_failure):
        return a_k*O_k-penalty*p_failure
    
    def expected_future_value(self,state, action, stage, p_fail):
        
        if state[0,0] == 0:
            return 0

        # Broken state contribution to expected future value
        broken_value = 0
        broken_contribution = broken_value*p_fail

        # Survival states controbution to expected future value
        stored_energy = self.state_to_energy(state)
        required_energy = self.get_required_energy(state,action)
        alpha_k,beta_k = self.get_beta_params(stage)
        # Get future values that are possible as a result of action from current state
        ROWS_PER_MODE = int((self.future_value_table.shape[0]-1)/2)
        if action == 0:
            possible_future_values = self.future_value_table[:ROWS_PER_MODE,stage+1]
            c_grid = np.arange(0,1.02,.01)*self.mdp.battery_capacity_joules
        if action == 1:
            possible_future_values = self.future_value_table[ROWS_PER_MODE:-1,stage+1]
            c_grid = np.arange(0,1.02,.01)*self.mdp.battery_capacity_joules

        unweighted_survival = self.compute_survival_contribution(stored_energy,
                                                    required_energy,
                                                    self.G_MAX,
                                                    alpha_k,beta_k,
                                                    p_fail,
                                                    possible_future_values, # future value will be at the next stage
                                                    c_grid)
        survival_contribution = (1-p_fail)*unweighted_survival
        
        return broken_contribution + survival_contribution

    def state_to_energy(self,state):
        return self.mdp.transition_logic.soc_to_energy(state[0,0])
    
    def get_required_energy(self, state, action):
        return self.mdp.transition_logic.get_required_energy(state,action)
    
    def get_beta_params(self,stage):
        alpha_k = self.mdp.env_provider.get_solar_alpha(stage)
        beta_k = self.mdp.env_provider.get_solar_beta(stage)
        return alpha_k, beta_k

    @staticmethod
    def compute_survival_contribution(C_k, E_k, G_max, alpha_k, beta_k, p_fail, V_next, c_grid):
        """
        Compute the 'survival contribution' term in the Bellman expectation:
        sum_j (1 - p_fail) * DeltaP_j * V_next[j]
        where DeltaP_j = BetaCDF((c_j +/- Δc/2 - (C_k - E_k)) / G_max)

        Parameters:
        - C_k (float): current stored energy
        - E_k (float): energy required for the action
        - G_max (float): maximum possible solar gain in stage k
        - alpha, beta (float): parameters of the Beta distribution
        - p_fail (float): total failure probability at stage k
        - V_next (np.ndarray): array of V_{k+1}(c_j, m*) values for each bin j
        - c_grid (np.ndarray): array of bin centers c_j

        Returns:
        - float: the survival contribution to E[V_{k+1} | s, a]
        """
    # Shift due to energy consumption
        delta = C_k - E_k

        # c_grid is an array of bin edges, length M+1 if there are M bins
        edges = c_grid
        e_lower = edges[:-1]   # shape = (M,)
        e_upper = edges[1:]    # shape = (M,)

        u_lower = (e_lower - delta) / G_max   # shape = (M,)
        u_upper = (e_upper - delta) / G_max   # shape = (M,)

        # Clamp between 0 and 1
        u_upper = np.clip(u_upper, 0, 1)
        u_lower = np.clip(u_lower, 0, 1)

        # Regularized incomplete beta (Beta CDF)
        F_upper = betainc(alpha_k, beta_k, u_upper)
        F_lower = betainc(alpha_k, beta_k, u_lower)

        # Probability mass in each bin
        deltaP = F_upper - F_lower

        # Survival mass allocated to each bin
        survival_mass = (1 - p_fail) * deltaP

        # Compute weighted sum of V_next
        survival_contribution = np.dot(survival_mass, V_next)
        return survival_contribution#, deltaP, survival_mass



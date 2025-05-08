import numpy as np
from typing import Callable, Union
from tqdm import tqdm
from BaseClasses.plotting_utils_base import PlottingUtils
from BaseClasses.mdp_base import AbstractMDP, stochasticMDP
from scipy.stats import beta, weibull_min

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
        filename = f"future_value_table_{self.mdp.battery_capacity_wh}Wh_{self.horizon}horizon.html"
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

    def _initialize_future_value_table(self) -> np.ndarray:
        num_states = self.states.shape[0]
        return np.zeros((num_states, self.horizon))

    def _get_integration_grids(self, a_s,b_s,k_w,lam, t: int):
        """
        Build solar and wind grids and their pdf values for stage t.

        Solar: Beta on [0,1] with step 0.01
        Wind: Weibull on [0,30] with step 0.5 m/s
        """
        env = self.mdp.transition_logic.env_provider
        # solar
        # a_s = env.get_solar_alpha(t)
        # b_s = env.get_solar_beta(t)
        solar_dist = beta(a=a_s, b=b_s, loc=0.0, scale=1.0)
        solar_step = 1.0/10.0
        solar_nodes = np.arange(0.0, 1.0 + solar_step, solar_step)
        solar_pdf = solar_dist.pdf(solar_nodes)
        # wind
        # k_w = env.get_wind_shape(t)
        # lam  = env.get_wind_scale(t)
        wind_dist = weibull_min(c=k_w, loc=0.0, scale=lam)
        wind_step = 0.05
        wind_nodes = np.arange(0.0, 20.0 + wind_step, wind_step)
        wind_pdf   = wind_dist.pdf(wind_nodes)
        return solar_nodes, solar_pdf, wind_nodes, wind_pdf

    def _get_next_states(self,states:np.ndarray, action:int, solar_samples: np.ndarray):
        # return next_states


    def _expected_action_value(self, s: np.ndarray, a: int, t: int,
                               solar_nodes, solar_pdf,
                               wind_nodes, wind_pdf) -> float:
        """
        Compute E[r + gamma * V] for state s, action a at stage t
        via vectorized Riemann sum over solar and wind grids.
        """
        env = self.mdp.transition_logic.env_provider
        a_s = env.get_solar_alpha(t)
        b_s = env.get_solar_beta(t)
        k_w = env.get_wind_shape(t)
        lam  = env.get_wind_scale(t)
        solar_nodes, solar_pdf, wind_nodes, wind_pdf = self._get_integration_grids(a_s,b_s,k_w,lam,t)
        next_states = self._get_next_states(s, a, solar_nodes)
        success_values = self.lookup_future_values(next_states, t+1)
        failure_values = np.full_like(success_values, 0.0)
        
        raise NotImplementedError("This method is not implemented yet.")

    def _value(self, state: np.ndarray, a: int, next_state:np.ndarray, t: int,):
        """
        Compute the value of a state-action pair at a given time step.
        """
        # Placeholder for actual value computation
        return 0.0
    
    def _failure_probability(self,
                              states: np.ndarray,
                              actions: np.ndarray,
                              solar_samples:np.ndarray,
                              wind_samples:np.ndarray,
                              t: int
                            ) -> float:
        """
        Determine the total probability of a failure occuring given solar and wind samples.
        """



        p_f_battery = self._battery_failure_probability(stored_energy=stored_energy,
                                                        required_energy=requried_energy,
                                                        solar_gain_cdf=solar_dist.cdf
                                                        )
        
        p_f_mechanical = self._mechanical_failure_probability(states,actions,wind_samples)
        p_f_total = p_f_battery + p_f_mechanical - (p_f_battery * p_f_mechanical)
        return p_f_total

    @staticmethod
    def _battery_failure_probability(
        stored_energy: Union[float, np.ndarray],
        required_energy: Union[float, np.ndarray],
        solar_gain_cdf: Callable[[np.ndarray], np.ndarray],
    ) -> float:
        """
        Vectorized calculation of battery‐failure probability due to insufficient solar gain.

        A battery failure occurs whenever the random solar gain G_k is
        less than or equal to the energy deficit:
            threshold = capacity_energy - required_energy

        This function works element‐wise on arrays.

        Parameters
        ----------
        capacity_energy : float or array_like of float
            Available energy in the battery at stage k (e.g., in Wh or J).
        required_energy : float or array_like of float
            Energy required to complete the chosen action at stage k.
        solar_gain_cdf : Callable[[np.ndarray], np.ndarray]
            Vectorized CDF of the solar‐gain random variable G_k.
            Must accept and return a NumPy array of the same shape.

        Returns
        -------
        np.ndarray
            Probability of battery failure for each element:
            - If required_energy > capacity_energy, returns 1.0 (certain failure).
            - Otherwise, returns solar_gain_cdf(capacity_energy - required_energy).
        """
        # Convert inputs to arrays for broadcasting
        capacity = np.asarray(stored_energy, dtype=float)
        required = np.asarray(required_energy, dtype=float)

        # Compute the energy‐deficit threshold
        threshold = capacity - required

        # Clip negatives to zero so CDF is only evaluated on [0, ∞)
        clipped = np.clip(threshold, a_min=0.0, a_max=None)

        # Evaluate the CDF at each clipped threshold
        failure_prob = solar_gain_cdf(clipped)

        # Where required > capacity (threshold < 0), failure is certain
        failure_prob = np.where(threshold < 0.0, 1.0, failure_prob)

        return failure_prob

    @staticmethod
    def mechanical_failure_probability(
        model,
        action: Union[int, np.ndarray],
        state: np.ndarray,
        weibull_shape: float,
        weibull_scale: float,
        grid_size: int = 800
    ) -> np.ndarray:
        """
        Vectorized calculation of mechanical‐failure probability due to wind‐dependent failures.

        The mechanical failure probability for each (action, state) pair is:
            p_M = ∫₀^∞ [1 - p_succ(w; action, state)] f_W(w) dw

        where:
        - p_succ(w; action, state) = model.compute_probability(w, action, state)
        - f_W(w) is the Weibull PDF with parameters (weibull_shape, weibull_scale).

        Numerical integration is done via the trapezoidal rule on a fixed grid.

        Parameters
        ----------
        model : object
            Must implement
                compute_probability(
                    wind_speed: float|np.ndarray,
                    action: int|np.ndarray,
                    state: np.ndarray
                ) → np.ndarray
            returning P(success) for each input tuple.
        action : int or array_like of shape (n,)
            Action(s) taken (e.g. 0 for passive, 1 for active). Broadcastable to length n.
        state : array_like, shape (n, d)
            State vectors for each action; each row must match the format expected by `model`.
        weibull_shape : float
            Shape parameter (k) of the Weibull wind‐speed distribution.
        weibull_scale : float
            Scale parameter (λ) of the Weibull wind‐speed distribution.
        grid_size : int, optional
            Number of points in the wind‐speed grid (default: 800).

        Returns
        -------
        np.ndarray, shape (n,)
            Mechanical failure probabilities p_M for each (action, state) pair.
        """
        # --- Prep inputs ---
        # --- 1) Normalize state to shape (n, d) ---
        state_arr = np.asarray(state, dtype=float)
        if state_arr.ndim == 1:
            state_arr = state_arr[np.newaxis, :]
        n, d = state_arr.shape

        # --- 2) Normalize action to shape (n,) ---
        # Use ndmin=1 so scalars become array([scalar])
        action_arr = np.array(action, ndmin=1)
        if action_arr.size == 1:
            # broadcast scalar to all n states
            action_arr = np.full(n, action_arr.item(), dtype=action_arr.dtype)
        elif action_arr.size != n:
            raise ValueError(
                f"Length mismatch: action has size {action_arr.size}, expected {n}"
            )

        # --- Set up Weibull wind‐speed distribution & grid ---
        wind_dist = weibull_min(c=weibull_shape, scale=weibull_scale)
        w_max = wind_dist.ppf(0.9999)                # cover 99.9% of the mass
        w = np.linspace(0.0, w_max, grid_size)      # shape: (grid_size,)
        pdf = wind_dist.pdf(w)                      # shape: (grid_size,)

        # --- Broadcast to evaluate p_succ(w, a, s) for all combos ---
        # Flatten into vectors of length (grid_size * n)
        w_mat      = np.repeat(w[:, None],      n, axis=1).ravel()
        action_mat = np.repeat(action_arr[None, :], grid_size, axis=0).ravel()
        state_mat  = np.repeat(state_arr[None, :, :], grid_size, axis=0) \
                        .reshape(-1, state_arr.shape[1])

        # --- Compute success probabilities & convert to failure ---
        succ = model.compute_probability(w_mat, action_mat, state_mat)  # length grid_size*n
        succ = succ.reshape(grid_size, n)
        fail_cond = 1.0 - succ  # shape: (grid_size, n)

        # --- Integrate over wind speed ---
        integrand = fail_cond * pdf[:, None]            # shape: (grid_size, n)
        p_M = np.trapz(integrand, w, axis=0)            # shape: (n,)

        return p_M

    
    def solve(self) -> None:
        """
        Perform backward induction, filling future_value_table.
        """
        states = self.states
        Ptab = self.future_value_table
        for t in tqdm(range(self.horizon-2, -1, -1)):
            solar_nodes, solar_pdf, wind_nodes, wind_pdf = self._get_integration_grids(t)
            for i, s in tqdm(enumerate(states[:-1])):
                values = [self._expected_action_value(s, a, t,
                            solar_nodes, solar_pdf,
                            wind_nodes, wind_pdf)
                          for a in (0,1)]
                Ptab[i, t] = max(values)
        print("Future-value table:\n", Ptab)

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


# class mdpAnalyticalBackwardSolver:
#     """
#     Analytically solves a finite-horizon stochastic MDP via backward induction.

#     At each stage, integrates over solar and wind distributions to compute
#     expected action values and fills a future-value table, which can then be
#     visualized as a surface plot.

#     Parameters
#     ----------
#     mdp : stochasticMDP
#         The MDP model, providing transition, reward, and environment data.
#     horizon : int
#         Number of discrete time steps in the planning horizon.
#     """

#     def __init__(self, mdp: stochasticMDP, horizon: int):
#         """
#         Initialize the backward solver.

#         Parameters
#         ----------
#         mdp : stochasticMDP
#             The MDP model, providing transition, reward, and environment data.
#         horizon : int
#             Total number of decision epochs.
#         """
#         self.mdp = mdp
#         self.horizon = horizon
#         self.states = self.mdp._get_states()
#         self._GAMMA = 1.0
#         self.future_value_table = self._initialize_future_value_table()

#     def _initialize_future_value_table(self) -> np.ndarray:
#         """
#         Allocate and zero-initialize the future value table.

#         Returns
#         -------
#         numpy.ndarray
#             A zero matrix of shape (num_states, horizon).
#         """
#         num_states = self.states.shape[0]
#         T = self.horizon
#         return np.zeros((num_states, T))

#     def solve(self) -> None:
#         """
#         Perform backward induction to fill the future value table.

#         Loops backward over each stage, approximates integrals over solar (Beta)
#         and wind (Weibull) distributions using Riemann sums, evaluates the two
#         discrete actions per state, and stores the maximal expected value.

#         After filling the table, prints it and plots a surface of value vs. state
#         & stage using PlottingUtils.plot_surface_plotly.
#         """
        
#         raise NotImplementedError("Analytical backward induction is not implemented.")
    
#     def battery_failure_probability(self, solar_energy_distribution, stored_energy, required_energy) -> float:
#         """
#         Calculate the probability of battery failure based on the solar energy distribution.

#         Args:
#             solar_energy_distribution (scipy.stats distribution): The distribution of solar energy.
#             stored_energy (float): The amount of energy currently stored in the battery.
#             required_energy (float): The amount of energy required to avoid failure.

#         Returns:
#             float: Probability of battery failure.
#         """
#         # P(failure) = P(G ≤ required − stored)
#         threshold = required_energy - stored_energy
#         return solar_energy_distribution.cdf(threshold)

#     def mechanical_failure_probability(self, wind_speed_mps, action, state) -> float:
#         """
#         Calculate the probability of mechanical failure based on wind speed and action.

#         Args:
#             wind_speed_mps (float): Wind speed in meters per second.
#             action (int): Action taken (0 for float, 1 for fly).
#             state (np.ndarray): Current state of the system.

#         Returns:
#             float: Probability of mechanical failure.
#         """
#         failure_probability = self.mdp.transition_model.compute_probability(
#                                                             wind_speed=wind_speed_mps,
#                                                             action=action,
#                                                             state=state
#                                                             )
#         return failure_probability
    
#     def total_failure_probability(self, mechanical_failure_prob, battery_failure_prob) -> float:
#         """
#         Calculate the total probability of failure.

#         Args:
#             mechanical_failure_prob (float): Probability of mechanical failure.
#             battery_failure_prob (float): Probability of battery failure.

#         Returns:
#             float: Total probability of failure.
#         """
#         return mechanical_failure_prob + battery_failure_prob - (mechanical_failure_prob * battery_failure_prob)

#     def lookup_future_values(self, states: np.ndarray, stages: np.ndarray) -> np.ndarray:
#         """
#         Retrieve precomputed future values for given states and stages.
#         Any state with mode==2 is treated as the broken state [-1.0, 2].

#         Args:
#             states (np.ndarray): Array of states to look up (shape: [n, state_dim]).
#             stages (np.ndarray): Array of time indices corresponding to each state (shape: [n]).

#         Returns:
#             np.ndarray: Array of future values for each (state, stage) pair.

#         Raises:
#             ValueError: If any non-broken state isn’t in the solver’s state table.
#         """
#         # 1) Locate the broken‐state index
#         is_broken_state = (self.states[:, 1] == 2) & (self.states[:, 0] == -1.0)
#         try:
#             broken_idx = np.nonzero(is_broken_state)[0][0]
#         except IndexError:
#             raise ValueError("Broken state [-1.0, 2] not found in state table.")

#         n = states.shape[0]
#         state_idxs = np.empty(n, dtype=int)

#         # 2) Map all mode==2 inputs to the broken index
#         broken_mask = (states[:, 1] == 2)
#         state_idxs[broken_mask] = broken_idx

#         # 3) Handle the remaining (mode 0 or 1) inputs
#         normal_idxs = np.nonzero(~broken_mask)[0]
#         if normal_idxs.size > 0:
#             norm_states = states[normal_idxs]
#             # exact match on soc and mode
#             mask = np.all(self.states[None, :, :] == norm_states[:, None, :], axis=2)

#             # check that every normal state was found
#             if not np.all(mask.any(axis=1)):
#                 missing = normal_idxs[~mask.any(axis=1)]
#                 bad_states = states[missing]
#                 raise ValueError(f"States {bad_states.tolist()} at indices {missing.tolist()} not found.")

#             # pick the first matching index for each
#             matched_idxs = np.argmax(mask, axis=1)
#             state_idxs[normal_idxs] = matched_idxs

#         # 4) Finally, pull from the table
#         return self.future_value_table[state_idxs, stages]

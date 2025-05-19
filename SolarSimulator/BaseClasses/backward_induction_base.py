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

    def _value(self, state: np.ndarray, a: int, next_state:np.ndarray, t: int,):
        """
        Compute the value of a state-action pair at a given time step.
        """
        # Placeholder for actual value computation
        return 0.0
    
    def _compute_failure_probability(
        self,
        states: np.ndarray,    # shape (n,2): [SoC, mode]
        actions: np.ndarray,   # shape (n,): 0 or 1
        t: int
    ) -> np.ndarray:
        """
        Compute p_fail(s_k,a_k) = p_B + p_M - p_B*p_M for each (state,action) pair.

        Step 1:  p_B = P(battery failure) = F_beta(required - stored; α,β)
        Step 2:  p_M = ∫₀^∞ [1 - P_success(w; a_k,s_k)] f_Weibull(w) dw
        Step 3:  p_fail = p_B + p_M - p_B*p_M
        """

        # --- grab helpers and parameters ---
        tl  = self.mdp.transition_logic
        env = tl.env_provider

        # --- Step 1: battery failure via Beta CDF ---
        a_s, b_s = env.get_solar_alpha(t), env.get_solar_beta(t)
        solar_dist = beta(a=a_s, b=b_s, loc=0.0, scale=1357*.66*.01*15*60)
        # stored and required energy (J or Wh)
        stored   = tl.soc_to_energy(states[:, 0])
        required = tl._calculate_energy_consumption(states, actions)
        deficit  = required - stored
        p_B      = solar_dist.cdf(deficit)

        # --- Step 2: mechanical failure via Weibull‐sigmoid integration ---
        k_w, lam = env.get_wind_shape(t), env.get_wind_scale(t)
        wind_dist = weibull_min(c=k_w, loc=0.0, scale=lam)
        grid_size = 800
        w_max     = wind_dist.ppf(0.9999)
        w         = np.linspace(0.0, w_max, grid_size)
        pdf       = wind_dist.pdf(w)

        n = states.shape[0]
        # Broadcast wind × (state,action) to shape (grid_size*n,)
        w_mat      = np.repeat(w[:, None],        n, axis=1).ravel()
        action_mat = np.repeat(actions[None, :],  grid_size, axis=0).ravel()
        state_mat  = np.repeat(states[None, :, :], grid_size, axis=0) \
                          .reshape(-1, states.shape[1])

        # success probability P_success(w, a, s)
        succ = tl.transition_model.compute_probability(
                    w_mat, action_mat, state_mat
               )
        succ = succ.reshape(grid_size, n)
        fail_cond = 1.0 - succ

        # integrate to get p_M
        p_M = np.trapz(fail_cond * pdf[:, None], w, axis=0)

        # --- Step 3: combine independent failures ---
        p_fail = p_B + p_M - (p_B * p_M)

        return p_fail
    
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


import unittest
from BaseClasses.transition_model_base import RealisticSuccessProbability
class TestFailureProbability(unittest.TestCase):
    def setUp(self):
        # common dummy parameters and solver setup
        alpha, beta_param = 2.0, 5.0
        k_w, lam = 1.5, 3.0
        G_max = 1357*.66*.01*15*60

        class DummyEnv:
            def get_solar_alpha(self, t): return alpha
            def get_solar_beta(self, t): return beta_param
            def get_wind_shape(self, t): return k_w
            def get_wind_scale(self, t): return lam

        env = DummyEnv()
        class StubLogic:
            def __init__(self):
                from transition_model_base import RealisticSuccessProbability
                self.env_provider = env
                self.transition_model = RealisticSuccessProbability()
                self._battery_capacity_joules = G_max
            def soc_to_energy(self, soc):
                return (soc / 100.0) * self._battery_capacity_joules
            def _calculate_energy_consumption(self, states, actions):
                # constant consumption of 30 units
                return np.full(states.shape[0], 30.0)

        self.solver = mdpAnalyticalBackwardSolver.__new__(mdpAnalyticalBackwardSolver)
        self.solver.mdp = type('M', (), {})()
        self.solver.mdp.transition_logic = StubLogic()
        self.G_max = G_max
        self.alpha = alpha
        self.beta = beta_param
        self.k_w = k_w
        self.lam = lam

    def monte_carlo_failure(self, soc, action, N=50000):
        # helper to compute Monte Carlo p_fail for given scenario
        beta_dist = beta(a=self.alpha, b=self.beta, loc=0.0, scale=1.0)
        wind_dist = weibull_min(c=self.k_w, loc=0.0, scale=self.lam)
        stored = self.solver.mdp.transition_logic.soc_to_energy(np.array([soc]))[0]
        required = self.solver.mdp.transition_logic._calculate_energy_consumption(
            np.array([[soc, 0]]), np.array([action]))[0]
        model = self.solver.mdp.transition_logic.transition_model
        rng = np.random.default_rng(123)

        failures = 0
        for _ in range(N):
            G = beta_dist.rvs(random_state=rng) * self.G_max
            w = wind_dist.rvs(random_state=rng)
            succ = model.compute_probability(np.array([w]), np.array([action]), np.array([[soc,0]]))[0]
            B = (G < (required - stored))
            M = rng.random() > succ
            if B or M:
                failures += 1
        return failures / N

    def test_single_scenario(self):
        # single reference scenario
        soc, action, t = 50.0, 1, 0
        p_analytical = self.solver._compute_failure_probability(
            np.array([[soc, 0]]), np.array([action]), t)[0]
        p_mcs = self.monte_carlo_failure(soc, action)
        self.assertAlmostEqual(p_analytical, p_mcs, delta=0.02)

    def test_multiple_scenarios(self):
        # test a variety of SoC and action combinations
        scenarios = [(0.0, 0), (0.0,1), (50.0,1), (100.0,1)]
        t = 0
        for soc, action in scenarios:
            with self.subTest(soc=soc, action=action):
                p_analytical = self.solver._compute_failure_probability(
                    np.array([[soc, 0]]), np.array([action]), t)[0]
                p_mcs = self.monte_carlo_failure(soc, action)
                # for full battery (soc=100), ensure battery failure is zero
                if soc == 100.0:
                    self.assertAlmostEqual(
                        p_analytical, 
                        self.solver._compute_failure_probability(
                            np.array([[soc,0]]), np.array([action]), t)[0],
                        delta=1e-6
                    )
                self.assertAlmostEqual(p_analytical, p_mcs, delta=0.03)

if __name__ == '__main__':
    unittest.main()

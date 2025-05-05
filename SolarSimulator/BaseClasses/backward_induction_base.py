import numpy as np
from tqdm import tqdm
from BaseClasses.plotting_utils_base import PlottingUtils
from BaseClasses.mdp_base import AbstractMDP, stochasticMDP
from scipy.stats import beta, weibull_min

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
        PlottingUtils.plot_surface_plotly(self.future_value_table, self.mdp.battery_capacity_wh, filename)

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

    At each stage, integrates over solar and wind distributions to compute
    expected action values and fills a future-value table, which can then be
    visualized as a surface plot.

    Parameters
    ----------
    mdp : stochasticMDP
        The MDP model, providing transition, reward, and environment data.
    horizon : int
        Number of discrete time steps in the planning horizon.
    """

    def __init__(self, mdp: stochasticMDP, horizon: int):
        """
        Initialize the backward solver.

        Parameters
        ----------
        mdp : stochasticMDP
            The MDP model, providing transition, reward, and environment data.
        horizon : int
            Total number of decision epochs.
        """
        self.mdp = mdp
        self.horizon = horizon
        self.states = self.mdp._get_states()
        self._GAMMA = 1.0
        self.future_value_table = self._initialize_future_value_table()

    def _initialize_future_value_table(self) -> np.ndarray:
        """
        Allocate and zero-initialize the future value table.

        Returns
        -------
        numpy.ndarray
            A zero matrix of shape (num_states, horizon).
        """
        num_states = self.states.shape[0]
        T = self.horizon
        return np.zeros((num_states, T))

    def solve(self) -> None:
        """
        Perform backward induction to fill the future value table.

        Loops backward over each stage, approximates integrals over solar (Beta)
        and wind (Weibull) distributions using Riemann sums, evaluates the two
        discrete actions per state, and stores the maximal expected value.

        After filling the table, prints it and plots a surface of value vs. state
        & stage using PlottingUtils.plot_surface_plotly.
        """
        NUM_POINTS_SOLAR = 800
        NUM_POINTS_WIND = 800
        for stage in tqdm(range(self.horizon - 1, -1, -1)):
            c_stage = self.mdp.env_provider.get_wind_shape(stage)
            scale_stage = self.mdp.env_provider.get_wind_scale(stage)
            a_stage = self.mdp.env_provider.get_solar_alpha(stage)
            b_stage = self.mdp.env_provider.get_solar_beta(stage)
            distSolar = beta(a_stage, b_stage, scale=1.0)
            distWind = weibull_min(c_stage, scale=scale_stage)

            for i, state in enumerate(self.states[:-1, :]):
                values = np.zeros(2)
                for action in [0, 1]:
                    solar_vals = np.linspace(0, 1, NUM_POINTS_SOLAR)
                    dx = solar_vals[1] - solar_vals[0]
                    y_max = 50.0
                    wind_vals = np.linspace(0, y_max, NUM_POINTS_WIND)
                    dy = wind_vals[1] - wind_vals[0]

                    X_mesh, Y_mesh = np.meshgrid(solar_vals, wind_vals, indexing='xy')
                    states = np.full((X_mesh.size, 2), state)
                    actions = np.full(X_mesh.size, action)
                    next_states = self.mdp.transition_logic.nofail_transition(states, actions, X_mesh)
                    failure_states = np.full_like(next_states, np.array([-1.0, 2]))

                    rewards = self.mdp.reward(states, actions, next_states, stage)
                    V_vec = self.value_function(stage, rewards, next_states)

                    p_success = self.mdp.transition_model.compute_probability(Y_mesh.flatten(), action, state)
                    failure_rewards = np.full_like(p_success, -self.mdp.failure_penalty)
                    future_fail = self.lookup_future_values(failure_states, np.full(p_success.shape, stage + 1))
                    failure_value = failure_rewards + self._GAMMA * future_fail

                    fX = distSolar.pdf(X_mesh.flatten())
                    fY = distWind.pdf(Y_mesh.flatten())
                    joint_pdf = fX * fY
                    integrand = (p_success * V_vec + (1 - p_success) * failure_value) * joint_pdf

                    values[action] = np.sum(integrand) * dx * dy

                self.future_value_table[i, stage] = np.max(values)

        print(self.future_value_table)
        PlottingUtils.plot_surface_plotly(self.future_value_table, self.mdp.battery_capacity_wh)

    def value_function(self, stage: int, rewards: np.ndarray, next_states: np.ndarray) -> np.ndarray:
        """
        Compute the stage-t value for each next state instance.

        Groups identical next states to avoid redundant table lookups, then
        returns γ·V_{t+1}(s') + r for each sample.

        Parameters
        ----------
        stage : int
            The current stage index (0-based).
        rewards : numpy.ndarray
            Immediate rewards for each sample, shape (N,).
        next_states : numpy.ndarray
            Array of next states for each sample, shape (N, state_dim).

        Returns
        -------
        numpy.ndarray
            Values per sample, shape (N,).
        """
        unique_next_states, inverse_indices, _ = np.unique(
            next_states, axis=0, return_inverse=True, return_counts=True
        )
        next_stage = stage + 1
        if next_stage < self.horizon:
            future_values = self.lookup_future_values(
                unique_next_states, np.full(unique_next_states.shape[0], next_stage)
            )
        else:
            future_values = np.zeros_like(rewards)

        values = np.zeros_like(rewards)
        for i in range(unique_next_states.shape[0]):
            idx = np.where(inverse_indices == i)[0]
            values[idx] = self._GAMMA * future_values[i] + rewards[idx]
        return values

    def lookup_future_values(self, states: np.ndarray, stages: np.ndarray) -> np.ndarray:
        """
        Retrieve future values V_t(s) from the table for given states and stages.

        Parameters
        ----------
        states : numpy.ndarray
            Array of query states, shape (M, state_dim).
        stages : numpy.ndarray
            Array of stage indices for each state, shape (M,).

        Returns
        -------
        numpy.ndarray
            Future-value entries, shape (M,).

        Raises
        ------
        ValueError
            If any query state is not in the precomputed state list or if
            requested stages exceed the horizon.
        """
        if np.any(stages >= self.horizon):
            return np.zeros_like(stages, dtype=float)

        stages = stages.astype(int)
        mask = np.all(self.states[None, :, :] == states[:, None, :], axis=2)
        if not np.all(mask.any(axis=1)):
            missing = np.where(~mask.any(axis=1))[0]
            raise ValueError(f"States at indices {missing} not found in the state table.")

        state_indices = np.argmax(mask, axis=1)
        return self.future_value_table[state_indices, stages]

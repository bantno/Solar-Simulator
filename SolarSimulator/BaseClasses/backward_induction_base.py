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

        PlottingUtils.plot_surface_plotly(self.future_value_table, self.mdp.battery_capacity_wh)

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

        Args:
            states (np.ndarray): Array of states to look up (shape: [n, state_dim]).
            stages (np.ndarray): Array of time indices corresponding to each state (shape: [n]).

        Returns:
            np.ndarray: Array of future values for each (state, stage) pair.

        Raises:
            ValueError: If any requested state is not in the solver's state table.
        """
        # Build boolean mask matching requested states to self.states
        mask = np.all(self.states[None, :, :] == states[:, None, :], axis=2)
        if not np.all(mask.any(axis=1)):
            missing = np.where(~mask.any(axis=1))[0]
            raise ValueError(f"States at indices {missing} not found in the state table.")

        # Map each requested state to its index in self.states
        state_idxs = np.argmax(mask, axis=1)
        return self.future_value_table[state_idxs, stages]

class mdpAnalyticalBackwardSolver:
    def __init__(self, mdp: stochasticMDP, horizon):
        self.mdp = mdp
        self.horizon = horizon
        self.states = self.mdp._get_states()
        self._GAMMA = 1.0
        self.future_value_table = self._initialize_future_value_table()

    def _initialize_future_value_table(self):
        num_states = self.states.shape[0]
        T = self.horizon
        future_value_table = np.zeros((num_states, T))
        return future_value_table

    def solve(self):
        NUM_POINTS_SOLAR = 800
        NUM_POINTS_WIND = 800
        for stage in tqdm(range(self.horizon - 1, -1, -1)):
            c_stage = self.mdp.env_provider.get_wind_shape(stage)
            scale_stage = self.mdp.env_provider.get_wind_scale(stage)
            a_stage = self.mdp.env_provider.get_solar_alpha(stage)
            b_stage = self.mdp.env_provider.get_solar_beta(stage)
            # Define distributions for the current stage
            distSolar = beta(a_stage,b_stage, scale=1.0)
            distWind = weibull_min(c_stage, scale=scale_stage)
            for i, state in enumerate(self.states[:-1, :]):
                values = np.zeros(2)
                for action in [0,1]:
                    solar_vals = np.linspace(0, 1, NUM_POINTS_SOLAR)
                    dx = solar_vals[1] - solar_vals[0]
                    # fX_vals = distSolar.pdf(solar_vals)

                    # For Y ~ Weibull(2,1), domain is [0,∞), but we truncate:
                    y_max = 50.0
                    wind_vals = np.linspace(0, y_max, NUM_POINTS_WIND)
                    dy = wind_vals[1] - wind_vals[0]
                    # fY_vals = distWind.pdf(wind_vals)

                    X_mesh, Y_mesh = np.meshgrid(solar_vals, wind_vals, indexing='xy')
                    states = np.full((X_mesh.shape[0]*X_mesh.shape[1], 2), state)
                    actions = np.full_like(X_mesh.flatten(), action)
                    next_states = self.mdp.transition_logic.nofail_transition(states,actions,X_mesh)
                    failure_states = np.full_like(next_states, np.array([-1.0, 2]))
                    # It is better to rewrite V for vectorized operations:

                    rewards = self.mdp.reward(states, actions, next_states, stage)
                    # need to set reward to be negative where battery is empty
                    
                    V_vec = self.value_function(stage,rewards,next_states)

                    p_success = self.mdp.transition_model.compute_probability(Y_mesh.flatten(), action, state)
                    failure_rewards = np.full_like(p_success, -self.mdp.failure_penalty)

                    # 2) Compute the true failure‐state value (penalty + γ·future( failure_state ))
                    #    Use the MDP’s reward() + lookup_future_values() on failure_states
                    future_fail = self.lookup_future_values(failure_states, np.full_like(p_success, stage+1))
                    failure_value = failure_rewards + self._GAMMA * future_fail
                    # Assume V(C) is already scalar.
                    fX = distSolar.pdf( X_mesh.flatten() )
                    fY = distWind.pdf( Y_mesh.flatten() )
                    joint_pdf = fX * fY
                    integrand = (p_success * V_vec
                        + (1 - p_success) * failure_value) \
                        * joint_pdf
                    values[action] = np.sum(integrand) * dx * dy

                self.future_value_table[i, stage] = max(values)
        print(self.future_value_table)
        PlottingUtils.plot_surface_plotly(self.future_value_table, self.mdp.battery_capacity_wh)

    def value_function(self, stage, rewards, next_states) -> float:
        unique_next_states, inverse_indices, counts = np.unique(next_states, axis=0, return_inverse=True, return_counts=True)
        next_stage = stage + 1
        if next_stage < self.horizon:
            future_values = self.lookup_future_values(unique_next_states, np.full(unique_next_states.shape[0], next_stage))
        else:
            future_values = np.full_like(rewards,0.)
        values = np.zeros_like(rewards)
        for i, _ in enumerate(unique_next_states):
            indices = np.where(inverse_indices == i)[0]
            values[indices] = self._GAMMA * future_values[i] + rewards[indices]
        return values

    def lookup_future_values(self, states: np.ndarray, stages: np.ndarray) -> np.ndarray:
        if np.any(stages >= self.horizon):
            return np.full_like(stages,0.0)
        stages = stages.astype(int)
        mask = np.all(self.states[None, :, :] == states[:, None, :], axis=2)
        if not np.all(mask.any(axis=1)):
            missing_indices = np.where(~mask.any(axis=1))[0]
            raise ValueError(f"States at indices {missing_indices} not found in the state table.")
        state_indices = np.argmax(mask, axis=1)
        future_values = self.future_value_table[state_indices, stages]
        return future_values


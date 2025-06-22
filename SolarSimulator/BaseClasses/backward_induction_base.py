import numpy as np
from tqdm import tqdm
from BaseClasses.mdp_base import AbstractMDP,stochasticMDP
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
        filename = f"future_value_table_{self.mdp.battery_capacity_wh}Wh_{self.horizon}h_{self.mdp.failure_penalty}p_{self.mdp.env_provider.lat}lat.npy"
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

    def __init__(self, mdp:stochasticMDP, horizon: int):
        self.mdp = mdp
        self.horizon = horizon
        # ─── derive dynamic SoC grid from passed-in soc_increment ───────────────
        # (this replaces the old 1%/101-bin assumption)
        self.soc_increment: float = float(self.mdp.soc_increment)
        # build percent grid [0, Δ%, 2Δ%, …, 100]
        self.soc_levels: np.ndarray = np.arange(
            0.0,
            100.0 + self.soc_increment,
            self.soc_increment
        )
        # number of SoC bins per mode
        self.n_soc_levels: int = len(self.soc_levels)
        # energy width of one SoC bin (J)
        self.Δ_energy: float = (
            self.mdp.battery_capacity_joules
            * (self.soc_increment / 100.0)
        )
        # ─────────────────────────────────────────────────────────────────────────
        self.states = self.mdp._get_states()
        self._GAMMA = 1.0
        self.future_value_table = self._initialize_future_value_table()
        self.optimal_action_table = np.zeros_like(self.future_value_table)
        self.G_MAX = self.mdp.env_provider._energy_gain_from_solar(1.)
        soc_values = self.states[:-1, 0]  # exclude the 'broken' terminal state
        self._soc_grid = np.array(sorted(set(np.round(soc_values, 6))))
        self._num_modes = 2  # assume modes 0 and 1, mode==2 is broken

    def _initialize_future_value_table(self) -> np.ndarray:
        num_states = self.states.shape[0]
        return np.zeros((num_states, self.horizon))


    def set_start_date(self,start_date):
        self.start_date = start_date

    def set_location(self,location):
        self.location = location

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
        grid_size = 100
        w_max = wind_dist.ppf(0.9999999)
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

    def _compute_failure_probability(
        self,
        states: np.ndarray,    # shape (n,2): [SoC (in %), mode]
        actions: np.ndarray,   # shape (n,), each 0 or 1
        t: int
    ) -> np.ndarray:
        """
        Compute the true p_fail = p_B + (1 - p_B)*p_{M|B=0} for each (state,action).
        This matches the MCS order: (1) draw solar G -> check battery -> if survive, check wind.
        """

        n = states.shape[0]
        p_fail = np.zeros(n, dtype=float)

        # 1) Extract current stored energy (joules) and required energy (joules)
        #    for each state,action at time t.
        #    Note: state[:,0] is SoC in percent; we must convert to joules.
        C_joules = self.mdp.transition_logic.soc_to_energy(states[:, 0])
        required = self.mdp.transition_logic.get_required_energy(states, actions)

        # 2) Build the Beta distribution for solar at time t:
        α = self.mdp.env_provider.get_solar_alpha(t)
        β = self.mdp.env_provider.get_solar_beta(t)
        G_max = self.G_MAX  # maximum possible solar gain in joules

        solar_dist = beta(a=α, b=β, loc=0.0, scale=G_max)

        # 3) For each i, compute p_B[i] = P{ G < (required - C) }.
        #    If (required[i] - C[i]) <= 0, then p_B[i] = 0 (no battery fail possible).
        deficits = required - C_joules   # shape = (n,)
        # To avoid negative arguments to Beta CDF, clip:
        u = np.clip(deficits / G_max, 0.0, 1.0)  # dimensionless
        p_B = solar_dist.cdf(u)                  # shape = (n,)

        # 4) Now compute p_M_conditional[i] = P{ mech fail | G >= (required[i] - C[i]) }.
        #    If required[i] <= C[i], then debthreshold = 0, i.e. G >= 0 always, so p_M_conditional = p_M_uncond.
        #    If required[i] > C[i], then battery only survives if G >= (required - C).  We must reweight wind‐fail over that region.
        #
        #    We will do a 2D numerical integral:
        #      numerator_i   = ∫_{G = max(0, required[i]-C[i])}^{G_max} [P(mech fail | action, state)] * f_G(G) dG
        #      denominator_i = ∫_{G = max(0, required[i]-C[i])}^{G_max} f_G(G) dG   = 1 - p_B[i]
        #
        #    Since mechanical failure prob does NOT depend on G, but DOES depend on the original state,
        #    we can actually factor out “∫f_G(G) dG” if (required[i] <= C[i]) OR if we assume independence.
        #    But to mirror exactly what MC does, we explicitly exclude the region G < (required-C).
        #

        # Precompute unconditional mechanical‐fail p_M_uncond[i] for each i:
        p_M_uncond = self._mechanical_failure_probability(states, actions, t)
        # Note: p_M_uncond[i] = P{ mech fail } if you attempt the action, regardless of solar.

        # Now build p_M_conditional array of length n.
        p_M_cond = np.zeros(n, dtype=float)

        # If required[i] <= C[i], then p_B[i] = 0 and “battery survive” region is G ∈ [0, G_max].
        # So p_M_cond[i] = p_M_uncond[i].
        surv_all = (deficits <= 0)   # boolean mask
        p_M_cond[surv_all] = p_M_uncond[surv_all]

        # For the indices where deficits > 0, the battery‐survive region is G ≥ deficits[i].
        # Then mechanical_fail only “counts” over that tail of the Beta.
        idx = np.nonzero(deficits > 0)[0]
        if idx.size > 0:
            # We'll do a Gauss‐Legendre quadrature over G ∈ [0, G_max], but we only integrate
            # from G_min = deficits[i] to G_max.  We'll build a single grid in G and then loop.
            m = 200
            # Gauss‐Legendre nodes x_j ∈ [−1, 1], weights w_j ∈ [0,2].
            x, w = leggauss(m)
            # Map to [0, G_max]:
            G_nodes = 0.5 * (x + 1) * G_max      # shape (m,)
            pdf_G   = solar_dist.pdf(G_nodes)   # shape (m,)

            for i_ in idx:
                thr = deficits[i_]            # the cutoff = (E_i - C_i)
                if thr >= G_max:
                    # Then no G ∈ [thr, G_max] exists except measure zero, so the denominator→0,
                    # and effectively p_M_cond[i_] = 0 (since you never survive battery).
                    p_M_cond[i_] = 0.0
                    continue

                # Build a mask of which G_nodes lie in [thr, G_max].
                survive_mask = (G_nodes >= thr)
                if not survive_mask.any():
                    p_M_cond[i_] = 0.0
                    continue

                # ∫_{G=thr}^{G_max} f_G(G) dG  ≈  sum_{j: G_j≥thr}  pdf_G[j] * (w_j * (G_max/2))
                denom = np.sum(pdf_G[survive_mask] * w[survive_mask]) * (G_max / 2.0)

                # For ∫ P(mech fail) * f_G(G) dG over G ∈ [thr, G_max]:
                # Since mechanical‐fail(i_) does NOT depend on G (it only depends on (state[i_], action[i_])),
                # we can just multiply p_M_uncond[i_] by denom.  (Wind‐failure and solar are independent.)
                num = p_M_uncond[i_] * denom

                # Finally:
                p_M_cond[i_] = num / (denom + 1e-16)  # = p_M_uncond[i_] (but ensures denominator>0)
                # In fact, p_M_cond[i_] == p_M_uncond[i_], because wind‐fail is independent of G.
                # We do this explicitly only to illustrate the conditioning.

        # Now build the final p_fail array:
        p_fail = p_B + (1.0 - p_B) * p_M_cond
        return p_fail

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
                # self.optimal_action_table[i,t] = np.argmax(values)
        filename = f"future_value_table_{self.mdp.battery_capacity_wh}Wh_{self.horizon}h_{self.mdp.failure_penalty}p_{self.start_date[0:12]}.npy"
        np.save(filename, self.future_value_table)
        print("Value function table saved to:", filename)

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

    @staticmethod
    def expected_reward(a_k,O_k,penalty,p_failure):
        return a_k*O_k-penalty*p_failure
    
    def expected_future_value(self,state, action, stage, p_fail):
        
        # Broken state contribution to expected future value
        broken_value = 0
        broken_contribution = broken_value*p_fail

        # Survival states controbution to expected future value
        stored_energy = self.state_to_energy(state)
        required_energy = self.get_required_energy(state,action)
        alpha_k,beta_k = self.get_beta_params(stage)
        # Get future values that are possible as a result of action from current state
        ROWS_PER_MODE = int((self.future_value_table.shape[0]-1)/2)

        # Determine how many SoC‐bins per mode from the percent grid:
        soc_levels = self._soc_grid                    # e.g. [0, 3.33, 6.67, …, 100]
        n_levels   = len(soc_levels)                   # dynamic number of bins
        # use dynamic energy‐bin width:
        Δ: float = self.Δ_energy

        
        # slice the future‐value table by the dynamic number of bins:
        if action[0] == 0:
            V_next = self.future_value_table[
                0 : self.n_soc_levels,
                stage + 1
            ]
        elif action[0] == 1:
            V_next = self.future_value_table[
                self.n_soc_levels : 2 * self.n_soc_levels,
                stage + 1
            ]

        # Then call (make sure V_next has shape (101,) exactly!)
        survival_contribution = self.compute_survival_contribution(
            stored_energy,            # C_k
            required_energy,          # E_k
            self.G_MAX,               # G_max
            alpha_k,                  # Beta α
            beta_k,                   # Beta β
            p_fail,                   # failure probability
            V_next,                   # array of length 101
            Δ                         # capacity/100
        )
        
        return (0.0 * p_fail) + survival_contribution

    def state_to_energy(self,state):
        return self.mdp.transition_logic.soc_to_energy(state[0,0])
    
    def get_required_energy(self, state, action):
        return self.mdp.transition_logic.get_required_energy(state,action)
    
    def get_beta_params(self,stage):
        alpha_k = self.mdp.env_provider.get_solar_alpha(stage)
        beta_k = self.mdp.env_provider.get_solar_beta(stage)
        return alpha_k, beta_k

    @staticmethod
    def compute_survival_contribution(
        Ck, Ek, G_max, α, β, p_fail, V_next, Δ
    ):
        """
        Interpret solar Gₖ arriving *before* you pay Eₖ.
        """

        # 1) Before receiving solar, you have Cₖ.
        #    After you get solar Gₖ, your new continuous energy is Cₖ + Gₖ.
        #    To survive (i.e. not battery‐fail), you need (Cₖ + Gₖ) ≥ Eₖ.
        δ = Ck  # “base” before solar

        # 2) build N bins of width Δ, then a catch‐all upper bin
        N     = len(V_next)
        edges = np.concatenate((
            np.arange(N) * Δ,  # 0·Δ, 1·Δ, …, (N-1)·Δ
            [np.inf]           # final edge
        ))
        e_lower = edges[:-1]
        e_upper = edges[1:]

        # 3) We want P{ n·Δ ≤ (Cₖ + Gₖ − Eₖ) < (n+1)·Δ } for each bin n=0..100.
        #    Equivalently, P{ Gₖ ∈ [ (n·Δ + Eₖ − δ),  ((n+1)·Δ + Eₖ − δ) ) }.
        #    So we shift edges by +Eₖ−δ:
        u_lower = (e_lower + Ek - δ) / G_max
        u_upper = (e_upper + Ek - δ) / G_max

        # 4) Clip into [0,1] and compute Beta‐CDF:
        u_lower = np.clip(u_lower, 0.0, 1.0)
        u_upper = np.clip(u_upper, 0.0, 1.0)

        F_lower = betainc(α, β, u_lower)  # P{ Gₖ ≤ (e_lower+Eₖ−δ) }
        F_upper = betainc(α, β, u_upper)  # P{ Gₖ ≤ (e_upper+Eₖ−δ) }

        # 5) Bin‐probabilities = F_upper − F_lower:
        deltaP = F_upper - F_lower  # length = 101

        # 6) Multiply by survival‐mass and dot with V_next:
        survival_mass = (1.0 - p_fail) * deltaP  # shape = (101,)
        survival_contribution = np.dot(survival_mass, V_next)

        return survival_contribution

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



import numpy as np
from abc import ABC, abstractmethod
from BaseClasses.mdp_base import DeterministicMDP  # our deterministic MDP from mdp_base.py
from BaseClasses.backward_induction_base import DeterministicMDPBackwardSolver  # backward induction solver

# =============================================================================
# Policy Interface and Implementations
# =============================================================================
class AbstractPolicy(ABC):
    @abstractmethod
    def select_action(self, state: np.ndarray, t: int, mdp: DeterministicMDP) -> int:
        """
        Given the current state and time step t, select an action.

        Parameters:
            state (np.ndarray): Current state.
            t (int): Current time step.
            mdp (DeterministicMDP): The MDP providing the dynamics.

        Returns:
            int: The selected action.
        """
        pass

class AlwaysFlyPolicy(AbstractPolicy):
    def select_action(self, state: np.ndarray, t: int, mdp: DeterministicMDP) -> int:
        return 1

class AlwaysFloatPolicy(AbstractPolicy):
    def select_action(self, state: np.ndarray, t: int, mdp: DeterministicMDP) -> int:
        return 0

class DeterministicOptimalPolicy(AbstractPolicy):
    def __init__(self, solver: DeterministicMDPBackwardSolver):
        """
        Initialize the optimal policy with a backward induction solver.
        
        Parameters:
            solver (DeterministicMDPBackwardSolver): The solver that computes the value function.
        """
        self.solver = solver

    def select_action(self, state: np.ndarray, t: int, mdp: DeterministicMDP) -> int:
        """
        For each possible action, simulate the next state, evaluate its value,
        and select the action with the highest expected value.
        """
        # Initialize with very low values
        value_list = [-np.inf, -np.inf]
        for action in [0, 1]:
            next_state, reward = mdp.step(np.array([state]), np.array([action]), t)
            value = self.solver.value_function(t, reward, next_state)
            value_list[action] = value
        return int(np.argmax(value_list))

# =============================================================================
# Simulation Runner (Decoupled from environmental data input)
# =============================================================================
class SimulationRunner:
    def __init__(self, mdp: DeterministicMDP, horizon: int, initial_state: np.ndarray, policy: AbstractPolicy):
        """
        Initialize the simulation runner.

        Parameters:
            mdp (DeterministicMDP): The MDP instance to simulate.
            horizon (int): Total number of time steps to simulate.
            initial_state (np.ndarray): The starting state.
            policy (AbstractPolicy): Policy used to select actions.
        """
        self.mdp = mdp
        self.horizon = horizon
        self.initial_state = initial_state
        self.policy = policy

    def simulate_episode(self) -> (list, list, list):
        """
        Simulate a single episode using the provided policy.

        Returns:
            trajectory (list): States visited.
            actions (list): Actions taken.
            rewards (list): Rewards received.
        """
        state = self.initial_state
        trajectory = [state]
        actions = []
        rewards = []
        for t in range(self.horizon):
            # Here the simulation can use the MDP’s sampling functions if needed.
            # For example, one might use:
            # solar = self.mdp.sample_sunlight(t, 1)[0]
            # wind = self.mdp.sample_wind_speed(t, 1)[0]
            # and include these in policy decision if desired.
            action = self.policy.select_action(state, t, self.mdp)
            actions.append(action)
            next_state, reward = self.mdp.step(np.array([state]), np.array([action]), t)
            state = next_state[0]
            trajectory.append(state)
            rewards.append(reward[0])
            # Stop simulation if we enter the broken mode.
            if state[1] == 2:
                break
        return trajectory, actions, rewards

    def simulate_multiple_episodes(self, num_episodes: int) -> list:
        """
        Simulate multiple episodes.

        Parameters:
            num_episodes (int): Number of episodes to simulate.
            
        Returns:
            list: A list of episodes, each a dict with trajectory, actions, and rewards.
        """
        episodes = []
        for _ in range(num_episodes):
            traj, acts, rews = self.simulate_episode()
            episodes.append({'trajectory': traj, 'actions': acts, 'rewards': rews})
        return episodes

# =============================================================================
# Example Usage
# =============================================================================
if __name__ == "__main__":
    # ----- Setup dummy parameters for the MDP -----
    # Create time series data for 10 time steps.
    solar_rate_series = np.full(10, 100000)      # Dummy constant solar rate
    wind_series = np.full(10, 5.0)                 # Dummy constant wind speed
    whale_reward_series = np.full(10, 1)           # Dummy whale reward

    battery_capacity_wh = 200 * 60 * 60 * 5 / 3600  # Battery capacity in watt-hours.
    idle_power = 0                                  # Energy consumption when moored.
    cruise_power = 200                              # Energy consumption while flying.
    takeoff_power = 200                             # Additional energy consumption for takeoff.
    failure_penalty = 1000                          # Penalty for a failed transition.
    delta_t = 15                                    # Time step duration (in minutes).
    gamma = 1.0                                   # Discount factor.
    transition_model_name = "moderate"              # Transition model name.
    soc_increment = 5.0                             # Increment for state-of-charge (SOC)

    # ----- Instantiate the MDP -----
    mdp = DeterministicMDP(
        battery_capacity_wh, idle_power, cruise_power, takeoff_power,
        solar_rate_series, wind_series, whale_reward_series,
        failure_penalty, delta_t, gamma, transition_model_name, soc_increment
    )

    horizon = 10
    initial_state = np.array([100, 0])  # [SoC, mode] with full battery and moored

    # ----- Option 1: Simulation with a fixed AlwaysFlyPolicy -----
    fly_policy = AlwaysFlyPolicy()
    sim_runner_fly = SimulationRunner(mdp, horizon, initial_state, fly_policy)
    traj_fly, actions_fly, rewards_fly = sim_runner_fly.simulate_episode()
    print("AlwaysFlyPolicy Episode:")
    print("Trajectory:", traj_fly)
    print("Actions:", actions_fly)
    print("Rewards:", rewards_fly)

    # ----- Option 2: Simulation with an optimal policy using backward induction -----
    solver = DeterministicMDPBackwardSolver(mdp, horizon)
    solver.solve()  # Precompute future values offline
    optimal_policy = DeterministicOptimalPolicy(solver)
    sim_runner_optimal = SimulationRunner(mdp, horizon, initial_state, optimal_policy)
    traj_opt, actions_opt, rewards_opt = sim_runner_optimal.simulate_episode()
    print("\nDeterministicOptimalPolicy Episode:")
    print("Trajectory:", traj_opt)
    print("Actions:", actions_opt)
    print("Rewards:", rewards_opt)

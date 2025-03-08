from abc import ABC, abstractmethod
import numpy as np

class AbstractSimulation(ABC):
    """
    Abstract base class for simulating decision-making policies in an MDP.

    This class ensures that all simulation algorithms:
      - Use the same reward function defined in the MDP.
      - Employ identical energy dynamics and state transitions as specified by the MDP.
      
    The simulation loop delegates state transitions and reward computation to the MDP's step function,
    ensuring consistency. Subclasses must implement the abstract method `choose_action`, which defines
    the decision-making process (e.g. optimal policy via backward induction or a future threshold-based algorithm).
    """
    
    def __init__(self, mdp, horizon: int, initial_state: np.ndarray):
        """
        Initialize the simulation.

        Parameters:
            mdp : AbstractMDP
                An instance of a class that implements the MDP (e.g., DeterministicMDP).
            horizon : int
                Total number of simulation time steps.
            initial_state : np.ndarray
                The starting state for the simulation (must match the MDP's state representation).
        """
        self.mdp = mdp
        self.horizon = horizon
        self.initial_state = initial_state

    @abstractmethod
    def choose_action(self, state: np.ndarray, t: int) -> int:
        """
        Select an action given the current state and time step.

        Parameters:
            state : np.ndarray
                The current state.
            t : int
                The current time step.

        Returns:
            int: The chosen action (e.g., 0 for mooring, 1 for flying).
        """
        pass

    def simulate_episode(self):
        """
        Simulate a single episode (trajectory) of the MDP using the policy determined by choose_action.

        Returns:
            tuple:
                trajectory (list of np.ndarray): The list of states visited during the simulation.
                actions (list of int): The actions taken at each time step.
                rewards (list of float): The rewards obtained at each time step.
        """
        state = self.initial_state
        trajectory = [state]
        actions = []
        rewards = []
        for t in range(self.horizon):
            action = self.choose_action(state, t)
            actions.append(action)
            # Delegate to the MDP's step function to get the next state and reward.
            next_state, reward = self.mdp.step(np.array([state]), np.array([action]), t)
            state = next_state[0]  # Extract the state from the returned array.
            trajectory.append(state)
            rewards.append(reward[0])
            if next_state[0,1] == 2:
                break
        return trajectory, actions, rewards

    def simulate_multiple_episodes(self, num_episodes: int):
        """
        Simulate multiple episodes to evaluate policy performance.

        Parameters:
            num_episodes : int
                The number of episodes to simulate.

        Returns:
            list of dict:
                Each dictionary contains 'trajectory', 'actions', and 'rewards' for one episode.
        """
        episodes = []
        for _ in range(num_episodes):
            traj, acts, rews = self.simulate_episode()
            episodes.append({
                'trajectory': traj,
                'actions': acts,
                'rewards': rews
            })
        return episodes

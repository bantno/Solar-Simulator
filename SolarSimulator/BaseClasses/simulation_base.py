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
    def choose_action(self, **kwargs) -> int:
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

    def simulate_episode(self,
                         solar_samples_w:float,
                         wind_samples_ms:float,
                         whale_observations:float,
                         ):
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

            # Get data for timestep
            solar_sample = solar_samples_w[t]
            wind_sample = wind_samples_ms[t]
            whale_observation = whale_observations[t]
            action = self.choose_action(state=state,
                                        solar_sample_w=solar_sample,
                                        wind_sample_ms=wind_sample,
                                        whale_observation=whale_observation)
            actions.append(action)
            # Delegate to the MDP's step function to get the next state and reward.
            next_state, reward = self.mdp.step(np.array([state]), np.array([action]), t)
            state = next_state[0]  # Extract the state from the returned array.
            trajectory.append(state)
            rewards.append(reward[0])
            if next_state[0,1] == 2:
                break
        return trajectory, actions, rewards

    def simulate_multiple_episodes(self, solar_data, wind_data, whale_data, num_episodes: int):
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
        if num_episodes > min([wind_data.shape[0],solar_data.shape[0],whale_data.shape[0]]):
            raise IndexError("Number of requested episodes exceeds provided data.")

        for i in range(num_episodes):
            solar_samples_w = solar_data[i,:]
            wind_samples_ms = wind_data[i,:]
            whale_observations = whale_data[i,:]
            traj, acts, rews = self.simulate_episode(solar_samples_w,wind_samples_ms,whale_observations)
            episodes.append({
                'trajectory': traj,
                'actions': acts,
                'rewards': rews
            })
        return episodes


class AlwaysFlySimulation(AbstractSimulation):
    def choose_action(self,**kwargs) -> int:
        # Always choose to fly (action 1)
        return 1

class AlwaysFloatSimulation(AbstractSimulation):
    def choose_action(self,**kwargs) -> int:
        # Always choose to float (action 0)
        return 0
    
class ObservationThresholdSimulation(AbstractSimulation):
    def __init__(self, mdp, horizon: int, initial_state: np.ndarray, observation_threshold: float, wind_threshold: float):
        # Initialize the base simulation attributes.
        super().__init__(mdp, horizon, initial_state)
        # Add the new wind_threshold attribute.
        self.observation_threshold = observation_threshold
        self.wind_threshold = wind_threshold
        self.low_battery_threshold = 5. # Set this based on energy required to fly such that failure due to running out of battery will never occur

    def choose_action(self,
                      state,
                      solar_sample_w,
                      wind_sample_ms,
                      whale_observation) -> int:
        
    
        
        action = 0
        is_wind_acceptable = wind_sample_ms < self.wind_threshold
        is_observation_sufficient = whale_observation > self.observation_threshold
        is_battery_sufficient = state[0] > self.low_battery_threshold

        if is_wind_acceptable and is_observation_sufficient and is_battery_sufficient:
            action = 1

        return action
        
class DeterministicOptimalSimulation(AbstractSimulation):
    def __init__(self, mdp_solver, horizon: int, initial_state: np.ndarray):
        # Initialize the base simulation attributes.
        super().__init__(mdp_solver.mdp, horizon, initial_state)
        mdp_solver.solve()
        self.mdp_solver = mdp_solver

    def choose_action(self,
                      state,solar_sample_w,
                      wind_sample_ms,
                      whale_observation,
                      t,
                      ) -> int:
        
        value_list = [-10000,-10000]
        for action in [0,1]:
            next_state, reward = self.mdp.step(state[np.newaxis,:],np.array(action),t)
            value = self.mdp_solver.value_function(t,reward,next_state)
            value_list[action] = value

        action = np.argmax(value_list)
        return action
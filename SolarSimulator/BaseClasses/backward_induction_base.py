import numpy as np
from tqdm import tqdm
from BaseClasses.mdp_base import AbstractMDP

class DeterministicMDPBackwardSolver:
    def __init__(self,mdp:AbstractMDP,horizon):
        self.mdp = mdp
        self.horizon = horizon
        self.states = self.mdp._get_states()
        self._GAMMA = 1.
        self.future_value_table = self._initialize_future_value_table()

    def _initialize_future_value_table(self):
        """
        Creates an empty table for storing the total expected future value for every possible state 
        at every stage from 0 to T. In this table, each row corresponds to a state (as given in self.states)
        and each column corresponds to a stage index (from 0 to self.horizon).

        Assumptions:
        - self.states is a NumPy array of shape (num_states, state_dim), where each row is a state
            of the form np.array([SoC, mode]).
        - self.horizon contains the total number of stages T.

        The table is initialized with zeros.
        """
        num_states = self.states.shape[0]
        T = self.horizon
        future_value_table = np.zeros((num_states, T))
        return future_value_table

    def solve(self):
        """
        Solve the MDP using backward induction.
        """
        NUM_STATE_SAMPLES = 20000

        for stage in tqdm(range(self.horizon-1,-1,-1)):
            for i, state in enumerate(self.states[:-1,:]):
                states = np.full((NUM_STATE_SAMPLES,2),state)
                actions = np.full(NUM_STATE_SAMPLES,0)
                actions[NUM_STATE_SAMPLES//2:] = 1
                next_states,rewards = self.mdp.step(states,actions,stage)

                float_rewards = rewards[:NUM_STATE_SAMPLES//2-1]
                float_next_states = next_states[:NUM_STATE_SAMPLES//2-1]
                fly_rewards = rewards[NUM_STATE_SAMPLES//2:]
                fly_next_states = next_states[NUM_STATE_SAMPLES//2:]
                
                if stage == self.horizon-1:
                    float_reward = np.mean(float_rewards)
                    fly_reward = np.mean(fly_rewards)
                    value = max(float_reward,fly_reward)
                    
                else:
                    float_value = self.value_function(stage,float_rewards,float_next_states)
                    fly_value = self.value_function(stage,fly_rewards,fly_next_states)
                    value = max(float_value,fly_value)
                
                self.future_value_table[i,stage] = value


    def value_function(self,stage,rewards,next_states)->float:
        """
        Evaluate the Bellman equation, given by V(s) =  [ R(s, a) + γ * ∑_{s'} P(s' | s, a) * V(s') ]
        for a set of rewards and next states.

        Parameters:
            rewards: np.array
                The rewards for each state, s.
            next_states: np.array
                The next states, s'.
        """
        unique_next_states, inverse_indices, counts = np.unique(next_states, axis=0, return_inverse=True, return_counts=True)
        total = len(next_states)
        next_stage = stage + 1
        next_stages = np.full(unique_next_states.shape[0],next_stage)
        future_values = self.lookup_future_values(unique_next_states,next_stages)
        expected_future = 0.
        for i, unique_state in enumerate(unique_next_states):
            # Find indices corresponding to the unique state.
            indices = np.where(inverse_indices == i)[0]
            # Compute probability as frequency count divided by total number of outcomes.
            p = counts[i] / total
            # Compute the average immediate reward for this outcome.
            avg_reward = np.mean(rewards[indices])
            # Accumulate the weighted contribution from this outcome.
            expected_future += p * (self._GAMMA * future_values[i])
        return np.mean(rewards)+expected_future

    def lookup_future_values(self,states:np.ndarray,stages:np.ndarray) -> np.ndarray:
        """
        Vectorized lookup of future values for given states and stages from the value table.
        The returned array will have the same length as the input states, which should match the
        length of the rewards array you use elsewhere in your calculations.

        Parameters:
            states: np.ndarray
                An array of states (shape: (m, state_dim)) for which to retrieve the future value.
                Each state should match one of the rows in self.states, which is of shape (num_states, state_dim).
            stages: np.ndarray
                An array of stage indices (shape: (m,)) corresponding to each state. Each value should be
                an integer between 0 and self.horizon (inclusive).

        Returns:
            np.ndarray: An array of future values (shape: (m,)) corresponding to the provided states and stages.

        Raises:
            ValueError: If any state in the input is not found in self.states.
        """
        # Create a boolean mask where each row corresponds to an input state and each column to a state in self.states.
        mask = np.all(self.states[None, :, :] == states[:, None, :], axis=2)  # Shape: (m, num_states)

        # Check if every input state was found at least once.
        if not np.all(mask.any(axis=1)):
            missing_indices = np.where(~mask.any(axis=1))[0]
            raise ValueError(f"States at indices {missing_indices} not found in the state table.")

        # For each input state, find the index of the first matching state in self.states.
        state_indices = np.argmax(mask, axis=1)  # Shape: (m,)

        # Use fancy indexing to retrieve the corresponding values from the future value table.
        future_values = self.future_value_table[state_indices, stages]

        return future_values


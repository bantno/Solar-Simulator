import numpy as np
from tqdm import tqdm
from BaseClasses.mdp_base import AbstractMDP

class DeterministicMDPBackwardSolver:
    def __init__(self, mdp: AbstractMDP, horizon):
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
        NUM_STATE_SAMPLES = 20000
        for stage in tqdm(range(self.horizon - 1, -1, -1)):
            for i, state in enumerate(self.states[:-1, :]):
                states = np.full((NUM_STATE_SAMPLES, 2), state)
                actions = np.full(NUM_STATE_SAMPLES, 0)
                actions[NUM_STATE_SAMPLES // 2:] = 1
                next_states, rewards = self.mdp.step(states, actions, stage)
                float_rewards = rewards[:NUM_STATE_SAMPLES // 2 - 1]
                float_next_states = next_states[:NUM_STATE_SAMPLES // 2 - 1]
                fly_rewards = rewards[NUM_STATE_SAMPLES // 2:]
                fly_next_states = next_states[NUM_STATE_SAMPLES // 2:]
                if stage == self.horizon - 1:
                    float_reward = np.mean(float_rewards)
                    fly_reward = np.mean(fly_rewards)
                    value = max(float_reward, fly_reward)
                else:
                    float_value = self.value_function(stage, float_rewards, float_next_states)
                    fly_value = self.value_function(stage, fly_rewards, fly_next_states)
                    value = max(float_value, fly_value)
                self.future_value_table[i, stage] = value

    def value_function(self, stage, rewards, next_states) -> float:
        unique_next_states, inverse_indices, counts = np.unique(next_states, axis=0, return_inverse=True, return_counts=True)
        total = len(next_states)
        next_stage = stage + 1
        if next_stage < self.horizon:
            future_values = self.lookup_future_values(unique_next_states, np.full(unique_next_states.shape[0], next_stage))
        else:
            future_values = np.full_like(rewards,0.)
        expected_future = 0.0
        for i, _ in enumerate(unique_next_states):
            indices = np.where(inverse_indices == i)[0]
            p = counts[i] / total
            avg_reward = np.mean(rewards[indices])
            expected_future += p * (self._GAMMA * future_values[i])
        return np.mean(rewards) + expected_future

    def lookup_future_values(self, states: np.ndarray, stages: np.ndarray) -> np.ndarray:
        mask = np.all(self.states[None, :, :] == states[:, None, :], axis=2)
        if not np.all(mask.any(axis=1)):
            missing_indices = np.where(~mask.any(axis=1))[0]
            raise ValueError(f"States at indices {missing_indices} not found in the state table.")
        state_indices = np.argmax(mask, axis=1)
        future_values = self.future_value_table[state_indices, stages]
        return future_values

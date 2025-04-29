import numpy as np

class DummyEnvProvider:
    def __init__(self, seed=None):
        self.rng = np.random.default_rng(seed)

    def sample_whale_observation(self, t: int, n: int) -> np.ndarray:
        # This will be monkey‑patched in the test loop
        return self.rng.random(n)

class RewardTester:
    def __init__(self, failure_penalty: float, seed=None):
        self.failure_penalty = failure_penalty
        self.env_provider = DummyEnvProvider(seed)

    def original_reward(self,
                        states: np.ndarray,
                        actions: np.ndarray,
                        next_states: np.ndarray,
                        t: int) -> np.ndarray:
        whale_reward = np.where(
            actions == 1,
            self.env_provider.sample_whale_observation(t, len(actions)),
            0.0
        )
        failure_penalty = np.where(
            next_states[:, 1] == 2,
            self.failure_penalty,
            0.0
        )
        return whale_reward - failure_penalty

    def optimized_reward(self,
                         states: np.ndarray,
                         actions: np.ndarray,
                         next_states: np.ndarray,
                         t: int) -> np.ndarray:
        samples = self.env_provider.sample_whale_observation(t, len(actions))
        rewards = actions * samples
        fail_mask = next_states[:, 1] == 2
        rewards[fail_mask] -= self.failure_penalty
        return rewards

def run_tests(num_tests=10, batch_size=100, seed=42):
    tester = RewardTester(failure_penalty=5.0, seed=seed)
    rng = np.random.default_rng(seed+1)

    for i in range(num_tests):
        # generate a fresh sample array and stub the provider to always return it
        samples = rng.random(batch_size)
        tester.env_provider.sample_whale_observation = (
            lambda t, n, _s=samples: _s
        )

        t = rng.integers(0, 100)
        states = rng.random((batch_size, 2))
        actions = rng.integers(0, 2, size=batch_size)
        next_states = np.column_stack([
            rng.random(batch_size),
            rng.integers(0, 3, size=batch_size)
        ])

        out1 = tester.original_reward(states, actions, next_states, t)
        out2 = tester.optimized_reward(states, actions, next_states, t)

        if not np.allclose(out1, out2, atol=1e-8):
            print(f"Mismatch on test #{i}")
            print("actions:         ", actions)
            print("next_states[:,1]:", next_states[:,1])
            print("original:        ", out1)
            print("optimized:       ", out2)
            raise AssertionError("Outputs differ!")
    print(f"All {num_tests} tests passed—both implementations agree.")

if __name__ == "__main__":
    run_tests()

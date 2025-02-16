"""Class to test functionality of transition models."""

import unittest
import numpy as np
from BaseClasses.transition_model_base import (
    ProbabilityModelFactory,
    LinearSuccessProbability,
    OnlySuccessProbability,
    TestSuccessProbability,
    RealisticSuccessProbability,
    OptimisticSuccessProbability,
)


class TestTransitionModelVectorization(unittest.TestCase):
    """Testing class to evaluate broadcasting and vectorization capabilities
      of transition model classes."""
    def setUp(self):
        """Set up test fixtures."""
        self.wind_speeds = np.array([5.0, 10.0, 15.0])
        self.actions = np.array([0, 1, 0])
        self.states = [(50, 0), (75, 1), (30, 0)]
        self.single_wind_speed = 10.0
        self.single_action = 1
        self.single_state = (50, 0)

    def test_linear_success_probability(self):
        """Evaluate vectorization of linear success model."""
        model = LinearSuccessProbability(failure_slope=0.05)
        result = model.compute_probability(self.wind_speeds, self.actions, self.states)
        self.assertEqual(len(result), len(self.wind_speeds))

    def test_only_success_probability(self):
        """Evaluate vectorization of only success model."""
        model = OnlySuccessProbability()
        result = model.compute_probability(self.wind_speeds, self.actions, self.states)
        self.assertEqual(len(result), len(self.wind_speeds))

    def test_test_success_probability(self):
        """Evaluate vectorization of testing success model."""
        model = TestSuccessProbability()
        result = model.compute_probability(self.wind_speeds, self.actions, self.states)
        self.assertEqual(len(result), len(self.wind_speeds))

    def test_realistic_success_probability(self):
        """Evaluate vectorization of realistic transition model."""
        model = RealisticSuccessProbability()
        result = model.compute_probability(self.wind_speeds, self.actions, self.states)
        self.assertEqual(len(result), len(self.wind_speeds))

    def test_optimistic_success_probability(self):
        """Evaluate vectorization of optimistic succcess probability."""
        model = OptimisticSuccessProbability()
        result = model.compute_probability(self.wind_speeds, self.actions, self.states)
        self.assertEqual(len(result), len(self.wind_speeds))

    def test_broadcasting(self):
        """Evaluate input broadcasting behavior for all available transition success models."""
        factory = ProbabilityModelFactory
        models = factory.list_models()

        for model_name in models:
            model = factory.select_probability_model(model_name)
            # Test broadcasting single wind_speed
            result = model.compute_probability(self.single_wind_speed, self.actions, self.states)
            self.assertEqual(len(result), len(self.actions))

            # Test broadcasting single action
            result = model.compute_probability(self.wind_speeds, self.single_action, self.states)
            self.assertEqual(len(result), len(self.wind_speeds))

            # Test broadcasting single state
            result = model.compute_probability(self.wind_speeds, self.actions, self.single_state)
            self.assertEqual(len(result), len(self.wind_speeds))


class TestTransitionModelValues(unittest.TestCase):
    """Evalute return values of transition success models."""
    def setUp(self):
        """Set up test fixtures."""
        self.wind_speeds = np.array([0.0, 0.0, 10.0, 10.0, 100.0, 100.0])
        self.actions = np.array([0, 1, 0, 1, 0, 1])
        self.states = [(50, 0), (50, 0), (50, 0), (25, 1), (25, 1), (25, 1)]
        self.single_wind_speed = 10.0
        self.single_action = 1
        self.single_state = (50, 0)

    def test_linear_success_probability(self):
        """Evaluate vectorization of linear success transition model."""
        failure_slope = 0.05
        model = LinearSuccessProbability(failure_slope=failure_slope)
        probabilities = model.compute_probability(
            wind_speed=self.wind_speeds, action=self.actions, state=self.states
        )
        expected_probabilities = 1 - self.wind_speeds * failure_slope
        expected_probabilities = [x if x >= 0 else 0 for x in expected_probabilities]

        self.assertEqual(len(expected_probabilities), len(probabilities))
        self.assertTrue(np.array_equal(probabilities, expected_probabilities))

    def test_realistic_success_probability(self):
        """Ensure that transition probabilities from realistic model
          are approximately accurate."""
        raise NotImplementedError()
    
    def test_optimistic_success_probability(self):
        """Ensure that transition probabilities from optimistic model are
          approximately accurate."""
        raise NotImplementedError()


if __name__ == "__main__":
    unittest.main()

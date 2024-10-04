import unittest
import pandas as pd
import numpy as np
import sys
import os
import numpy as np
import unittest

# Add the parent directory to the sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../BaseClasses')))
from mdp_example import MDP

class Plane:
    """ A mock Plane class for testing purposes. """
    def __init__(self, voltage, capacity):
        self.voltage = voltage  # Voltage in volts
        self.capacity = capacity  # Capacity in amp-hours

    def get_required_power(self, speed, efficiency):
        """ Returns a mock required power value for flying. """
        return 200  # Assume 200 watts for simplicity


class TestMDP(unittest.TestCase):

    def setUp(self):
        # Plane parameters
        voltage = 48  # Voltage in volts
        capacity = 5  # Capacity in amp-hours
        self.plane = Plane(voltage, capacity)

        # MDP parameters
        self.soc_increment = 20
        self.vehicle_states = ["moored", "flying"]
        self.max_stages = 24  # 10 stages for testing
        self.actions = ["float", "fly"]
        self.expected_solar_power = np.random.uniform(55, 60, self.max_stages)
        self.dt = 60  # 15-minute time steps
        self.gamma = .9  # Discount factor
        self.epsilon = 1e-5  # Convergence threshold for value iteration
        self.start_time = 0  # Start at 0 minutes

        # Initialize the MDP class
        self.mdp = MDP(
            plane=self.plane,
            soc_increment=self.soc_increment,
            vehicle_states=self.vehicle_states,
            max_stages=self.max_stages,
            actions=self.actions,
            expected_solar_power=self.expected_solar_power,
            dt=self.dt,
            start_time=self.start_time,
            gamma=self.gamma,
            epsilon=self.epsilon
        )


    def test_initialization(self):
        """ Test if MDP initializes correctly and expected values are set. """
        NUM_STATES = (100/self.soc_increment+1)*2
        self.assertEqual(len(self.mdp.states), NUM_STATES)  # 6 SoC levels * 2 vehicle states = 12 states
        self.assertEqual(self.mdp.ev_table.shape, (NUM_STATES, self.max_stages))  # Shape of EV table
        self.assertEqual(self.mdp.policy_table.shape, (NUM_STATES, self.max_stages))  # Shape of policy table
        print("Showing Expected Value Table")
        print(self.mdp.ev_table)
        self.mdp.plot_surfaces_by_state(self.plane.capacity,144)


    def test_reward_calculation(self):
        """ Test if rewards are calculated correctly based on action and state. """
        soc = 60
        state = (soc, "moored")
        stage = 12
        action = "fly"
        
        # Test reward during daytime
        self.assertTrue(self.mdp.is_daytime(self.start_time, self.dt, stage))
        reward = self.mdp.R(state, action, stage)
        self.assertNotEqual(reward, 0)  # Should get a non-zero reward during daytime

        # Test reward during nighttime
        night_stage = 5  # Assuming this is nighttime
        self.assertFalse(self.mdp.is_daytime(self.start_time, self.dt, night_stage))
        reward_night = self.mdp.R(state, action, night_stage)
        self.assertEqual(reward_night, 0)  # No reward during nighttime

    def test_state_transition(self):
        """ Test if state transitions are handled correctly. """
        state = (60, "moored")
        action = "fly"
        stage = 5
        new_state = self.mdp.calculate_new_state(state, action, stage)
        
        self.assertEqual(new_state[1], "flying")  # State should change to 'flying'
        self.assertGreaterEqual(new_state[0], 0)  # SoC should not be negative
        self.assertLessEqual(new_state[0], 100)  # SoC should not exceed 100

    def test_feasibility(self):
        state = (0,"flying")
        action = "fly"
        stage = 48
        solar_power = 0 # Assume no solar power
        self.assertFalse(self.mdp.is_action_feasible(action,state,stage,solar_power))

    def test_value_iteration_convergence(self):
        """ Test if value iteration converges within a reasonable number of iterations. """
        self.mdp.value_iteration()
        
        # Check if the EV and policy tables are fully populated
        ev_values = self.mdp.ev_table.isnull().sum().sum()
        policy_values = self.mdp.policy_table.isnull().sum().sum()

        self.assertEqual(ev_values, 0)  # No NaN values should be present in EV table
        self.assertEqual(policy_values, 0)  # No NaN values should be present in policy table

    def test_policy_consistency(self):
        """ Test if policy is consistent after running value iteration. """
        self.mdp.value_iteration()

        # Check if the policy actions are reasonable (only 'float' or 'fly')
        for state in self.mdp.states:
            for stage in range(self.max_stages):
                action = self.mdp.policy_table.loc[state, stage]
                self.assertIn(action, self.actions)  # Action should be either 'float' or 'fly'
        print(self.mdp.policy_table)

    def test_future_reward(self):
        """ Test the future reward calculation for a given state and action. """
        state = (60, "moored")
        action = "fly"
        stage = 3
        
        future_reward = self.mdp.get_future_reward(state, action, stage)
        self.assertIsInstance(future_reward, (int, float))  # Should return a numeric value
        self.assertGreaterEqual(future_reward, 0)  # Future rewards should be non-negative

if __name__ == '__main__':
    unittest.main()

import sys
import os

# Add the parent directory to the sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'BaseClasses')))

import pandas as pd
import numpy as np
import unittest

from mdp import mdp


class TestControlReward(unittest.TestCase):
    def test_get_control_reward(self):
        # Test case 1: u = 'float' should result in a reward of 0
        self.assertEqual(mdp.get_control_reward('float', [2, 3, 4]), 0)

        # Test case 2: u = 'fly' and w = [2, 3, 4] should result in a reward of 24 (1 * 2 * 3 * 4)
        self.assertEqual(mdp.get_control_reward('fly', [2, 3, 4]), 24)

        # Test case 3: u = 'fly' and w = [1, 10, 100] should result in a reward of 1000 (1 * 1 * 10 * 100)
        self.assertEqual(mdp.get_control_reward('fly', [1, 10, 100]), 1000)

        # Test case 4: u = 'fly' and w = [] (empty list) should result in a reward of 1 (multiplicative identity)
        self.assertEqual(mdp.get_control_reward('fly', []), 1)

        # Test case 5: u = 'float' and w = [] (empty list) should result in a reward of 0
        self.assertEqual(mdp.get_control_reward('float', []), 0)

if __name__ == '__main__':
    unittest.main()

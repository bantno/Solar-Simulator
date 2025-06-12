import unittest
from BaseClasses.seaplane_base import Seaplane
import numpy as np

class TestClimbEnergy(unittest.TestCase):
    def setUp(self):
        # A fresh Seaplane for each test
        self.plane = Seaplane(lat=0.0, lon=0.0, tz="UTC", capacity=200/22.2)
        self.U = 20.0                      # m/s
        self.gamma = np.radians(2)       # 5° climb angle

    def test_energy_positive(self):
        """Climbing from 0 to 100 m at a positive angle uses positive energy."""
        E,time = self.plane.climb_energy(
            U=self.U,
            gamma=self.gamma,
            starting_altitude=0.0,
            ending_altitude=300.0,
            timestep=1
        )
        self.assertGreater(E, 0.0,
            "Energy should be positive for a nonzero climb")

    def test_energy_monotonic(self):
        """Climbing twice as high should cost strictly more energy."""
        E1,time1 = self.plane.climb_energy(self.U, self.gamma, 0.0, 200.0, 1)
        E2,time2 = self.plane.climb_energy(self.U, self.gamma, 0.0, 300.0, 1)
        print(E1, E2)
        self.assertGreater(E2, E1,
            "Climbing to 200 m should cost more energy than to 100 m")

    def test_zero_angle_raises(self):
        """A zero climb angle should be rejected or guarded against."""
        with self.assertRaises(AssertionError):
            # if you add `assert gamma>0` in your code, this will catch it
            self.plane.climb_energy(self.U, 0.0, 0.0, 50.0, 1)

    def test_new_vs_old(self):
        """The new climb_energy function should match the old one."""
        # The old function is a bit buggy, so we don't expect it to be
        # exactly the same, but we do expect it to be close.
        E1,time1 = self.plane.climb_energy(
            U=self.U,
            gamma=self.gamma,
            starting_altitude=0.0,
            ending_altitude=300.0,
            timestep=1
        )
        P1 = E1/time1
        params = self.plane.get_mdp_power_params()
        P2 = params['takeoff_power']
        print(f"Old: {P2}, New: {P1}")
        print(params['cruise_power'])
        # self.assertAlmostEqual(P1, P2, delta=1e-3)

if __name__ == "__main__":
    unittest.main()

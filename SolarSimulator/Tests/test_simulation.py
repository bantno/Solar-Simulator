import os
import unittest
import pytz
from datetime import datetime
from BaseClasses.simulation_base import SingleCaseSimulation
from BaseClasses.seaplane_base import Seaplane


class TestSingleCaseSimulation(unittest.TestCase):
    def setUp(self):
        # Define test parameters
        self.plane = Seaplane(
            lat=30,
            lon=-90,
            tz="Etc/GMT-6",
            pdc0=0,
            gamma=-0.0047,
            tracking=False,
            cs=False,
            cd0=0.01,
            cdtot=0.06,
            n_tot=0.75,
            S=1,
            af_mass=6,
            voltage=22.2,
            capacity=30,
        )
        self.lat = 30
        self.lon = -90
        self.tz = "Etc/GMT-6"
        self.expected_file = r"Data\TEST_CASES\Wind\expected_fake_weather_data_low_wind.pkl"
        self.actual_file = r"Data\TEST_CASES\Wind\fake_weather_data_low_wind.pkl"
        self.save_history = True
        self.use_expected = False
        self.simulate_failure = True
        self.transition_model_name = "Realistic"

        # Initialize SingleCaseSimulation
        self.simulation = SingleCaseSimulation(
            self.plane,
            self.lat,
            self.lon,
            self.tz,
            self.expected_file,
            self.actual_file,
            self.save_history,
            self.use_expected,
            self.simulate_failure,
            self.transition_model_name,
        )

        # Define simulation parameters

        tz = pytz.timezone("Etc/GMT+6")  # UTC-6 timezone

        self.start_date = datetime(2025, 6, 1, tzinfo=tz)
        self.end_date = datetime(2025, 6, 5, tzinfo=tz)
        self.dt = 15

    def test_run_single_optimal_case(self):
        algo = "Optimal"
        threshold = None
        # Run the single case simulation
        result = self.simulation.run_single_case(
            self.start_date,
            self.end_date,
            self.dt,
            algo,
            threshold,
        )

        # Save the result to a file
        filename = f"{algo}_c{self.plane.capacity}_test_simulation_result.pkl"
        result.to_pickle(filename)
        print(f"Test simulation completed and result saved to '{filename}'.")

        # Check if the result file is created
        self.assertTrue(os.path.exists(filename))

    def test_run_single_threshold_case(self):
        algo = "Threshold"
        threshold = 0.25
        # Run the single case simulation
        result = self.simulation.run_single_case(
            self.start_date,
            self.end_date,
            self.dt,
            algo,
            threshold,
        )

        # Save the result to a file
        filename = f"{algo}_c{self.plane.capacity}_t{threshold}_test_simulation_result.pkl"
        result.to_pickle(filename)
        print(f"Test simulation completed and result saved to '{filename}'.")


if __name__ == "__main__":
    unittest.main()

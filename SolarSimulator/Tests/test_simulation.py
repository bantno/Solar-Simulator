import os
import unittest
import pytz
from datetime import datetime
from BaseClasses.simulation_base import SingleCaseSimulation
from BaseClasses.seaplane_base import Seaplane
from BaseClasses.transition_model_base import ProbabilityModelFactory


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

        self.save_history = True
        self.use_expected = False
        self.simulate_failure = False

        self.transition_models = ProbabilityModelFactory.list_models()
        self.algorithms = ["Threshold","Optimal"]

        # Define simulation parameters

        tz = pytz.timezone("Etc/GMT-6")  # UTC-6 timezone

        self.start_date = datetime(2025, 1, 1, tzinfo=tz)
        self.end_date = datetime(2025, 1, 4, tzinfo=tz)
        self.dt = 15

    def test_run_single_optimal_case(self):
        """Test single optimal case simulation using fake data and test transition model."""
               # Set data files
        data_file = r"SolarSimulator\Tests\test_data\sample_cases\data\data_wind-constant_low_whale-constant.pkl"

        # Set transition model
        transition_model_name = "realistic"

        # Initialize SingleCaseSimulation
        simulation = SingleCaseSimulation(
            plane=self.plane,
            lat=self.lat,
            lon=self.lon,
            tz=self.tz,
            save_history=self.save_history,
            use_expected=self.use_expected,
            simulate_failure=self.simulate_failure,
            transition_model_name=transition_model_name,
        )

        algo = "Optimal"
        threshold = 0.25
        # Run the single case simulation
        result = simulation.run_single_case(
            start_date=self.start_date,
            end_date=self.end_date,
            dt=self.dt,
            algo=algo,
            threshold=threshold,
            data_file=data_file,
        )

        # Save the result to a file
        filename = f"{algo}_c{self.plane.capacity}_t{threshold}_test_simulation_result.pkl"
        result.to_pickle(filename)
        print(f"Test simulation completed and result saved to '{filename}'.")

    def test_run_single_threshold_case(self):
        """Test single threshold case simulation using fake data and test transition model."""
        # Set data files
        data_file = r"SolarSimulator\Tests\test_data\sample_cases\data\data_wind-constant_low_whale-constant.pkl"

        # Set transition model
        transition_model_name = "realistic"

        # Initialize SingleCaseSimulation
        simulation = SingleCaseSimulation(
            plane=self.plane,
            lat=self.lat,
            lon=self.lon,
            tz=self.tz,
            save_history=self.save_history,
            use_expected=self.use_expected,
            simulate_failure=self.simulate_failure,
            transition_model_name=transition_model_name,
        )

        algo = "Threshold"
        threshold = 0.25
        # Run the single case simulation
        result = simulation.run_single_case(
            start_date=self.start_date,
            end_date=self.end_date,
            dt=self.dt,
            algo=algo,
            threshold=threshold,
            data_file=data_file,
        )

        # Save the result to a file
        filename = f"{algo}_c{self.plane.capacity}_t{threshold}_test_simulation_result.pkl"
        result.to_pickle(filename)
        print(f"Test simulation completed and result saved to '{filename}'.")

    def test_run_single_all_cases(self):
        """Run test cases for three example scenarios for each valid transition model."""
        data_directory = r"SolarSimulator/Tests/test_data/sample_cases/data/"
        models = ["realistic", "optimistic"]
        case_files = [f for f in os.listdir(data_directory) if f.endswith('.pkl')]
        for case in case_files:
            for model in models:
                for algo in self.algorithms:
                    with self.subTest(msg=f"{model},{algo}"):
                        simulation = SingleCaseSimulation(
                            self.plane,
                            self.lat,
                            self.lon,
                            self.tz,
                            self.save_history,
                            self.use_expected,
                            self.simulate_failure,
                            transition_model_name=model,
                        )
                        threshold = 0.25
                        result = simulation.run_single_case(
                            self.start_date,
                            self.end_date,
                            self.dt,
                            data_directory+case,
                            algo,
                            threshold,
                        )
                        filename = f"{algo}_c{self.plane.capacity}_model-{model}_test_simulation_result.pkl"
                        result.to_pickle(filename)
                        print(f"Test simulation for model '{model}' completed and result saved to '{filename}'.")
                        self.assertTrue(os.path.exists(filename))

    def test_run_single_many_runs(self):
        """Test the ability to run a single set of parameters for multiple iterations."""
        # Set data files
        data_file = r"SolarSimulator\Tests\test_data\sample_cases\data\data_wind-constant_low_whale-constant.pkl"

        # Set transition model
        transition_model_name = "realistic"

        # Initialize SingleCaseSimulation
        simulation = SingleCaseSimulation(
            plane=self.plane,
            lat=self.lat,
            lon=self.lon,
            tz=self.tz,
            save_history=False,
            use_expected=self.use_expected,
            simulate_failure=True,
            transition_model_name=transition_model_name,
        )
        num_runs = 20000
        algos = ["Threshold","Optimal"]
        for algo in algos:
            threshold = 0.25
            # Run the single case simulation
            result = simulation.run_single_case(
                start_date=self.start_date,
                end_date=self.end_date,
                dt=self.dt,
                algo=algo,
                threshold=threshold,
                data_file=data_file,
                num_runs=num_runs,
                failure_penalty=2
            )
            # Save the result to a file
            filename = f"{algo}_Data_c{self.plane.capacity}_t{threshold}_{self.dt}min_{self.start_date.timetuple().tm_yday}-{self.end_date.timetuple().tm_yday}_{num_runs}_lat{self.lat}.pkl"
            result.to_pickle(filename)
            print(f"Test simulation completed and result saved to '{filename}'.")
if __name__ == "__main__":
    unittest.main()

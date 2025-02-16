import unittest
import BaseClasses.run


class TestParametrization(unittest.TestCase):
    def setUp(self):
        self.sim_executor = BaseClasses.run
        self.test_simulation_params_path = r"SolarSimulator\Tests\test_data\simulation_params.yaml"

    def test_read_params_from_yaml(self):
        """Test that parameters are correctly read from yaml file."""
        params = self.sim_executor.load_simulations_config(self.test_simulation_params_path)
        self.assertEqual(len(params),2)
        keys = params[0].keys()
        self.assertIn("capacities", keys)
        self.assertIn("charge_thresholds", keys)
        self.assertIn("thresholds", keys)
        self.assertIn("dt", keys)
        self.assertIn("num_runs", keys)
        self.assertIn("start_date", keys)
        self.assertIn("end_date", keys)
        self.assertIn("latitude", keys)
        self.assertIn("longitude", keys)
        self.assertIn("save_dir", keys)
        self.assertIn("save_states", keys)
        self.assertIn("use_expected", keys)
        self.assertIn("simulate_failure", keys)
        self.assertIn("transition_model", keys)
        self.assertIn("use_multiprocessing", keys)
        self.assertIn("wind_threshold", keys)

    def test_simulation_call(self):
        """Test that simulation class is initialized correctly and run method is called with correct 
        arguments based on parameters in simulation configuration file"""
        raise NotImplementedError()
    
    def test_whale_threshold(self):
        """Test that whale observation case simulation is called with
          correct threshold for whale observation."""
        raise NotImplementedError()
    
    def test_charge_threshold(self):
        """Test that charge case simulation is called with correct threshold 
        for takeoff battery."""
        raise NotImplementedError()
    
    def test_wind_threshold(self):
        """Test that threshold simulation cases us correct threshold for acceptable wind speed."""
        raise NotImplementedError()


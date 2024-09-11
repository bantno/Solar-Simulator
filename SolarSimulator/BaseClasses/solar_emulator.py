import numpy as np

class SolarEmulator:
    """
    A class that simulates stochastic solar power production and cloudiness over a number of stages.
    """

    def __init__(self, num_stages, irradiance_mean, irradiance_std, cloud_prob, time_of_day_func, max_solar_power=5):
        """
        Initializes the SolarEmulator with the given parameters.

        Parameters:
            num_stages (int): Number of stages (decision times) in the simulation.
            irradiance_mean (float): Mean solar irradiance (in W/m^2).
            irradiance_std (float): Standard deviation of solar irradiance.
            cloud_prob (float): Probability of cloudiness at any given stage.
            time_of_day_func (function): A function that provides a time of day factor (0 to 1) for each stage.
            max_solar_power (float): Maximum power output of the solar panel/system in kW (default is 5 kW).
        """
        self.num_stages = num_stages
        self.irradiance_mean = irradiance_mean
        self.irradiance_std = irradiance_std
        self.cloud_prob = cloud_prob
        self.time_of_day_func = time_of_day_func
        self.max_solar_power = max_solar_power

    def simulate_stochastic_solar_power(self):
        """
        Simulates stochastic solar power production and cloudiness over all stages.

        Returns:
            list: A list of dictionaries with 'irradiance', 'cloudy', 'solar_power', and 'reward' for each stage.
        """
        results = []
        
        for stage in range(self.num_stages):
            # Simulate solar irradiance using a normal distribution
            irradiance = np.random.normal(self.irradiance_mean, self.irradiance_std)
            irradiance = max(0, irradiance)  # Irradiance cannot be negative
            
            # Adjust solar irradiance based on the time of day
            time_of_day_factor = self.time_of_day_func(stage,num_stages)
            adjusted_irradiance = irradiance * time_of_day_factor
            
            # Simulate cloudiness as a Bernoulli random variable
            cloudy = np.random.binomial(1, self.cloud_prob)  # 1 if cloudy, 0 if clear
            
            # If cloudy, reduce the solar power by 50%
            solar_power = adjusted_irradiance if cloudy == 0 else adjusted_irradiance * 0.5
            
            # Normalize the solar power output to the maximum solar panel capacity
            solar_power = min(self.max_solar_power, solar_power / 1000 * self.max_solar_power)  # Convert irradiance to kW
            
            # Reward is based on the actual solar power output
            reward = solar_power
            
            results.append({
                'stage': stage,
                'irradiance': adjusted_irradiance,
                'cloudy': bool(cloudy),
                'solar_power': solar_power,
                'reward': reward
            })
        
        return results

    @staticmethod
    def expected_solar_power(irradiance_mean, cloud_prob, time_of_day_factor, max_solar_power=50):
        """
        Calculates the expected solar power output for a given stage.
        
        Parameters:
            irradiance_mean (float): Mean solar irradiance (in W/m^2) at the given stage.
            cloud_prob (float): Probability of cloudiness at the stage.
            time_of_day_factor (float): A factor (0 to 1) representing the intensity of sunlight for the time of day.
            max_solar_power (float): Maximum power output of the solar system in W (default is 50 W).
        
        Returns:
            float: Expected solar power output in kW.
        """
        # Calculate the expected irradiance adjusted by time of day
        expected_irradiance = irradiance_mean * time_of_day_factor
        
        # Convert irradiance (W/m^2) to power in kW (assuming 1000 W/m^2 gives maximum power)
        expected_power_clear = min(max_solar_power, expected_irradiance / 1000 * max_solar_power)
        
        # Calculate the expected power output, accounting for cloudiness
        expected_power = expected_power_clear * (1 - 0.5 * cloud_prob)
        
        return expected_power

if __name__ == '__main__':
    # Example time_of_day_func
    def time_of_day_func(stage,daily_stages):
        """
        A simple time of day function that simulates more solar power around noon.
        Returns a factor (0 to 1) representing the sunlight intensity during the day.
        """
        return max(0, np.sin(np.pi * (np.mod(stage,daily_stages) / daily_stages)))  # Peaks around noon (stage 5 out of 10)


    # Example usage
    num_stages = 100
    irradiance_mean = 800  # mean irradiance (W/m^2)
    irradiance_std = 200   # standard deviation of irradiance (W/m^2)
    cloud_prob = 0.3  # 30% chance of cloudiness at each stage

    solar_emulator = SolarEmulator(num_stages, irradiance_mean, irradiance_std, cloud_prob, time_of_day_func)
    results = solar_emulator.simulate_stochastic_solar_power()

    for stage_result in results:
        print(stage_result)

    # Example usage of expected_solar_power
    time_of_day_factor = 0.9  # simulates morning/afternoon sunlight, peak at noon
    expected_power = SolarEmulator.expected_solar_power(irradiance_mean, cloud_prob, time_of_day_factor)
    print(f"Expected solar power output: {expected_power:.2f} W")

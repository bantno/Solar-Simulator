import numpy as np

def simulate_stochastic_solar_power(num_stages, irradiance_mean, irradiance_std, cloud_prob, time_of_day_func):
    """
    Simulates stochastic solar power production and cloudiness over a number of stages.
    
    Parameters:
        num_stages (int): Number of stages (decision times) in the simulation.
        irradiance_mean (float): Mean solar irradiance (in W/m^2).
        irradiance_std (float): Standard deviation of solar irradiance.
        cloud_prob (float): Probability of cloudiness at any given stage.
        time_of_day_func (function): A function that provides a time of day factor (0 to 1) for each stage.
        
    Returns:
        results (list of dict): A list of dictionaries with 'irradiance', 'cloudy', 'solar_power', and 'reward' for each stage.
    """
    max_solar_power = 5  # Maximum power output of the solar panel/system in kW

    results = []
    
    for stage in range(num_stages):
        # Simulate solar irradiance using a normal distribution
        irradiance = np.random.normal(irradiance_mean, irradiance_std)
        irradiance = max(0, irradiance)  # Irradiance cannot be negative
        
        # Adjust solar irradiance based on the time of day (e.g., higher during noon, lower in the morning/evening)
        time_of_day_factor = time_of_day_func(stage)
        adjusted_irradiance = irradiance * time_of_day_factor
        
        # Simulate cloudiness as a Bernoulli (binary) random variable
        cloudy = np.random.binomial(1, cloud_prob)  # 1 if cloudy, 0 if clear
        
        # If cloudy, reduce the solar power by 50%
        solar_power = adjusted_irradiance if cloudy == 0 else adjusted_irradiance * 0.5
        
        # Normalize the solar power output to the maximum solar panel capacity
        solar_power = min(max_solar_power, solar_power / 1000 * max_solar_power)  # Convert irradiance (W/m^2) to kW
        
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

def expected_solar_power(irradiance_mean, cloud_prob, time_of_day_factor, max_solar_power=5):
    """
    Calculates the expected solar power output for a given stage.
    
    Parameters:
        irradiance_mean (float): Mean solar irradiance (in W/m^2) at the given stage.
        cloud_prob (float): Probability of cloudiness at the stage.
        time_of_day_factor (float): A factor (0 to 1) representing the intensity of sunlight for the time of day.
        max_solar_power (float): Maximum power output of the solar system in kW (default is 5 kW).
    
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

# Example usage

def time_of_day_func(stage):
    """
    A simple time of day function that simulates more solar power around noon.
    Returns a factor (0 to 1) representing the sunlight intensity during the day.
    """
    return max(0, np.sin(np.pi * (stage / 10)))  # Peaks around noon (stage 5 out of 10)

num_stages = 10
irradiance_mean = 800  # mean irradiance (W/m^2)
irradiance_std = 200   # standard deviation of irradiance (W/m^2)
cloud_prob = 0.3  # 30% chance of cloudiness at each stage

results = simulate_stochastic_solar_power(num_stages, irradiance_mean, irradiance_std, cloud_prob, time_of_day_func)

for stage_result in results:
    print(stage_result)

# Example usage
irradiance_mean = 800  # mean irradiance (W/m^2)
cloud_prob = 0.3  # 30% chance of cloudiness
time_of_day_factor = 0.9  # simulates morning/afternoon sunlight, peak at noon

expected_power = expected_solar_power(irradiance_mean, cloud_prob, time_of_day_factor)
print(f"Expected solar power output: {expected_power:.2f} kW")

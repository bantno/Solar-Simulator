# probability_visualizer.py
import matplotlib.pyplot as plt
import numpy as np

def plot_success_probability(model, wind_speed_range=(0, 40), num_points=400):
    """
    Visualize the success probability for different actions and states
    using the provided probability model.

    Parameters:
        model: An instance of ActionSuccessProbabilityModel.
        wind_speed_range: Tuple (min, max) defining wind speeds.
        num_points: Number of points to sample in the wind speed range.
    """
    wind_speeds = np.linspace(wind_speed_range[0], wind_speed_range[1], num_points)
    actions = [0, 1]
    states = [(0, 0), (0, 1)]
    labels = ["Float", "Land", "Takeoff", "Fly"]
    plt.figure(figsize=(10, 6))
    label_index = 0
    for action in actions:
        for state in states:
            # Create an array of repeated states for broadcasting.
            state_array = [state] * num_points
            probabilities = model.compute_probability(wind_speeds, action, state_array)
            plt.plot(wind_speeds, probabilities, label=labels[label_index])
            label_index += 1
    plt.xlabel("Wind Speed [m/s]")
    plt.ylabel("Success Probability")
    plt.title("Success Probability vs Wind Speed")
    plt.legend()
    plt.grid(True)
    plt.show()

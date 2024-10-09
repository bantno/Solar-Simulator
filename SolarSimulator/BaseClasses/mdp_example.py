import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from datetime import datetime
from tqdm import tqdm

class MDP:
    """
    Class representing a Markov Decision Process (MDP) problem.
    """

    def __init__(self, plane, soc_increment, vehicle_states, max_stages, actions, expected_solar_power, dt=10, start_time=0, gamma=0.9, epsilon=1e-3):
        self.vehicle_states = vehicle_states
        self.plane = plane
        self.soc_increment = soc_increment
        self.states = self.create_states(soc_increment, vehicle_states)
        self.actions = actions
        self.dt = dt
        self.gamma = gamma
        self.epsilon = epsilon
        self.start_time = start_time

        # Verify expected solar power length
        if len(expected_solar_power) != max_stages :
            raise ValueError(f"Expected length {max_stages-1}, but got {len(expected_solar_power)}.")
        self.expected_solar_power = expected_solar_power

        self.ev_table = pd.DataFrame(
            np.nan, index=pd.MultiIndex.from_tuples(self.states), columns=range(max_stages)
        )

        self.policy_table = pd.DataFrame(
            np.nan,
            index=pd.MultiIndex.from_tuples(self.states),
            columns=range(max_stages),
            dtype=object
        )

        self.create_ev_table(max_stages, expected_solar_power)

    def create_states(self, soc_increment: int, vehicle_states: list) -> list:
        """
        Generate a list of states based on state of charge (SoC) increments and vehicle states.
        """
        states = []
        for state in vehicle_states:
            for soc in range(0, 101, soc_increment):
                states.append((soc, state))
        return states

    def T(self, state, action, stage):
        """
        Returns the possible next states and their transition probabilities for a given action and stage.
        """
        success_prob, failure_prob = self.calculate_maneuver_probabilities(state[1], action, stage)
        new_state_success = self.calculate_new_state(state, action, stage)
        new_state_failure = state  # Stay in the same state on failure
        return [(success_prob, new_state_success), (failure_prob, new_state_failure)]

    def R(self, state, action, stage, wind_speed, whale_prob_table):
        """
        Calculates the reward for performing the given action in the current state at the current stage.
        Includes stochastic rewards based on the probability of finding whales (time-dependent) and wind speed.
        
        Parameters:
        - state: Current state as a tuple (SoC, vehicle_state)
        - action: The action being taken ('float', 'fly')
        - stage: The current stage in the simulation
        - wind_speed: Expected wind speed at the current stage
        - whale_prob_table: Table that maps the time of day to the probability of finding whales
        
        Returns:
        - Reward value considering both deterministic and stochastic factors.
        """

        prob_success, prob_failure = self.calculate_maneuver_probabilities(state[1], action, stage)
        survival_reward = prob_success * 1 + prob_failure * (-100)

        # Determine whale sighting probability based on time of day
        if self.is_daytime(self.start_time, self.dt, stage):
            whale_prob = whale_prob_table.get(stage, 0)  # Hourly whale probability
            # Calculate rewards based on the action
            if action == 'float':
                # Whale reward based on probability of randomly finding whale using hydrophone, assume 0 for now
                whale_reward = whale_prob * 0
            
            elif action == 'fly':
                # Whale reward based on probability of finding whale during survey, assume 5 for now
                whale_reward = whale_prob * 5
        else: # Night time
            if action == 'float':
                # Whale reward based on probability of randomly finding whale using hydrophone, assume 0 for now
                whale_reward = whale_prob * 0
            elif action == 'fly':
                # Whale reward based on probability of finding whale during flight, assume 0 because we cant see at night
                whale_reward = whale_prob * 0

        return survival_reward + whale_reward


    def calculate_new_state(self, state, action, stage):
        """
        Calculate the new state of charge after performing the action.
        """
        soc = state[0]
        delta_soc = self.calculate_soc_update(self.plane, action, self.dt, self.expected_solar_power[stage])
        new_soc = soc + delta_soc
        new_soc = min(new_soc, 100)  # Keep SoC less than 100

        if action == "fly":
            new_vehicle_state = "flying"
        elif action == "float":
            new_vehicle_state = "moored"
        return (new_soc, new_vehicle_state)

    def create_ev_table(self, max_stages, expected_solar_power):
        """
        Creates an expectation value (EV) table with the given number of stages.
        """
        print("Creating expected value table...\n")
        for stage in tqdm(range(max_stages-1, -1, -1)):
            w = self.is_daytime(self.start_time, self.dt, stage)
            for state in self.states:
                max_reward = -np.inf
                best_action = None
                for action in self.actions:
                    if self.is_action_feasible(action, state, stage, expected_solar_power[stage]):
                        reward = self.R(state, action, stage)
                        future_reward = self.get_future_reward(state, action, stage)
                        total_reward = reward + self.gamma * future_reward
                        if total_reward > max_reward:
                            max_reward = total_reward
                            best_action = action
                self.ev_table.loc[state, stage] = max_reward
                self.policy_table.loc[state, stage] = best_action
        print("Done!")

    def value_iteration(self, max_iterations=1000):
        """
        Performs value iteration to find the optimal policy.
        """
        print("Starting value iteration...")
        for iteration in range(max_iterations):
            delta = 0
            for stage in tqdm(range(self.ev_table.shape[1])):
                for state in self.states:
                    v = self.ev_table.loc[state, stage]
                    max_reward = -np.inf
                    best_action = None
                    for action in self.actions:
                        if self.is_action_feasible(action, state, stage, self.expected_solar_power[stage]):
                            reward = self.R(state, action, stage)
                            future_reward = self.get_future_reward(state, action, stage)
                            total_reward = reward + self.gamma * future_reward
                            if total_reward > max_reward:
                                max_reward = total_reward
                                best_action = action
                    self.ev_table.loc[state, stage] = max_reward
                    self.policy_table.loc[state, stage] = best_action
                    delta = max(delta, abs(v - max_reward))

            if delta < self.epsilon:
                print(f"Convergence achieved after {iteration+1} iterations.")
                break
        else:
            print(f"Value iteration terminated after reaching max iterations ({max_iterations}).")

    def get_future_reward(self, state, action, stage):
        """
        Returns the future reward for transitioning to a new state after performing the given action.
        """
        next_stage = stage + 1
        if next_stage not in self.ev_table.columns:
            return 0  # No future reward beyond the last stage
        new_state = self.calculate_new_state(state, action,stage)
        return self.ev_table.loc[new_state, next_stage]

    def is_action_feasible(self, action, state, stage, solar_power):
        """
        Checks whether the given action is feasible from the current state at the given stage.
        """
        delta_soc = self.calculate_soc_update(self.plane, action, self.dt, solar_power)
        new_soc = min(state[0] + delta_soc,100)
        return 0 <= new_soc <= 100
    
    def plot_surface(self, df, title, battery_capacity_ah, max_stages):
        """
        Plot a surface for a specific vehicle state based on the EV table.
        """
        # Ensure the 'Figures' folder exists
        if not os.path.exists('Figures'):
            os.makedirs('Figures')

        # Extracting the MultiIndex levels
        X = df.index.get_level_values(0).values.astype(float)  # State of Charge
        Y = df.columns.values.astype(float)  # Stages
        
        # Converting the DataFrame values into a 2D numpy array
        Z = df.values
        
        # Creating a meshgrid for X and Y
        X, Y = np.meshgrid(Y, X)
        
        # Plotting the surface
        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot(111, projection='3d')
        
        surf = ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none')
        
        # Adding labels
        ax.set_xlabel('Stages')
        ax.set_ylabel('State of Charge (%)')
        ax.set_zlabel('Expected Value')
        ax.set_title(f'Surface Plot for state: {title} \nBattery Capacity: {battery_capacity_ah} Ah')
        
        # Adding a color bar to show the color scale
        fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
        
        # Generate a timestamp and create a filename
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        filename = f"Figures/surface_plot_{max_stages}_{title}_{timestamp}.png"
        
        # Save the figure to the 'Figures' folder
        plt.savefig(filename)
        plt.show()
        plt.close()  # Close the plot to avoid displaying it
    
    def plot_surfaces_by_state(self, battery_capacity_ah, max_stages):
        """
        Plot surfaces for each vehicle state from the EV table.
        """
        # Get the unique values in the second level of the MultiIndex (Vehicle State)
        states = self.ev_table.index.get_level_values(1).unique()
        
        for state in states:
            # Filter the DataFrame based on the state
            df_state = self.ev_table.xs(state, level=1)
            
            # Plot the surface for this state
            self.plot_surface(df_state, state, battery_capacity_ah, max_stages)

    def calculate_soc_update(self, plane, action, dt, solar_power):
        """
        Calculates the change in SoC after performing the given action.
        """
        if action == "float":
            required_power = 0
        elif action == "fly":
            required_power = plane.get_required_power(20, 1.2)  # Assumed constants for flight

        avionics_power = 10
        net_power = solar_power - required_power - avionics_power
        energy_change = net_power * dt * 60  # Convert power (W) to energy (Joules)
        soc_change = energy_change / (plane.voltage * plane.capacity * 3600) * 100
        return self.soc_increment * round(soc_change / self.soc_increment)

    def calculate_maneuver_probabilities(self, current_state, action, stage):
        """
        Calculate the success and failure probabilities for the given maneuver, adjusting continuously based on wind speed.
        """
        wind_speed = self.wind_speed_table[stage]  # Retrieve wind speed for the current stage

        # Base probabilities
        if current_state == "moored" and action == "float":
            base_success_prob, base_failure_prob = 0.99, 0.01  # High success rate for floating
        elif current_state == "moored" and action == "fly":
            base_success_prob, base_failure_prob = 0.90, 0.10  # Higher failure risk for taking off
        elif current_state == "flying" and action == "float":
            base_success_prob, base_failure_prob = 0.90, 0.10  # Moderate risk for flying to floating
        elif current_state == "flying" and action == "fly":
            base_success_prob, base_failure_prob = 0.95, 0.05  # Low failure risk for continuous flying
        else:
            return 1.0, 0.0  # Default to guaranteed success

        # Wind speed influence
        # Define thresholds for low and high wind speed ranges
        low_wind_threshold = 5
        high_wind_threshold = 20

        # Adjust failure probability based on wind speed in a continuous manner
        if wind_speed <= low_wind_threshold:
            wind_factor = 0  # No adjustment for low wind (below or equal to threshold)
        elif wind_speed >= high_wind_threshold:
            wind_factor = 1  # Max adjustment for high wind (above or equal to threshold)
        else:
            # Linearly scale between the low and high wind thresholds
            wind_factor = (wind_speed - low_wind_threshold) / (high_wind_threshold - low_wind_threshold)

        # Adjust the failure probability continuously based on wind_factor
        failure_prob = base_failure_prob + wind_factor * (0.5 - base_failure_prob)  # Scale up to a max of 50% failure
        success_prob = 1 - failure_prob  # Success probability is the complement of failure

        return success_prob, failure_prob


    @staticmethod
    def is_daytime(start_time, time_step, stage):
        """
        Determine if it's daytime (6 AM to 6 PM) based on the simulation stage.
        """
        minutes_per_day = 1440
        current_time = (start_time + time_step * stage) % minutes_per_day
        return 360 <= current_time < 1080

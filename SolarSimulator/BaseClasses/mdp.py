import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from datetime import datetime
from mpl_toolkits.mplot3d import Axes3D

from tqdm import tqdm

class mdp:
    """
    Class representing a Markov Decision Process (MDP) problem.
    """

    def __init__(self,
                 plane,
                 soc_increment,
                 vehicle_states,
                 max_stages,
                 actions,
                 start_index,
                 end_index,
                 whale_prob,
                 dt,
                 no_failure_prob,
                 start_time=0,
                 gamma=1.0,
                 epsilon=1e-3,
                 wind_mag_filename=r"Data\DISTRIBUTIONS\wind_mag_ev.pkl",
                 expected_solar_power_filename = r"Data\DISTRIBUTIONS\solar_ev.pkl"):
        
        self.show_progress = False
        self.vehicle_states = vehicle_states
        self.plane = plane
        self.soc_increment = soc_increment
        self.states = self.create_states(soc_increment, vehicle_states)
        self.actions = actions
        self.dt = dt
        self.gamma = gamma
        self.epsilon = epsilon
        self.start_time = start_time
        self.max_stages = max_stages
        self.stepwise_failure_prob = self.calculate_step_transition_prob(max_stages*dt,no_failure_prob,dt)
        print(f"Step failure probability: {self.stepwise_failure_prob}")

        lat_lon_pair = [(29.25,  -85.0)] # TODO: make this not hard coded
        wind_table = pd.read_pickle(wind_mag_filename)[lat_lon_pair].sort_index()
        self.wind_speed_table = wind_table.loc[start_index:end_index]
        solar_table = pd.read_pickle(expected_solar_power_filename)[lat_lon_pair].sort_index()
        expected_solar_power = solar_table.loc[start_index:end_index]

        # Verify expected solar power length
        if len(expected_solar_power)-1 != max_stages :
            raise ValueError(f"Expected length {max_stages}, but got {len(expected_solar_power)-1}.")
        self.expected_solar_power = expected_solar_power
        self.whale_prob_table = whale_prob

        # Initialize tables
        self.ev_table = pd.DataFrame(
            np.nan, index=pd.MultiIndex.from_tuples(self.states), columns=range(self.max_stages)
        )

        self.policy_table = pd.DataFrame(
            np.nan,
            index=pd.MultiIndex.from_tuples(self.states),
            columns=range(self.max_stages),
            dtype=object
        )


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
        new_state_success = self.calculate_new_state(state, action, stage, self.expected_solar_power.iloc[stage])
        new_state_failure = state  # Stay in the same state on failure
        return [(success_prob, new_state_success), (failure_prob, new_state_failure)]

    def R(self, state, action, stage):
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
        survival_reward = prob_success * 0.0 + prob_failure * (-50)
        minutes = (self.start_time + self.dt * stage) % 1440
        whale_prob = self.whale_prob_table.loc[minutes // 120]["Sighting Probability"]

        # Determine whale sighting probability based on time of day
        if self.is_daytime(self.start_time, self.dt, stage):
            # Calculate rewards based on the action
            if action == 'float':
                # Whale reward based on probability of randomly finding whale using hydrophone, assume 0 for now
                whale_reward = whale_prob * 0
            
            elif action == 'fly':
                # Whale reward based on probability of finding whale during survey, assume 5 for now
                whale_reward = whale_prob * 100
        else: # Night time
            if action == 'float':
                # Whale reward based on probability of randomly finding whale using hydrophone, assume 0 for now
                whale_reward = whale_prob * 0
            elif action == 'fly':
                # Whale reward based on probability of finding whale during flight, assume 0 because we cant see at night
                whale_reward = whale_prob * 0

        return survival_reward + whale_reward




    def calculate_new_state(self, state, action, stage, solar_power):
        """
        Calculate the new state of charge after performing the action.
        """
        soc = state[0]
        delta_soc = self.calculate_soc_update(self.plane, action, self.dt, solar_power)
        new_soc = soc + delta_soc
        new_soc = min(new_soc, 100)  # Keep SoC less than 100

        if action == "fly":
            new_vehicle_state = "flying"
        elif action == "float":
            new_vehicle_state = "moored"
        else:
            raise ValueError(f"Expected action 'float' or 'fly'. Got {action}.")
        return (new_soc, new_vehicle_state)

    def create_ev_table(self):
        """
        Creates an expectation value (EV) table with the given number of stages.
        """

        # print("Creating expected value table...\n")
        iterator = tqdm(range(self.max_stages-1, -1, -1)) if self.show_progress else range(self.max_stages-1, -1, -1)
        
        for stage in iterator:
            w = self.is_daytime(self.start_time, self.dt, stage)
            for state in self.states:
                max_reward = -np.inf
                best_action = None
                for action in self.actions:
                    if self.is_action_feasible(action, state, stage, self.expected_solar_power.iloc[stage].values[0]):
                        reward = self.R(state, action, stage)
                        future_reward = self.get_future_reward(state, action, stage)
                        total_reward = reward + self.gamma * future_reward
                        if total_reward > max_reward:
                            max_reward = total_reward
                            best_action = action
                self.ev_table.loc[state, stage] = max_reward
                self.policy_table.loc[state, stage] = best_action
        # print("Done!")

    def value_iteration(self, max_iterations=10):
        """
        Performs value iteration to find the optimal policy.
        """
        print("Starting value iteration...")

        self.policy_table = pd.DataFrame(
            np.nan,
            index=pd.MultiIndex.from_tuples(self.states),
            columns=range(self.max_stages),
            dtype=object
        )
        iterator = tqdm(range(self.ev_table.shape[1])) if self.show_progress else range(self.ev_table.shape[1])
        for iteration in range(max_iterations):
            delta = 0
            for stage in iterator:
                for state in self.states:
                    v = self.ev_table.loc[state, stage]
                    max_reward = -np.inf
                    best_action = "float"
                    for action in self.actions:
                        if self.is_action_feasible(action, state, stage, self.expected_solar_power.iloc[stage].values[0]):
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
        """snip
        Returns the future reward for transitioning to a new state after performing the given action.
        """
        next_stage = stage + 1
        if next_stage not in self.ev_table.columns:
            return 0  # No future reward beyond the last stage
        new_state = self.calculate_new_state(state, action,stage, self.expected_solar_power.iloc[stage].values[0])
        return self.ev_table.loc[new_state, next_stage]

    def is_action_feasible(self, action, state, stage, solar_power):
        """
        Checks whether the given action is feasible from the current state at the given stage.
        """
        delta_soc = self.calculate_soc_update(self.plane, action, self.dt, solar_power)
        new_soc = min(state[0] + delta_soc,100)
        return 0 <= new_soc <= 100

    def calculate_soc_update(self, plane, action, dt, solar_power):
        """
        Calculates the change in SoC after performing the given action.
        """
        panel_efficiency = 0.15 # TODO: use PVWATTS FOR THIS
        if action == "float":
            required_power = 0
        elif action == "fly":
            required_power = plane.get_required_power(20, 1.2)  # Assumed constants for flight
        else :
            raise ValueError(f"Expected action 'float' or 'fly'. Got {action}.")

        avionics_power = 8
        net_power = solar_power*panel_efficiency - required_power - avionics_power
        energy_change = net_power * dt * 60  # Convert power (W) to energy (Joules)
        soc_change = energy_change / (plane.voltage * plane.capacity * 3600) * 100
        return self.soc_increment * round(soc_change / self.soc_increment)

    def calculate_maneuver_probabilities(self, current_state, action, stage):
        """
        Calculate the success and failure probabilities for the given maneuver, adjusting continuously based on wind speed.
        """
        wind_speed = self.wind_speed_table.iloc[stage].values[0]  # Retrieve wind speed for the current stage
        base_failure_prob = self.stepwise_failure_prob
        # Base probabilities
        if current_state == "moored" and action == "float":
            state_action_factor = 0.75  # High success rate for floating
        elif current_state == "moored" and action == "fly":
            state_action_factor = 2  # Higher failure risk for taking off
        elif current_state == "flying" and action == "float":
            state_action_factor = 2  # Moderate risk for flying to floating
        elif current_state == "flying" and action == "fly":
            state_action_factor = 1.5  # Low failure risk for continuous flying
        else:
            return 0.0, 1.0  # Default to guaranteed failure
        base_failure_prob = base_failure_prob * state_action_factor

        # Wind speed influence
        # Define thresholds for low and high wind speed ranges
        low_wind_threshold = 5  # m/s
        high_wind_threshold = 20  # m/s

        # Adjust failure probability based on wind speed in a continuous manner
        if wind_speed <= low_wind_threshold:
            wind_factor = 1  # No adjustment for low wind (below or equal to threshold)
        elif wind_speed >= high_wind_threshold:
            wind_factor = 1.2  # Max adjustment for high wind (above or equal to threshold)
        else:
            # Linearly scale between the low and high wind thresholds
            wind_factor = 1+(wind_speed - low_wind_threshold) / (high_wind_threshold - low_wind_threshold)

        # Adjust the failure probability continuously based on wind_factor
        failure_prob = base_failure_prob * wind_factor  # Scale up to a max of 20% failure
        success_prob = 1 - failure_prob  # Success probability is the complement of failure

        return success_prob, failure_prob


    @staticmethod
    def is_daytime(start_time, time_step, stage):
        """
        Check if the current simulation stage is within daytime hours (6 AM to 6 PM).

        Parameters:
            start_time (int): Start time of the simulation in minutes from midnight (0-1439).
            time_step (int): Duration of each simulation step in minutes.
            stage (int): The current stage of the simulation.

        Returns:
            bool: True if the time falls between 6:00 AM (360) and 6:00 PM (1080), 
                otherwise False.
        """
        minutes_per_day = 1440
        current_time = (start_time + time_step * stage) % minutes_per_day
        return 360 <= current_time < 1080
    
    @staticmethod
    def calculate_step_transition_prob(period_min, no_failure_probability, step_length_min):
        """
        Calculate the stepwise transition probability for each step within a specified period.

        This method computes the probability of failure for a single step given the total 
        failure probability over a period and the number of steps within that period. 
        It ensures that the compounded stepwise failure matches the specified total failure 
        probability over the entire period.

        Parameters:
            period_min (float): The total length of the period in minutes.
            failure_probability (float): The overall failure probability for the entire period 
                (value between 0 and 1).
            step_length_min (float): The length of each step in minutes.

        Returns:
            float: The stepwise failure probability for each individual step.

        Example:
            If the total period is 60 minutes with a failure probability of 0.5 and 
            step length is 15 minutes, this method returns the stepwise probability for 
            each 15-minute interval.

        Raises:
            ValueError: If any input is non-positive or the failure_probability is not in [0, 1].
        """
        # Validate inputs
        if period_min <= 0:
            raise ValueError("Period_min must be a positive number.")
        if not (0 <= no_failure_probability <= 1):
            raise ValueError("Failure_probability must be between 0 and 1, inclusive.")
        if step_length_min <= 0:
            raise ValueError("Step_length_min must be a positive number.")

        # Calculate the number of steps and the stepwise failure probability
        num_steps = np.ceil(period_min / step_length_min)
        stepwise_failure_probability = 1 - (no_failure_probability ** (1 / num_steps))

        return stepwise_failure_probability
    
    
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
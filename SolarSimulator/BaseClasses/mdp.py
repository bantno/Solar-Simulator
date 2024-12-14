import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from scipy.stats import beta
from tqdm import tqdm

class mdp:
    """
    Class representing a Markov Decision Process (MDP) problem.
    """

    def __init__(self, plane, soc_increment, vehicle_states, actions, start_date: datetime,
                 end_date: datetime, expected_data, whale_surface_probs, dt, mission_success_prob,
                 gamma=1.0, epsilon=1e-3):
        
        # Simulation settings and parameters
        self.vehicle_states = vehicle_states
        self.actions = actions
        self.dt = dt
        self.gamma = 1.0
        self.epsilon = epsilon
        self.show_progress = False

        # Plane and state of charge settings
        self.plane = plane
        self.soc_increment = soc_increment
        self.states = self.create_states(soc_increment, vehicle_states)

        # Expected data and validation
        expected_data.sort_index()
        self.expected_data = expected_data
        self.expected_solar_power = expected_data['expected_solar_rad'].values
        self.max_stages = len(pd.date_range(start_date, end_date, freq=f"{dt}min"))
        if self.max_stages != len(self.expected_data):
            raise ValueError(f"Max stages and data lengths do not match. {self.max_stages}!={len(self.expected_data)}")

        # Failure probabilities and whale sighting probabilities
        self.stepwise_failure_prob = 1-mission_success_prob
        self.whale_surface_probs = whale_surface_probs
        self.start_time = start_date.minute + 60 * start_date.hour

        # Initialize tables for expected rewards and optimal policy
        self.ev_table = pd.DataFrame(
            np.nan, index=pd.MultiIndex.from_tuples(self.states), columns=range(self.max_stages)
        )
        self.policy_table = pd.DataFrame(
            np.nan, index=pd.MultiIndex.from_tuples(self.states),
            columns=range(self.max_stages), dtype=object
        )

    def create_states(self, soc_increment: int, vehicle_states: list) -> list:
        """
        Generate a list of states based on state of charge (SoC) increments and vehicle states.
        """
        states = [(soc, state) for state in vehicle_states for soc in range(0, 101, soc_increment)]
        return states

    def T(self, state, action, stage):
        """
        Returns failure probability for a given action and stage.
        """
        _, failure_prob = self.calculate_maneuver_probabilities(state[1], action)
        return failure_prob

    def calculate_new_state(self, state, action, solar_power):
        """
        Calculate the new state of charge after performing the action.
        """
        soc = state[0]
        delta_soc = self.calculate_soc_update(self.plane,state, action, self.dt, solar_power)
        new_soc = min(soc + delta_soc, 100)  # Limit SoC to 100
        new_vehicle_state = "flying" if action == "fly" else "moored"

        # Set state to "broken" if SoC falls below 0
        if new_soc < 0:
            new_soc, new_vehicle_state = -1, "broken"
        
        return (new_soc, new_vehicle_state)

    def create_ev_table(self):
        """
        Creates an expectation value (EV) table with the given number of stages.
        """
        S, dt, efficiency = self.plane.S, self.dt, 0.10
        required_cruise_energy = self.plane.required_cruise_power * 60 * dt
        required_takeoff_energy = self.plane.required_takeoff_energy
        capacity_j = self.plane.voltage * self.plane.capacity * 3600
        alphas, betas = self.expected_data['beta_alpha'].values, self.expected_data['beta_beta'].values

        failure_penalty, whale_found_reward = 25, 1
        stages = tqdm(range(self.max_stages-1, -1, -1), desc="EV", leave=False) if self.show_progress else range(self.max_stages-1, -1, -1)

        for stage in stages:
            for state in self.states:
                reward_list = [-1e12] * len(self.actions)

                for idx, action in enumerate(self.actions):
                    required_energy = required_cruise_energy if idx else 0
                    if state[1] == "moored" and action=="fly":
                        required_energy = required_cruise_energy+required_takeoff_energy
                    current_energy = state[0] / 100 * capacity_j
                    max_collected_energy = 1367 * S * dt * 60 * efficiency

                    solar_alpha, solar_beta = alphas[stage], betas[stage]
                    whale_surface_probability = self.get_sighting_probability(stage, dt, self.start_time)
                    broken_probability = self.T(state, action, stage)
                    k = whale_found_reward if idx else 0

                    reward = self.expected_reward(
                        required_energy, current_energy, max_collected_energy, failure_penalty,
                        k, solar_alpha, solar_beta, whale_surface_probability, broken_probability
                    )
                    total_reward = reward + self.gamma * self.get_future_reward(state, action, stage)
                    reward_list[idx] = total_reward

                max_reward = np.max(reward_list)
                self.ev_table.loc[state, stage] = max_reward
                self.policy_table.loc[state, stage] = self.actions[np.argmax(reward_list)]
    
    @staticmethod
    def expected_reward(P, C, I, k, l, solar_alpha, solar_beta, p_H_1:float, p_B_1:float)->float:
        """
        Calculate the expected reward E[R(X, H)].
        
        Parameters:
        - P (float): Required energy.
        - C (float): Stored energy.
        - I (float): Maximum collected energy.
        - k (float): Absolute values of penalty for vehicle failure.
        - l (float): Reward if X > 0 and H = 1.
        - alpha (float): Shape parameter for the Beta distribution.
        - beta (float): Shape parameter for the Beta distribution.
        - P_H_1 (float): Probability that whale is at the surface.
        - P_B_1 (float): Probability that B = 1.
        
        Returns:
        - float: Expected reward E[R(X, H)].
        """
        # Probability that H = 1
        p_H_0 = 1 - p_H_1
        p_B_0 = 1 - p_B_1

        # Calculate the threshold for X <= 0 condition (S <= (P - C) / I)
        threshold = ((P-C) / I)
        if solar_alpha==0 or solar_beta==0:
            # If no energy is collected, handle the penalty based on stored energy
            if C < P:
                # If stored energy is insufficient to meet required energy, apply penalty
                F_S = 1
            else:
                # If stored energy is sufficient, no penalty
                F_S = 0
        else:
            # Probability that S <= threshold, i.e., F_S
            F_S = beta.cdf(threshold, solar_alpha, solar_beta)

        # Calculate the expected rewards for each case
        reward_H0_B0 = -k * F_S
        reward_H0_B1 = -k
        reward_H1_B0 = l - k * F_S
        reward_H1_B1 = l - k

        expected_reward = (reward_H0_B0 * p_H_0 * p_B_0 +
                        reward_H0_B1 * p_H_0 * p_B_1 +
                        reward_H1_B0 * p_H_1 * p_B_0 +
                        reward_H1_B1 * p_H_1 * p_B_1)

        return expected_reward

    def get_future_reward(self, state, action, stage):
        """
        Calculate the future reward after transitioning to a new state by performing an action.

        Parameters:
        - state (tuple): Current state as (SoC, vehicle_state).
        - action (str): Action to perform, e.g., "fly" or "float".
        - stage (int): Current stage in the simulation.

        Returns:
        - float: Future reward value for the next state.
        """
        # Define the next stage in the sequence
        next_stage = stage + 1

        # Return 0 if this is the last stage, as there is no future reward
        if next_stage not in self.ev_table.columns:
            return 0  

        # Calculate the new state based on the current state, action, and solar power for the stage
        new_state = self.calculate_new_state(state, action, self.expected_solar_power[stage])

        # Determine the future reward based on the state of charge in the new state
        # If the state of charge (SoC) is non-negative, retrieve the reward from the EV table
        if new_state[0] >= 0:
            future_reward = self.ev_table.loc[new_state, next_stage]
        else:
            # If SoC is negative, set future reward to 0
            future_reward = 0

        return future_reward

    def calculate_soc_update(self, plane, state, action, dt, solar_power):
        """
        Calculate the change in State of Charge (SoC) after performing the specified action.

        Parameters:
        - plane: The plane object containing power and battery specifications.
        - action (str): Action to perform, either "float" or "fly".
        - dt (float): Time step in minutes.
        - solar_power (float): Available solar power in watts per square meter.

        Returns:
        - int: The rounded change in SoC based on the action and environmental conditions.
        """
        panel_efficiency = 0.15  # TODO: Update using PVWATTS for more accurate efficiency
        required_takeoff_energy = 0
        # Determine required power based on action
        if action == "float":
            required_power = 0
        elif action == "fly":
            required_power = plane.required_cruise_power
            if state[1] == "moored":
                required_takeoff_energy = plane.required_takeoff_energy
        else:
            raise ValueError(f"Expected action 'float' or 'fly'. Got {action}.")

        # Calculate the net power balance
        avionics_power = plane.idle_power
        solar_input = solar_power * panel_efficiency * plane.S
        net_power = solar_input - required_power - avionics_power

        # Convert power (W) to energy (Joules) and then to change in SoC (%)
        energy_change = net_power * dt * 60 - required_takeoff_energy # Convert power to energy
        soc_change = (energy_change / (plane.voltage * plane.capacity * 3600)) * 100  # Energy to SoC %

        # Round to the nearest SoC increment and return
        return self.soc_increment * round(soc_change / self.soc_increment)

    def calculate_maneuver_probabilities(self, current_state, action):
        """
        Calculate the success and failure probabilities for a maneuver based on the current state, 
        action, and base failure probability.

        Parameters:
        - current_state (str): The current state of the plane, e.g., "moored" or "flying".
        - action (str): The maneuver to be performed, e.g., "float" or "fly".

        Returns:
        - tuple: Success and failure probabilities.
        """
        base_failure_prob = self.stepwise_failure_prob

        # Determine failure probability factor based on state-action combinations
        state_action_factors = {
            ("moored", "float"): 1.0,
            ("moored", "fly"): 5.0,
            ("flying", "float"): 5.0,
            ("flying", "fly"): 2.0
        }

        # Retrieve factor or default to guaranteed failure if combination is invalid
        state_action_factor = state_action_factors.get((current_state, action), None)
        if state_action_factor is None:
            return 0.0, 1.0  # Invalid action for the given state: return guaranteed failure

        # Calculate probabilities
        failure_prob = base_failure_prob * state_action_factor
        success_prob = 1 - failure_prob

        return success_prob, failure_prob
    
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
    
    def get_sighting_probability(self, current_step, timestep, start_time):
        """
        Calculate the whale sighting probability for the given step and time.

        This method computes the probability of a whale sighting based on the current time,
        adjusting for the nearest time block (every 120 minutes) for the given step.

        Parameters:
            current_step (int): The current step in the simulation.
            timestep (int): The time interval between each step in minutes.
            start_time (int): The start time in minutes.

        Returns:
            float: The probability of whale sighting at the nearest time block.
        """
        current_time = (start_time + (current_step * timestep)+60) % 1440
        nearest_start = (current_time // 120) * 120
        return self.whale_surface_probs.get(nearest_start)
    
    
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

if __name__ == "__main__":
    expected_data = pd.read_pickle(r"Data\EXPECTED_DATA\data_expected_60min.pkl")
    print(expected_data.columns)
    alphas = expected_data['beta_alpha'].values
    betas = expected_data['beta_beta'].values
    print(alphas)
    print(betas)
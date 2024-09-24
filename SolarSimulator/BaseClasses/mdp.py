import pandas as pd
import numpy as np
from tqdm import tqdm

class mdp:
    """
    Class representing a markov decision process problem
    """

    def __init__(self, plane, soc_increment, vehicle_states, max_stages, actions, stm, dt=10, start_time=0):
        self.vehicle_states = vehicle_states
        self.plane = plane
        self.battery_capacity = self.plane.voltage*self.plane.capacity*3600
        self.stm = stm
        self.soc_increment = soc_increment
        self.states = self.create_states(soc_increment, vehicle_states)
        self.actions = actions
        self.start_time=0
        self.dt = dt

        self.create_ev_table(max_stages)
        

    @staticmethod
    def create_states(soc_increment: int, vehicle_states: list) -> list:
        """
        Generate a list of states based on state of charge (SoC) increments and specified vehicle states.
        """
        states = []
        for state in vehicle_states:
            for soc in range(0, 101, soc_increment):
                states.append((soc, state))
        return states
    
    @staticmethod
    def calculate_maneuver_probabilities(current_state, action, stage):
        """
        Calculate the probabilities of successfully executing a maneuver and failing.
        """
        if current_state == "moored" and action == "float":
            # Constant failure probability of 5%
            failure_prob = 0.001
            success_prob = 1 - failure_prob
        else:
            
            # Adjust lambda based on current state and action
            if current_state == "moored" and action == "fly":
                success_prob = 0.95
            elif current_state == "flying" and action == "float":
                success_prob = 0.95
            elif current_state == "flying" and action == "fly":
                success_prob = 0.99
            
            failure_prob = 1 - success_prob
        
        return success_prob, failure_prob
    
    @staticmethod
    def time_of_day_func(stage, timestep):
        """
        Compute the time of day as a sinusoidal function based on the current stage and timestep.
        """
        daily_stages = 24*60/timestep
        normalized_stage = (np.mod(stage, daily_stages) / daily_stages)
        factor = np.sin(np.pi * normalized_stage)
        if factor < 0.6:
            factor = 0
        return max(0, factor)
    
    @staticmethod
    def expected_solar_power(irradiance_mean, cloud_prob, time_of_day_factor, max_solar_power=80):
        """
        Calculates the expected solar power output for a given stage.
        """
        expected_irradiance = irradiance_mean * time_of_day_factor
        expected_power_clear = min(max_solar_power, expected_irradiance / 1000 * max_solar_power)
        expected_power = expected_power_clear * (1 - 0.5 * cloud_prob)
        
        return expected_power
    
    @staticmethod
    def is_daytime(start_time: int, time_step: int = 10, stage: int = 0) -> int:
        """
        Determines if the current stage is during the day or night, accounting for simulations
        that span multiple days.

        Args:
            start_time (int): The time in minutes from the start of the day (0-1439).
                            For example, 0 is 12:00 AM, 720 is 12:00 PM, and 1439 is 11:59 PM.
            time_step (int): The time step duration in minutes. Default is 10 minutes.
            stage (int): The current stage of the simulation.

        Returns:
            int: 1 if the stage is during the day (6 AM to 6 PM), otherwise 0.
        """
        # Number of minutes in a day
        minutes_per_day = 24 * 60
        
        # Calculate the current time in minutes, accounting for multiple days
        total_time = start_time + time_step * stage
        current_time = total_time % minutes_per_day
        
        # Convert minutes to determine day (6:00 AM = 360 minutes, 6:00 PM = 1080 minutes)
        if 360 <= current_time < 1080:
            return 1
        else:
            return 0

    
    def create_ev_table(self, max_stages):
        """
        Creates an expectation value (EV) table with a specified number of stages.
        """
        print("Creating expected value table...\n")
        num_columns = max_stages
        self.ev_table = pd.DataFrame(
            np.nan,
            index=pd.MultiIndex.from_tuples(self.states),
            columns=range(num_columns),
        )

        for k in tqdm(range(max_stages-1,-1,-1)):
            # Write new function for calculating W
            w = self.is_daytime(self.start_time,self.dt,k)

            for s in self.states:
                max_reward = -np.inf
                for u in self.actions:
                    if self.is_action_feasible(u,s,k):
                        control_reward = self.get_control_reward(u,w,s,k)
                        future_reward = self.get_future_reward(s,u,k,w)
                        reward = control_reward+future_reward
                        if reward > max_reward:
                            max_reward = reward
                self.ev_table.loc[s,k] = max_reward

        print("Done!")

    def get_activation_vector(self, u, w, k):
        """
        Retrieves activation vector.
        """
        return self.calculate_soc_update(self.plane, u, self.dt, k, self.soc_increment)

    def get_control_reward(self, u: str, w: list, current_state: tuple, k):
        """
        Determines the reward for a given action.
        """
        if w == 0:  # Night time case
            reward = 0
        elif w == 1:  # Day time case
            prob_success, prob_failure = self.calculate_maneuver_probabilities(current_state[1], u, k)
            reward = prob_success*(1) + prob_failure*(-100)
            
        return reward
    
    def is_action_feasible(self, action, state, k):
        """
        Checks whether the action is feasible for the current state.
        """
        a = self.calculate_soc_update(self.plane, action, self.dt, k, self.soc_increment)
        soc = state[0]
        if action == self.actions[0]:
            vehicle_state = self.vehicle_states[0]
        elif action == self.actions[1]:
            vehicle_state = self.vehicle_states[1]
        
        new_state = (soc + a, vehicle_state)
        feasible = new_state[0] <= 100 and new_state[0] > 0

        return feasible

    @staticmethod
    def calculate_soc_update(plane, action: str, dt: int, stage: int, use_solar_approx=True, solar_power=0, soc_increment=1) -> int:
        """
        Static method to determine update to state of charge when executing a given action.
        """
        irradiance_mean = 1000  # W/m^2
        cloud_prob = 0.3  # Probability of cloud cover
        daily_stages = 24*60/dt
        time_of_day_factor = max(0, np.sin(np.pi * (np.mod(stage, daily_stages) / daily_stages)))

        if action == "float":
            required_power = 0
        elif action == "fly":
            u = 20
            rho = 1.2
            required_power = plane.get_required_power(u, rho)  # Watts

        if use_solar_approx:
            solar_power = mdp.expected_solar_power(irradiance_mean, cloud_prob, time_of_day_factor, max_solar_power=80)

        avionics_power = 10
        total_power = solar_power - required_power - avionics_power  # Watts
        delta_energy = total_power * dt * 60  # Joules
        
        soc_update = delta_energy / (plane.voltage * plane.capacity * 3600) * 100
        rounded_soc_update = soc_increment * np.round(soc_update/soc_increment)
        return int(rounded_soc_update)

    def get_future_reward(self, state, action, stage: int, w):
        """
        Retrieves the expected reward for a given state and stage.
        """
        if action == self.actions[0]:
            u = 0
        elif action == self.actions[1]:
            u = 1

        new_stage = stage + 1

        if new_stage not in self.ev_table.columns:
            reward = 0
        else:
            a = self.get_activation_vector(action, w, stage)
            soc = state[0]
            if u == 0:
                vehicle_state = self.vehicle_states[0]
            elif u == 1:
                vehicle_state = self.vehicle_states[1]
            
            new_state = (soc + a, vehicle_state)
            if new_state[0] < 0 or new_state[0] > 100:
                reward = -np.inf
            else:
                reward = self.ev_table.loc[new_state, new_stage]

        return reward

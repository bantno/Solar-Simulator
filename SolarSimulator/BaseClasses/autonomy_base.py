import numpy as np
import random
from tqdm import tqdm

class Autonomy:
    """Represents the autonomy module for a solar-powered seaplane."""

    def __init__(self,dt,mdp_model,data,whale_probabilities):
        self.actual_environmental_data = data
        self.dt = dt
        self.mdp_model = mdp_model
        self.data = data
        self.whale_prob = whale_probabilities

    def simulate_simple_behavior(self,
                                 initial_state,
                                 true_success_prob,
                                 simulate_failure = False,
                                 save_history = False,
                                 threshold = 0.1):

        reward = 0
        max_stages = len(self.data)-1
        # print(f"Threshold = {threshold}\n")

        self.stepwise_failure_prob = self.calculate_step_transition_prob(self.dt*max_stages,true_success_prob,self.dt)
        self.wind_speed_table = self.data["wind_speed_10m"]
        
        night_hours = 12
        nightly_idle_soc = np.ceil((self.mdp_model.plane.idle_power*night_hours*3600)/(self.mdp_model.plane.capacity*self.mdp_model.plane.voltage*3600)*100)
        single_flight_soc = np.ceil(self.mdp_model.plane.get_required_power(20,1.2)*self.dt*60/(self.mdp_model.plane.capacity*self.mdp_model.plane.voltage*3600)*100)
        battery_capacity_J = self.mdp_model.plane.capacity*self.mdp_model.plane.voltage*3600
        # Preallocate arrays with a fixed size
        state_history_list = np.empty(max_stages,dtype=tuple)  # Adjust dimensions based on the state size
        energy_history_list = np.empty(max_stages)
        solar_power_list = np.empty(max_stages)
        whale_list = np.empty(max_stages)

        # Initialize the first elements
        state_history_list[0] = initial_state
        energy_history_list[0] = initial_state[0] / 100 * battery_capacity_J
        solar_power_list[0] = 0.0
        whale_list[0] = 0.0
        shortwave_radiation = self.data["shortwave_radiation"].values

        for k in range(max_stages-1):
            current_state = state_history_list[k]
            current_energy = energy_history_list[k]
            solar_power_wpm2 = shortwave_radiation[k]
            whale_prob = self.get_sighting_probability(self.whale_prob,k,self.dt,0)
            is_reward_sufficient = whale_prob>threshold and solar_power_wpm2>0
            is_battery_sufficient = current_state[0] > nightly_idle_soc*2 + single_flight_soc # TODO: Create better way to determine this

            if is_reward_sufficient and is_battery_sufficient:
                best_action = "fly"
            else :
                best_action = "float"

            if simulate_failure:
                _,failure_prob = self.calculate_maneuver_probabilities(current_state=current_state[1],
                                                                                    action=best_action,
                                                                                    stage=k)
            else:
                failure_prob = -1

            is_action_successful = np.random.uniform(0,1) > failure_prob
            
            if is_action_successful :
                new_energy = current_energy + self.calculate_energy_update(self.mdp_model.plane,best_action,self.dt,solar_power_wpm2)
                new_state = self.calculate_new_state(best_action,new_energy,battery_capacity_J)
                reward+=self.R(current_state,best_action,k,whale_prob)
                state_history_list[k+1]=new_state
                energy_history_list[k+1]=new_energy
                whale_list[k+1]=whale_prob
                solar_power_list[k+1] = solar_power_wpm2
            else:
                reward = reward-25 # TODO MAKE whale penalty a parameter
                break
        if save_history:
            return reward, k, state_history_list[:k + 1], solar_power_list[:k + 1], whale_list
        else:
            return reward,k


    def simulate_mdp_behavior(self,
                              initial_state,
                              true_success_prob,
                              simulate_failure = False,
                              save_history = False):

        battery_capacity_J = self.mdp_model.plane.capacity*self.mdp_model.plane.voltage*3600
        reward = 0
        max_stages = len(self.data)-1
        self.stepwise_failure_prob = self.calculate_step_transition_prob(self.dt*max_stages,true_success_prob,self.dt)
        self.wind_speed_table = self.data["wind_speed_10m"]
        optimal_policy = self.mdp_model.policy_table

        # Preallocate arrays with a fixed size
        state_history_list = np.empty(max_stages,dtype=tuple)  # Adjust dimensions based on the state size
        energy_history_list = np.empty(max_stages)
        solar_power_list = np.empty(max_stages)
        whale_list = np.empty(max_stages)

        # Initialize the first elements
        state_history_list[0] = initial_state
        energy_history_list[0] = initial_state[0] / 100 * battery_capacity_J
        solar_power_list[0] = 0.0
        whale_list[0] = 0.0
        shortwave_radiation = self.data["shortwave_radiation"].values

        night_hours = 12
        nightly_idle_soc = np.ceil((self.mdp_model.plane.idle_power*night_hours*3600)/(self.mdp_model.plane.capacity*self.mdp_model.plane.voltage*3600)*100)
        single_flight_soc = np.ceil(self.mdp_model.plane.get_required_power(20,1.2)*self.dt*60/(self.mdp_model.plane.capacity*self.mdp_model.plane.voltage*3600)*100)
        

        for k in range(max_stages-1):
            current_state = state_history_list[k]
            best_action = optimal_policy.loc[current_state,k]
            current_energy = energy_history_list[k]
            solar_power_wpm2 = shortwave_radiation[k]
            whale_prob = self.get_sighting_probability(self.whale_prob,k,self.dt,0)

            if simulate_failure :
                _,failure_prob = self.calculate_maneuver_probabilities(current_state=current_state[1],
                                                                                    action=best_action,
                                                                                    stage=k)
            else:
                failure_prob = -1

            if np.random.uniform(0,1) > failure_prob :
                new_energy = current_energy + self.calculate_energy_update(self.mdp_model.plane,best_action,self.dt,solar_power_wpm2)
                new_state = self.calculate_new_state(best_action,new_energy,battery_capacity_J)
                reward+=self.R(current_state,best_action,k,whale_prob)
                state_history_list[k+1]=new_state
                energy_history_list[k+1]=new_energy
                whale_list[k+1]=whale_prob
                solar_power_list[k+1] = solar_power_wpm2
            else :
                reward = reward-5
                break
        if save_history:
            return reward, k, state_history_list[:k + 1], solar_power_list[:k + 1], whale_list
        else:
            return reward,k

    

    def R(self,state,action,stage,whale_prob):
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
    
    
        whale_reward = 0

        # Determine whale sighting probability based on time of day
        if self.data["shortwave_radiation"].iloc[stage]>0:
            # Calculate rewards based on the action
            if action == 'float':
                pass
            elif action == 'fly':
                # Whale reward based on probability of finding whale during survey
                if np.random.uniform(0,1) < whale_prob:
                    whale_reward = 1
        else: # Night time
            if action == 'float':
                pass
            elif action == 'fly':
                pass

        return whale_reward
    
    def calculate_new_state(self,best_action,energy,max_capacity):
        if best_action == "float":
            state = "moored"
        elif best_action=="fly":
            state = "flying"
        soc = min(round(energy/max_capacity*100),100)
        soc= max(0,soc)
        return (soc,state)


    def calculate_maneuver_probabilities(self,current_state,action,stage):
        """
        Calculate the success and failure probabilities for the given maneuver, adjusting continuously based on wind speed.
        """
        wind_speed = self.wind_speed_table.iloc[stage]  # Retrieve wind speed for the current stage
        base_failure_prob = self.stepwise_failure_prob
        # Base probabilities
        if current_state == "moored" and action == "float":
            state_action_factor = 1.0  # High success rate for floating
        elif current_state == "moored" and action == "fly":
            state_action_factor = 10  # Higher failure risk for taking off
        elif current_state == "flying" and action == "float":
            state_action_factor = 10  # Moderate risk for flying to floating
        elif current_state == "flying" and action == "fly":
            state_action_factor = 2  # Low failure risk for continuous flying
        else:
            return 0.0, 1.0  # Default to guaranteed failure
        failure_prob = base_failure_prob * state_action_factor
        success_prob = 1-failure_prob
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
        # print(stepwise_failure_probability)
        return stepwise_failure_probability
    
    def calculate_energy_update(self, plane, action, dt, solar_power):
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

        avionics_power = plane.idle_power
        net_power = solar_power*panel_efficiency*plane.S - required_power - avionics_power
        energy_change = net_power * dt * 60  # Convert power (W) to energy (Joules)
        return energy_change
    
    @staticmethod
    def get_sighting_probability(probability_map, current_step, timestep, start_time):
        # Calculate the current time in minutes
        current_time = (start_time + (current_step * timestep)) % 1440
        
        # Find the nearest start time by rounding down to the closest 120-minute mark
        nearest_start = (current_time // 120) * 120
        
        # Return the probability, or None if out of range
        return probability_map.get(nearest_start)

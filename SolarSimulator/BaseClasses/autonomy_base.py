import numpy as np

class Autonomy:
    """Represents the autonomy module for a solar-powered seaplane."""

    def __init__(self,dt,mdp_model,data,whale_probabilities):
        self.actual_environmental_data = data
        self.dt = dt
        self.mdp_model = mdp_model
        self.data = data
        self.whale_prob = whale_probabilities
        self.failure_penalty = 25
        self.soc_increment = 1

    def simulate_simple_behavior(self,
                                 initial_state,
                                 true_success_prob,
                                 simulate_failure = False,
                                 save_history = False,
                                 threshold = 0.1):

        reward = 0
        max_stages = len(self.data)-1
        # print(f"Threshold = {threshold}\n")

        self.stepwise_failure_prob = 1-true_success_prob
        
        night_hours = 12
        nightly_idle_soc = np.ceil((self.mdp_model.plane.idle_power*night_hours*3600)/(self.mdp_model.plane.capacity*self.mdp_model.plane.voltage*3600)*100)
        single_flight_soc = np.ceil((self.mdp_model.plane.get_required_power(20,1.2)*self.dt*60+self.mdp_model.plane.required_takeoff_energy)/(self.mdp_model.plane.capacity*self.mdp_model.plane.voltage*3600)*100)
        battery_capacity_J = self.mdp_model.plane.capacity*self.mdp_model.plane.voltage*3600
        
        # Preallocate arrays with a fixed size
        state_history_list = np.empty(max_stages,dtype=tuple)  # Adjust dimensions based on the state size
        energy_history_list = np.empty(max_stages)
        solar_power_list = np.empty(max_stages)
        whale_list = np.empty(max_stages)

        # Initialize the first elements
        state_history_list[0] = initial_state
        energy_history_list[0] = initial_state[0] / 100 * battery_capacity_J
        shortwave_radiation = self.data["shortwave_radiation"].values
        solar_power_list[0] = shortwave_radiation[0]
        whale_list[0] = 0.0
        flight_minutes = 0.0
        

        for k in range(max_stages-1):
            current_state = state_history_list[k]
            current_energy = energy_history_list[k]
            solar_power_wpm2 = shortwave_radiation[k]
            whale_prob = self.get_sighting_probability(self.whale_prob,k,self.dt,0)
            is_reward_sufficient = whale_prob>threshold and solar_power_wpm2>0
            is_battery_sufficient = current_state[0] > nightly_idle_soc + single_flight_soc*2 # TODO: Create better way to determine this

            if is_reward_sufficient and is_battery_sufficient:
                best_action = "fly"
                flight_minutes += self.dt
            else :
                best_action = "float"

            if simulate_failure:
                _,failure_prob = self.calculate_maneuver_probabilities(current_state=current_state[1],
                                                                                    action=best_action,
                                                                                    stage=k)
            else:
                failure_prob = -1

            is_action_successful = np.random.uniform(0,1) > failure_prob
            
            new_energy = current_energy + self.calculate_energy_update(self.mdp_model.plane,current_state,best_action,self.dt,solar_power_wpm2)
            new_state = self.calculate_new_state(best_action,new_energy,battery_capacity_J)
            reward+=self.R(current_state,best_action,k,whale_prob)
            state_history_list[k+1]=new_state
            energy_history_list[k+1]=new_energy
            whale_list[k+1]=whale_prob
            solar_power_list[k+1] = solar_power_wpm2
            if not is_action_successful :
                reward -= self.failure_penalty # TODO MAKE whale penalty a parameter
                break
            if new_state[0] < 0 :
                reward = reward-self.failure_penalty
                break
        
        # print(reward)
        if save_history:
            return reward, k, state_history_list[1:k], solar_power_list[1:k], whale_list
        else:
            return reward,k,flight_minutes
        
    def simulate_fullcharge_behavior(self,
                                initial_state,
                                true_success_prob,
                                simulate_failure = False,
                                save_history = False,
                                capacity_threshold = 0.95):

        reward = 0
        max_stages = len(self.data)-1
        # print(f"Threshold = {threshold}\n")

        self.stepwise_failure_prob = 1-true_success_prob
        
        # night_hours = 12
        # nightly_idle_soc = np.ceil((self.mdp_model.plane.idle_power*night_hours*3600)/(self.mdp_model.plane.capacity*self.mdp_model.plane.voltage*3600)*100)
        single_flight_soc = np.ceil((self.mdp_model.plane.get_required_power(20,1.2)*self.dt*60+self.mdp_model.plane.required_takeoff_energy)/(self.mdp_model.plane.capacity*self.mdp_model.plane.voltage*3600)*100)
        battery_capacity_J = self.mdp_model.plane.capacity*self.mdp_model.plane.voltage*3600
        

        state_history_list = np.empty(max_stages,dtype=tuple)  # Adjust dimensions based on the state size
        energy_history_list = np.empty(max_stages)
        solar_power_list = np.empty(max_stages)
        whale_list = np.empty(max_stages)

        # Initialize the first elements
        state_history_list[0] = initial_state
        energy_history_list[0] = initial_state[0] / 100 * battery_capacity_J
        solar_power_list[0] = 0.0
        whale_list[0] = 0.0
        flight_minutes = 0.0
        shortwave_radiation = self.data["shortwave_radiation"].values

        for k in range(max_stages-1):
            current_state = state_history_list[k]
            current_energy = energy_history_list[k]
            solar_power_wpm2 = shortwave_radiation[k]
            whale_prob = self.get_sighting_probability(self.whale_prob,k,self.dt,0)
            # is_reward_sufficient = whale_prob>threshold and solar_power_wpm2>0
            is_battery_sufficient = current_energy > battery_capacity_J*capacity_threshold or (current_state[1] == "flying" and current_state[0] > single_flight_soc)

            if is_battery_sufficient:
                best_action = "fly"
                flight_minutes += self.dt
            else :
                best_action = "float"

            if simulate_failure:
                _,failure_prob = self.calculate_maneuver_probabilities(current_state=current_state[1],
                                                                                    action=best_action,
                                                                                    stage=k)
            else:
                failure_prob = -1

            is_action_successful = np.random.uniform(0,1) > failure_prob
            
            # Update state history
            new_energy = current_energy + self.calculate_energy_update(self.mdp_model.plane,current_state,best_action,self.dt,solar_power_wpm2)
            new_state = self.calculate_new_state(best_action,new_energy,battery_capacity_J)
            reward+=self.R(current_state,best_action,k,whale_prob)
            state_history_list[k+1]=new_state
            energy_history_list[k+1]=new_energy
            whale_list[k+1]=whale_prob
            solar_power_list[k+1] = solar_power_wpm2

            if not is_action_successful or new_state[0] < 0:
                reward -= self.failure_penalty
                break
        
        if save_history:
            return reward, k, state_history_list[1:k], solar_power_list[1:k], whale_list
        else:
            return reward,k,flight_minutes
        


    def simulate_mdp_behavior(self,
                              initial_state,
                              true_success_prob,
                              simulate_failure = False,
                              save_history = False):

        battery_capacity_J = self.mdp_model.plane.capacity*self.mdp_model.plane.voltage*3600
        required_cruise_energy = self.mdp_model.plane.required_cruise_power*self.dt*60
        required_takeoff_energy = self.mdp_model.plane.required_takeoff_energy
        reward = 0
        max_stages = len(self.data)-1
        self.stepwise_failure_prob = 1-true_success_prob

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
        flight_minutes = 0.0
        action_list = ["float","fly"]
        shortwave_radiation = self.data["shortwave_radiation"].values
        
        for k in range(max_stages-1):
            current_state = state_history_list[k]
            current_energy = energy_history_list[k]
            solar_power_wpm2 = shortwave_radiation[k]
            whale_prob = self.get_sighting_probability(self.whale_prob,k,self.dt,0)

            # best_action = self.mdp_model.policy_table.loc[current_state,k]
            collected_energy = self.mdp_model.plane.S*solar_power_wpm2*self.dt*60
            value_list = []
            for action in action_list:
                required_energy = required_cruise_energy if action=="fly" else 0
                if current_state[1] == "moored" and action=="fly":
                    required_energy = required_cruise_energy+required_takeoff_energy
                if action == "fly" :
                    whale_surface_prob = whale_prob
                else :
                    whale_surface_prob = 0
                current_reward = self.current_reward(current_state,action,k,required_energy,current_energy,collected_energy,
                                                     self.failure_penalty,1,whale_surface_prob)
                future_state = self.calculate_next_state(current_state=current_state,action=action,solar_power_wpm2=solar_power_wpm2)
                expected_future_reward = self.mdp_model.ev_table.loc[future_state,k+1]
                value = current_reward + expected_future_reward
                value_list.append(value)

            # TODO: Determine which action to take based on value
            best_action = action_list[np.argmax(value_list)]

            _,failure_prob = self.calculate_maneuver_probabilities(current_state=current_state[1],
                                                                                action=best_action,
                                                                                stage=k)

            if best_action == "fly" :
                flight_minutes += self.dt

            if simulate_failure:
                _,failure_prob = self.calculate_maneuver_probabilities(current_state=current_state[1],
                                                                                    action=best_action,
                                                                                    stage=k)
            else:
                failure_prob = -1

            is_action_successful = np.random.uniform(0,1) > failure_prob
            
            # Update state history
            new_energy = current_energy + self.calculate_energy_update(self.mdp_model.plane,current_state,best_action,self.dt,solar_power_wpm2)
            new_state = self.calculate_new_state(best_action,new_energy,battery_capacity_J)
            reward+=self.R(current_state,best_action,k,whale_prob)
            state_history_list[k+1]=new_state
            energy_history_list[k+1]=new_energy
            whale_list[k+1]=whale_prob
            solar_power_list[k+1] = solar_power_wpm2

            if not is_action_successful or new_state[0] < 0:
                reward -= self.failure_penalty
                break
        
        if save_history:
            return reward, k, state_history_list[1:k], solar_power_list[1:k], whale_list
        else:
            return reward,k, flight_minutes

    

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
        return (soc,state)

    def calculate_next_state(self,current_state,action,solar_power_wpm2):
        soc = current_state[0]
        delta_soc = self.calculate_soc_update(self.mdp_model.plane,current_state,action,self.dt,solar_power_wpm2)
        new_soc = min(soc + delta_soc, 100)  # Limit SoC to 100
        new_vehicle_state = "flying" if action == "fly" else "moored"

        # Set state to "broken" if SoC falls below 0
        if new_soc < 0:
            new_soc, new_vehicle_state = -1, "broken"
        return (new_soc, new_vehicle_state)
    
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
        panel_efficiency = 0.10  # TODO: Update using PVWATTS for more accurate efficiency
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

    def calculate_maneuver_probabilities(self,current_state,action,stage):
        """
        Calculate the success and failure probabilities for the given maneuver, adjusting continuously based on wind speed.
        """
        base_failure_prob = self.stepwise_failure_prob
        # Base probabilities
        if current_state == "moored" and action == "float":
            state_action_factor = 1.0  # High success rate for floating
        elif current_state == "moored" and action == "fly":
            state_action_factor = 15  # Higher failure risk for taking off
        elif current_state == "flying" and action == "float":
            state_action_factor = 15  # Moderate risk for flying to floating
        elif current_state == "flying" and action == "fly":
            state_action_factor = 2  # Low failure risk for continuous flying
        else:
            return 0.0, 1.0  # Default to guaranteed failure
        failure_prob = base_failure_prob * state_action_factor
        success_prob = 1-failure_prob
        return success_prob, failure_prob
    
    def reward(self,state,current_action,stage,collected_solar_energy,current_whale_surface_prob):
        if current_action == "fly":
            whale_finding_reward = 1*current_whale_surface_prob+0*(1-current_whale_surface_prob)
        else:
            whale_finding_reward = 0

    def current_reward(self,current_state,current_action,current_stage,
                       P, C, I, failure_penalty,whale_finding_reward,
                       p_H_1:float)->float:
        """
        Calculate the expected reward E[R(X, H)].
        
        Parameters:
        - P (float): Required energy.
        - C (float): Stored energy.
        - I (float): Collected energy.
        - k (float): Absolute values of penalty for vehicle failure.
        - l (float): Reward if X > 0 and H = 1.
        - alpha (float): Shape parameter for the Beta distribution.
        - beta (float): Shape parameter for the Beta distribution.
        - P_H_1 (float): Probability that whale is at the surface.
        - P_B_1 (float): Probability that B = 1.
        
        Returns:
        - float: Expected reward E[R(X, H)].
        """

        if P > C + I:
            energy_failure_probability = 1
        else:
            energy_failure_probability = 0

        transition_failure_probability = self.mdp_model.T(current_state,current_action,current_stage)
        failure_probability = 1-(1-energy_failure_probability)*(1-transition_failure_probability)

        reward = p_H_1*whale_finding_reward+failure_probability*(-failure_penalty)

        return reward
        

    def decision(self):
        pass
        # if the current reward is greater than the difference between the current and future values in the expected value table, choose to fly.
        # otherwise, choose to do nothing?
        

    
    # @staticmethod
    # def calculate_step_transition_prob(period_min, no_failure_probability, step_length_min):
    #     """
    #     Calculate the stepwise transition probability for each step within a specified period.

    #     This method computes the probability of failure for a single step given the total 
    #     failure probability over a period and the number of steps within that period. 
    #     It ensures that the compounded stepwise failure matches the specified total failure 
    #     probability over the entire period.

    #     Parameters:
    #         period_min (float): The total length of the period in minutes.
    #         failure_probability (float): The overall failure probability for the entire period 
    #             (value between 0 and 1).
    #         step_length_min (float): The length of each step in minutes.

    #     Returns:
    #         float: The stepwise failure probability for each individual step.

    #     Example:
    #         If the total period is 60 minutes with a failure probability of 0.5 and 
    #         step length is 15 minutes, this method returns the stepwise probability for 
    #         each 15-minute interval.

    #     Raises:
    #         ValueError: If any input is non-positive or the failure_probability is not in [0, 1].
    #     """
    #     # Validate inputs
    #     if period_min <= 0:
    #         raise ValueError("Period_min must be a positive number.")
    #     if not (0 <= no_failure_probability <= 1):
    #         raise ValueError("Failure_probability must be between 0 and 1, inclusive.")
    #     if step_length_min <= 0:
    #         raise ValueError("Step_length_min must be a positive number.")

    #     # Calculate the number of steps and the stepwise failure probability
    #     num_steps = np.ceil(period_min / step_length_min)
    #     stepwise_failure_probability = 1 - (no_failure_probability ** (1 / num_steps))
    #     # print(stepwise_failure_probability)
    #     return stepwise_failure_probability
    
    def calculate_energy_update(self, plane, state, action, dt, solar_power):
        """
        Calculates the change in SoC after performing the given action.
        """
        panel_efficiency = 0.10 # TODO: use PVWATTS FOR THIS
        
        required_takeoff_energy=0
        required_cruise_power=0
        
        if action == "float":
            required_cruise_power = 0
        elif action == "fly":
            required_cruise_power = plane.required_cruise_power  # Assumed constants for flight
            if state[1] == "moored":
                required_takeoff_energy = plane.required_takeoff_energy
        else :
            raise ValueError(f"Expected action 'float' or 'fly'. Got {action}.")
        

        avionics_power = plane.idle_power
        net_power = solar_power*panel_efficiency*plane.S - required_cruise_power - avionics_power
        energy_change = net_power * dt * 60  - required_takeoff_energy # Convert power (W) to energy (Joules)
        return energy_change
    
    @staticmethod
    def get_sighting_probability(probability_map, current_step, timestep, start_time):
        # Calculate the current time in minutes
        current_time = (start_time + (current_step * timestep)+60)%1440
        
        # Find the nearest start time by rounding down to the closest 120-minute mark
        nearest_start = (current_time // 120) * 120
        
        # Return the probability, or None if out of range
        return probability_map.get(nearest_start)
    
    @staticmethod
    def expected_reward(P, C, k, l, collected_energy, p_H_1:float, p_B_1:float)->float:
        """
        Calculate the expected reward E[R(X, H)].
        
        Parameters:
        - P (float): Required energy.
        - C (float): Stored energy.
        - k (float): Absolute values of penalty for vehicle failure.
        - l (float): Reward if X > 0 and H = 1.
        - P_H_1 (float): Probability that whale is at the surface.
        - P_B_1 (float): Probability that B = 1.
        
        Returns:
        - float: Expected reward E[R(X, H)].
        """
        # Probability that H = 1
        p_H_0 = 1 - p_H_1
        p_B_0 = 1 - p_B_1
        
        # If no energy is collected, handle the penalty based on stored energy
        if C + collected_energy < P:
            # If stored energy is insufficient to meet required energy, apply penalty
            F_S = 1
        else:
            # If stored energy + collected energy is sufficient, no penalty
            F_S = 0

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

import numpy as np
from BaseClasses.mdp import mdp
import random

class Autonomy:
    """Represents the autonomy module for a solar-powered seaplane."""

    def __init__(self,dt,actual_solar_power,whale_probabilities):
        self.solar_power_actual = actual_solar_power
        self.whale_prob_table = whale_probabilities
        self.dt = dt

    def simulate_simple_behavior(self,
                                 plane,
                                 soc_increment,
                                 start_index,
                                 end_index,
                                 max_stages,
                                 initial_state,
                                 actual_solar_power,
                                 avail_wind_mag,
                                 whale_probabilities,
                                 no_fail_prob,
                                 simulate_failure = False):
        """
        Simulates simple plane behavior over time with random failures during state transitions.

        Parameters:
        solar_power (pd.Series): Solar power available at each time step.
        is_daytime (pd.Series): Boolean series indicating daytime (True) or nighttime (False).
        cruise_power (float): Power required for cruising.
        battery_capacity (float): Total battery energy capacity in joules.
        landing_threshold (float): Battery fraction at which the plane must land.
        takeoff_threshold (float): Battery fraction required for takeoff.
        timestep_minutes (float): Simulation time step in minutes.
        min_flight_minutes (float): Minimum flight time after takeoff.
        takeoff_penalty_fn (function): Function to compute energy penalty for takeoff.

        Returns:
        tuple: duty_cycle, energy_history, state_history, num_takeoffs, failure_occurred
        """

        vehicle_states = ["moored", "flying"]
        actions = ["float", "fly"]
        reward = 0



        mdp_model = mdp(plane,
                        soc_increment,
                        vehicle_states,
                        max_stages,
                        actions,
                        start_index=start_index,
                        end_index=end_index,
                        whale_prob=whale_probabilities,
                        dt=60,
                        no_failure_prob=no_fail_prob
                        )
        
        state_history_list = [initial_state]
        solar_power_list = [0.0]

        for k in range(len(actual_solar_power)-1):
            current_state = state_history_list[-1]
            solar_power = actual_solar_power.iloc[k][0]
            if mdp_model.is_action_feasible("fly",current_state,k,solar_power) and self.R(current_state,"fly",k)>25 and current_state[0] > 30 :
                best_action = "fly"
            else :
                best_action = "float"

            if simulate_failure:
                success_prob,failure_prob = mdp_model.calculate_maneuver_probabilities(current_state=current_state[1],
                                                                                    action=best_action,
                                                                                    stage=k)
            else:
                failure_prob = -1
            if np.random.uniform(0,1) > failure_prob and mdp_model.is_action_feasible(best_action,current_state,k,solar_power) :
                new_state = mdp_model.calculate_new_state(state=current_state,
                                        action=best_action,
                                        stage=k,
                                        solar_power=solar_power)
                reward+=mdp_model.R(current_state,best_action,k)

                state_history_list.append(new_state)
                solar_power_list.append(solar_power)
            else:
                print("Failure!")
                break

        return state_history_list,solar_power_list,reward


    def simulate_mdp_behavior(self,
                              plane,
                              soc_increment,
                              start_index,
                              end_index,
                              max_stages,
                              initial_state,
                              actual_solar_power,
                              avail_wind_mag,
                              whale_probabilities,
                              no_fail_prob,
                              simulate_failure = False):
        """
        Simulates plane behavior using an MDP to determine the optimal flight policy.

        Parameters:
        plane: The plane object containing relevant attributes like battery and power.
        soc_increment (int): State of charge increment in percentages.
        max_stages (int): Number of stages or time steps in the simulation.
        initial_state (tuple): Starting state as (SoC, vehicle_state).
        solar_power (pd.Series): Solar power available at each time step.
        is_daytime (pd.Series): Boolean series indicating daytime (True) or nighttime (False).

        Returns:
        tuple: duty_cycle, energy_history, state_history, num_takeoffs
        """
        vehicle_states = ["moored", "flying"]
        actions = ["float", "fly"]
        reward = 0



        mdp_model = mdp(plane,
                        soc_increment,
                        vehicle_states,
                        max_stages,
                        actions,
                        start_index=start_index,
                        end_index=end_index,
                        whale_prob=whale_probabilities,
                        dt=60,
                        no_failure_prob=no_fail_prob
                        )
        state_history_list = [initial_state]
        solar_power_list = [0.0]

        mdp_model.create_ev_table()
        # mdp_model.value_iteration()
        
        optimal_policy = mdp_model.policy_table

        for k in range(len(actual_solar_power)-1):
            current_state = state_history_list[-1]
            best_action = optimal_policy.loc[current_state,k]
            solar_power = actual_solar_power.iloc[k][0]

            if simulate_failure :
                success_prob,failure_prob = mdp_model.calculate_maneuver_probabilities(current_state=current_state[1],
                                                                                    action=best_action,
                                                                                    stage=k)
            else:
                failure_prob = -1

            if np.random.uniform(0,1) > failure_prob and mdp_model.is_action_feasible(best_action,current_state,k,solar_power) :
                new_state = mdp_model.calculate_new_state(state=current_state,
                                                        action=best_action,
                                                        stage=k,
                                                        solar_power=solar_power)
                reward+=self.R(current_state,best_action,k)
                state_history_list.append(new_state)
                solar_power_list.append(solar_power)
            else :
                reward = reward-50
                print("Failure!")
                break
        return state_history_list,solar_power_list,reward
    

    def R(self,state,action,stage):
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
    
        minutes = (self.dt * stage) % 1440
        whale_prob = self.whale_prob_table.loc[minutes // 120]["Sighting Probability"]
        whale_reward = 0

        # Determine whale sighting probability based on time of day
        if self.solar_power_actual.iloc[stage][0]>0:
            # Calculate rewards based on the action
            if action == 'float':
                pass
            elif action == 'fly':
                # Whale reward based on probability of finding whale during survey, assume 5 for now
                if np.random.uniform(0,1) < whale_prob:
                    whale_reward = 100
        else: # Night time
            if action == 'float':
                pass
            elif action == 'fly':
                pass

        return whale_reward


import numpy as np
from BaseClasses.mdp import mdp
import random

class Autonomy:
    """Represents the autonomy module for a solar-powered seaplane."""

    def __init__(self):
        pass

    def simulate_simple_behavior(self,
                                 plane,
                                 soc_increment,
                                 start_index,
                                 end_index,
                                 max_stages,
                                 initial_state,
                                 actual_solar_power,
                                 avail_wind_mag,
                                 whale_probabilities):
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
                        dt=60
                        )
        
        state_history_list = [initial_state]
        solar_power_list = [0.0]

        for k in range(len(actual_solar_power)-1):
            current_state = state_history_list[-1]
            solar_power = actual_solar_power.iloc[k][0]
            if mdp_model.is_action_feasible("fly",current_state,k,solar_power) and mdp_model.is_daytime(0,mdp_model.dt,k):
                best_action = "fly"
            else :
                best_action = "float"

            success_prob,failure_prob = mdp_model.calculate_maneuver_probabilities(current_state=current_state,
                                                                                   action=best_action,
                                                                                   stage=k)
            if np.random.uniform(0,1) > failure_prob and not  mdp_model.is_action_feasible(best_action,current_state,k,solar_power) :
                new_state = mdp_model.calculate_new_state(state=current_state,
                                        action=best_action,
                                        stage=k,
                                        solar_power=solar_power)
                reward+=mdp_model.R(current_state,best_action,k)

                state_history_list.append(new_state)
                solar_power_list.append(solar_power)
            else:
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
                              whale_probabilities):
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
                        dt=60
                        )
        
        state_history_list = [initial_state]
        solar_power_list = [0.0]
        # energy_history = [initial_state[0]]
        # state_history = [1 if initial_state[1] == "Flying" else 0]
        mdp_model.value_iteration()
        optimal_policy = mdp_model.policy_table

        for k in range(len(actual_solar_power)-1):
            current_state = state_history_list[-1]
            best_action = optimal_policy.loc[current_state,k]
            solar_power = actual_solar_power.iloc[k][0]
            success_prob,failure_prob = mdp_model.calculate_maneuver_probabilities(current_state=current_state,
                                                                                   action=best_action,
                                                                                   stage=k)
            if np.random.uniform(0,1) > failure_prob :
                new_state = mdp_model.calculate_new_state(state=current_state,
                                                        action=best_action,
                                                        stage=k,
                                                        solar_power=solar_power)
                reward+=mdp_model.R(current_state,best_action,k)
                state_history_list.append(new_state)
                solar_power_list.append(solar_power)
            else :
                break
        return state_history_list,solar_power_list,reward

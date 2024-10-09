import numpy as np
from BaseClasses.mdp import mdp
import random

class Autonomy:
    """Represents the autonomy module for a solar-powered seaplane."""

    def __init__(self):
        pass

    def simulate_simple_behavior(self, solar_power, is_daytime, cruise_power, battery_capacity, landing_threshold, takeoff_threshold, timestep_minutes, min_flight_minutes, takeoff_penalty_fn):
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
        state = "Moored"
        energy_joules = battery_capacity
        energy_history = []
        state_history = []
        num_takeoffs = 0
        flight_time = 0
        failure_occurred = False

        for i in range(len(solar_power)):
            if state == "Flying":
                # Flying state logic
                flight_time += timestep_minutes / 60
                energy_joules -= (cruise_power - solar_power.iloc[i]) * timestep_minutes * 60

                # Transition to Moored state if energy is too low or it's nighttime
                if energy_joules <= landing_threshold * battery_capacity or not is_daytime[i]:
                    # Check if the transition to 'moored' fails
                    if random.random() > 0.95:  # Failure probability for "flying" -> "moored"
                        failure_occurred = True
                        break  # End simulation on failure
                    state = "Moored"
                state_history.append(1)

            elif state == "Moored":
                # Moored state logic
                energy_joules = min(energy_joules + solar_power.iloc[i] * timestep_minutes * 60, battery_capacity)

                # Check if conditions for takeoff are met
                if energy_joules >= takeoff_threshold * battery_capacity and is_daytime[i]:
                    # Check for flight feasibility based on available energy and flight time
                    if energy_joules >= cruise_power * 60 * min_flight_minutes:
                        # Check if the transition to 'flying' fails
                        if random.random() > 0.95:  # Failure probability for "moored" -> "flying"
                            failure_occurred = True
                            break  # End simulation on failure

                        # Take off
                        state = "Flying"
                        energy_joules -= takeoff_penalty_fn() + (cruise_power - solar_power.iloc[i]) * timestep_minutes * 60
                        num_takeoffs += 1
                state_history.append(0)

            energy_history.append(energy_joules / battery_capacity * 100)

        # Handle the failure case by filling remaining steps with -10 for energy and -1 for state
        if failure_occurred:
            remaining_steps = len(solar_power) - (i + 1)  # Ensure correct number of remaining steps
            energy_history.extend([-10] * remaining_steps)
            state_history.extend([-1] * remaining_steps)
            energy_history.append(-10)
            state_history.append(-1)
            print("Failure")

        total_daytime_minutes = sum(is_daytime) * timestep_minutes
        duty_cycle = (flight_time / (total_daytime_minutes / 60) * 100) if total_daytime_minutes > 0 else 0

        return duty_cycle, energy_history, state_history, num_takeoffs, failure_occurred


    def simulate_mdp_behavior(self,
                              plane,
                              soc_increment,
                              max_stages,
                              initial_state,
                              expected_solar_power,
                              actual_solar_power,
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

        mdp_model = mdp(plane,
                        soc_increment,
                        vehicle_states,
                        max_stages,
                        actions,
                        expected_solar_power,
                        whale_probabilities,
                        )
        
        state_history_list = [initial_state]
        # energy_history = [initial_state[0]]
        # state_history = [1 if initial_state[1] == "Flying" else 0]
        mdp_model.value_iteration()
        optimal_policy = mdp_model.policy_table

        for k in range(len(actual_solar_power)):
            current_state = state_history_list[-1]
            best_action = optimal_policy[current_state]

            new_state = mdp_model.calculate_new_state(state=current_state,
                                                      action=best_action,
                                                      stage=k)
            state_history_list.append(new_state)
        return True

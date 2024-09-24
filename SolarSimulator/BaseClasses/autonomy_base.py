import numpy as np
from BaseClasses.mdp import mdp

class Autonomy:
    """Class representing a the autonomy module for a seaplane"""
    def __init__(self):
        pass

    def simple_plane_behavior(self, P_solar, is_daytime, P_cruise, capacity_j, landing_capacity, takeoff_capacity, dt, min_flight_hr, calc_takeoff_penalty):
        """
        Simulates the behavior of a solar-powered plane over time.

        Parameters:
        P_solar (pd.Series): A pandas Series representing the solar power available at each time step.
        is_daytime (pd.Series): A pandas Series indicating whether it is daytime (True) or nighttime (False) at each time step.
        P_cruise (float): The power required to cruise the plane.
        capacity_j (float): The energy capacity of the plane's battery in joules.
        landing_capacity (float): The battery capacity threshold at which the plane needs to land, expressed as a fraction of capacity_j.
        takeoff_capacity (float): The battery capacity threshold required for the plane to take off, expressed as a fraction of capacity_j.
        dt (float): The time step size in hours.
        min_flight_hr (float): The minimum number of hours the plane must be able to fly after taking off.
        calc_takeoff_penalty (function): A function that calculates the energy penalty for taking off.

        Returns:
        tuple: A tuple containing the following elements:
            dc (float): The duty cycle, expressed as the percentage of time the plane spends flying during daytime.
            energy_history (list): A list representing the percentage of battery capacity over time.
            state_history (list): A list representing the state history over time, where 1 indicates flying and 0 indicates moored.
            num_takeoff (int): The number of takeoffs performed by the plane.
        """
        state = "Moored"
        energy_j = capacity_j
        state_history = []
        energy_history = []
        num_takeoff = 0
        flying = 0

        for i in range(len(P_solar)):
            if state == "Flying":
                state_history.append(1)
                flying += 1
                energy_j -= (P_cruise - P_solar.iloc[i]) * dt * 60
                if energy_j <= capacity_j * landing_capacity or not is_daytime[i]:
                    state = "Moored"
            elif state == "Moored":
                state_history.append(0)
                if energy_j <= capacity_j:
                    energy_j += P_solar.iloc[i] * dt * 60
                if energy_j >= takeoff_capacity * capacity_j and is_daytime[i]:
                    if energy_j > P_cruise * 60 * 60 * min_flight_hr:
                        state = "Flying"
                        energy_j -= calc_takeoff_penalty()
                        energy_j -= (P_cruise - P_solar.iloc[i]) * dt * 60
                        num_takeoff += 1
            if energy_j > capacity_j:
                energy_j = capacity_j
            energy_history.append(energy_j / capacity_j * 100)

        total = sum(is_daytime)
        if total == 0.0:
            dc = 0
        else:
            dc = flying / total * 100

        return dc, energy_history, state_history, num_takeoff
    
    # def mdp_behavior(self,plane,soc_increment,max_stages,start_state,P_solar, is_daytime):
    #     """
    #     Returns:
    #     tuple: A tuple containing the following elements:
    #         dc (float): The duty cycle, expressed as the percentage of time the plane spends flying during daytime.
    #         energy_history (list): A list representing the percentage of battery capacity over time.
    #         state_history (list): A list representing the state history over time, where 1 indicates flying and 0 indicates moored.
    #         num_takeoff (int): The number of takeoffs performed by the plane.
    #     """
    #     vehicle_states = ["moored", "flying"]
    #     actions = ["float", "fly"]
    #     stm = [0, 0, 0, 0]

    #     mdp_instance = mdp(plane,soc_increment, vehicle_states, max_stages, actions, stm)
    #     state_list = [start_state]

    #     for k in range(max_stages):
    #         state = state_list[-1]
    #         reward = -np.inf
            
    #         # Choose which action to take
    #         for a in actions:
    #             if mdp_instance.is_action_feasible(a,state,k):
    #                 control_reward = mdp_instance.get_control_reward(a,w,state,k)
    #                 future_reward = mdp_instance.get_future_reward(state,a,k,w)
    #                 v = control_reward + future_reward
    #                 if v > reward :
    #                     reward = v
    #                     new_state = 
            
    #         # Calculate effect of that action
    #         state_list.append(new_state)

    #     return dc, energy_history, state_history, num_takeoff

    def mdp_behavior(self, plane, soc_increment, max_stages, start_state, P_solar, is_daytime):
        """
        Simulates the behavior of a solar-powered plane using MDP to determine the optimal policy.

        Parameters:
        plane: The plane object with necessary attributes such as voltage and capacity.
        soc_increment (int): The increment of state of charge (SoC) in percentages.
        max_stages (int): The number of stages (or time steps) for the simulation.
        start_state (tuple): The initial state (SoC, vehicle_state).
        P_solar (pd.Series): A pandas Series representing the solar power available at each time step.
        is_daytime (pd.Series): A pandas Series indicating whether it is daytime (True) or nighttime (False) at each time step.

        Returns:
        tuple: A tuple containing the following elements:
            dc (float): The duty cycle, expressed as the percentage of time the plane spends flying during daytime.
            energy_history (list): A list representing the percentage of battery capacity over time.
            state_history (list): A list representing the state history over time, where 1 indicates flying and 0 indicates moored.
            num_takeoff (int): The number of takeoffs performed by the plane.
        """
        vehicle_states = ["moored", "flying"]
        actions = ["float", "fly"]
        stm = [0, 0, 0, 0]

        # Create an instance of the MDP class
        mdp_instance = mdp(plane, soc_increment, vehicle_states, max_stages, actions, stm)

        state_list = [start_state]
        energy_history = [start_state[0]]
        state_history = [1 if start_state[1] == "flying" else 0]
        num_takeoff = 0
        flying = 0

        # Loop through the stages to calculate optimal actions based on the EV table
        for k in range(max_stages-1):
            current_state = state_list[-1]
            max_reward = -np.inf
            best_action = None

            w = mdp_instance.is_daytime(mdp_instance.start_time, mdp_instance.dt, k)

            # Determine the optimal action at this stage
            for action in actions:
                if mdp_instance.is_action_feasible(action, current_state, k):
                    control_reward = mdp_instance.get_control_reward(action, w, current_state, k)
                    future_reward = mdp_instance.get_future_reward(current_state, action, k, w)
                    total_reward = control_reward + future_reward
                    
                    if total_reward > max_reward:
                        max_reward = total_reward
                        best_action = action

            # Perform the chosen action and update the state
            soc_update = mdp_instance.calculate_soc_update(
                plane, best_action, mdp_instance.dt, k,
                False,solar_power=P_solar.iloc[k], soc_increment=mdp_instance.soc_increment
            )
            new_soc = current_state[0] + soc_update
            new_state = (new_soc, "flying" if best_action == "fly" else "moored")

            # Ensure the new state is valid and update histories
            if new_soc <= 100 and new_soc >= 0:
                state_list.append(new_state)
                energy_history.append(new_soc)
                state_history.append(1 if new_state[1] == "flying" else 0)
                
                if new_state[1] == "flying":
                    flying += 1
                if best_action == "fly" and current_state[1] == "moored":
                    num_takeoff += 1
            else:
                break  # Exit if the SoC becomes invalid (e.g., battery drained)

        # Calculate duty cycle (percentage of time spent flying during daytime)
        total_daytime = sum(is_daytime)
        dc = (flying / total_daytime * 100) if total_daytime > 0 else 0

        return dc, energy_history, state_history, num_takeoff


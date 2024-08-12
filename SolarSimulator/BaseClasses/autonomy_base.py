
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
                if energy_j <= capacity_j * landing_capacity or not is_daytime.iloc[i]:
                    state = "Moored"
            elif state == "Moored":
                state_history.append(0)
                if energy_j <= capacity_j:
                    energy_j += P_solar.iloc[i] * dt * 60
                if energy_j >= takeoff_capacity * capacity_j and is_daytime.iloc[i]:
                    if energy_j > P_cruise * 60 * 60 * min_flight_hr:
                        state = "Flying"
                        energy_j -= calc_takeoff_penalty()
                        energy_j -= (P_cruise - P_solar.iloc[i]) * dt * 60
                        num_takeoff += 1
            if energy_j > capacity_j:
                energy_j = capacity_j
            energy_history.append(energy_j / capacity_j * 100)

        total = is_daytime.sum()
        if total == 0.0:
            dc = 0
        else:
            dc = flying / total * 100

        return dc, energy_history, state_history


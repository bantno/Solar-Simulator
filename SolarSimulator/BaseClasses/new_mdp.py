import sys
import numpy as np
import time
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.stats import beta as betaDist
from scipy.stats import weibull_min
from scipy.integrate import quad, simpson
sys.path.append(r"C:\Users\brian\OneDrive\Documents\Georgia Tech\Research\Whale Plane\SolarSim\SolarSimulator\BaseClasses")
from seaplane_base import Seaplane

class ExpectedValueTable:
    def __init__(self,plane: Seaplane,expected_solar_data,expected_wind_data, whale_observation_data, soc_increment:int,timestep_min: int):
        solar_panel_efficiency = 0.1
        self.plane = plane
        self.battery_capacity_wh = self.plane.capacity*self.plane.voltage
        self.max_collected_power = 1367*solar_panel_efficiency*plane.S
        self.dt = timestep_min
        self.soc_increment = soc_increment
        self.expected_solar = expected_solar_data
        self.expected_wind = expected_wind_data
        self.states = self._create_states(soc_increment,[0,1])

        if 100 % soc_increment != 0:
            raise ValueError("Specified state of charge increment does not divide evenly into 100%.")
        else:
            self.ev_table = np.zeros((int(2*(100/soc_increment+1)+1),expected_solar_data.shape[0]))
        self.whale_probability_data = whale_observation_data

    
    def _create_states(self, soc_increment: int, vehicle_states: list) -> list:
        """
        Generate a list of states based on state of charge (SoC) increments and vehicle states.
        """
        states = [(soc, state) for state in vehicle_states for soc in range(0, 101, soc_increment)]
        states.append((-1,2))
        return states

    def generate_ev_table(self):
        for k in tqdm(range(self.ev_table.shape[1]-1,-1,-1)):
            for idx,state in enumerate(self.states[:-1]):
                self.ev_table[idx,k] = self._ev_entry(k,state)

        # self.plot_surface(self.ev_table)
    
    def _calculate_case_probabilities(self,stage,state,reward_k):
        """
        Calculate the probabilities of different energy and reward sufficiency cases.

        This method computes the probabilities associated with four cases of 
        energy and reward sufficiency based on solar energy availability, current 
        state of charge, and required energy.

        Args:
            stage (int): The current stage in the decision-making process.
            state (tuple): The current state of the system, typically including
                the state of charge (SOC) as the first element.
            reward_k (float): The reward threshold used to assess sufficiency.

        Returns:
            tuple: A tuple containing:
                - probabilities (tuple): A tuple of four probabilities (p0, p1, p2, p3) 
                where:
                    p0: Probability of insufficient solar and insufficient reward.
                    p1: Probability of insufficient solar and sufficient reward.
                    p2: Probability of sufficient solar and insufficient reward.
                    p3: Probability of sufficient solar and sufficient reward.
                - alpha_u_0 (float): The updated alpha parameter for insufficient reward.
                - alpha_u_1 (float): The updated alpha parameter for sufficient reward.

        Notes:
            - Solar energy availability is modeled using a Beta distribution
            parameterized by `alpha_k` and `beta_k`.
            - Required energy, collected energy, and current state of charge
            are converted to Joules for consistency.
            - The method uses helper functions to calculate probabilities for 
            sufficient solar energy and sufficient reward.
        """
        alpha_k = self.expected_solar[stage,0]
        beta_k  = self.expected_solar[stage,1]
        max_collected_energy_J = self.max_collected_power*self.dt*60
        current_energy_J = self.soc_to_joules(state[0])
        required_energy_J = self._calculate_required_energy(state,action=1)
        p_sufficient_solar = self._calculate_sufficient_solar_probability(required_energy_J,current_energy_J,max_collected_energy_J,alpha_k,beta_k)
        p_sufficient_reward,alpha_u_0,alpha_u_1 = self._calculate_sufficient_reward_probability(stage,state,reward_k,alpha_k,beta_k)
        
        p0 = (1-p_sufficient_solar)*(1-p_sufficient_reward)
        p1 = (1-p_sufficient_solar)*(p_sufficient_reward)
        p2 = (p_sufficient_solar)*(1-p_sufficient_reward)
        p3 = (p_sufficient_solar)*(p_sufficient_reward)

        return (p0,p1,p2,p3),alpha_u_0,alpha_u_1

    def _calculate_sufficient_solar_probability(self,required_energy,current_energy,max_collected_energy, alpha, beta):
        """
        Calculate the probability of having sufficient solar energy to meet requirements.

        This method computes the probability that the collected solar energy will 
        be sufficient to meet the energy shortfall based on the system's current 
        state and a Beta distribution model for solar energy availability.

        Args:
            required_energy (float): The energy required to complete the task (in Joules).
            current_energy (float): The current stored energy (in Joules).
            max_collected_energy (float): The maximum energy that can be collected 
                during the time interval (in Joules).
            alpha (float): The alpha parameter of the Beta distribution modeling solar energy.
            beta (float): The beta parameter of the Beta distribution modeling solar energy.

        Returns:
            float: The probability of having sufficient solar energy, 
            computed as 1 - F_S, where F_S is the cumulative distribution function 
            (CDF) of the Beta distribution evaluated at the calculated threshold.

        Notes:
            - The threshold represents the normalized shortfall in energy required to meet 
            the demand, scaled by the maximum possible collected energy.
            - The Beta distribution is used to model the variability in solar energy collection.
        """
        threshold = ((required_energy-current_energy) / max_collected_energy) # Calculate the threshold for X <= 0 condition (S <= (P - C) / I)
        F_S = betaDist.cdf(threshold, alpha, beta)
        return 1-F_S
    
    def _calculate_sufficient_reward_probability(self,stage,state,reward_k,alpha_k,beta_k,n=1000):
        """
        Calculate the probability of obtaining sufficient reward for a given action.

        This method estimates the probability that the reward difference between 
        two actions will exceed a specified threshold, based on sampled solar power 
        scenarios modeled using a Beta distribution.

        Args:
            stage (int): The current stage in the decision-making process.
            state (tuple): The current state of the system.
            reward_k (float): The reward threshold to evaluate sufficiency.

        Returns:
            tuple: A tuple containing:
                - p_sufficient_reward (float): The probability that the reward 
                difference (d_alpha) is greater than or equal to `reward_k`.
                - mean_ev_0 (float): The mean value of the alpha function for action 0.
                - mean_ev_1 (float): The mean value of the alpha function for action 1.

        Notes:
            - Samples are drawn from a Beta distribution parameterized by `alpha_k` 
            and `beta_k` to simulate solar power variability.
            - The `_alpha` method computes the reward for a given action under the 
            sampled solar power conditions.
            - The method evaluates the proportion of samples where the reward 
            difference `d_alpha` is greater than or equal to the threshold `reward_k`.
        """
        samples = np.random.beta(alpha_k,beta_k,size=n)*self.max_collected_power
        
        ev_0 = np.array(self._alpha(stage, state, 0, samples))
        ev_1 = np.array(self._alpha(stage, state, 1, samples))

        d_alpha = ev_0-ev_1
        p_sufficient_reward = np.mean(reward_k >= d_alpha)

        return p_sufficient_reward,np.mean(ev_0),np.mean(ev_1)

    def _calculate_required_energy(self,state,action):
        """
        Calculate the energy required for a given action.

        This method computes the energy (in Joules) required based on the specified 
        action and the current state of the system.

        Args:
            state (tuple): The current state of the system, where `state[1]` indicates 
                whether the plane is 0 or in another state.
            action (int): The action to evaluate:
                - 0: Idle.
                - 1: Cruise (with potential takeoff if moored).

        Returns:
            float: The required energy for the specified action (in Joules).

        Raises:
            ValueError: If an invalid action is specified.

        Notes:
            - Energy calculations are based on power requirements and the time step 
            duration (`self.dt`), converted to seconds.
            - For action 1, if the plane is "moored," additional energy for takeoff 
            is added to the cruise power requirement.
        """
        required_energy_J = 0
        timestep_s = self.dt*60
        if action == 0 :
            required_energy_J += self.plane.idle_power*timestep_s
        elif action == 1 :
            required_energy_J += self.plane.required_cruise_power*timestep_s
            if state[1] == 0:
                required_energy_J += self.plane.required_takeoff_energy
        else:
            raise ValueError("Invlaid action specified.")
        
        return required_energy_J

    def _alpha(self,stage:int,state:tuple,action:int,solar_power_w):
        """
        Compute the expected value for a given action and state.

        This method calculates the expected value of taking a specific action 
        from the current state, considering the resulting next state and the 
        associated expected value from a lookup table.

        Args:
            state (tuple): The current state of the system.
            action (int): The action to evaluate.
            solar_power_w (float): The solar power collected by the plane's solar array (in Watts).

        Returns:
            float: The expected value for the specified action and state.

        Notes:
            - The next state is determined using the `calculate_next_state` method, 
            which incorporates the effects of the action and available solar power.
            - The expected value for the next state is retrieved from a precomputed 
            lookup table (`self.ev_table`) for the subsequent stage (`stage + 1`).
        """

        next_state = self.calculate_next_state(state,action,solar_power_w)
        ev = self.lookup_expected_value(self.ev_table,stage+1,next_state)
        # print(state_time-start_time,time.time()-state_time)
        return ev
    
   
    def soc_to_joules(self,soc):
        """
        Convert state of charge (SOC) to energy in Joules.

        This method converts the state of charge (SOC) percentage into the 
        equivalent energy stored in the battery, expressed in Joules.

        Args:
            soc (float): The state of charge as a percentage (0 to 100).

        Returns:
            int: The equivalent energy stored in the battery (in Joules).

        Notes:
            - The conversion uses the battery capacity in watt-hours (`self.battery_capacity_wh`) 
            and converts it to Joules (1 watt-hour = 3600 Joules).
            - The result is returned as an integer.
        """
        joules = soc/100*self.battery_capacity_wh*3600
        return int(joules)
    
    def calculate_next_state(self, state, action, solar_power_w):
        """
        Calculate the next state of the system based on the current state, action, and solar power.

        This method determines the updated state of charge (SOC) and the vehicle's 
        state after performing a specified action, taking into account the time step 
        duration and available solar power.

        Args:
            state (tuple): The current state of the system, where:
                - `state[0]` represents the SOC as a percentage (0 to 100).
                - `state[1]` represents the vehicle's current state (1, 0, or 2).
            action (str): The action to perform ("fly" or "float").
            solar_power_w (float): The solar power collected by the plane's solar array (in Watts).

        Returns:
            tuple: The next state of the system as:
                - new_soc (float): The updated SOC (limited to a maximum of 100, 
                or set to -1 if the SOC falls below 0, indicating a 2 state).
                - new_vehicle_state (str): The updated vehicle state (1, 0, or 2).

        Raises:
            ValueError: If an invalid action is specified.

        Notes:
            - The SOC update is calculated using `_calculate_soc_update`, which incorporates 
            energy consumption and solar power input over the time step (`self.dt`).
            - The vehicle state transitions based on the action and current state:
                - 2 state is maintained if already broken or if the SOC falls below 0.
                - "fly" action transitions the vehicle to 1.
                - "float" action transitions the vehicle to 0.
        """
        soc = state[0]

        # Handle broken state upfront
        if state[1] == 2:
            return (-1, 2)

        # Calculate SoC update
        delta_soc = self._calculate_soc_update(self.plane, state, action, self.dt, solar_power_w)
        new_soc = np.minimum(soc + delta_soc, 100)  # Limit SoC to 100

        # Determine new vehicle state
        if action == 1:
            new_vehicle_state = 1
        elif action == 0:
            new_vehicle_state = 0
        else:
            raise ValueError(f"Invalid action. Expected action 0 or 1, got {action}.")

        # Set state to 2 if SoC falls below 0, and handle the vehicle state update
        new_soc, new_vehicle_states = np.where(new_soc < 0, -1, new_soc), np.where(new_soc < 0, 2, new_vehicle_state)

        if new_soc.ndim == 0:
            return (new_soc, new_vehicle_states)

        # Return updated states as a list of tuples
        return np.column_stack((new_soc, new_vehicle_states))

    def _calculate_soc_update(self, plane, state, action, dt, solar_power):
        """
        Vectorized version of _calculate_soc_update to compute SOC changes for multiple solar_power values.

        Args:
            plane (object): The plane object containing operational parameters such as 
                required cruise power, takeoff energy, idle power, and wing area (`S`).
            state (tuple): The current state of the system, where `state[1]` indicates 
                whether the plane is 0 or in another state.
            action (np.ndarray): Array of actions (0 for "float", 1 for "fly") of shape (n,).
            solar_power (np.ndarray): Collected solar power values (in Watts) of shape (n, 1).
            dt (float): The duration of the time step (in minutes).

        Returns:
            np.ndarray: Array of SOC changes as percentages, rounded to the nearest SOC increment, shape (n,).
        """
        # start_time = time.time()
        # Ensure solar_power is a numpy array with correct dimensions
        solar_power = np.atleast_2d(solar_power).flatten()  # Ensure 1D array

        # Initialize constants
        required_takeoff_energy = plane.required_takeoff_energy if state[1] == 0 and action == 1 else 0

        # Map actions to required power
        required_power = plane.required_cruise_power if action == 1 else 0

        # Calculate net power balance
        avionics_power = plane.idle_power
        solar_input = solar_power
        net_power = solar_input - required_power - avionics_power

        # Convert power (W) to energy (Joules) and then to change in SoC (%)
        energy_change = net_power * dt * 60 - required_takeoff_energy  # Power to energy
        soc_change = (energy_change / (self.battery_capacity_wh * 3600)) * 100  # Energy to SoC %

        # Round to the nearest SoC increment and return
        rounded_soc_change = self.soc_increment * np.round(soc_change / self.soc_increment)
        # print(time.time()-start_time)
        return rounded_soc_change
    
    def _P_S_given_w(self, w, u_k, state, p_f=1.0):
        """
        Compute the conditional probability P(S|w_k) based on the state and wind speed.
        
        Parameters:
        - w (float): Wind speed.
        - u_k (int): Control input (1 for active action, 0 for passive action).
        - state (tuple): State of vehicle at stage k.
        - p_f (float): Failure probability (default: 0.1 for floating mode).
        
        Returns:
        - float: Probability of success P(S|w_k).
        """
        if state[1] == 0:
            x_2 = 0
        elif state[1] == 1:
            x_2 = 1
        elif state[1] == 2:
            return 0
        else:
            raise ValueError("Invalid state.")
        
        if u_k == 1 and x_2 == 0:  # Takeoff
            return 1 - 1 / (1 + np.exp(15 - 0.35 * w))
        elif u_k == 0 and x_2 == 0:  # Floating
            return 1 - np.full_like(w,p_f)
        elif u_k == 1 and x_2 == 1:  # Flying
            return 1 - np.full_like(w,p_f)
        elif u_k == 0 and x_2 == 1:  # Landing
            return 1 - 1 / (1 + np.exp(10 - 0.35 * w))
        else:
            raise ValueError("Invalid combination of u_k and x_2.")

    def f_W(self, w, c_k, scale_k):
        """
        Compute the Weibull distribution PDF for a given wind speed.
        
        Parameters:
        - w (float): Wind speed.
        
        Returns:
        - float: Probability density f_W(w).
        """
        return (c_k / scale_k) * (w / scale_k)**(c_k - 1) * np.exp(-(w / scale_k)**c_k)
        
    def _compute_success_probability(self, u_k, state_k,c_k,scale_k):
        """
        Compute the overall probability P(S) by integrating P(S|w_k) * f_W(w).
        
        Parameters:
        - u_k (int): Control input (1 for active action, 0 for passive action).
        - state_k (tuple): Mode (0 for ground/floating, 1 for air/flying).
        
        Returns:
        - float: Overall probability of success.
        """
        # Define the integrand
        def integrand(w):
            return self._P_S_given_w(w, u_k, state_k,p_f=0.0) * self.f_W_vectorized(w,c_k,scale_k)
        
        # Integrate over the domain of the Weibull distribution [0, ∞)

        x = np.linspace(0.001, 45, 501) 
        y = integrand(x)
        result = simpson(x=x,y=y)
        return result
        

    def _ev_entry(self,k,state):

        reward_k = self.whale_probability_data[k]*1
        wind_shape_k = self.expected_wind[k,0]
        wind_scale_k = self.expected_wind[k,1]

        probabilities,alpha_u_0,alpha_u_1 = self._calculate_case_probabilities(k,state,reward_k)
        
        p_4 = probabilities[3]
        
        p_success_u_0 = self._compute_success_probability(0,state,wind_shape_k,wind_scale_k)
        p_success_u_1 = self._compute_success_probability(1,state,wind_shape_k,wind_scale_k)
        
        E_J_k = (1-p_4)*(p_success_u_0)*(alpha_u_0) + p_4*(reward_k + p_success_u_1*alpha_u_1)
        return E_J_k
    
    def f_W_vectorized(self, w, c_k, scale_k):
        """
        Compute the Weibull distribution PDF for a vector of wind speeds.

        Parameters:
        - w (np.ndarray): Array of wind speeds.
        - c_k (float): Shape parameter of the Weibull distribution.
        - scale_k (float): Scale parameter of the Weibull distribution.

        Returns:
        - np.ndarray: Array of PDF values corresponding to the input wind speeds.
        """
        w = np.asarray(w)  # Ensure w is a numpy array
        pdf = np.zeros_like(w)  # Initialize result with zeros
        valid = w >= 0  # Boolean mask for valid wind speeds (w >= 0)
        pdf[valid] = (c_k / scale_k) * (w[valid] / scale_k)**(c_k - 1) * np.exp(-(w[valid] / scale_k)**c_k)
        return pdf

    @staticmethod
    def lookup_expected_value(array, stage, states, discretization=0.01):
        """
        Look up values in a numpy array based on the stage, state of charge, and vehicle states for multiple states.
        
        Parameters:
            array (np.ndarray): The 2*n*1 by k numpy array to look up values from.
            stage (int): The column index in the array.
            states (list): A list of tuples, each containing:
                - state_of_charge (float): The state of charge percentage (0-100 range).
                - vehicle_state (str): The state of the vehicle, either 0, 1, or 2.
            discretization (float): The discretization step for the state of charge. Default is 0.01.
        
        Returns:
            list: A list of values from the array corresponding to each input state tuple.
        """

        # Convert states to a numpy array for vectorization
        states_array = np.array(states)
        if states_array.ndim == 1:
            states_array = np.expand_dims(states_array, axis=0)
        state_of_charge = states_array[:, 0].astype(float)  # Extract the state of charge values
        vehicle_state = states_array[:, 1]    # Extract the vehicle state values
        valid_options = [0, 1, 2]

        invalid_values = ~np.isin(vehicle_state, valid_options)
        if np.any(invalid_values):
            raise ValueError("Invalid vehicle state values:", vehicle_state[invalid_values])


        # Validate state_of_charge
        if np.any(state_of_charge < -1) or np.any(state_of_charge > 100):
            raise ValueError("State of charge not within valid range. Expected 0 <= soc <= 100.")

        # Convert state_of_charge from percentage to decimal
        state_of_charge = state_of_charge / 100.0

        # Calculate the number of discrete states based on discretization
        n = int(1 / discretization) + 1

        # Map vehicle state to row indices using numpy vectorized operations
        row_indices = np.zeros_like(state_of_charge, dtype=int)
        row_indices[vehicle_state == 0] = (state_of_charge[vehicle_state == 0] / discretization).astype(int)
        row_indices[vehicle_state == 1] = n + (state_of_charge[vehicle_state == 1] / discretization).astype(int)
        row_indices[vehicle_state == 2] = 2 * n

        # Ensure the stage index is within bounds
        if stage < 0 or stage > array.shape[1]:
            raise IndexError(f"Stage index out of bounds. Stage {stage} > max stage {array.shape[1]-1}.")
        elif stage == array.shape[1]:
            return np.zeros_like(row_indices)

        # Fetch values for all states
        results = np.zeros_like(row_indices, dtype=float)
        results[row_indices < array.shape[0]] = array[row_indices[row_indices < array.shape[0]], stage]
        
        if results.size == 1:
            return results[0]

        return results.tolist()

        # Separate data
    
    @staticmethod
    def plot_surface(data, capacity=50):
        """
        Plots separate surface plots for the 'moored,' 'flying,' and 'broken' states.

        Parameters:
            data (numpy.ndarray): A 2D array where:
                - Rows 0-100 represent battery percentages for the 'moored' state.
                - Rows 101-201 represent battery percentages for the 'flying' state.
                - Row 202 represents the 'broken' state.
            capacity (int): Battery capacity in Ah (used for the plot titles).
        """
        # Extract data for each state
        moored_data = data[:101, :]
        flying_data = data[101:202, :]
        broken_data = data[202:, :]

        # Generate grids
        time_steps = np.arange(data.shape[1])  # Time steps (x-axis)
        battery_percentages = np.linspace(0, 100, 101)  # Battery percentages (y-axis)

        # Plot for 'moored' state
        fig = plt.figure(figsize=(12, 8))
        ax = plt.subplot(projection='3d')
        X, Y = np.meshgrid(time_steps, battery_percentages)
        surf = ax.plot_surface(X, Y, moored_data, cmap="viridis", edgecolor='none')
        cbar = plt.colorbar(surf, ax=ax, shrink=0.5, aspect=10)
        cbar.set_label("Expected Value")
        ax.set_title(f"Surface Plot for State: Moored\nBattery Capacity: {capacity} Ah")
        ax.set_xlabel("Stages")
        ax.set_ylabel("State of Charge (%)")
        ax.set_zlabel("Expected Value")
        plt.tight_layout()
        plt.show()

        # Plot for 'flying' state
        fig = plt.figure(figsize=(12, 8))
        ax = plt.subplot(projection='3d')
        surf = ax.plot_surface(X, Y, flying_data, cmap="plasma", edgecolor='none')
        cbar = plt.colorbar(surf, ax=ax, shrink=0.5, aspect=10)
        cbar.set_label("Expected Value")
        ax.set_title(f"Surface Plot for State: Flying\nBattery Capacity: {capacity} Ah")
        ax.set_xlabel("Stages")
        ax.set_ylabel("State of Charge (%)")
        ax.set_zlabel("Expected Value")
        plt.tight_layout()
        plt.show()

        # Plot for 'broken' state (single row, flatten data)
        fig = plt.figure(figsize=(12, 8))
        ax = plt.subplot(projection='3d')
        ax.plot(np.arange(data.shape[1]), [0] * data.shape[1], broken_data.flatten(), label="Broken State", color="red")
        ax.set_title(f"Surface Plot for State: Broken\nBattery Capacity: {capacity} Ah")
        ax.set_xlabel("Stages")
        ax.set_ylabel("State of Charge (%)")
        ax.set_zlabel("Expected Value")
        ax.legend()
        plt.tight_layout()
        plt.show()


    
if __name__ == "__main__":
    class SeaplaneMock(Seaplane):
        def __init__(self):
            # Initialize with some example values
            self.capacity = 50  # Battery capacity (Ah)
            self.voltage = 24   # Battery voltage (V)
            self.S = 50         # Wing area (m^2)
            self.idle_power = 100  # Idle power consumption (W)
            self.required_cruise_power = 300  # Required power during cruise (W)
            self.required_takeoff_energy = 2000  # Energy needed for takeoff (J)

    # Sample data for solar, wind, and whale observation

    stages = 96*30

    # Solar: [Stage, Alpha, Beta]
    solar_data = np.random.uniform(5, 15, size=(stages,3))

    # Wind: Random wind data for testing
    wind_data = np.random.uniform(5, 15, size=(stages,3))

    # Whale Observation Data: Random probabilities (dummy values)
    whale_data = np.random.random(size=(stages,))
    whale_data[stages-1] = 0.

    # Initialize SeaplaneMock
    plane = SeaplaneMock()

    # Create ExpectedValueTable instance with a smaller SOC increment and timestep
    ev_table_instance = ExpectedValueTable(plane, solar_data, wind_data, whale_data, soc_increment=1, timestep_min=10)

    # Generate the expected value table
    ev_table_instance.generate_ev_table()

    # Print the generated table
    print("Expected Value Table (EV Table):")
    print(ev_table_instance.ev_table)
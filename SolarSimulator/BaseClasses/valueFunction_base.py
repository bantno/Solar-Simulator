"""This module contains the implementation of the ExpectedValueTable class for the Seaplane MDP."""

from abc import ABC, abstractmethod
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from tqdm import tqdm
from scipy.stats import beta as betaDist
from scipy.integrate import simpson
from BaseClasses.seaplane_base import Seaplane
from BaseClasses.transition_model_base import ActionSuccessProbabilityModel


class AbstractValueFunction(ABC):
    """Abstract base class for computing P(S=1 | w_k), the probability of action success given wind speed."""
    def __init__(
        self,
        plane: Seaplane,
        expected_solar_data: np.ndarray,
        expected_wind_data: np.ndarray,
        whale_observation_data: np.ndarray,
        soc_increment: int,
        timestep_min: int,
        transition_model: ActionSuccessProbabilityModel,
        failure_penalty: float = 0.0,
    ):
        self.plane = plane
        self.battery_capacity_wh = self.plane.capacity * self.plane.voltage
        self.max_collected_power = 1367 * self.plane.solar_panel_efficieny * plane.S
        self.dt = timestep_min
        self.soc_increment = soc_increment

        self.expected_solar = expected_solar_data
        self.expected_wind = expected_wind_data
        self.expected_whale = whale_observation_data

        self.states = self._create_states(soc_increment, [0, 1])
        self.transition_model = transition_model
        self.failure_penalty = failure_penalty

        self.value_table = np.zeros(
                (int(2 * (100 / soc_increment + 1) + 1), expected_solar_data.shape[0])
            )
        self.value_table[-1, :] = -self.failure_penalty


    def _create_states(self, soc_increment: int, vehicle_states: list) -> np.ndarray:
        """
        Generate a 2D NumPy array representing all possible states of the system.

        The states are defined as pairs of (state of charge, vehicle state) and 
        are grouped by vehicle state. The state of charge (SoC) values range from 
        0 to 100 in increments of `soc_increment`. An additional terminal state (-1, 2) 
        is appended at the end.

        Parameters:
        -----------
        soc_increment : int
            The step size for discretizing the state of charge (SoC) from 0 to 100.
        vehicle_states : list
            A list of possible vehicle states.

        Returns:
        --------
        np.ndarray
            A 2D NumPy array where each row represents a state as [SoC, vehicle_state].
            The states are grouped by vehicle state.
        """
        
        soc_values = np.arange(0, 101, soc_increment)  # Generate SoC values
        state_values = np.array(vehicle_states)[:, None]  # Convert to column vector

        # Repeat SoC values for each vehicle state (grouped)
        soc_repeated = np.tile(soc_values, (len(vehicle_states), 1)).T
        state_repeated = np.repeat(state_values, len(soc_values), axis=1).T

        # Stack the results into a 2D array
        states = np.column_stack((soc_repeated.ravel(), state_repeated.ravel()))

        # Append the additional (-1, 2) state
        states = np.vstack([states, [-1, 2]])

        return states



    @abstractmethod
    def _value_table_entry(self,full_state, stage)->float:
        """
        Calculate the entry in the value table for a specified state and stage.

        Parameters:
        - full_state (numpy array with two entries): first entry represents the state of charge of the vehicle, second entry represents the vehicle state, either 0 or 1
        - stage (int): the stage of the simulation

        Returns:
        - value: float, value table entry for the specified parameters
        """
        
    
    def _value_table_column(self,stage)->np.ndarray:
        """
        Calculate all the entries in the column of the value table that corresponds to a given stage

        Parameters:
        - stage (int): the stage of the value table that should be calculated

        Returns:
        - column (np.ndarray): column array that represents the specified column of the value table
        """
        num_states = self.states.shape[0]  # Number of possible states
        column = np.zeros(num_states)
        
        for i in range(num_states):
            full_state = self.states
            column[i] = self._value_table_entry(full_state, stage)
        
        return column

    def _generate_value_table(self)->None:
        """
        Populate the value table.
        """
        for stage in reversed(range(self.value_table.shape[1])):
            self.value_table[:,stage] = self._value_table_column(stage)

        return


class ValueFunction:
    """The ExpectedValueTable class for the Seaplane MDP."""
    def __init__(
        self,
        plane: Seaplane,
        expected_solar_data: np.ndarray,
        expected_wind_data: np.ndarray,
        whale_observation_data: np.ndarray,
        soc_increment: int,
        timestep_min: int,
        transition_model: ActionSuccessProbabilityModel,
        failure_penalty: float = 0.0,
    ):
        solar_panel_efficiency = 0.1
        self.plane = plane
        self.battery_capacity_wh = self.plane.capacity * self.plane.voltage
        self.max_collected_power = 1367 * solar_panel_efficiency * plane.S
        self.dt = timestep_min
        self.soc_increment = soc_increment
        self.expected_solar = expected_solar_data
        self.expected_wind = expected_wind_data
        self.states = self._create_states(soc_increment, [0, 1])
        self.transition_model = transition_model
        self.failure_penalty = failure_penalty

        if 100 % soc_increment != 0:
            raise ValueError("Given state of charge increment does not divide evenly into 100%.")
        else:
            self.ev_table = np.zeros(
                (int(2 * (100 / soc_increment + 1) + 1), expected_solar_data.shape[0])
            )
            self.ev_table[-1, :] = -self.failure_penalty

        self.whale_probability_data = whale_observation_data

    def _create_states(self, soc_increment: int, vehicle_states: list) -> list:
        """
        Generate a list of states based on state of charge (SoC) increments and vehicle states.
        """
        states = [(soc, state) for state in vehicle_states for soc in range(0, 101, soc_increment)]
        states.append((-1, 2))
        return states

    def generate_ev_table(self):
        """Fill in expected value table."""
        for k in tqdm(range(self.ev_table.shape[1] - 1, -1, -1)):
            for idx, state in enumerate(self.states[:-1]):
                self.ev_table[idx, k] = self._ev_entry(k, state)

        filename = f"evTable_{self.plane.capacity}_{self.plane.lat}_{self.transition_model.name}"
        self.plot_surface_plotly(self.ev_table, self.plane.capacity, self.failure_penalty, filename)
        np.savetxt("Results\\" + filename + ".csv", self.ev_table, delimiter=",")

    def _calculate_case_probabilities(self, stage, state, reward_k):
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
        alpha_k = self.expected_solar[stage, 0]
        beta_k = self.expected_solar[stage, 1]
        max_collected_energy_j = self.max_collected_power * self.dt * 60
        current_energy_j = self.soc_to_joules(state[0])
        required_energy_j = self._calculate_required_energy(state, action=1)
        p_sufficient_solar = self._calculate_sufficient_solar_probability(
            required_energy_j, current_energy_j, max_collected_energy_j, alpha_k, beta_k
        )

        # TODO: Check if these are the correct index...
        shape_k = self.expected_wind[stage, 0]
        scale_k = self.expected_wind[stage, 1]

        p_sufficient_reward, alpha_u_0, alpha_u_1 = self._calculate_sufficient_reward_probability(
            stage, state, reward_k, alpha_k, beta_k, shape_k, scale_k)

        p0 = (1 - p_sufficient_solar) * (1 - p_sufficient_reward)
        p1 = (1 - p_sufficient_solar) * (p_sufficient_reward)
        p2 = (p_sufficient_solar) * (1 - p_sufficient_reward)
        p3 = (p_sufficient_solar) * (p_sufficient_reward)

        return (p0, p1, p2, p3), alpha_u_0, alpha_u_1

    def _calculate_sufficient_solar_probability(
        self, required_energy, current_energy, max_collected_energy, alpha, beta
    ):
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
        threshold = (
            required_energy - current_energy
        ) / max_collected_energy  # Calculate the threshold for X <= 0 condition (S <= (P - C) / I)
        insufficient_energy_probability = betaDist.cdf(threshold, alpha, beta)
        return 1 - insufficient_energy_probability

    def _calculate_sufficient_reward_probability(
        self, stage, state, reward_k, alpha_k, beta_k, shape, scale, n=5000
    ):
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
        # TODO Add wind sampling to this method and vectorize.
        if beta_k == 1000:
            solar_samples = np.zeros(n)
        else:
            solar_samples = np.random.beta(alpha_k, beta_k, size=n) * self.max_collected_power
        wind_samples = np.random.weibull(shape, size=n) * scale

        ev_0 = np.array(
            self._alpha(stage, state, 0, solar_power_w=solar_samples, wind_speed_ms=wind_samples)
        )
        ev_1 = np.array(
            self._alpha(stage, state, 1, solar_power_w=solar_samples, wind_speed_ms=wind_samples)
        )

        d_alpha = (ev_0 - ev_1)
        p_sufficient_reward = np.mean(reward_k > d_alpha)

        return p_sufficient_reward, np.mean(ev_0), np.mean(ev_1)

    def _calculate_required_energy(self, state, action):
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
        required_energy_j = 0
        timestep_s = self.dt * 60
        if action == 0:
            required_energy_j += self.plane.idle_power * timestep_s
        elif action == 1:
            required_energy_j += self.plane.required_cruise_power * timestep_s
            if state[1] == 0:
                required_energy_j += self.plane.required_takeoff_energy
        else:
            raise ValueError("Invlaid action specified.")

        return required_energy_j

    def _alpha(self, stage: int, state: tuple, action: int, solar_power_w, wind_speed_ms):
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
        # TODO: Make this fully account for the stochastic nature of the transition.
        #  As is, only solar irradiation is considered, leaving out the probability of action failure
        next_state = self.calculate_next_state(state, action, solar_power_w)
        p_success_u = self.transition_model.compute_probability(wind_speed=wind_speed_ms, action=action, state=state)
        ev = self.lookup_expected_value(self.ev_table, stage + 1, next_state) * p_success_u - self.failure_penalty * (1 - p_success_u)
        return ev

    def soc_to_joules(self, soc):
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
        joules = soc / 100 * self.battery_capacity_wh * 3600
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
        new_soc, new_vehicle_states = np.where(new_soc < 0, -1, new_soc), np.where(
            new_soc < 0, 2, new_vehicle_state
        )

        if new_soc.ndim == 0:
            return (new_soc, new_vehicle_states)

        # Return updated states as a list of tuples
        return np.column_stack((new_soc, new_vehicle_states))

    def _calculate_soc_update(self, plane, state, action, dt, solar_power):
        """
        Vectorized version of _calculate_soc_update to compute SOC changes for multiple
        solar_power values.

        Args:
            plane (object): The plane object containing operational parameters such as
                required cruise power, takeoff energy, idle power, and wing area (`S`).
            state (tuple): The current state of the system, where `state[1]` indicates
                whether the plane is 0 or in another state.
            action (np.ndarray): Array of actions (0 for "float", 1 for "fly") of shape (n,).
            solar_power (np.ndarray): Collected solar power values (in Watts) of shape (n, 1).
            dt (float): The duration of the time step (in minutes).

        Returns:
            np.ndarray: Array of SOC changes as percentages, rounded to the nearest SOC
            increment, shape (n,).
        """
        # start_time = time.time()
        # Ensure solar_power is a numpy array with correct dimensions
        solar_power = np.atleast_2d(solar_power).flatten()  # Ensure 1D array

        # Initialize constants
        required_takeoff_energy = (
            plane.required_takeoff_energy if state[1] == 0 and action == 1 else 0
        )

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

    def compute_conditional_success_probability(self, w, u_k, state):
        """
        Compute the conditional probability P(S|w_k) based on wind speed, action, and current state.

        Parameters:
        - w (float): Wind speed.
        - u_k (int): Control input (1 for active action, 0 for passive action).
        - state (tuple): State of vehicle at stage k.

        Returns:
        - float: Probability of success P(S|w_k).
        """
        return self.transition_model.compute_probability(w, u_k, state)

    def _compute_success_probability(self, u_k, state_k, c_k, scale_k):
        """
        Compute the overall probability P(S) by integrating P(S|w_k) * f_W(w).

        Parameters:
        - u_k (int): Control input (1 for active action, 0 for passive action).
        - state_k (tuple): Mode (0 for ground/floating, 1 for air/flying).

        Returns:
        - float: Overall probability of success.
        """

        w = np.linspace(0.00000001, 60, 501)
        conditional_success_prob = self.compute_conditional_success_probability(w, u_k, state_k)
        wind_pdf = self.f_W_vectorized(w, c_k, scale_k)
        result = simpson(x=w, y=conditional_success_prob * wind_pdf)
        return result

    def last_entry(self,k,state):
        reward_k = self.whale_probability_data[k] * 1

        wind_shape_k = self.expected_wind[k, 0]
        wind_scale_k = self.expected_wind[k, 1]
        p_success_u_0 = self._compute_success_probability(0, state, wind_shape_k, wind_scale_k)
        return -self.failure_penalty*(1-p_success_u_0)

    def _ev_entry(self, k, state):

        if state[0] == 0:
            return -self.failure_penalty
        elif k == self.ev_table.shape[1] - 1:
            return self.last_entry(k,state)

        reward_k = self.whale_probability_data[k] * 1
        case_probs, alpha_u_0, alpha_u_1 = self._calculate_case_probabilities(
            k, state, reward_k)

        # Removed probabilities from here because I think i was double counting them. Alpha already accounts for probability of failure
        # Really before i fixed this i was tripple counting the probability of failure
        case_1_2_3 = (1-case_probs[3])*(alpha_u_0)
        case_4 = case_probs[3]*(reward_k + alpha_u_1)

        # Need to include the negative effect of penalty
        e_j_k =  case_1_2_3 + case_4
        return e_j_k

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
        pdf = (c_k / scale_k) * (w / scale_k) ** (c_k - 1) * np.exp(-((w / scale_k) ** c_k))
        return pdf

    @staticmethod
    def lookup_expected_value(array, stage, states, soc_increment=1):
        """
        Look up values in a numpy array based on the stage, state of charge, and vehicle
        states for multiple states.

        Parameters:
            array (np.ndarray): The 2*n*1 by k numpy array to look up values from.
            stage (int): The column index in the array.
            states (list): A list of tuples, each containing:
                - state_of_charge (float): The state of charge percentage (0-100 range).
                - vehicle_state (str): The state of the vehicle, either 0, 1, or 2.
            soc_increment (int): The increment step for the state of charge (default is 1).

        Returns:
            list: A list of values from the array corresponding to each input state tuple.
        """
        states_array = np.atleast_2d(states)
        state_of_charge = states_array[:, 0].astype(float)
        vehicle_state = states_array[:, 1]

        if not set(np.unique(vehicle_state)).issubset({0, 1, 2}):
            raise ValueError("Invalid vehicle state values detected.")
        if np.any((state_of_charge < -1) | (state_of_charge > 100)):
            raise ValueError("State of charge not within valid range (-1 to 100).")

        n = 100 // soc_increment + 1
        row_indices = np.zeros(len(state_of_charge), dtype=int)

        # Handle special case where state_of_charge = -1 -> last row
        row_indices[state_of_charge == -1] = array.shape[0] - 1

        # Handle other states normally
        normal_states = state_of_charge != -1
        row_indices[normal_states] = (state_of_charge[normal_states] / soc_increment).astype(int)
        row_indices[normal_states] += np.where(vehicle_state[normal_states] == 1, n, 0)
        row_indices[normal_states] += np.where(vehicle_state[normal_states] == 2, 2 * n, 0)

        if stage < 0 or stage >= array.shape[1]:
            raise IndexError(
                f"Stage index {stage} out of bounds for array with {array.shape[1]} stages."
            )

        results = np.zeros(len(row_indices), dtype=float)
        valid_indices = row_indices < array.shape[0]
        results[valid_indices] = array[row_indices[valid_indices], stage]

        return results[0] if results.size == 1 else results

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
        ax = plt.subplot(projection="3d")
        x, y = np.meshgrid(time_steps, battery_percentages)
        surf = ax.plot_surface(x, y, moored_data, cmap="viridis", edgecolor="none")
        cbar = plt.colorbar(surf, ax=ax, shrink=0.5, aspect=10)
        cbar.set_label("Expected Value")
        ax.set_title(f"Surface Plot for State: Moored\nBattery Capacity: {capacity} Ah")
        ax.set_xlabel("Stages")
        ax.set_ylabel("State of Charge (%)")
        ax.set_zlabel("Expected Value")
        plt.tight_layout()
        plt.savefig("ev_table_moored.png")

        # Plot for 'flying' state
        fig = plt.figure(figsize=(12, 8))
        ax = plt.subplot(projection="3d")
        surf = ax.plot_surface(x, y, flying_data, cmap="plasma", edgecolor="none")
        cbar = plt.colorbar(surf, ax=ax, shrink=0.5, aspect=10)
        cbar.set_label("Expected Value")
        ax.set_title(f"Surface Plot for State: Flying\nBattery Capacity: {capacity} Ah")
        ax.set_xlabel("Stages")
        ax.set_ylabel("State of Charge (%)")
        ax.set_zlabel("Expected Value")
        plt.tight_layout()
        plt.savefig("ev_table_flying.png")

        # Plot for 'broken' state (single row, flatten data)
        fig = plt.figure(figsize=(12, 8))
        ax = plt.subplot(projection="3d")
        ax.plot(
            np.arange(data.shape[1]),
            [0] * data.shape[1],
            broken_data.flatten(),
            label="Broken State",
            color="red",
        )
        ax.set_title(f"Surface Plot for State: Broken\nBattery Capacity: {capacity} Ah")
        ax.set_xlabel("Stages")
        ax.set_ylabel("State of Charge (%)")
        ax.set_zlabel("Expected Value")
        ax.legend()
        plt.tight_layout()
        plt.savefig("ev_table_broken.png")

    @staticmethod
    def plot_surface_plotly(data, capacity=50, failure_penalty=None, filename=None):
        """
        Plots an interactive 3D surface plot for the 'moored,' 'flying,' and 'broken'
        states using Plotly.

        Parameters:
            data (numpy.ndarray): A 2D array where:
                - Rows 0-100 represent battery percentages for the 'moored' state.
                - Rows 101-201 represent battery percentages for the 'flying' state.
                - Row 202 represents the 'broken' state.
            capacity (int): Battery capacity in Ah (used for the plot title).
        """
        # Extract data for each state
        moored_data = data[:101, :]
        flying_data = data[101:202, :]
        broken_data = data[202, :]  # Single row for broken state

        # Generate grids
        time_steps = np.arange(data.shape[1])  # Time steps (x-axis)
        battery_percentages = np.linspace(0, 100, 101)  # Battery percentages (y-axis)
        x, y = np.meshgrid(time_steps, battery_percentages)

        # Create figure
        fig = go.Figure()

        # Add Moored State surface
        fig.add_trace(
            go.Surface(
                z=moored_data, x=x, y=y, colorscale="Blues", opacity=0.9, name="Moored", showlegend=True
            )
        )

        # Add Flying State surface
        fig.add_trace(
            go.Surface(
                z=flying_data, x=x, y=y, colorscale="Magma", opacity=0.8, name="Flying", showlegend=True
            )
        )

        # # Add Broken State line
        # fig.add_trace(
        #     go.Scatter3d(
        #         x=time_steps,
        #         y=[0] * len(time_steps),
        #         z=broken_data,
        #         mode="lines",
        #         line=dict(color="red", width=4),
        #         name="Broken",
        #     )
        # )

        # Update layout
        fig.update_layout(
            title=(
                "Surface Plot for Moored, Flying, and Broken States "
                f"({capacity} Ah, Penalty: {failure_penalty})"
            ),
            scene=dict(
                xaxis_title="Stages",
                yaxis_title="State of Charge (%)",
                zaxis_title="Expected Value",
            ),
        )

        # Save plot
        fig.write_html("Figures\\" + filename + ".html")


if __name__ == "__main__":
    pass

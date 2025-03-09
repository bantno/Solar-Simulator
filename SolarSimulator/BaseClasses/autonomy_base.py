"""This module contains the base class for the autonomy module of the solar-powered seaplane simulator."""

import numpy as np
from BaseClasses.valueFunction_base import ValueFunction


class Autonomy:
    """Represents the autonomy module for a solar-powered seaplane."""

    def __init__(self, dt, mdp_model: ValueFunction, use_expected_reward: bool = False, wind_threshold: float = 100.):
        self.dt = dt
        self.mdp_model = mdp_model
        self.failure_penalty = mdp_model.failure_penalty
        self.plane = mdp_model.plane
        self.soc_increment = 1
        self.panel_efficiency = 0.10
        self.max_capacity_J = self.plane.capacity * self.plane.voltage * 3600
        self.use_expected_reward = use_expected_reward
        self.transition_model = mdp_model.transition_model
        self.wind_threshold = wind_threshold

    def simulate_observation_threshold_mission(
        self,
        initial_state,
        solar_data,
        wind_data,
        whale_data,
        simulate_failure=False,
        save_history=False,
        threshold=None,
    ):
        """
        Simulate a mission using a threshold-based decision for flying.
        All simulation steps (state update, energy update, failure checking, etc.) are handled
        identically to the optimal simulation except that the decision to fly is made using
        _determine_observation_threshold_action.
        """
        return self._simulate_mission(
            initial_state,
            solar_data,
            wind_data,
            whale_data,
            simulate_failure,
            save_history,
            threshold,
            decision_type="threshold",
        )

    def simulate_optimal_mission(
        self,
        initial_state,
        solar_data,
        wind_data,
        whale_data,
        simulate_failure=False,
        save_history=False,
        threshold=None,
    ):
        """
        Simulate a mission using an optimal decision for flying.
        All simulation steps (state update, energy update, failure checking, etc.) are handled
        identically to the threshold simulation except that the decision to fly is made using
        _determine_optimal_action.
        Note: The 'threshold' parameter is unused in the optimal case.
        """
        return self._simulate_mission(
            initial_state,
            solar_data,
            wind_data,
            whale_data,
            simulate_failure,
            save_history,
            threshold,
            decision_type="optimal",
        )

    def _simulate_mission(
        self,
        initial_state,
        solar_data,
        wind_data,
        whale_data,
        simulate_failure=False,
        save_history=False,
        threshold=None,
        decision_type="threshold",
    ):
        """
        Unified simulation loop for both threshold and optimal missions.
        All steps are executed identically (energy update, reward calculation, state history,
        failure checking, etc.), except for how the decision to fly is made.

        Parameters:
            initial_state: The starting state of the simulation.
            solar_data, wind_data, whale_data: Lists of environmental data.
            simulate_failure: Whether to simulate failures probabilistically.
            save_history: If True, detailed history is returned.
            threshold: Threshold value (used only for threshold-based decision).
            decision_type: "threshold" for threshold simulation or "optimal" for optimal simulation.
        """
        max_stages = self._validate_data_lengths(solar_data, wind_data, whale_data)
        battery_capacity_J, nightly_idle_soc, single_flight_soc = self._compute_energy_parameters()
        state_history_list, energy_history_list, u_k_list, failure_prob_list = (
            self._initialize_state_history(initial_state, max_stages, battery_capacity_J)
        )
        samples = self._generate_mcs_samples(max_stages)
        flight_minutes, reward = 0.0, 0
        is_failure = False
        failure_type = 0

        if decision_type == "optimal":
            action_list = [0, 1]
            value_list = [-1000, -1000]

        for k in range(max_stages - 1):
            current_state, current_energy, solar_power_wpm2, wind_speed, whale_prob = (
                self._extract_step_data(
                    k,
                    state_history_list,
                    energy_history_list,
                    solar_data,
                    wind_data,
                    whale_data,
                )
            )

            if decision_type == "optimal":
                collected_solar_power = self.plane.S * solar_power_wpm2 * self.panel_efficiency
                best_action = self._determine_optimal_action(
                    k,
                    current_state,
                    action_list,
                    value_list,
                    collected_solar_power,
                    wind_speed,
                    whale_prob,
                )
            elif decision_type == "threshold":
                best_action = self._determine_observation_threshold_action(
                    whale_prob,
                    solar_power_wpm2,
                    current_state,
                    nightly_idle_soc,
                    single_flight_soc,
                    threshold,
                    wind_speed,
                )
            else:
                raise ValueError("Unknown decision type provided.")

            if best_action == 1:
                flight_minutes += self.dt

            failure_prob = self._compute_failure_prob(wind_speed, best_action, current_state)
            is_action_successful = self._is_action_successful(
                samples[k], failure_prob, simulate_failure
            )
            new_energy, new_state = self._update_energy_and_state(
                current_state,
                current_energy,
                best_action,
                solar_power_wpm2,
                battery_capacity_J,
            )
            reward += self.simulate_stochastic_reward(
                current_state, best_action, k, whale_prob, self.use_expected_reward
            )

            state_history_list[k + 1] = new_state
            energy_history_list[k + 1] = new_energy
            u_k_list[k + 1] = best_action
            failure_prob_list[k + 1] = failure_prob

            if not is_action_successful:
                state_history_list[k:] = [(-10, 2)] * (len(state_history_list) - k)
                reward -= self.failure_penalty
                is_failure = True
                failure_type = 1
                break
            elif new_state[0] < 0:
                state_history_list[k:] = [(-30, 2)] * (len(state_history_list) - k)
                reward -= self.failure_penalty
                is_failure = True
                failure_type = 2
                break

        return self._finalize_simulation(
            save_history,
            reward,
            k,
            state_history_list,
            u_k_list,
            failure_prob_list,
            solar_data,
            wind_data,
            whale_data,
            flight_minutes,
            is_failure,
            failure_type,
        )

    def simulate_charge_threshold_mission(
        self,
        initial_state,
        solar_data,
        wind_data,
        whale_data,
        simulate_failure=False,
        save_history=False,
        threshold=None,
    ):
        raise NotImplementedError("Charge threshold simulation has not been implemented.")
        
    ### Helper Functions ###

    def _validate_data_lengths(self, solar_data, wind_data, whale_data):
        """Ensure all input data lists have the same length."""
        if len(whale_data) == len(wind_data) == len(solar_data):
            return len(wind_data)
        raise ValueError(
            f"Data lengths are not equal. Wind: {len(wind_data)}, Solar: {len(solar_data)}, Whale: {len(whale_data)}."
        )

    def _is_action_successful(self, random_sample, failure_prob, simulate_failure=True):
        """Determine if the action was successful based on the failure probability."""
        return random_sample > failure_prob if simulate_failure else True

    def _generate_mcs_samples(self, max_stages, seed=None):
        """Generate random samples for Monte Carlo simulation."""
        return np.random.uniform(0, 1, max_stages)

    def _compute_energy_parameters(self):
        """Compute battery capacity and energy thresholds."""
        night_hours = 12
        battery_capacity_J = self.plane.capacity * self.plane.voltage * 3600
        nightly_idle_soc = np.ceil(
            (self.plane.idle_power * night_hours * 3600) / battery_capacity_J * 100
        )
        single_flight_soc = np.ceil(
            (
                self.plane.get_required_power(20, 1.2) * self.dt * 60
                + self.plane.required_takeoff_energy
            )
            / battery_capacity_J
            * 100
        )
        return battery_capacity_J, nightly_idle_soc, single_flight_soc

    def _initialize_state_history(self, initial_state, max_stages, battery_capacity_J):
        """
        Initialize state and energy history arrays.

        Parameters:
        initial_state (tuple): The initial state of the system.
        max_stages (int): The maximum number of stages for the simulation.
        battery_capacity_J (float): The battery capacity in Joules.

        Returns:
        tuple: A tuple containing:
            - state_history_list (np.ndarray): An array to store the state history.
            - energy_history_list (np.ndarray): An array to store the energy history.
            - u_k_list (np.ndarray): An array to store control inputs.
        """
        state_history_list = np.empty((max_stages, 2))
        energy_history_list = np.zeros(max_stages)
        u_k_list = np.zeros(max_stages)
        failure_prob_list = np.zeros(max_stages)
        state_history_list[0, :] = np.array(initial_state)
        energy_history_list[0] = initial_state[0] / 100 * battery_capacity_J
        return state_history_list, energy_history_list, u_k_list, failure_prob_list

    def _extract_step_data(
        self,
        k,
        state_history_list,
        energy_history_list,
        solar_data,
        wind_data,
        whale_data,
    ):
        """Extract data for the current simulation step."""
        return (
            state_history_list[k],
            energy_history_list[k],
            solar_data[k],
            wind_data[k],
            whale_data[k],
        )

    def _determine_observation_threshold_action(
        self,
        whale_prob,
        solar_power_wpm2,
        current_state,
        nightly_idle_soc,
        single_flight_soc,
        threshold,
        wind_speed,
    ):
        """Determine the best action based on energy and reward conditions."""
        is_reward_sufficient = whale_prob > threshold and solar_power_wpm2 > 0
        is_battery_sufficient = current_state[0] > (nightly_idle_soc + single_flight_soc)
        is_wind_low = wind_speed < self.wind_threshold
        decide_flight = np.all([is_reward_sufficient, is_battery_sufficient, is_wind_low])
        return 1 if decide_flight else 0

    def _determine_charge_threshold_action(
        self,
        current_state,
        nightly_idle_soc,
        single_flight_soc,
        charge_threshold,
    ):
        """Determine the best action based on stored energy."""
        is_battery_sufficient = current_state[0] > charge_threshold * 100 or (
            current_state[1] == 1 and current_state[0] > single_flight_soc + nightly_idle_soc
        )
        return 1 if is_battery_sufficient else 0

    def _determine_optimal_action(
        self,
        k,
        current_state,
        action_list,
        value_list,
        collected_solar_power,
        wind_speed,
        whale_prob,
    ):
        """Determine the best action using the MDP model."""
        for idx, action in enumerate(action_list):
            next_state = self.mdp_model.calculate_next_state(current_state, action, collected_solar_power)
            alpha = self.mdp_model.lookup_expected_value(self.mdp_model.ev_table, k + 1, next_state, self.mdp_model.soc_increment)
            value_list[idx] = self.reward(current_state, action, next_state, wind_speed, whale_prob) + alpha[0] * self.transition_model.compute_probability(wind_speed, action, current_state)
        return np.argmax(value_list)

    def reward(self, state, action, next_state, wind_speed, whale_prob):
        """"Calculate the reward for a given stage, state, action, and wind speed."""
        whale_reward = 0 if action == 0 else whale_prob

        # Account for failure that will occur if the plane runs out of battery
        if next_state[0, 0] <= 1:
            failure_reward = -self.failure_penalty
        else:
            failure_reward = -self.failure_penalty * (1 - self.transition_model.compute_probability(wind_speed, action, state)[0])
        reward_k = whale_reward + failure_reward
        return reward_k

    def _compute_failure_prob(self, wind_speed, best_action, current_state):
        """Compute the probability of failure given the wind conditions and action."""
        return 1 - self.transition_model.compute_probability(wind_speed, best_action, current_state)

    def _update_energy_and_state(
        self,
        current_state,
        current_energy,
        best_action,
        solar_power_wpm2,
        battery_capacity_J,
    ):
        """Compute new energy and state after applying the action."""
        new_energy = min(
            current_energy
            + self.calculate_energy_update(
                self.mdp_model.plane,
                state=current_state,
                action=best_action,
                dt=self.dt,
                solar_power_wpm2=solar_power_wpm2,
            ),
            self.max_capacity_J,
        )
        return new_energy, self.calculate_new_state(best_action, new_energy, battery_capacity_J)

    def _finalize_simulation(
        self,
        save_history,
        reward,
        k,
        state_history_list,
        action_list,
        failure_prob_list,
        solar_data,
        wind_data,
        whale_data,
        flight_minutes,
        is_failure,
        failure_type,
    ):
        """Finalize the simulation results."""
        k += 1
        if save_history:
            return (
                reward,
                k,
                state_history_list,
                action_list,
                failure_prob_list,
                solar_data,
                wind_data,
                whale_data,
                flight_minutes,
                is_failure,
                failure_type,
            )
        return reward, k, flight_minutes, is_failure, failure_type

    def simulate_stochastic_reward(
        self, state, action, stage, whale_prob, use_expected_value=False
    ):
        """
        Calculates the reward for performing the given action in the current state at the current stage.
        Includes stochastic rewards based on the probability of finding whales (time-dependent) and wind speed.

        Parameters:
        - state: Current state as a tuple (SoC, vehicle_state)
        - action: The action being taken ('float', 'fly')
        - stage: The current stage in the simulation

        Returns:
        - Reward value considering both deterministic and stochastic factors.
        """
        whale_reward = 0
        if action == 1:
            if use_expected_value:
                whale_reward = whale_prob
            else:
                if np.random.uniform(0, 1) < whale_prob:
                    whale_reward = 1
        return whale_reward

    def calculate_new_state(self, best_action, energy, max_capacity):
        if best_action == 0:
            state = 0
        elif best_action == 1:
            state = 1
        else:
            raise ValueError(f"Action: {best_action} is not a valid action.")
        soc = min(round(energy / max_capacity * 100), 100)
        return (soc, state)

    def calculate_next_state(self, current_state, action, solar_power_wpm2):
        soc = current_state[0]
        delta_soc = self.calculate_soc_update(
            self.mdp_model.plane, current_state, action, self.dt, solar_power_wpm2
        )
        new_soc = min(soc + delta_soc, 100)  # Limit SoC to 100
        new_vehicle_state = 1 if action == 1 else 0

        # Set state to "broken" if SoC falls below 0
        if new_soc < 0:
            new_soc, new_vehicle_state = -1, 2
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
        solar_input = solar_power * self.panel_efficiency * plane.S
        net_power = solar_input - required_power - avionics_power

        # Convert power (W) to energy (Joules) and then to change in SoC (%)
        energy_change = net_power * dt * 60 - required_takeoff_energy  # Convert power to energy
        soc_change = (
            energy_change / (plane.voltage * plane.capacity * 3600)
        ) * 100  # Energy to SoC %

        # Round to the nearest SoC increment and return
        return self.soc_increment * round(soc_change / self.soc_increment)

    def calculate_energy_update(self, plane, state, action, dt, solar_power_wpm2):
        """
        Calculates the change in SoC after performing the given action.
        """
        required_takeoff_energy = 0
        required_cruise_power = 0

        if action == 0:
            required_cruise_power = 0
        elif action == 1:
            required_cruise_power = plane.required_cruise_power  # Assumed constants for flight
            if state[1] == 0:
                required_takeoff_energy = plane.required_takeoff_energy
        else:
            raise ValueError(f"Expected action 0 (float) or 1 (fly). Got {action}.")

        avionics_power = plane.idle_power
        collected_power = solar_power_wpm2 * self.panel_efficiency * plane.S
        net_power = collected_power - required_cruise_power - avionics_power
        energy_change = (
            net_power * dt * 60 - required_takeoff_energy
        )  # Convert power (W) to energy (Joules)
        return energy_change

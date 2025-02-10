"""This module contains the base class for the autonomy module of the solar-powered seaplane simulator."""

import numpy as np


class Autonomy:
    """Represents the autonomy module for a solar-powered seaplane."""

    def __init__(self, dt, mdp_model, use_expected_reward: bool = False):
        self.dt = dt
        self.mdp_model = mdp_model
        self.plane = mdp_model.plane
        self.soc_increment = 1
        self.panel_efficiency = 0.10
        self.max_capacity_J = self.plane.capacity * self.plane.voltage * 3600
        self.use_expected_reward = use_expected_reward
        self.transition_model = mdp_model.transition_model

    def simulate_observation_threshold_mission(
        self,
        initial_state,
        solar_data,
        wind_data,
        whale_data,
        true_success_prob,
        simulate_failure=False,
        save_history=False,
        threshold=None,
    ):

        # Ensure all data lists have the same length
        max_stages = self._validate_data_lengths(solar_data, wind_data, whale_data)

        self.stepwise_failure_prob = 1 - true_success_prob
        battery_capacity_J, nightly_idle_soc, single_flight_soc = (
            self._compute_energy_parameters()
        )

        # Initialize history arrays
        state_history_list, energy_history_list, u_k_list, failure_prob_list = (
            self._initialize_state_history(
                initial_state, max_stages, battery_capacity_J
            )
        )
        flight_minutes, reward = 0.0, 0

        # Simulation loop
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

            best_action = self._determine_observation_threshold_action(
                whale_prob,
                solar_power_wpm2,
                current_state,
                nightly_idle_soc,
                single_flight_soc,
                threshold,
            )
            if best_action == 1:
                flight_minutes += self.dt

            failure_prob = self._compute_failure_prob(
                wind_speed, best_action, current_state
            )
            if simulate_failure:
                is_action_successful = np.random.uniform(0, 1) > failure_prob
            else:
                is_action_successful = True

            new_energy, new_state = self._update_energy_and_state(
                current_state,
                current_energy,
                best_action,
                solar_power_wpm2,
                battery_capacity_J,
            )
            reward += self.simulate_stochastic_reward(
                current_state,
                best_action,
                k,
                whale_prob,
                use_expected_value=self.use_expected_reward,
            )

            (
                state_history_list[k + 1],
                energy_history_list[k + 1],
                u_k_list[k + 1],
                failure_prob_list[k + 1],
            ) = (new_state, new_energy, best_action, failure_prob)

            if not is_action_successful:
                state_history_list[k:] = [(-10, 2)] * (len(state_history_list) - k)
                break
            elif new_state[0] < 0:
                state_history_list[k:] = [(-15, 2)] * (len(state_history_list) - k)
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
        )

    def simulate_optimal_mission(
        self,
        initial_state,
        solar_data,
        wind_data,
        whale_data,
        true_success_prob,
        simulate_failure=False,
        save_history=False,
        threshold=None,
    ):

        # Ensure all data lists have the same length
        max_stages = self._validate_data_lengths(solar_data, wind_data, whale_data)

        battery_capacity_J = self.plane.capacity * self.mdp_model.plane.voltage * 3600
        state_history_list, energy_history_list, u_k_list, failure_prob_list = (
            self._initialize_state_history(
                initial_state, max_stages, battery_capacity_J
            )
        )
        flight_minutes, reward = 0.0, 0
        action_list = [0, 1]
        value_list = [-1000, -1000]

        # Simulation loop
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

            collected_solar_power = (
                self.plane.S * solar_power_wpm2 * self.panel_efficiency
            )
            best_action = self._determine_optimal_action(
                k,
                current_state,
                action_list,
                value_list,
                collected_solar_power,
                wind_speed,
                whale_prob,
            )

            failure_prob = self._compute_failure_prob(
                wind_speed, best_action, current_state
            )
            if simulate_failure:
                is_action_successful = np.random.uniform(0, 1) > failure_prob
            else:
                is_action_successful = True

            new_energy, new_state = self._update_energy_and_state(
                current_state,
                current_energy,
                best_action,
                solar_power_wpm2,
                battery_capacity_J,
            )
            reward += self.simulate_stochastic_reward(
                current_state, best_action, k, whale_prob
            )

            (
                state_history_list[k + 1],
                energy_history_list[k + 1],
                u_k_list[k + 1],
                failure_prob_list[k + 1],
            ) = (new_state, new_energy, best_action, failure_prob)

            if not is_action_successful:
                state_history_list[k:] = [(-10, 2)] * (len(state_history_list) - k)
                break
            elif new_state[0] < 0:
                state_history_list[k:] = [(-15, 2)] * (len(state_history_list) - k)
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
        )

    def simulate_charge_threshold_mission(
                self,
        initial_state,
        solar_data,
        wind_data,
        whale_data,
        true_success_prob,
        simulate_failure=False,
        save_history=False,
        threshold=None,
    ):

        # Ensure all data lists have the same length
        max_stages = self._validate_data_lengths(solar_data, wind_data, whale_data)

        self.stepwise_failure_prob = 1 - true_success_prob
        battery_capacity_J, nightly_idle_soc, single_flight_soc = (
            self._compute_energy_parameters()
        )

        # Initialize history arrays
        state_history_list, energy_history_list, u_k_list, failure_prob_list = (
            self._initialize_state_history(
                initial_state, max_stages, battery_capacity_J
            )
        )
        flight_minutes, reward = 0.0, 0

        # Simulation loop
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

            best_action = self._determine_charge_threshold_action(
                current_state,
                nightly_idle_soc,
                single_flight_soc,
                threshold,
            )
            if best_action == 1:
                flight_minutes += self.dt

            failure_prob = self.transition_model.compute_probability(
                wind_speed, best_action, current_state
            )
            if simulate_failure:
                is_action_successful = np.random.uniform(0, 1) > failure_prob
            else:
                is_action_successful = True

            new_energy, new_state = self._update_energy_and_state(
                current_state,
                current_energy,
                best_action,
                solar_power_wpm2,
                battery_capacity_J,
            )
            reward += self.simulate_stochastic_reward(
                current_state,
                best_action,
                k,
                whale_prob,
                use_expected_value=self.use_expected_reward,
            )

            (
                state_history_list[k + 1],
                energy_history_list[k + 1],
                u_k_list[k + 1],
                failure_prob_list[k + 1],
            ) = (new_state, new_energy, best_action, failure_prob)

            if not is_action_successful:
                state_history_list[k:] = [(-10, 2)] * (len(state_history_list) - k)
                break
            elif new_state[0] < 0:
                state_history_list[k:] = [(-15, 2)] * (len(state_history_list) - k)
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
        )

    ### Helper Functions ###

    def _validate_data_lengths(self, solar_data, wind_data, whale_data):
        """Ensure all input data lists have the same length."""
        if len(whale_data) == len(wind_data) == len(solar_data):
            return len(wind_data)
        raise ValueError(
            f"Data lengths are not equal. Wind: {len(wind_data)}, Solar: {len(solar_data)}, Whale: {len(whale_data)}."
        )

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
        state_history_list = np.empty(max_stages, dtype=tuple)
        energy_history_list = np.zeros(max_stages)
        u_k_list = np.zeros(max_stages)
        failure_prob_list = np.zeros(max_stages)
        state_history_list[0] = initial_state
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
    ):
        """Determine the best action based on energy and reward conditions."""
        is_reward_sufficient = whale_prob > threshold and solar_power_wpm2 > 0
        is_battery_sufficient = current_state[0] > (
            nightly_idle_soc + single_flight_soc
        )
        return 1 if is_reward_sufficient and is_battery_sufficient else 0

    def _determine_charge_threshold_action(
        self,
        current_state,
        nightly_idle_soc,
        single_flight_soc,
        charge_threshold,
        ):
        """Determine the best action based on stored energy."""
        is_battery_sufficient = current_state[0] > charge_threshold*100 or (
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
            value_list[idx] = self.mdp_model._alpha(
                k, current_state, action, collected_solar_power
            ) * self.transition_model.compute_probability(
                wind_speed, action, current_state
            )
        return 1 if whale_prob >= (value_list[0] - value_list[1]) else 0

    def _compute_failure_prob(
        self, wind_speed, best_action, current_state
    ):
        """Compute the probability of failure given the wind conditions and action."""
        return 1 - self.transition_model.compute_probability(
            wind_speed, best_action, current_state
        )

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
        return new_energy, self.calculate_new_state(
            best_action, new_energy, battery_capacity_J
        )

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
    ):
        """Finalize the simulation results."""
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
            )
        return reward, k, flight_minutes

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
        - wind_speed: Expected wind speed at the current stage
        - whale_prob_table: Table that maps the time of day to the probability of finding whales

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
        energy_change = (
            net_power * dt * 60 - required_takeoff_energy
        )  # Convert power to energy
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
            required_cruise_power = (
                plane.required_cruise_power
            )  # Assumed constants for flight
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

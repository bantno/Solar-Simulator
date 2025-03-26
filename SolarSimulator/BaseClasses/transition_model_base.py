from abc import ABC, abstractmethod
import numpy as np
import matplotlib.pyplot as plt
from BaseClasses.environment_provider_base import DeterministicEnvironmentProvider, AbstractEnvironmentProvider


class ActionSuccessProbabilityModel(ABC):
    """Abstract base class for computing P(S=1 | w_k), the probability of action success given wind speed."""

    @abstractmethod
    def compute_probability(self, wind_speed, action, state) -> np.ndarray:
        """
        Compute the probability of action success.

        Parameters:
        - wind_speed (float): Current wind speed.
        - action (int): Control action (1 = active action, 0 = passive action).
        - state (tuple): System state at the current stage.

        Returns:
        - float: Probability of success P(S=1 | w_k).
        """

    @staticmethod
    def check_and_broadcast(wind_speed, action, state) -> tuple:
        """
        Helper function to check the length of input variables and broadcast scalars
        to match the length of the largest input.

        Parameters:
        - wind_speed: Either a scalar or vector of length n.
        - action: Either a scalar or vector of length n.
        - state: Either a tuple (x,y) or a list of tuples [(x,y),(a,b)...] or a numpy array [[x,y],[a,b],...]

        Returns:
        - list: List of broadcasted inputs, ensuring they all have the same length.
        """
        wind_speed = np.atleast_1d(wind_speed)
        action = np.atleast_1d(action)
        state = np.atleast_2d(state)

        # Determine the maximum length from the inputs
        max_len = max(len(wind_speed), len(action), state.shape[0])

        # Broadcast the scalars or adjust the arrays to match max_len
        wind_speed = np.broadcast_to(wind_speed, max_len)
        action = np.broadcast_to(action, max_len)
        state = np.broadcast_to(state, (max_len, state.shape[1])) if state.shape[0] == 1 else state

        return wind_speed, action, state

    def visualize_success_probability(self):
        """
        Visualize the success probability for wind speeds from 0-40 for all combinations of action and state.
        """
        wind_speeds = np.linspace(0, 40, 400)
        actions = [0, 1]
        states = [(0, 0), (0, 1)]
        labels = [
            "Float",
            "Land",
            "Takeoff",
            "Fly"
        ]

        plt.figure(figsize=(12, 8))
        i=0
        for action in actions:
            for state in states:
                probabilities = self.compute_probability(wind_speeds, action, np.array([state] * len(wind_speeds)))
                plt.plot(wind_speeds, probabilities, label=labels[i])
                i+=1

        plt.xlabel("Wind Speed [m/s]")
        plt.ylabel("Success Probability")
        plt.title("Success Probability vs Wind Speed for Different Actions and States")
        plt.legend()
        plt.grid(True)
        plt.show()

class SigmoidSuccessProbability(ActionSuccessProbabilityModel):
    """Abstract base class for probability models using a sigmoid function."""

    @abstractmethod
    def compute_probability(self, wind_speed, action, state):
        """
        Compute the probability of success based on wind speed, action, and state.

        Parameters:
        - wind_speed (float): Current wind speed.
        - action (int): Control action (e.g., 1 = active action, 0 = passive action).
        - state (tuple): System state at the current stage.

        Returns:
        - float: Probability of success.
        """
        pass

    @abstractmethod
    def sigmoid(self, x, a, b):
        """
        Compute the sigmoid function.

        Parameters:
        - x (float): Input value (e.g., wind speed).
        - a (float): Sigmoid parameter for shift.
        - b (float): Sigmoid parameter for scale.

        Returns:
        - float: Computed sigmoid value.
        """
        pass

class LinearSuccessProbability(ActionSuccessProbabilityModel):
    """Alternative probability model where failure probability increases linearly with wind speed."""

    def __init__(self, failure_slope=0.05,name="linear"):
        """
        Initialize the linear failure probability model.

        Parameters:
        - failure_slope (float): Slope controlling how fast failure probability increases with wind speed.
        """
        self.failure_slope = failure_slope
        self.name=name

    def compute_probability(self, wind_speed, action, state):
        """
        Compute P(S=1 | w_k) using a linear failure probability model.

        Parameters:
        - wind_speed (float or np.array): Current wind speed or an array of wind speeds.
        - action (int or np.array): Control action (1 = active action, 0 = passive action), or an array of actions.
        - state (tuple or np.array): System state at the current stage, or an array of states.

        Returns:
        - np.array: Probabilities of success for each element.
        """
        # Use the helper function to handle broadcasting
        wind_speed, action, state = self.check_and_broadcast(wind_speed, action, state)

        vehicle_mode = state[:, 1]  # Mode: 0 = Floating, 1 = Flying, 2 = Broken

        success_prob = np.where(vehicle_mode == 2, 0, 1 - self.failure_slope * wind_speed)
        success_prob = np.where(success_prob < 0, 0, success_prob)

        return success_prob

class OnlySuccessProbability(ActionSuccessProbabilityModel):
    """Probability model using where a valid vehicle state and action always leads to a successful transition."""

    def __init__(self,name="nofail"):
        """
        Initialize the probability model with default parameters.
        """
        self.name=name

    def compute_probability(self, wind_speed, action, state):
        """
        Compute P(S=1 | w_k) based on the vehicle's state and action.

        Parameters:
        - wind_speed (float or np.array): Current wind speed or an array of wind speeds.
        - action (int or np.array): Control action (1 = active action, 0 = passive action), or an array of actions.
        - state (tuple or np.array): System state at the current stage, or an array of states.

        Returns:
        - np.array: Probabilities of success for each element.
        """

        wind_speed, action, state = self.check_and_broadcast(wind_speed, action, state)
        vehicle_mode = state[:, 1]

        # Initialize probabilities assuming success
        success_prob = np.ones_like(vehicle_mode, dtype=float)

        # Handle the broken vehicle case
        success_prob[vehicle_mode == 2] = 0

        # Handle action-state combinations with vectorized indexing
        # Takeoff (action == 1 and vehicle_mode == 0)
        success_prob[(action == 1) & (vehicle_mode == 0)] = 1
        # Floating (action == 0 and vehicle_mode == 0)
        success_prob[(action == 0) & (vehicle_mode == 0)] = 1
        # Flying (action == 1 and vehicle_mode == 1)
        success_prob[(action == 1) & (vehicle_mode == 1)] = 1
        # Landing (action == 0 and vehicle_mode == 1)
        success_prob[(action == 0) & (vehicle_mode == 1)] = 1

        # Check for invalid combinations in a single step
        invalid_combinations = (action == 1) & (
            vehicle_mode == 2
        )  # Invalid: action 1 and broken vehicle
        invalid_combinations |= (action == 0) & (
            vehicle_mode == 2
        )  # Invalid: action 0 and broken vehicle
        if np.any(invalid_combinations):
            raise ValueError("Invalid combination of action and vehicle mode.")

        return success_prob

class WindIndependentSuccessProbability(ActionSuccessProbabilityModel):
    """Probability model using where a valid vehicle state and action always leads to a successful transition."""

    def __init__(self,name="nowind"):
        """
        Initialize the probability model with default parameters.
        """
        self.name=name

    def compute_probability(self, wind_speed, action, state):
        """
        Compute P(S=1 | w_k) based on the vehicle's state and action.

        Parameters:
        - wind_speed (float or np.array): Current wind speed or an array of wind speeds.
        - action (int or np.array): Control action (1 = active action, 0 = passive action), or an array of actions.
        - state (tuple or np.array): System state at the current stage, or an array of states.

        Returns:
        - np.array: Probabilities of success for each element.
        """

        wind_speed, action, state = self.check_and_broadcast(wind_speed, action, state)
        vehicle_mode = state[:, 1]

        # Initialize probabilities assuming success
        success_prob = np.ones_like(vehicle_mode, dtype=float)

        # Handle the broken vehicle case
        success_prob[vehicle_mode == 2] = 0

        # Handle action-state combinations with vectorized indexing
        # Takeoff (action == 1 and vehicle_mode == 0)
        success_prob[(action == 1) & (vehicle_mode == 0)] = 0.99
        # Floating (action == 0 and vehicle_mode == 0)
        success_prob[(action == 0) & (vehicle_mode == 0)] = 0.9999
        # Flying (action == 1 and vehicle_mode == 1)
        success_prob[(action == 1) & (vehicle_mode == 1)] = 0.999
        # Landing (action == 0 and vehicle_mode == 1)
        success_prob[(action == 0) & (vehicle_mode == 1)] = 0.95

        # Check for invalid combinations in a single step
        invalid_combinations = (action == 1) & (
            vehicle_mode == 2
        )  # Invalid: action 1 and broken vehicle
        invalid_combinations |= (action == 0) & (
            vehicle_mode == 2
        )  # Invalid: action 0 and broken vehicle
        if np.any(invalid_combinations):
            raise ValueError("Invalid combination of action and vehicle mode.")

        return success_prob

class TestSuccessProbability(ActionSuccessProbabilityModel):
    """Probability model using where a valid vehicle state and action always leads to a successful transition."""

    def __init__(self, name="test"):
        """
        Initialize the probability model with default parameters.
        """
        self.name = name

    def compute_probability(self, wind_speed, action, state):
        """
        Compute P(S=1 | w_k) based on the vehicle's state and action.

        Parameters:
        - wind_speed (float or np.array): Current wind speed or an array of wind speeds.
        - action (int or np.array): Control action (1 = active action, 0 = passive action), or an array of actions.
        - state (tuple or np.array): System state at the current stage, or an array of states.

        Returns:
        - np.array: Probabilities of success for each element.
        """
        wind_speed, action, state = self.check_and_broadcast(wind_speed, action, state)

        vehicle_mode = state[:, 1]

        success_prob = np.ones_like(vehicle_mode, dtype=float)

        success_prob[vehicle_mode == 2] = 0

        # Handle action-state combinations with vectorized indexing
        # Takeoff (action == 1 and vehicle_mode == 0) -> probability 0.25
        success_prob[(action == 1) & (vehicle_mode == 0)] = 0.25
        # Floating (action == 0 and vehicle_mode == 0) -> probability 1
        success_prob[(action == 0) & (vehicle_mode == 0)] = 1
        # Flying (action == 1 and vehicle_mode == 1) -> probability 0.75
        success_prob[(action == 1) & (vehicle_mode == 1)] = 0.75
        # Landing (action == 0 and vehicle_mode == 1) -> probability 0.25
        success_prob[(action == 0) & (vehicle_mode == 1)] = 0.25

        return success_prob

class RealisticSuccessProbability(SigmoidSuccessProbability):
    """Sigmoid probability model using a logistic function for takeoff and landing failure probabilities."""

    def __init__(
        self,
        floating_failure=0.001,
        flying_failure=0.01,
        takeoff_params=(5, 0.5),
        landing_params=(4, 0.7),
        name = "realistic"
    ):
        self.a1, self.b1 = takeoff_params
        self.a2, self.b2 = landing_params
        self.floating_failure = floating_failure
        self.flying_failure = flying_failure
        self.name = name

    def compute_probability(self, wind_speed, action, state):
        """
        Compute P(S=1 | w_k) based on the vehicle's state and action.

        Parameters:
        - wind_speed (float or np.array): Current wind speed or an array of wind speeds.
        - action (int or np.array): Control action (1 = active action, 0 = passive action), or an array of actions.
        - state (tuple or np.array): System state at the current stage, or an array of states.

        Returns:
        - np.array: Probabilities of success for each element.
        """
        wind_speed, action, state = self.check_and_broadcast(wind_speed, action, state)
        vehicle_mode = state[:, 1]  # Mode: 0 = Floating, 1 = Flying, 2 = Broken

        # Initialize probabilities
        success_prob = np.ones_like(vehicle_mode, dtype=float)

        # Handle broken vehicle case (mode == 2)
        success_prob[vehicle_mode == 2] = 0

        # Compute failure probabilities for each action-state combination
        takeoff_mask = (action == 1) & (vehicle_mode == 0)
        floating_mask = (action == 0) & (vehicle_mode == 0)
        flying_mask = (action == 1) & (vehicle_mode == 1)
        landing_mask = (action == 0) & (vehicle_mode == 1)

        # Takeoff (action == 1 and vehicle_mode == 0)
        failure_prob_takeoff = self.sigmoid(wind_speed, self.a1, self.b1)
        success_prob[takeoff_mask] = 0.99 - failure_prob_takeoff[takeoff_mask]

        # Floating (action == 0 and vehicle_mode == 0)
        failure_prob_floating = self.floating_failure
        success_prob[floating_mask] = 1 - failure_prob_floating

        # Flying (action == 1 and vehicle_mode == 1)
        failure_prob_flying = self.sigmoid(wind_speed, self.a1, 0.35)
        success_prob[flying_mask] = 0.99 - failure_prob_flying[flying_mask]

        # Landing (action == 0 and vehicle_mode == 1)
        failure_prob_landing = self.sigmoid(wind_speed, self.a2, self.b2)
        success_prob[landing_mask] = 0.99 - failure_prob_landing[landing_mask]

        # Check for invalid combinations and raise error if found
        invalid_combinations = (action == 1) & (vehicle_mode == 2) | (action == 0) & (
            vehicle_mode == 2
        )
        if np.any(invalid_combinations):
            raise ValueError("Invalid combination of action and vehicle mode.")

        return success_prob

    def sigmoid(self, x, a, b):
        return 1 / (1.1 + np.exp(a - b * x))

class ModerateSuccessProbability(SigmoidSuccessProbability):
    """Sigmoid probability model using a logistic function for takeoff and landing failure probabilities."""

    def __init__(
        self,
        floating_failure=0.00001,
        flying_failure=0.0001,
        takeoff_params=(8, 0.5),
        landing_params=(7, 0.7),
        name = "moderate"
    ):
        self.a1, self.b1 = takeoff_params
        self.a2, self.b2 = landing_params
        self.floating_failure = floating_failure
        self.flying_failure = flying_failure
        self.name = name

    def compute_probability(self, wind_speed, action, state):
        """
        Compute P(S=1 | w_k) based on the vehicle's state and action.

        Parameters:
        - wind_speed (float or np.array): Current wind speed or an array of wind speeds.
        - action (int or np.array): Control action (1 = active action, 0 = passive action), or an array of actions.
        - state (tuple or np.array): System state at the current stage, or an array of states.

        Returns:
        - np.array: Probabilities of success for each element.
        """
        wind_speed, action, state = self.check_and_broadcast(wind_speed, action, state)
        vehicle_mode = state[:, 1]  # Mode: 0 = Floating, 1 = Flying, 2 = Broken

        # Initialize success probability to 1
        success_prob = np.ones_like(vehicle_mode, dtype=float)

        # Precompute sigmoid values for efficiency
        failure_prob_takeoff = self.sigmoid(wind_speed, self.a1, self.b1)
        failure_prob_landing = self.sigmoid(wind_speed, self.a2, self.b2)

        # Assign probabilities efficiently
        success_prob = np.where(vehicle_mode == 2, 0, success_prob)  # Broken vehicle always fails
        success_prob = np.where((action == 1) & (vehicle_mode == 0), 1 - failure_prob_takeoff, success_prob)
        success_prob = np.where((action == 0) & (vehicle_mode == 0), 1 - self.floating_failure, success_prob)
        success_prob = np.where((action == 1) & (vehicle_mode == 1), 1 - self.flying_failure, success_prob)
        success_prob = np.where((action == 0) & (vehicle_mode == 1), 1 - failure_prob_landing, success_prob)

        # Raise error for invalid action-mode combinations
        if np.any(vehicle_mode == 2):  # Broken vehicle should not have valid actions
            raise ValueError("Invalid combination of action and vehicle mode.")

        return success_prob

    def sigmoid(self, x, a, b):
        return 1 / (1.1 + np.exp(a - b * x))
    
class SomeSuccessProbability(SigmoidSuccessProbability):
    """Sigmoid probability model using a logistic function for takeoff and landing failure probabilities."""

    def __init__(
        self,
        floating_failure=0.00001,
        flying_failure=0.0001,
        takeoff_params=(8, 0.5),
        landing_params=(7, 0.5),
        name = "some"
    ):
        self.a1, self.b1 = takeoff_params
        self.a2, self.b2 = landing_params
        self.floating_failure = floating_failure
        self.flying_failure = flying_failure
        self.name = name

    def compute_probability(self, wind_speed, action, state):
        """
        Compute P(S=1 | w_k) based on the vehicle's state and action.

        Parameters:
        - wind_speed (float or np.array): Current wind speed or an array of wind speeds.
        - action (int or np.array): Control action (1 = active action, 0 = passive action), or an array of actions.
        - state (tuple or np.array): System state at the current stage, or an array of states.

        Returns:
        - np.array: Probabilities of success for each element.
        """
        wind_speed, action, state = self.check_and_broadcast(wind_speed, action, state)
        vehicle_mode = state[:, 1]  # Mode: 0 = Floating, 1 = Flying, 2 = Broken

        # Initialize success probability to 1
        success_prob = np.ones_like(vehicle_mode, dtype=float)

        # Precompute sigmoid values for efficiency
        failure_prob_takeoff = self.sigmoid(wind_speed, self.a1, self.b1)
        failure_prob_landing = self.sigmoid(wind_speed, self.a2, self.b2)

        # Assign probabilities efficiently
        success_prob = np.where(vehicle_mode == 2, 0, success_prob)  # Broken vehicle always fails
        success_prob = np.where((action == 1) & (vehicle_mode == 0), 1 - failure_prob_takeoff, success_prob)
        success_prob = np.where((action == 0) & (vehicle_mode == 0), 1 - self.floating_failure, success_prob)
        success_prob = np.where((action == 1) & (vehicle_mode == 1), 1 - self.flying_failure, success_prob)
        success_prob = np.where((action == 0) & (vehicle_mode == 1), 1 - failure_prob_landing, success_prob)

        # Raise error for invalid action-mode combinations
        if np.any(vehicle_mode == 2):  # Broken vehicle should not have valid actions
            raise ValueError("Invalid combination of action and vehicle mode.")

        return success_prob

    def sigmoid(self, x, a, b):
        return 1 / (1.1 + np.exp(a - b * x))

class OptimisticSuccessProbability(SigmoidSuccessProbability):
    """Sigmoid probability model using a logistic function for takeoff and landing failure probabilities."""

    def __init__(
        self,
        floating_failure=0.0,
        flying_failure=0.0,
        takeoff_params=(17, 0.5),
        landing_params=(16, 0.5),
        name = "optimistic"
    ):
        self.a1, self.b1 = takeoff_params
        self.a2, self.b2 = landing_params
        self.floating_failure = floating_failure
        self.flying_failure = flying_failure
        self.name=name

    def compute_probability(self, wind_speed, action, state):
        """
        Compute P(S=1 | w_k) based on the vehicle's state and action.

        Parameters:
        - wind_speed (float or np.array): Current wind speed or an array of wind speeds.
        - action (int or np.array): Control action (1 = active action, 0 = passive action), or an array of actions.
        - state (tuple or np.array): System state at the current stage, or an array of states.

        Returns:
        - np.array: Probabilities of success for each element.
        """
        wind_speed, action, state = self.check_and_broadcast(wind_speed, action, state)
        vehicle_mode = state[:, 1]  # Mode: 0 = Floating, 1 = Flying, 2 = Broken

        success_prob = np.zeros_like(vehicle_mode, dtype=float)

        # Handle broken vehicle case (mode == 2)
        success_prob[vehicle_mode == 2] = 0

        # Compute failure probabilities for each action-state combination
        takeoff_mask = (action == 1) & (vehicle_mode == 0)
        floating_mask = (action == 0) & (vehicle_mode == 0)
        flying_mask = (action == 1) & (vehicle_mode == 1)
        landing_mask = (action == 0) & (vehicle_mode == 1)

        # Takeoff (action == 1 and vehicle_mode == 0)
        failure_prob_takeoff = self.sigmoid(wind_speed, self.a1, self.b1)
        success_prob[takeoff_mask] = 1 - failure_prob_takeoff[takeoff_mask]

        # Floating (action == 0 and vehicle_mode == 0)
        success_prob[floating_mask] = 1 - self.floating_failure

        # Flying (action == 1 and vehicle_mode == 1)
        success_prob[flying_mask] = 1 - self.flying_failure

        # Landing (action == 0 and vehicle_mode == 1)
        failure_prob_landing = self.sigmoid(wind_speed, self.a2, self.b2)
        success_prob[landing_mask] = 1 - failure_prob_landing[landing_mask]

        # Check for invalid combinations and raise an error if found
        invalid_combinations = (action == 1) & (vehicle_mode == 2) | (action == 0) & (
            vehicle_mode == 2
        )
        if np.any(invalid_combinations):
            raise ValueError("Invalid combination of action and vehicle mode.")

        return success_prob

    def sigmoid(self, x, a, b):
        return 1 / (1 + np.exp(a - b * x))

class ProbabilityModelFactory:
    """Factory class to select the appropriate action success probability model."""

    # Define a dictionary mapping model names to their corresponding classes
    models = {
        "linear": LinearSuccessProbability,
        "nofail": OnlySuccessProbability,
        "realistic": RealisticSuccessProbability,
        "optimistic": OptimisticSuccessProbability,
        "test": TestSuccessProbability,
        "moderate": ModerateSuccessProbability,
        "some": SomeSuccessProbability,
        "nowind": WindIndependentSuccessProbability,
        # Add more models here as needed
        # 'new_model': NewModelClass,
    }

    @staticmethod
    def select_probability_model(model_name: str, **kwargs) -> ActionSuccessProbabilityModel:
        """
        Select the probability model based on the given string.

        Parameters:
        - model_name (str): Name of the model (e.g., 'sigmoid', 'linear', etc.).
        - kwargs: Additional parameters for model initialization.

        Returns:
        - ActionSuccessProbabilityModel: An instance of the selected model.
        """
        model_name = model_name.lower()  # Convert to lowercase to handle case insensitivity

        # Check if the model exists in the dictionary
        if model_name in ProbabilityModelFactory.models:
            # Return an instance of the corresponding class
            return ProbabilityModelFactory.models[model_name](**kwargs)
        else:
            raise ValueError(f"Unknown model name: {model_name}")

    @staticmethod
    def list_models() -> list:
        """
        List available probability models.

        Returns:
        - list: Names of available models.
        """
        return list(ProbabilityModelFactory.models.keys())

    @staticmethod
    def is_model_available(model_name: str) -> bool:
        """
        Returns Boolean describing if the specified model name matches
        one of the defined models.
        """
        return model_name.lower() in ProbabilityModelFactory.list_models()

class AbstractTransitionLogic(ABC):
    @property
    @abstractmethod
    def battery_capacity_joules(self) -> float:
        pass

    @property
    @abstractmethod
    def soc_increment(self) -> float:
        pass

    @abstractmethod
    def transition(self, states: np.ndarray, actions: np.ndarray, t: int) -> np.ndarray:
        pass

    def soc_to_energy(self, soc: np.ndarray) -> np.ndarray:
        return (soc / 100.0) * self.battery_capacity_joules

    def energy_to_soc(self, next_energy: np.ndarray) -> np.ndarray:
        raw_soc = (next_energy / self.battery_capacity_joules) * 100.0
        floored_soc = np.floor(raw_soc / self.soc_increment) * self.soc_increment
        return floored_soc

    def min_to_seconds(self, minutes: float) -> float:
        return minutes * 60.

class DeterministicTransitionLogic(AbstractTransitionLogic):
    def __init__(self, battery_capacity_joules: float, soc_increment: float,
                 idle_power: float, cruise_power: float, takeoff_power: float,
                 delta_t: float, transition_model, env_provider: AbstractEnvironmentProvider):
        self._battery_capacity_joules = battery_capacity_joules
        self._soc_increment = soc_increment
        self.idle_power = idle_power
        self.cruise_power = cruise_power
        self.takeoff_power = takeoff_power
        self.delta_t = delta_t
        self.transition_model = transition_model
        self.env_provider = env_provider

    @property
    def battery_capacity_joules(self) -> float:
        return self._battery_capacity_joules

    @property
    def soc_increment(self) -> float:
        return self._soc_increment

    def sample_energy_gain(self, stage, n):
        return self.env_provider.sample_sunlight(stage, n)

    def sample_wind_speeds(self, stage, n):
        return self.env_provider.sample_wind_speed(stage, n)

    def transition(self, states: np.ndarray, actions: np.ndarray, t: int) -> np.ndarray:
        moored_float_energy = self.idle_power * self.min_to_seconds(self.delta_t)
        takeoff_energy = (self.cruise_power + self.takeoff_power) * self.min_to_seconds(self.delta_t)
        land_energy = self.cruise_power * self.min_to_seconds(self.delta_t) / 4
        continue_flight_energy = self.cruise_power * self.min_to_seconds(self.delta_t)
        energy_lookup = np.array([
            [moored_float_energy, takeoff_energy],
            [land_energy, continue_flight_energy],
            [0, 0]
        ])
        energy_consumption = energy_lookup[states[:, 1].astype(int), actions]
        energy_gain = self.sample_energy_gain(t, states.shape[0])
        current_energy = self.soc_to_energy(states[:, 0])
        next_energy = current_energy + energy_gain - energy_consumption
        next_soc = np.clip(self.energy_to_soc(next_energy), -1., 100.)
        next_mode = np.where(next_soc <= 0, 2, np.where(actions == 0, 0, 1))
        next_state = np.column_stack((next_soc, next_mode))
        mode2_mask = next_state[:, 1] == 2
        next_state[mode2_mask, 0] = -1.0
        return next_state


class StochasticTransitionLogic(AbstractTransitionLogic):
    def __init__(self, battery_capacity_joules: float, soc_increment: float,
                 idle_power: float, cruise_power: float, takeoff_power: float,
                 delta_t: float,
                 transition_model, env_provider):
        self._battery_capacity_joules = battery_capacity_joules
        self._soc_increment = soc_increment
        self.idle_power = idle_power
        self.cruise_power = cruise_power
        self.takeoff_power = takeoff_power
        self.delta_t = delta_t
        self.transition_model = transition_model
        self.env_provider = env_provider

    @property
    def battery_capacity_joules(self) -> float:
        return self._battery_capacity_joules

    @property
    def soc_increment(self) -> float:
        return self._soc_increment

    def sample_energy_gain(self, stage, n):
        return self.env_provider.sample_sunlight(stage, n)

    def sample_wind_speeds(self, stage, n):
        return self.env_provider.sample_wind_speed(stage, n)

    def transition(self, states: np.ndarray, actions: np.ndarray, t: int) -> np.ndarray:
        moored_float_energy = self.idle_power * self.min_to_seconds(self.delta_t)
        takeoff_energy = (self.cruise_power + self.takeoff_power) * self.min_to_seconds(self.delta_t)
        land_energy = self.cruise_power * self.min_to_seconds(self.delta_t) / 4
        continue_flight_energy = self.cruise_power * self.min_to_seconds(self.delta_t)
        energy_lookup = np.array([
            [moored_float_energy, takeoff_energy],
            [land_energy, continue_flight_energy],
            [0, 0]
        ])
        energy_consumption = energy_lookup[states[:, 1].astype(int), actions]
        energy_gain = self.sample_energy_gain(t, states.shape[0])
        current_energy = self.soc_to_energy(states[:, 0])
        next_energy = current_energy + energy_gain - energy_consumption
        next_soc = np.clip(self.energy_to_soc(next_energy), -1., 100.)
        next_mode = np.where(next_soc <= 0, 2, np.where(actions == 0, 0, 1))
        next_state = np.column_stack((next_soc, next_mode))
        wind_speeds = self.sample_wind_speeds(t, states.shape[0])
        success_probabilities = self.transition_model.compute_probability(wind_speeds, actions, states)
        false_states = np.tile(np.array([-1.0, 2]), (states.shape[0], 1))
        random_vals = np.random.rand(states.shape[0])[:, np.newaxis]
        next_states = np.where(random_vals < success_probabilities[:, np.newaxis],
                                 next_state,
                                 false_states)
        mode2_mask = next_states[:, 1] == 2
        next_states[mode2_mask, 0] = -1.0
        return next_states
    
    def transition_continuous_energy(self,current_energy, states: np.ndarray, actions: np.ndarray, t: int) -> np.ndarray:
        moored_float_energy = self.idle_power * self.min_to_seconds(self.delta_t)
        takeoff_energy = (self.cruise_power + self.takeoff_power) * self.min_to_seconds(self.delta_t)
        land_energy = self.cruise_power * self.min_to_seconds(self.delta_t) / 4
        continue_flight_energy = self.cruise_power * self.min_to_seconds(self.delta_t)
        energy_lookup = np.array([
            [moored_float_energy, takeoff_energy],
            [land_energy, continue_flight_energy],
            [0, 0]
        ])
        energy_consumption = energy_lookup[states[:, 1].astype(int), actions]
        energy_gain = self.sample_energy_gain(t, states.shape[0])
        next_energy = current_energy + energy_gain - energy_consumption
        next_soc = np.clip(self.energy_to_soc(next_energy), -1., 100.)
        next_mode = np.where(next_soc <= 0, 2, np.where(actions == 0, 0, 1))
        next_state = np.column_stack((next_soc, next_mode))
        wind_speeds = self.sample_wind_speeds(t, states.shape[0])
        success_probabilities = self.transition_model.compute_probability(wind_speeds, actions, states)
        false_states = np.tile(np.array([-1.0, 2]), (states.shape[0], 1))
        random_vals = np.random.rand(states.shape[0])[:, np.newaxis]
        next_states = np.where(random_vals < success_probabilities[:, np.newaxis],
                                 next_state,
                                 false_states)
        mode2_mask = next_states[:, 1] == 2
        next_states[mode2_mask, 0] = -1.0
        return next_states, next_energy

# Example usage:
if __name__ == "__main__":
    factory = ProbabilityModelFactory()
    model = factory.select_probability_model(
        "moderate",
    )
    print(model)
    model.visualize_success_probability()

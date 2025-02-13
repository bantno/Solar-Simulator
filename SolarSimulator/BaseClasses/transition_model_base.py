from abc import ABC, abstractmethod
import numpy as np

class ActionSuccessProbabilityModel(ABC):
    """Abstract base class for computing P(S=1 | w_k), the probability of action success given wind speed."""
    
    @abstractmethod
    def compute_probability(self, wind_speed, action, state):
        """
        Compute the probability of action success.

        Parameters:
        - wind_speed (float): Current wind speed.
        - action (int): Control action (1 = active action, 0 = passive action).
        - state (tuple): System state at the current stage.

        Returns:
        - float: Probability of success P(S=1 | w_k).
        """
        pass


class SigmoidSuccessProbability(ABC):
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
    
    def __init__(self, failure_slope=0.05):
        """
        Initialize the linear failure probability model.

        Parameters:
        - failure_slope (float): Slope controlling how fast failure probability increases with wind speed.
        """
        self.failure_slope = failure_slope

    def compute_probability(self, wind_speed, action, state):
        """
        Compute P(S=1 | w_k) using a linear failure probability model.

        Parameters:
        - wind_speed (float): Current wind speed.
        - action (int): Control action (1 = active action, 0 = passive action).
        - state (tuple): System state at the current stage.

        Returns:
        - float: Probability of success.
        """
        vehicle_mode = state[1]

        if vehicle_mode == 2:
            return 0  # No success if broken.

        base_success = max(0, 1 - self.failure_slope * wind_speed)
        return base_success

class OnlySuccessProbability(ActionSuccessProbabilityModel):
    """Probability model using where a valid vehicle state and action always leads to a successful transition."""
    
    def __init__(self):
        """
        Initialize the probability model with default parameters.
        """
        pass

    def compute_probability(self, wind_speed, action, state):
        """
        Compute P(S=1 | w_k) based on the vehicle's state and action.

        Parameters:
        - wind_speed (float): Current wind speed.
        - action (int): Control action (1 = active action, 0 = passive action).
        - state (tuple): System state at the current stage.

        Returns:
        - float: Probability of success.
        """
        vehicle_mode = state[1]  # Mode: 0 = Floating, 1 = Flying, 2 = Broken
        
        if vehicle_mode == 2:
            return 0  # No success if the vehicle is broken.

        if action == 1 and vehicle_mode == 0:  # Takeoff
            return 1
        elif action == 0 and vehicle_mode == 0:  # Floating
            return 1
        elif action == 1 and vehicle_mode == 1:  # Flying
            return 1
        elif action == 0 and vehicle_mode == 1:  # Landing
            return 1
        else:
            raise ValueError("Invalid combination of action and vehicle mode.")

class RealisticSuccessProbability(SigmoidSuccessProbability):
    """Sigmoid probability model using a logistic function for takeoff and landing failure probabilities."""

    def __init__(self, floating_failure=0.0, flying_failure=0.0, takeoff_params=(5, 0.5), landing_params=(4, 0.5)):
        self.a1, self.b1 = takeoff_params
        self.a2, self.b2 = landing_params
        self.floating_failure = floating_failure
        self.flying_failure = flying_failure

    def compute_probability(self, wind_speed, action, state):
        vehicle_mode = state[1]  # Mode: 0 = Floating, 1 = Flying, 2 = Broken
        
        if vehicle_mode == 2:
            return 0  # No success if the vehicle is broken.

        if action == 1 and vehicle_mode == 0:  # Takeoff
            failure_prob = self.sigmoid(wind_speed, self.a1, self.b1)
            return 0.99-failure_prob
        elif action == 0 and vehicle_mode == 0:  # Floating
            failure_prob = self.floating_failure
            return 1- failure_prob
        elif action == 1 and vehicle_mode == 1:  # Flying
            failure_prob = self.flying_failure
            return 1- failure_prob
        elif action == 0 and vehicle_mode == 1:  # Landing
            failure_prob = self.sigmoid(wind_speed, self.a2, self.b2)
            return 0.99-failure_prob
        else:
            raise ValueError("Invalid combination of action and vehicle mode.")
        
    def sigmoid(self, x, a, b):
        return 1 / (1.1 + np.exp(a - b * x))
    
class OptimisticSuccessProbability(SigmoidSuccessProbability):
    """Sigmoid probability model using a logistic function for takeoff and landing failure probabilities."""

    def __init__(self, floating_failure=0.0, flying_failure=0.0, takeoff_params=(17, 0.5), landing_params=(16, 0.5)):
        self.a1, self.b1 = takeoff_params
        self.a2, self.b2 = landing_params
        self.floating_failure = floating_failure
        self.flying_failure = flying_failure

    def compute_probability(self, wind_speed, action, state):
        vehicle_mode = state[1]  # Mode: 0 = Floating, 1 = Flying, 2 = Broken
        
        if vehicle_mode == 2:
            return 0  # No success if the vehicle is broken.

        if action == 1 and vehicle_mode == 0:  # Takeoff
            failure_prob = self.sigmoid(wind_speed, self.a1, self.b1)
        elif action == 0 and vehicle_mode == 0:  # Floating
            failure_prob = self.floating_failure
        elif action == 1 and vehicle_mode == 1:  # Flying
            failure_prob = self.flying_failure
        elif action == 0 and vehicle_mode == 1:  # Landing
            failure_prob = self.sigmoid(wind_speed, self.a2, self.b2)
        else:
            raise ValueError("Invalid combination of action and vehicle mode.")

        return 1 - failure_prob
        
    def sigmoid(self, x, a, b):
        return 1 / (1 + np.exp(a - b * x))
class ProbabilityModelFactory:
    """Factory class to select the appropriate action success probability model."""
    
    # Define a dictionary mapping model names to their corresponding classes
    models = {
        'linear': LinearSuccessProbability,
        'nofail': OnlySuccessProbability,
        'realistic': RealisticSuccessProbability,
        'optimistic': OptimisticSuccessProbability,
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
    def list_models():
        """
        List available probability models.

        Returns:
        - list: Names of available models.
        """
        return list(ProbabilityModelFactory.models.keys())

# Example usage:
if __name__ == "__main__":
    factory = ProbabilityModelFactory()
    model = factory.select_probability_model('sigmoid', failure_prob=0.8, takeoff_params=(18, 0.6), landing_params=(15, 0.7))
    print(model)


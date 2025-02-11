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

class SigmoidSuccessProbability(ActionSuccessProbabilityModel):
    """Default probability model using a logistic function for takeoff and landing failure probabilities."""
    
    def __init__(self, failure_prob=1.0, takeoff_params=(17, 0.5), landing_params=(16, 0.5)):
        """
        Initialize the probability model with default parameters.

        Parameters:
        - failure_prob (float): Base failure probability.
        - takeoff_params (tuple): (a1, b1) parameters for the takeoff logistic function.
        - landing_params (tuple): (a2, b2) parameters for the landing logistic function.
        """
        self.a1, self.b1 = takeoff_params
        self.a2, self.b2 = landing_params

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
            return 1 - 1 / (1 + np.exp(self.a1 - self.b1 * wind_speed))
        elif action == 0 and vehicle_mode == 0:  # Floating
            return 1  # Always successful (ignoring stochastic failures for now).
        elif action == 1 and vehicle_mode == 1:  # Flying
            return 1  # Always successful.
        elif action == 0 and vehicle_mode == 1:  # Landing
            return 1 - 1 / (1 + np.exp(self.a2 - self.b2 * wind_speed))
        else:
            raise ValueError("Invalid combination of action and vehicle mode.")

class LinearFailureProbability(ActionSuccessProbabilityModel):
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

class ProbabilityModelFactory:
    """Factory class to select the appropriate action success probability model."""
    
    # Define a dictionary mapping model names to their corresponding classes
    models = {
        'sigmoid': SigmoidSuccessProbability,
        'linear': LinearFailureProbability,
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


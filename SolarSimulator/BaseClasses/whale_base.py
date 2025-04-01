import numpy as np

class WhaleRewardSeries:
    """Abstract base class for creating whale reward series."""
    def create_series(self, horizon):
        raise NotImplementedError("Subclasses must implement create_series()")

class SinusoidalWhaleRewardSeries(WhaleRewardSeries):
    """Creates a sinusoidal whale reward series."""
    def create_series(self, horizon):
        x = np.linspace(np.pi, np.pi * 60, horizon)
        return 0.5 * np.sin(x) + 0.5

class ConstantWhaleRewardSeries(WhaleRewardSeries):
    """Creates a constant whale reward series."""
    def create_series(self, horizon):
        return np.full(horizon, 0.5)

def whale_reward_series_factory(series_type, horizon):
    """
    Factory method to create the desired whale reward series.
    
    Parameters:
        series_type (str): The type of series ('sinusoidal', 'constant', etc.)
        horizon (int): Number of time steps to create the series for.
    
    Returns:
        numpy.ndarray: The whale reward series.
    """
    if series_type == "sinusoidal":
        factory = SinusoidalWhaleRewardSeries()
    elif series_type == "constant":
        factory = ConstantWhaleRewardSeries()
    else:
        raise ValueError(f"Unknown whale reward series type: {series_type}")
    return factory.create_series(horizon)
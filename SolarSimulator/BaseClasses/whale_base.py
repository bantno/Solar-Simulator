import numpy as np
import matplotlib.pyplot as plt

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
    
class RealWhaleRewardSeries(WhaleRewardSeries):
    """Creates a real whale reward series."""
    def create_series(self, horizon):
        # Define the base values for each 2-hour block (12 blocks per day)
        block_values = np.array([
            0.00,  # 00:00-02:00
            0.00,  # 02:00-04:00
            0.00,  # 04:00-06:00
            0.082, # 06:00-08:00
            0.098, # 08:00-10:00
            0.095, # 10:00-12:00
            0.217, # 12:00-14:00
            0.215, # 14:00-16:00
            0.183, # 16:00-18:00
            0.278, # 18:00-20:00
            0.000,  # 20:00-22:00
            0.000   # 22:00-24:00
        ])
        
        # Each 2-hour block corresponds to 8 steps of 15 minutes (2 hours = 120 minutes, and 120/15 = 8)
        daily_series = np.repeat(block_values, 8)  # Creates an array of length 96 (12*8)
        
        # If the simulation horizon exceeds one day, tile the daily series accordingly.
        full_series = np.tile(daily_series, int(np.ceil(horizon / 96)))
        
        # Return the series truncated to the desired horizon length.
        return full_series[:horizon]

class WhaleRewardSeriesFactory:
    """Factory class to create whale reward series."""
    
    _series_mapping = {
        "sinusoidal": SinusoidalWhaleRewardSeries,
        "constant": ConstantWhaleRewardSeries,
        "real": RealWhaleRewardSeries,
    }
    
    @classmethod
    def create_series(self, series_type, horizon):
        """Creates the desired whale reward series.

        Args:
            series_type (str): The type of series ('sinusoidal', 'constant', 'real', etc.)
            horizon (int): Number of time steps to create the series for.

        Returns:
            numpy.ndarray: The whale reward series.
        """
        try:
            series_class = self._series_mapping[series_type]
        except KeyError:
            raise ValueError(f"Unknown whale reward series type: {series_type}")
        return series_class().create_series(horizon)
    
if __name__ == "__main__":
    # Generate one day's series (96 timesteps of 15 minutes each)
    horizon = 96
    series = WhaleRewardSeriesFactory.create_series('real', horizon)

    # Convert each timestep to hours of day
    hours = np.arange(horizon) * 0.25  # 0.25 hours per timestep

    plt.figure(figsize=(10, 4))
    plt.plot(hours, series, marker='.')
    plt.xlabel('Hour of Day')
    plt.ylabel('Whale Observation Probability')
    plt.title('Real Whale Observation Probability Over 24 Hours')
    plt.xticks(np.arange(0, 25, 2))
    plt.xlim(0, 24)
    plt.grid(True)
    plt.tight_layout()
    plt.show()
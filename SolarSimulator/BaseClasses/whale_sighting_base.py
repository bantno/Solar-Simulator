import pandas as pd

class WhaleSightingProbability:
    def __init__(self):
        # Define the time intervals in minutes from midnight (0-1439)
        self.time_intervals = [
            (0, 120),     # 0000-0200
            (120, 240),   # 0200-0400
            (240, 360),   # 0400-0600
            (360, 480),   # 0600-0800
            (480, 600),   # 0800-1000
            (600, 720),   # 1000-1200
            (720, 840),   # 1200-1400
            (840, 960),   # 1400-1600
            (960, 1080),  # 1600-1800
            (1080, 1200), # 1800-2000
            (1200, 1320), # 2000-2200
            (1320, 1440)  # 2200-2400
        ]
        
        # Define the sighting probabilities
        self.sighting_probabilities = [0.073, 0.093, 0.065, 0.082, 0.098, 0.217, 0.183, 0.278, 0.183, 0.204, 0.090, 0.090]

        # Create a pandas DataFrame to store the data
        self.df = pd.DataFrame({
            'Time Interval': self.time_intervals,
            'Sighting Probability': self.sighting_probabilities
        })

    def get_probability_by_minutes(self, minutes):
        """
        Get the whale sighting probability given the number of minutes. 
        Handles minutes greater than 1439 by wrapping around (mod 1440).
        
        :param minutes: Minutes from midnight (can exceed 1439)
        :return: Sighting probability (float)
        """
        if minutes < 0:
            raise ValueError("Minutes must be non-negative.")

        # Wrap around using modulo to keep minutes within the 0-1439 range
        minutes = minutes % 1440

        # Find the corresponding time interval for the given minutes
        for idx, (start, end) in enumerate(self.df['Time Interval']):
            if start <= minutes < end:
                return self.df['Sighting Probability'].iloc[idx]
class WhaleSighting:
    def __init__(self):
        # Define start times and corresponding probabilities as a dictionary
        self.probability_map = {
            0: 0.073, 120: 0.093, 240: 0.065, 360: 0.082,
            480: 0.098, 600: 0.217, 720: 0.183, 840: 0.278,
            960: 0.183, 1080: 0.204, 1200: 0.090, 1320: 0.090
        }

    def get_sighting_probability(self, current_step, timestep, start_time):
        # Calculate the current time in minutes
        current_time = start_time + (current_step * timestep)
        
        # Find the nearest start time by rounding down to the closest 120-minute mark
        nearest_start = (current_time // 120) * 120
        
        # Return the probability, or None if out of range
        return self.probability_map.get(nearest_start)
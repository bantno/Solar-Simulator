class WhaleSighting:
    def __init__(self):
        # Define start times and corresponding probabilities as a dictionary
        # self.probability_map = {
        #     0: 0.073, 120: 0.093, 240: 0.065, 360: 0.082,
        #     480: 0.098, 600: 0.217, 720: 0.183, 840: 0.278,
        #     960: 0.183, 1080: 0.204, 1200: 0.090, 1320: 0.090
        # }
        self.probability_map = {
        360: 0.,    # 0 (midnight) + 6 hours = 6:00 AM UTC
        480: 0.,    # 2:00 AM + 6 hours = 8:00 AM UTC
        600: 0.,    # 4:00 AM + 6 hours = 10:00 AM UTC
        720: 0.082,  # 6:00 AM + 6 hours = 12:00 PM (noon) UTC
        840: 0.098,  # 8:00 AM + 6 hours = 2:00 PM UTC
        960: 0.217,  # 10:00 AM + 6 hours = 4:00 PM UTC
        1080: 0.183,  # 12:00 PM + 6 hours = 6:00 PM UTC
        1200: 0.278,  # 2:00 PM + 6 hours = 8:00 PM UTC
        1320: 0.183,
        0: 0.,    # 4:00 PM + 6 hours = 10:00 PM UTC (wraps to 0 for midnight the next day)
        120: 0.,     # 6:00 PM + 6 hours = 12:00 AM (midnight) UTC (wraps to 120 minutes past midnight)
        240: 0.,     # 8:00 PM + 6 hours = 2:00 AM UTC
        }



    def get_sighting_probability(self, current_step, timestep, start_time):
        # Calculate the current time in minutes
        current_time = (start_time + (current_step * timestep))%1440
        
        # Find the nearest start time by rounding down to the closest 120-minute mark
        nearest_start = (current_time // 120) * 120
        
        # Return the probability, or None if out of range
        return self.probability_map.get(nearest_start)
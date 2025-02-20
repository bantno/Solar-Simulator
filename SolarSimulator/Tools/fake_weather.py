import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timezone, timedelta

class FakeDataGenerator:
    def __init__(self, start_time, end_time, freq="15min"):
        self.time_index = pd.date_range(start=start_time, end=end_time, freq=freq, tz="UTC-06:00")
        self.cases = self._initialize_cases()

    def _initialize_cases(self):
        return {
            "constant_low": ("constant_low", "constant"),
            "constant_high": ("constant_high", "constant"),
            "low_to_high": ("low_to_high", "low_to_high"),
            "high_to_low": ("high_to_low", "high_to_low"),
            "high_except_short_low": ("high_except_short_low", "constant")
        }

    def _generate_solar_radiation(self):
        hours = self.time_index.hour
        return np.where((hours >= 6) & (hours < 14), 342, 0)

    def _generate_wind_data(self, pattern, low_speed=5, high_speed=20):
        patterns = {
            'constant_low': np.full(self.time_index.size, low_speed),
            'constant_high': np.full(self.time_index.size, high_speed),
            'low_to_high': np.concatenate((np.full(len(self.time_index)//2, low_speed), np.full(len(self.time_index)-len(self.time_index)//2, high_speed))),
            'high_to_low': np.concatenate((np.full(len(self.time_index)//2, high_speed), np.full(len(self.time_index)-len(self.time_index)//2, low_speed))),
            'high_except_short_low': np.full(self.time_index.size, high_speed)
        }
        if pattern == 'high_except_short_low':
            patterns[pattern][len(self.time_index)//3:len(self.time_index)//2] = low_speed
        wind_speeds = patterns.get(pattern)
        wind_directions = np.random.uniform(0, 360, size=self.time_index.size)
        return wind_speeds, wind_directions

    def _generate_whale_probability(self, pattern, low_prob=0.1, high_prob=0.9):
        solar = self._generate_solar_radiation()
        patterns = {
            'constant': np.where(solar > 0, 0.5, 0),
            'low_to_high': np.concatenate((np.full(len(self.time_index)//2, low_prob), np.full(len(self.time_index)-len(self.time_index)//2, high_prob))),
            'high_to_low': np.concatenate((np.full(len(self.time_index)//2, high_prob), np.full(len(self.time_index)-len(self.time_index)//2, low_prob)))
        }
        whale_prob = patterns.get(pattern)
        whale_prob[solar == 0] = 0
        return whale_prob

    def generate_case(self, wind_pattern, whale_pattern):
        solar_radiation = self._generate_solar_radiation()
        wind_speeds, wind_directions = self._generate_wind_data(wind_pattern)
        whale_prob = self._generate_whale_probability(whale_pattern)
        
        hours = self.time_index.hour
        beta_alpha = np.where((hours >= 6) & (hours < 14), 10, 1)
        beta_beta = np.where((hours >= 6) & (hours < 14), 30, 40)
        k = [10] * len(self.time_index)
        scale = wind_speeds

        expected_data = pd.DataFrame({
            "month": self.time_index.month,
            "day": self.time_index.day,
            "hour": self.time_index.hour,
            "minute": self.time_index.minute,
            "expected_solar_rad": solar_radiation,
            "expected_wind_speed": wind_speeds,
            "expected_whale_prob": whale_prob,
            "beta_alpha": beta_alpha,
            "beta_beta": beta_beta,
            "weibull_k": k,
            "weibull_loc": [0] * len(self.time_index),
            "weibull_scale": scale,
        }, index=self.time_index)

        actual_data = pd.DataFrame({
            "wind_speed_10m": wind_speeds,
            "wind_direction_10m": wind_directions,
            "shortwave_radiation": solar_radiation,
            "whale_observation_probability": whale_prob
        }, index=self.time_index)

        combined_data = pd.concat([expected_data, actual_data], axis=1)
        filename = f"data_wind-{wind_pattern}_whale-{whale_pattern}.pkl"
        with open(filename, 'wb') as f:
            pd.to_pickle(combined_data, f)
        return expected_data, actual_data

    def visualize_data(self, expected, actual,  case_name="Case"):
        fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
        fig.suptitle(case_name, fontsize=16)

        expected[["expected_solar_rad"]].plot(ax=axes[0], label="Expected Solar Rad", linestyle='--')
        actual[["shortwave_radiation"]].plot(ax=axes[0], label="Actual Solar Rad")

        expected[["expected_wind_speed"]].plot(ax=axes[1], label="Expected Wind Speed", linestyle='--')
        actual[["wind_speed_10m"]].plot(ax=axes[1], label="Actual Wind Speed")

        expected[["expected_whale_prob"]].plot(ax=axes[2], label="Expected Whale Probability", linestyle='--')
        actual[["whale_observation_probability"]].plot(ax=axes[2], label="Actual Whale Probability")

        for ax in axes:
            ax.set_xlabel("Time")
            ax.grid(True)
            ax.legend()

        plt.tight_layout()
        plt.show()

def run_all_cases():
    start_time = datetime(2025, 1, 1, 0, 0, tzinfo=timezone(timedelta(hours=-6)))
    end_time = datetime(2025, 2, 1, 0, 0, tzinfo=timezone(timedelta(hours=-6)))

    generator = FakeDataGenerator(start_time, end_time)

    for case_name, (wind_pattern, whale_pattern) in generator.cases.items():
        expected, actual = generator.generate_case(wind_pattern, whale_pattern)
        print(f"Generated data for: {case_name}")
        # generator.visualize_data(expected, actual, case_name)

if __name__ == "__main__":
    run_all_cases()

from abc import ABC, abstractmethod
import numpy as np
from BaseClasses.solar_panel_base import SolarPanelFactory

class AbstractEnvironmentProvider(ABC):
    @abstractmethod
    def sample_sunlight(self, t: int, n: int = 1) -> np.ndarray:
        """
        Return an array of solar energy values at time step t.
        """

    @abstractmethod
    def sample_wind_speed(self, t: int, n: int = 1) -> np.ndarray:
        """
        Return an array of wind speed values at time step t.
        """

    @abstractmethod
    def sample_whale_observation(self, t: int, n: int = 1) -> np.ndarray:
        """
        Return an array of whale observation values at time step t.
        """

class DeterministicEnvironmentProvider(AbstractEnvironmentProvider):
    def __init__(self, solar_rate_series: np.ndarray, wind_series: np.ndarray, whale_reward_series: np.ndarray, delta_t: float):
        self.solar_rate_series = solar_rate_series
        self.wind_series = wind_series
        self.whale_reward_series = whale_reward_series
        self.delta_t = delta_t

    def sample_sunlight(self, t: int, n: int = 1) -> np.ndarray:
        if t >= len(self.solar_rate_series):
            raise IndexError("Time index t exceeds the length of solar_rate_series.")
        return np.full((n,), self.solar_rate_series[t] * self.delta_t)

    def sample_wind_speed(self, t: int, n: int = 1) -> np.ndarray:
        if t >= len(self.wind_series):
            raise IndexError("Time index t exceeds the length of wind_series.")
        return np.full((n,), self.wind_series[t])

    def sample_whale_observation(self, t: int, n: int = 1) -> np.ndarray:
        if t >= len(self.whale_reward_series):
            raise IndexError("Time index t exceeds the length of whale_reward_series.")
        return np.full((n,), self.whale_reward_series[t])

class StochasticWindEnvironmentProvider(AbstractEnvironmentProvider):
    """
    Environment provider that uses a stochastic Weibull distribution for wind speed.
    The solar and whale observation data are provided deterministically.
    """
    def __init__(self, solar_rate_series: np.ndarray, wind_distributions: np.ndarray,
                 whale_reward_series: np.ndarray, delta_t: float):
        self.solar_rate_series = solar_rate_series
        self.wind_shape = wind_distributions[:, 0]
        self.wind_scale = wind_distributions[:, 1]
        self.whale_reward_series = whale_reward_series
        self.delta_t = delta_t

    def sample_sunlight(self, t: int, n: int) -> np.ndarray:
        return np.array([self.solar_rate_series[t]] * n)

    def sample_wind_speed(self, t: int, n: int) -> np.ndarray:
        return self.weibull_wind_speed_dist(t, n)

    def sample_whale_observation(self, t: int, n: int) -> np.ndarray:
        return np.array([self.whale_reward_series[t]] * n)

    def weibull_wind_speed_dist(self, t: int, n: int) -> np.ndarray:
        # Define the Weibull shape parameter (k) and a time-varying scale parameter (λ)
        k = self.wind_shape[t]
        lam = self.wind_scale[t]
        return lam * np.random.weibull(k, size=n)

class StochasticWindSolarEnvironmentProvider(AbstractEnvironmentProvider):
    def __init__(self, solar_distributions: np.ndarray, wind_distributions: np.ndarray,
                 whale_reward_series: np.ndarray, delta_t_min: float, solar_panel_model: str = "constant", rng=None):
        self.wind_shape = wind_distributions[:, 0]
        self.wind_scale = wind_distributions[:, 1]
        self.solar_alpha = solar_distributions[:, 0]
        self.solar_beta = solar_distributions[:, 1]
        self.whale_reward_series = whale_reward_series
        self.DELTA_T_MIN = delta_t_min
        self.DELTA_T_SEC = delta_t_min * 60
        self.panel = SolarPanelFactory.create_solar_panel(solar_panel_model)
        # Use a provided generator or default to np.random.default_rng()
        self.rng = rng if rng is not None else np.random.default_rng()

    def set_seed(self, seed: int) -> None:
        """
        Set the random seed for reproducibility.
        """
        self.rng = np.random.default_rng(seed)

    def reset(self,seed: int = None) -> None:
        """
        Reset the random number generator to a specific seed.
        """
        self.set_seed(seed)

    def sample_sunlight(self, t: int, n: int) -> np.ndarray:
        w_p_m2 = self.beta_solar_energy_dist(t, n) * 1367.0  # W/m^2
        j_p_m2 = w_p_m2 * self.DELTA_T_SEC  # J/m^2
        a_m2 = self.panel.area
        return j_p_m2 * a_m2 * self.panel.efficiency

    def sample_wind_speed(self, t: int, n: int) -> np.ndarray:
        return self.weibull_wind_speed_dist(t, n)

    def sample_whale_observation(self, t: int, n: int) -> np.ndarray:
        return np.array([self.whale_reward_series[t]] * n)

    def weibull_wind_speed_dist(self, t: int, n: int) -> np.ndarray:
        k = self.wind_shape[t]
        lam = self.wind_scale[t]
        return lam * self.rng.weibull(k, size=n)
    
    def beta_solar_energy_dist(self, t: int, n: int) -> np.ndarray:
        a = self.solar_alpha[t]
        b = self.solar_beta[t]
        return self.rng.beta(a, b, size=n)


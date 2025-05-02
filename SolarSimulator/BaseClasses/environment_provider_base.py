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

    def get_wind_shape(self, stage):
        return self.wind_shape[stage]

    def get_wind_scale(self, stage):
        return self.wind_scale[stage]
    
    def get_solar_alpha(self, stage):
        return self.solar_alpha[stage]
    
    def get_solar_beta(self, stage):
        return self.solar_beta[stage]

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
    
    def _energy_gain_from_solar(self, solar_vals: np.ndarray) -> np.ndarray:
        """Calculate energy gain from solar values."""
        # Solar vals ranges from 0 to 1, representing the fraction of solar energy available
        w_p_m2 = solar_vals * 1367.0  # W/m^2
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

class RegimeSwitchingWindSolarEnvironmentProvider(StochasticWindSolarEnvironmentProvider):
    """
    Environment provider that uses two wind regimes (high and low) in sequence,
    while maintaining a constant diurnal solar cycle with small stochastic variations.

    Parameters
    ----------
    whale_reward_series : np.ndarray
        Time series of whale observation rewards (one value per stage).
    delta_t_min : float
        Timestep size in minutes.
    switch_stage : int
        Stage length (in stages) for each regime block.
    high_wind : tuple(float, float)
        Weibull parameters (shape k, scale λ) for the high-wind regime.
    low_wind : tuple(float, float)
        Weibull parameters (shape k, scale λ) for the low-wind regime.
    solar_concentration : float
        Concentration parameter φ for the Beta solar distribution (larger = less variance).
    start_with_high : bool, optional
        If True (default), the first block uses the high-wind regime;
        otherwise, the first block uses low-wind.
    repeat_pattern : bool, optional
        If True, alternate regimes every `switch_stage` indefinitely;
        if False (default), use two blocks only (high then low or vice versa).
    solar_panel_model : str, optional
        Identifier for the solar panel model, passed to the base provider.
    rng : np.random.Generator, optional
        Random number generator; if None, defaults to np.random.default_rng().
    """
    def __init__(
        self,
        whale_reward_series: np.ndarray,
        delta_t_min: float,
        switch_stage: int,
        high_wind: tuple,
        low_wind: tuple,
        solar_concentration: float,
        start_with_high: bool = True,
        repeat_pattern: bool = False,
        solar_panel_model: str = "constant",
        rng=None
    ):
        total_stages = len(whale_reward_series)

        # Prepare wind distributions array
        wind_distributions = np.zeros((total_stages, 2))
        k_high, lam_high = high_wind
        k_low, lam_low = low_wind

        if repeat_pattern:
            # Alternate regimes every switch_stage
            for t in range(total_stages):
                block = t // switch_stage
                if start_with_high:
                    use_high = (block % 2 == 0)
                else:
                    use_high = (block % 2 != 0)
                if use_high:
                    wind_distributions[t] = [k_high, lam_high]
                else:
                    wind_distributions[t] = [k_low, lam_low]
        else:
            # Single switch: first block then second block only
            if start_with_high:
                wind_distributions[:switch_stage] = [k_high, lam_high]
                wind_distributions[switch_stage:] = [k_low, lam_low]
            else:
                wind_distributions[:switch_stage] = [k_low, lam_low]
                wind_distributions[switch_stage:] = [k_high, lam_high]

        # Build solar distributions for a diurnal Beta profile
        steps_per_day = int(24 * 60 / delta_t_min)
        i = np.arange(steps_per_day)
        diurnal = np.maximum(0, np.sin(2 * np.pi * (i / steps_per_day - 0.25)))
        solar_distributions = np.zeros((total_stages, 2))
        phi = solar_concentration
        for t in range(total_stages):
            idx = t % steps_per_day
            mu = diurnal[idx]
            solar_distributions[t] = [mu * phi, (1 - mu) * phi]
        self.wind_distributions  = wind_distributions
        self.solar_distributions = solar_distributions
        # Initialize the base provider
        super().__init__(
            solar_distributions=solar_distributions,
            wind_distributions=wind_distributions,
            whale_reward_series=whale_reward_series,
            delta_t_min=delta_t_min,
            solar_panel_model=solar_panel_model,
            rng=rng
        )

class SinusoidalWindSolarEnvironmentProvider(StochasticWindSolarEnvironmentProvider):
    """
    Environment provider that uses a constant Weibull shape parameter for wind and
    a sinusoidally varying scale parameter, alongside a constant diurnal solar cycle.

    Parameters
    ----------
    whale_reward_series : np.ndarray
        Time series of whale observation rewards (one value per stage).
    delta_t_min : float
        Timestep size in minutes.
    wind_shape : float
        Weibull shape parameter k for all wind samples.
    base_scale : float
        Mean scale parameter λ around which the sinusoid oscillates.
    scale_amplitude : float
        Amplitude of the sine wave for wind scale variations.
    scale_period : int
        Period of the sine wave in number of stages.
    solar_concentration : float
        Concentration parameter φ for the Beta solar distribution (larger = less variance).
    solar_panel_model : str, optional
        Identifier for the solar panel model, passed to the base provider.
    rng : np.random.Generator, optional
        Random number generator; if None, defaults to np.random.default_rng().
    """
    def __init__(
        self,
        whale_reward_series: np.ndarray,
        delta_t_min: float,
        wind_shape: float,
        base_scale: float,
        scale_amplitude: float,
        scale_period: int,
        solar_concentration: float,
        solar_panel_model: str = "constant",
        rng=None
    ):
        total_stages = len(whale_reward_series)

        # Build wind distributions: constant shape, sinusoidal scale
        wind_distributions = np.zeros((total_stages, 2))
        k = wind_shape
        for t in range(total_stages):
            scale = base_scale + scale_amplitude * np.sin(2 * np.pi * (t / scale_period))
            wind_distributions[t] = [k, scale]

        # Build solar distributions for a diurnal Beta profile
        steps_per_day = int(24 * 60 / delta_t_min)
        i = np.arange(steps_per_day)
        diurnal = np.maximum(0, np.sin(2 * np.pi * (i / steps_per_day - 0.25)))
        solar_distributions = np.zeros((total_stages, 2))
        phi = solar_concentration
        for t in range(total_stages):
            idx = t % steps_per_day
            mu = diurnal[idx]
            solar_distributions[t] = [mu * phi, (1 - mu) * phi]

        super().__init__(
            solar_distributions=solar_distributions,
            wind_distributions=wind_distributions,
            whale_reward_series=whale_reward_series,
            delta_t_min=delta_t_min,
            solar_panel_model=solar_panel_model,
            rng=rng
        )
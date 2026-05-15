from abc import ABC, abstractmethod
import numpy as np
from BaseClasses.solar_panel_base import SolarPanelFactory

class AbstractEnvironmentProvider(ABC):
    """
    Abstract base class for environment providers in the solar simulator.

    Defines the interface for sampling environmental conditions (sunlight, wind, whale observations)
    at discrete time steps. Subclasses must implement methods for sampling each environmental variable.
    """
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
    """
    Deterministic environment provider with fixed time series data.

    This class provides deterministic environmental samples by directly indexing
    into pre-defined time series arrays for solar, wind, and whale observations.

    Parameters
    ----------
    solar_rate_series : np.ndarray
        Time series of solar irradiance rates (W/m^2).
    wind_series : np.ndarray
        Time series of wind speeds (m/s).
    whale_reward_series : np.ndarray
        Time series of whale observation rewards.
    delta_t : float
        Timestep size in appropriate time units.

    Notes
    -----
    This class is currently out of date and raises NotImplementedError.
    """
    def __init__(self, solar_rate_series: np.ndarray, wind_series: np.ndarray, whale_reward_series: np.ndarray, delta_t: float):
        self.solar_rate_series = solar_rate_series
        self.wind_series = wind_series
        self.whale_reward_series = whale_reward_series
        self.delta_t = delta_t
        raise NotImplementedError("This class is out of date.")
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

class HistoricalEnvironmentProvider(AbstractEnvironmentProvider):
    """
    Environment provider using historical time series data.

    Returns deterministic values from historical data, replicating the same
    value n times when multiple samples are requested. Properly converts
    solar irradiance to energy using the specified solar panel model.

    Parameters
    ----------
    solar_irradiance_series : np.ndarray
        Historical solar irradiance time series [W/m^2].
    wind_speed_series : np.ndarray
        Historical wind speed time series [m/s].
    whale_observation_series : np.ndarray
        Historical whale observation time series.
    delta_t_min : float
        Timestep size in minutes.
    solar_panel_model : str, optional
        Identifier for the solar panel model (default: "constant").
    rng : np.random.Generator, optional
        Random number generator (included for API consistency, not used).

    Attributes
    ----------
    DELTA_T_MIN : float
        Timestep size in minutes.
    DELTA_T_SEC : float
        Timestep size in seconds.
    panel : SolarPanel
        Solar panel instance created from the factory.

    Notes
    -----
    TODO: Add functionality to randomly select historical data from a given time period
    (day, week, month, etc.) while maintaining interoperability with other environment
    provider classes. This would allow sampling from different historical periods while
    keeping the same AbstractEnvironmentProvider interface.
    """
    def __init__(
        self,
        solar_irradiance_series: np.ndarray,
        wind_speed_series: np.ndarray,
        whale_observation_series: np.ndarray,
        delta_t_min: float,
        solar_panel_model: str = "constant",
        rng=None
    ):
        self.solar_irradiance_series = solar_irradiance_series
        self.wind_speed_series = wind_speed_series
        self.whale_observation_series = whale_observation_series
        self.DELTA_T_MIN = delta_t_min
        self.DELTA_T_SEC = delta_t_min * 60
        self.panel = SolarPanelFactory.create_solar_panel(solar_panel_model)
        self.rng = rng if rng is not None else np.random.default_rng()

    def set_seed(self, seed: int) -> None:
        """
        Set the random seed for API consistency.

        Note: This provider is deterministic, but this method is included
        for compatibility with stochastic providers.
        """
        self.rng = np.random.default_rng(seed)

    def reset(self, seed: int = None) -> None:
        """
        Reset the random number generator to a specific seed.

        Note: This provider is deterministic, but this method is included
        for compatibility with stochastic providers.
        """
        if seed is not None:
            self.set_seed(seed)

    def sample_sunlight(self, t: int, n: int = 1) -> np.ndarray:
        """
        Return an array of solar energy values at time step t.

        Converts historical irradiance [W/m^2] to energy [J] using the
        panel area, efficiency, and timestep duration.

        Parameters
        ----------
        t : int
            Time step index.
        n : int, optional
            Number of samples to return (all identical).

        Returns
        -------
        np.ndarray
            Array of shape (n,) with solar energy values in Joules.
        """
        if t >= len(self.solar_irradiance_series):
            raise IndexError(
                f"Time index t={t} exceeds the length of solar_irradiance_series "
                f"({len(self.solar_irradiance_series)})."
            )

        # Convert irradiance to energy
        w_p_m2 = self.solar_irradiance_series[t]  # W/m^2
        j_p_m2 = w_p_m2 * self.DELTA_T_SEC  # J/m^2
        energy = j_p_m2 * self.panel.area * self.panel.efficiency

        return np.full((n,), energy)

    def sample_wind_speed(self, t: int, n: int = 1) -> np.ndarray:
        """
        Return an array of wind speed values at time step t.

        Parameters
        ----------
        t : int
            Time step index.
        n : int, optional
            Number of samples to return (all identical).

        Returns
        -------
        np.ndarray
            Array of shape (n,) with wind speed values in m/s.
        """
        if t >= len(self.wind_speed_series):
            raise IndexError(
                f"Time index t={t} exceeds the length of wind_speed_series "
                f"({len(self.wind_speed_series)})."
            )

        return np.full((n,), self.wind_speed_series[t])

    def sample_whale_observation(self, t: int, n: int = 1) -> np.ndarray:
        """
        Return an array of whale observation values at time step t.

        Parameters
        ----------
        t : int
            Time step index.
        n : int, optional
            Number of samples to return (all identical).

        Returns
        -------
        np.ndarray
            Array of shape (n,) with whale observation values.
        """
        if t >= len(self.whale_observation_series):
            raise IndexError(
                f"Time index t={t} exceeds the length of whale_observation_series "
                f"({len(self.whale_observation_series)})."
            )

        return np.full((n,), self.whale_observation_series[t])

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
        raise NotImplementedError("This Class is out of date.")

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
    """
    Environment provider with stochastic wind and solar energy sampling.

    Uses Weibull distributions for wind speed and Beta distributions for solar irradiance,
    allowing for realistic stochastic environmental variations while maintaining diurnal patterns.

    Parameters
    ----------
    solar_distributions : np.ndarray
        Array of shape (n_stages, 3) containing Beta distribution parameters (alpha, beta, clearsky).
        Each row specifies [alpha, beta, clearsky_irradiance] for that stage.
    wind_distributions : np.ndarray
        Array of shape (n_stages, 2) containing Weibull distribution parameters (shape k, scale λ).
        Each row specifies [k, λ] for that stage.
    whale_reward_series : np.ndarray
        Time series of whale observation rewards (one value per stage).
    delta_t_min : float
        Timestep size in minutes.
    solar_panel_model : str, optional
        Identifier for the solar panel model (default: "constant").
    rng : np.random.Generator, optional
        Random number generator for reproducibility; if None, defaults to np.random.default_rng().

    Attributes
    ----------
    DELTA_T_MIN : float
        Timestep size in minutes.
    DELTA_T_SEC : float
        Timestep size in seconds.
    panel : SolarPanel
        Solar panel instance created from the factory.
    rng : np.random.Generator
        Random number generator instance.
    """
    def __init__(self, solar_distributions: np.ndarray, wind_distributions: np.ndarray,
                 whale_reward_series: np.ndarray, delta_t_min: float, solar_panel_model: str = "constant", rng=None):
        self.wind_shape = wind_distributions[:, 0]
        self.wind_scale = wind_distributions[:, 1]
        self.solar_alpha = solar_distributions[:, 0]
        self.solar_beta = solar_distributions[:, 1]
        self.solar_cs = solar_distributions[:, 2]
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
    
    def get_solar_cs_irad(self, stage):
        return self.solar_cs[stage]
    
    def get_solar_cs_joules(self, stage):
        return self._energy_gain_from_solar(self.solar_cs[stage])

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
        w_p_m2 = self.beta_solar_energy_dist(t, n) # W/m^2
        j_p_m2 = w_p_m2 * self.DELTA_T_SEC  # J/m^2
        a_m2 = self.panel.area
        return j_p_m2 * a_m2 * self.panel.efficiency
    
    def _energy_gain_from_solar(self, solar_vals: np.ndarray) -> np.ndarray:
        """Calculate energy gain from solar values."""
        # Solar vals ranges from 0 to 1, representing the fraction of solar energy available
        w_p_m2 = solar_vals # W/m^2
        j_p_m2 = w_p_m2 * self.DELTA_T_SEC  # J/m^2
        a_m2 = self.panel.area
        return j_p_m2 * a_m2 * self.panel.efficiency

    def sample_wind_speed(self, t: int, n: int) -> np.ndarray:
        return self.weibull_wind_speed_dist(t, n)

    def sample_whale_observation(self, t: int, n: int=1) -> np.ndarray:
        return np.array([self.whale_reward_series[t]] * n)

    def weibull_wind_speed_dist(self, t: int, n: int) -> np.ndarray:
        k = self.wind_shape[t]
        lam = self.wind_scale[t]
        return lam * self.rng.weibull(k, size=n)
    
    def beta_solar_energy_dist(self, t: int, n: int) -> np.ndarray:
        a = self.solar_alpha[t] # Irradiance beta distribution alpha param
        b = self.solar_beta[t] # Irradiance beta distribution beta param
        cs = self.solar_cs[t] # Clearsky irradiance [W/m^2]
        return self.rng.beta(a, b, size=n) * cs
    
    def get_solar_alpha(self,stage):
        return self.solar_alpha[stage]
    
    def get_solar_beta(self, stage):
        return self.solar_beta[stage]

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
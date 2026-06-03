from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
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
                 whale_reward_series: np.ndarray, delta_t_min: float, solar_panel_model: str = "constant", rng=None,
                 wind_bin_edges: np.ndarray = None, wind_transition: np.ndarray = None):
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

        # --- Wind Markov-chain (persistence) state ---
        # When wind_transition is None the provider falls back to the original i.i.d.
        # Weibull sampling and behaves as a single-bin chain (n_wind_bins == 1), so the
        # solver's bin machinery reduces exactly to the i.i.d. case.
        self.use_wind_chain = wind_transition is not None
        if self.use_wind_chain:
            self.wind_bin_edges = np.asarray(wind_bin_edges, dtype=float)
            self.wind_transition = np.asarray(wind_transition, dtype=float)  # (T, n_bins, n_bins)
            self.n_wind_bins = self.wind_transition.shape[-1]
        else:
            self.wind_bin_edges = np.array([0.0, np.inf])
            self.wind_transition = None
            self.n_wind_bins = 1
        self._wind_bins = None       # per-lane current bin (set lazily on first sample)
        self.last_wind_bins = None   # bins used in the most recent sample_wind_speed call

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
        Reset the random number generator to a specific seed and clear chain state.
        """
        self.set_seed(seed)
        self._wind_bins = None
        self.last_wind_bins = None

    def get_wind_transition(self, t: int) -> np.ndarray:
        """Per-stage wind-bin transition matrix (n_bins x n_bins). Identity-like [[1]] when i.i.d."""
        if self.use_wind_chain:
            return self.wind_transition[t]
        return np.array([[1.0]])

    def _weibull_cdf(self, w, k, scale):
        # F(w) = 1 - exp(-(w/scale)^k); F(inf)=1, F(0)=0
        out = np.where(np.isinf(w), 1.0, 1.0 - np.exp(-np.power(np.clip(w, 0, None) / scale, k)))
        return out

    def _init_wind_bins(self, n: int) -> np.ndarray:
        """Draw initial wind bins for n lanes from the stage-0 Weibull bin masses."""
        k = self.wind_shape[0]
        scale = self.wind_scale[0]
        F = self._weibull_cdf(self.wind_bin_edges, k, scale)
        p = np.diff(F)
        p = p / p.sum()
        return self.rng.choice(self.n_wind_bins, size=n, p=p)

    def _sample_within_bin(self, t: int, bins: np.ndarray) -> np.ndarray:
        """Sample wind from the stage-t Weibull truncated to each lane's current bin."""
        k = self.wind_shape[t]
        scale = self.wind_scale[t]
        lo = self.wind_bin_edges[bins]
        hi = self.wind_bin_edges[bins + 1]
        F_lo = self._weibull_cdf(lo, k, scale)
        F_hi = self._weibull_cdf(hi, k, scale)
        u = F_lo + self.rng.random(size=bins.shape[0]) * (F_hi - F_lo)
        u = np.clip(u, 0.0, 1.0 - 1e-12)
        return scale * np.power(-np.log(1.0 - u), 1.0 / k)

    def _advance_bins(self, bins: np.ndarray, P: np.ndarray) -> np.ndarray:
        """Sample next bin per lane from transition rows P[bins]."""
        cum = np.cumsum(P[bins], axis=1)
        r = self.rng.random(size=bins.shape[0])
        return (cum > r[:, None]).argmax(axis=1)

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
        if not self.use_wind_chain:
            return self.weibull_wind_speed_dist(t, n)
        # Markov-chain path: sample within the current bin, expose it, then advance.
        if self._wind_bins is None or self._wind_bins.shape[0] != n:
            self._wind_bins = self._init_wind_bins(n)
        bins = self._wind_bins
        self.last_wind_bins = bins.copy()
        w = self._sample_within_bin(t, bins)
        self._wind_bins = self._advance_bins(bins, self.wind_transition[t])
        return w

    def sample_whale_observation(self, t: int, n: int=1) -> np.ndarray:
        return np.full(n, self.whale_reward_series[t])

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


class HistoricalBootstrapEnvironmentProvider(AbstractEnvironmentProvider):
    """
    Environment provider that drives MC episodes from real historical weather via block bootstrap.

    An episode's timeline keeps the mission's exact calendar dates.  The H-step horizon is
    split into consecutive blocks of `block_length_steps`.  Each block's weather is drawn from
    a randomly chosen year at the same calendar dates, independently per lane (MC episode).
    Wind and solar use the *same* chosen year per (lane, block), preserving their real
    cross-correlation.  Concatenating the blocks gives a real-within-block, season-correct,
    diverse sequence (~n_years^n_blocks combinations).

    For chain-solved policy evaluation: pass `wind_bin_edges` and `wind_transition` (copied
    from the distributional solve-side provider).  `sample_wind_speed` then digitizes the
    realized wind into bins (exposing `last_wind_bins`) so `simulate_episode_batch` and
    `choose_action_batch` work unchanged for both i.i.d.-solved and chain-solved policies.

    Parameters
    ----------
    cube : dict
        Calendar cube from `build_historical_cube_artifact`; keys `wind_cube`
        (slots_per_year x n_years), `solar_cube`, `slots_per_year`, `years`, `delta_t_min`.
    start_dt : pd.Timestamp or str
        Mission start datetime (only calendar position in the year matters; year ignored).
    horizon : int
        Episode length in steps.
    delta_t_min : float
        Timestep in minutes.  Must equal cube['delta_t_min'].
    block_length_steps : int, optional
        Bootstrap block size in steps (default: 7 days = 7*96 steps at 15 min).
    solar_panel_model : str
        Solar panel model identifier (passed to SolarPanelFactory).
    whale_reward_series : np.ndarray
        Pre-built whale series of length >= horizon.
    rng : np.random.Generator, optional
        RNG instance; defaults to np.random.default_rng().
    wind_bin_edges : np.ndarray, optional
        Full edge array (length n_bins+1) from the wind chain artifact.  When provided,
        enables chain-solved evaluation via bin digitization.
    wind_transition : np.ndarray, optional
        Per-stage transition matrices shaped (horizon, n_bins, n_bins).
    """

    # Cumulative days before each month in a non-leap year (0-indexed, Jan=0).
    _DAYS_BEFORE_MONTH = np.array([0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334])

    def __init__(
        self,
        cube: dict,
        start_dt,
        horizon: int,
        delta_t_min: float = 15.0,
        block_length_steps: int = None,
        solar_panel_model: str = "constant",
        whale_reward_series: np.ndarray = None,
        rng=None,
        wind_bin_edges: np.ndarray = None,
        wind_transition: np.ndarray = None,
    ):
        self.DELTA_T_MIN = float(delta_t_min)
        self.DELTA_T_SEC = self.DELTA_T_MIN * 60.0
        self.panel = SolarPanelFactory.create_solar_panel(solar_panel_model)
        self.whale_reward_series = whale_reward_series
        self.rng = rng if rng is not None else np.random.default_rng()

        cube_dt = cube.get("delta_t_min", delta_t_min)
        if int(cube_dt) != int(delta_t_min):
            raise ValueError(
                f"Cube delta_t_min ({cube_dt}) != requested delta_t_min ({delta_t_min})."
            )

        self._wind_cube = cube["wind_cube"]    # (slots_per_year, n_years)
        self._solar_cube = cube["solar_cube"]  # (slots_per_year, n_years)
        self._slots_per_year = int(cube["slots_per_year"])
        self._n_years = int(cube["n_years"])
        self.horizon = horizon

        # Default block = 7 days.
        if block_length_steps is None:
            block_length_steps = int(7 * 24 * 60 / delta_t_min)
        self.block_length_steps = int(block_length_steps)
        self._n_blocks = (horizon + self.block_length_steps - 1) // self.block_length_steps

        # Compute calendar slots for the mission window (wraps at year boundary).
        start_ts = pd.Timestamp(start_dt) if not isinstance(start_dt, pd.Timestamp) else start_dt
        start_slot = self._timestamp_to_slot(start_ts)
        self._window_slots = (np.arange(horizon, dtype=np.int64) + start_slot) % self._slots_per_year

        # Wind-chain support (core: chain-solved policy evaluated on historical weather).
        self.use_wind_chain = (wind_bin_edges is not None) and (wind_transition is not None)
        if self.use_wind_chain:
            self.wind_bin_edges = np.asarray(wind_bin_edges, dtype=float)
            self.wind_transition = np.asarray(wind_transition, dtype=float)
            self.n_wind_bins = self.wind_transition.shape[-1]
        else:
            self.wind_bin_edges = np.array([0.0, np.inf])
            self.wind_transition = None
            self.n_wind_bins = 1
        self.last_wind_bins = None

        # Lazy per-n year-assignment matrix (built on first sample call or after reset).
        # Shape (n, n_blocks): year_choice[lane, block] = year index in [0, n_years-1].
        # Storing only this tiny matrix (not the full (H, n) weather arrays) keeps memory
        # proportional to n*n_blocks rather than H*n, which matters for large episode counts.
        self._year_choice = None   # (n, n_blocks) int

    def _timestamp_to_slot(self, ts: pd.Timestamp) -> int:
        """Map a calendar datetime to its slot index in [0, slots_per_year-1], ignoring Feb 29."""
        if ts.month == 2 and ts.day == 29:
            raise ValueError("start_dt is Feb 29, which is excluded from the calendar cube.")
        spd = int(1440 // self.DELTA_T_MIN)  # slots per day
        sph = int(60 // self.DELTA_T_MIN)    # slots per hour
        doy_0 = int(self._DAYS_BEFORE_MONTH[ts.month - 1]) + ts.day - 1
        return doy_0 * spd + ts.hour * sph + ts.minute // int(self.DELTA_T_MIN)

    def _ensure_year_choice(self, n: int) -> None:
        """Draw per-lane block-year assignments if the batch size has changed."""
        if self._year_choice is not None and self._year_choice.shape[0] == n:
            return
        self._year_choice = self.rng.integers(0, self._n_years, size=(n, self._n_blocks))

    def _energy_gain_from_solar(self, ghi: np.ndarray) -> np.ndarray:
        """GHI [W/m^2] -> energy [J] using the panel model (broadcasts over any shape)."""
        return ghi * self.DELTA_T_SEC * self.panel.area * self.panel.efficiency

    def reset(self, seed=None) -> None:
        """Re-seed the RNG and clear year assignments so the next run draws fresh bootstrap data."""
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self._year_choice = None
        self.last_wind_bins = None

    def set_seed(self, seed: int) -> None:
        self.rng = np.random.default_rng(seed)
        self.reset()

    def get_wind_transition(self, t: int) -> np.ndarray:
        """Per-stage wind-bin transition matrix (n_bins x n_bins). Identity-like [[1]] when i.i.d."""
        if self.use_wind_chain:
            return self.wind_transition[t]
        return np.array([[1.0]])

    def sample_wind_speed(self, t: int, n: int) -> np.ndarray:
        self._ensure_year_choice(n)
        yr = self._year_choice[:, t // self.block_length_steps]   # (n,)
        wind_t = self._wind_cube[self._window_slots[t], yr]        # (n,)
        if self.use_wind_chain:
            self.last_wind_bins = np.digitize(wind_t, self.wind_bin_edges[1:-1]).astype(int)
        return wind_t

    def sample_sunlight(self, t: int, n: int) -> np.ndarray:
        self._ensure_year_choice(n)
        yr = self._year_choice[:, t // self.block_length_steps]    # (n,)
        ghi_t = self._solar_cube[self._window_slots[t], yr]        # (n,)
        return self._energy_gain_from_solar(ghi_t)

    def sample_whale_observation(self, t: int, n: int = 1) -> np.ndarray:
        return np.full(n, self.whale_reward_series[t])

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
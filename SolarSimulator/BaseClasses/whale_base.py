import numpy as np
import matplotlib.pyplot as plt

# The original hand-set diurnal profile: twelve 2-hour blocks starting at 00:00.
DEFAULT_BLOCK_VALUES = np.array([
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
    0.000, # 20:00-22:00
    0.000, # 22:00-24:00
])
DEFAULT_BLOCK_HOURS = 2.0


def block_series(block_values, horizon, *, block_hours=DEFAULT_BLOCK_HOURS,
                 delta_t=15, start_offset_min=0):
    """Expand per-block-of-day values into a per-stage series of length ``horizon``.

    Args:
        block_values: Sequence of values, one per block of the day, starting at 00:00.
            ``len(block_values) * block_hours`` must equal 24.
        horizon: Number of stages to return.
        block_hours: Duration of each block in hours.
        delta_t: Stage length in minutes; ``block_hours * 60`` must be a multiple of it.
        start_offset_min: Minutes after midnight at which stage 0 begins, so that
            index 0 of the returned series holds the value for the mission start time
            rather than for 00:00.

    With the defaults (``delta_t=15``, ``start_offset_min=0``) this reproduces the
    original ``np.repeat(block_values, 8)`` / ``np.tile`` construction exactly.
    """
    block_values = np.asarray(block_values, dtype=float)
    if block_values.ndim != 1 or block_values.size == 0:
        raise ValueError("block_values must be a non-empty 1-D sequence")
    if not np.isclose(block_values.size * block_hours, 24.0):
        raise ValueError(
            f"{block_values.size} blocks x {block_hours} h must span 24 h "
            f"(got {block_values.size * block_hours} h)")
    if np.any(block_values < 0.0) or np.any(block_values > 1.0):
        raise ValueError("block_values must lie in [0, 1]")
    steps_per_block = block_hours * 60.0 / delta_t
    if not np.isclose(steps_per_block, round(steps_per_block)):
        raise ValueError(
            f"block length {block_hours} h is not a whole number of {delta_t}-min stages")
    steps_per_block = int(round(steps_per_block))
    offset_steps = start_offset_min / delta_t
    if not np.isclose(offset_steps, round(offset_steps)):
        raise ValueError(
            f"start offset {start_offset_min} min is not a whole number of {delta_t}-min stages")
    daily = np.repeat(block_values, steps_per_block)
    daily = np.roll(daily, -(int(round(offset_steps)) % daily.size))
    full = np.tile(daily, int(np.ceil(horizon / daily.size)))
    return full[:horizon]


class WhaleRewardSeries:
    """Abstract base class for creating whale reward series."""
    def create_series(self, horizon, **kwargs):
        raise NotImplementedError("Subclasses must implement create_series()")


class SinusoidalWhaleRewardSeries(WhaleRewardSeries):
    """Creates a sinusoidal whale reward series."""
    def create_series(self, horizon, **kwargs):
        x = np.linspace(np.pi, np.pi * 60, horizon)
        return 0.5 * np.sin(x) + 0.5


class ConstantWhaleRewardSeries(WhaleRewardSeries):
    """Creates a constant whale reward series."""
    def create_series(self, horizon, **kwargs):
        return np.full(horizon, 0.5)


class RealWhaleRewardSeries(WhaleRewardSeries):
    """The original hand-set diurnal profile (``DEFAULT_BLOCK_VALUES``)."""
    def create_series(self, horizon, *, delta_t=15, start_offset_min=0, **kwargs):
        return block_series(DEFAULT_BLOCK_VALUES, horizon,
                            block_hours=DEFAULT_BLOCK_HOURS,
                            delta_t=delta_t, start_offset_min=start_offset_min)


class CustomWhaleRewardSeries(WhaleRewardSeries):
    """Diurnal block profile supplied from config (``whale_block_values`` / ``whale_block_hours``)."""
    def create_series(self, horizon, *, block_values=None, block_hours=DEFAULT_BLOCK_HOURS,
                      delta_t=15, start_offset_min=0, **kwargs):
        if block_values is None:
            raise ValueError("whale_series 'custom' requires whale_block_values")
        return block_series(block_values, horizon, block_hours=block_hours,
                            delta_t=delta_t, start_offset_min=start_offset_min)


class WhaleRewardSeriesFactory:
    """Factory class to create whale reward series."""

    _series_mapping = {
        "sinusoidal": SinusoidalWhaleRewardSeries,
        "constant": ConstantWhaleRewardSeries,
        "real": RealWhaleRewardSeries,
        "custom": CustomWhaleRewardSeries,
    }

    @classmethod
    def create_series(cls, series_type, horizon, **kwargs):
        """Creates the desired whale reward series.

        Args:
            series_type (str): 'sinusoidal', 'constant', 'real' or 'custom'.
            horizon (int): Number of time steps to create the series for.
            **kwargs: Series-specific options. Block-based series accept ``delta_t``
                (minutes), ``start_offset_min`` (minutes after midnight of stage 0),
                and, for 'custom', ``block_values`` and ``block_hours``.

        Returns:
            numpy.ndarray: The whale reward series.
        """
        try:
            series_class = cls._series_mapping[series_type]
        except KeyError:
            raise ValueError(f"Unknown whale reward series type: {series_type}")
        return series_class().create_series(horizon, **kwargs)


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

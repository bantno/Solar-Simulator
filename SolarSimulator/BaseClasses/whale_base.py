"""Whale observation-probability series.

The stage observation reward is the probability of a sighting while surveying. The
``encounter`` series builds it from conventional distance-sampling encounter theory
(Buckland et al. 2001, *Introduction to Distance Sampling*; Marsh & Sinclair 1989 for the
availability / perception split):

    lambda(t) = D * A * a(t) * g0 * light(t)          expected sightings per stage
    p(t)      = 1 - exp(-lambda(t))                    P(at least one sighting)

* ``D``     — density, animals / km^2, a property of the **location** (habitat-based
              density surfaces, e.g. Roberts et al. 2016 for the US Atlantic & Gulf of Mexico;
              NOAA stock assessments as a fallback).
* ``A``     — area surveyed per stage, km^2 = strip width x ground speed x stage length, a
              property of the **platform**.
* ``a(t)``  — availability, the fraction of time the animal is at the surface, a property of
              the **species** (tag studies).
* ``g0``    — perception, P(seen | surfaced in the strip), a property of the species (and
              sensor) — Barlow (2015) style trackline detection probabilities.
* ``light`` — the camera's daylight gate, a property of the **sensor**.

With no density given the series is the *focal* form ``p = a(t) * g0 * light(t)`` (a whale
is assumed to be under the aircraft); that is what the journal-results ``JoA2026`` profile
is: the *Kogia* card's availability (Scott et al. 2001, Table 2) with ``g0 = 1`` and a fixed
06:00-20:00 daylight gate.

Species cards (``SPECIES_CARDS``) hold the per-species numbers with their citations; a
config may name a card, override any of its fields, or supply the fields directly.
"""
import warnings

import numpy as np
import matplotlib.pyplot as plt

# Scott et al. (2001) Table 2: proportion of time at the surface by two-hour block, 00:00 start.
# Scott, M.D., Hohn, A.A., Westgate, A.J., Nicolas, J.R., Whitaker, B.R. & Campbell, W.B. (2001).
# A note on the release and tracking of a rehabilitated pygmy sperm whale (Kogia breviceps).
# J. Cetacean Res. Manage. 3(1):87-94. doi:10.47536/jcrm.v3i1.905. One rehabilitated animal,
# 4 days -- an example profile, not a population model.
SCOTT2001_SURFACE_PROPORTIONS = np.array([
    0.073,  # 00:00-02:00
    0.093,  # 02:00-04:00
    0.065,  # 04:00-06:00
    0.082,  # 06:00-08:00
    0.098,  # 08:00-10:00
    0.095,  # 10:00-12:00
    0.217,  # 12:00-14:00
    0.215,  # 14:00-16:00
    0.183,  # 16:00-18:00
    0.278,  # 18:00-20:00
    0.204,  # 20:00-22:00
    0.090,  # 22:00-24:00
])

# The journal-results profile: Scott 2001 daytime values, night (20:00-06:00) zeroed by hand.
JOA2026_BLOCK_VALUES = np.array([
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
DEFAULT_BLOCK_VALUES = JOA2026_BLOCK_VALUES  # backward-compatible name
DEFAULT_BLOCK_HOURS = 2.0

# Camera daylight gate that reproduces the JoA2026 night zeros.
DEFAULT_LIGHT = {"mode": "fixed_hours", "day_start": 6, "day_end": 20}

# Named availability profiles (per-block-of-day surface proportions, 00:00 start).
AVAILABILITY_PROFILES = {
    "scott2001": SCOTT2001_SURFACE_PROPORTIONS,
}

# Species cards: every number carries its source. ``availability_profile`` is a named
# profile, a scalar, or a per-block list; ``g0`` is P(seen | surfaced in the strip).
SPECIES_CARDS = {
    "kogia_spp": {
        "name": "kogia_spp",
        "common_name": "pygmy / dwarf sperm whale (Kogia spp.)",
        "availability_profile": "scott2001",
        "g0": 0.35,
        "citations": {
            "availability_profile": (
                "Scott et al. (2001) J. Cetacean Res. Manage. 3(1):87-94, Table 2 -- surface "
                "proportion by 2-h block, one VHF-tracked rehabilitated animal, 4 days"),
            "g0": (
                "Calm-sea (Beaufort 0-2) ship-observer trackline detection probability for Kogia "
                "(Barlow & Forney 2007 Fish. Bull. 105:509-526). Barlow (2015) Mar. Mamm. Sci. "
                "31:923-943: falls to <10% of the Beaufort-0 value by Beaufort 2; only ~3% of "
                "Kogia groups detected on the trackline over 1991-2014. No UAV-camera value "
                "exists -- treat as an upper bound / sensitivity."),
            "density_examples": (
                "Northern Gulf of Mexico: 0.00046 / km^2 visual, uncorrected for g(0) (Waring et "
                "al. 2013 via Hildebrand et al. 2019 Front. Mar. Sci. 6:66); 0.004-0.016 / km^2 "
                "acoustic at slope sites (Hildebrand et al. 2019). Western North Atlantic stock "
                "7,750 (CV 0.38) (NOAA SAR 2023). Map values: Roberts et al. (2016) Sci. Rep. "
                "6:22615 Kogia guild surfaces (OBIS-SEAMAP)."),
        },
    },
    "humpback": {
        "name": "humpback",
        "common_name": "humpback whale (Megaptera novaeangliae)",
        "availability_profile": 0.63,
        "g0": 0.91,
        "citations": {
            "availability_profile": (
                "Hodgson, Peel & Kelly (2017) Ecol. Appl. 27:1253-1267 -- availability 0.63 "
                "(0.86 mother/calf, 0.59 adults) for a fixed-wing UAV image footprint"),
            "g0": (
                "Hodgson et al. (2017): perception correction factor 1.10 for image review -> "
                "g0 ~ 0.91; Beaufort 1-5 had no significant effect on image-based detection"),
            "density_examples": (
                "Site-specific: Roberts et al. (2016) monthly East Coast surfaces; Hawaii "
                "breeding grounds from NOAA PIFSC / Lammers et al. (2023)."),
        },
    },
}


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


def stage_time_of_day_min(horizon, *, delta_t=15, start_offset_min=0):
    """Minutes after midnight at the start of each stage, shape (horizon,)."""
    return (start_offset_min + np.arange(horizon) * delta_t) % 1440


def _in_day_hours(tod_min, day_start, day_end):
    """Boolean mask: time-of-day (minutes) within [day_start, day_end) hours."""
    lo, hi = day_start * 60.0, day_end * 60.0
    if lo <= hi:
        return (tod_min >= lo) & (tod_min < hi)
    return (tod_min >= lo) | (tod_min < hi)  # window wraps midnight


def light_gate(horizon, light=None, *, delta_t=15, start_offset_min=0, solar_cs=None):
    """Per-stage daylight multiplier in {0, 1} for the camera.

    ``light``: ``{"mode": "fixed_hours", "day_start": 6, "day_end": 20}`` (default),
    ``{"mode": "clearsky_gate", "threshold_wm2": 0.0}`` (sun above the horizon from the
    location's clear-sky series ``solar_cs``), or ``{"mode": "none"}``.
    """
    light = dict(DEFAULT_LIGHT if light is None else light)
    mode = str(light.get("mode", "fixed_hours")).lower()
    if mode == "fixed_hours":
        tod = stage_time_of_day_min(horizon, delta_t=delta_t, start_offset_min=start_offset_min)
        return _in_day_hours(tod, float(light.get("day_start", 6)),
                             float(light.get("day_end", 20))).astype(float)
    if mode == "clearsky_gate":
        if solar_cs is None:
            raise ValueError("light mode 'clearsky_gate' needs the per-stage solar_cs series")
        cs = np.nan_to_num(np.asarray(solar_cs, dtype=float))
        if cs.size < horizon:
            raise ValueError(f"solar_cs has {cs.size} stages, horizon is {horizon}")
        return (cs[:horizon] > float(light.get("threshold_wm2", 0.0))).astype(float)
    if mode == "none":
        return np.ones(horizon)
    raise ValueError(f"unknown light mode {mode!r}")


def resolve_species_card(species):
    """Species card from a name, a dict naming a card plus overrides, or a bare dict."""
    if species is None:
        species = "kogia_spp"
    if isinstance(species, str):
        try:
            return dict(SPECIES_CARDS[species])
        except KeyError:
            raise ValueError(f"unknown species card {species!r}; known: {sorted(SPECIES_CARDS)}")
    if not isinstance(species, dict):
        raise ValueError("species must be a card name or a mapping")
    base = dict(SPECIES_CARDS.get(species.get("name"), {}))
    base.update({k: v for k, v in species.items() if v is not None})
    if "availability_profile" not in base:
        raise ValueError("species card needs availability_profile (name, scalar, or per-block list)")
    base.setdefault("g0", 1.0)
    return base


def availability_series(profile, horizon, *, block_hours=DEFAULT_BLOCK_HOURS, delta_t=15,
                        start_offset_min=0):
    """Per-stage surface proportion from a named profile, a scalar, or a per-block list."""
    if isinstance(profile, str):
        try:
            blocks = AVAILABILITY_PROFILES[profile.lower()]
        except KeyError:
            raise ValueError(f"unknown availability profile {profile!r}; known: {sorted(AVAILABILITY_PROFILES)}")
        return block_series(blocks, horizon, block_hours=block_hours, delta_t=delta_t,
                            start_offset_min=start_offset_min)
    if np.isscalar(profile):
        a = float(profile)
        if not 0.0 <= a <= 1.0:
            raise ValueError("availability must lie in [0, 1]")
        return np.full(horizon, a)
    return block_series(profile, horizon, block_hours=block_hours, delta_t=delta_t,
                        start_offset_min=start_offset_min)


def swept_area_km2(strip_width_km, ground_speed_ms, delta_t):
    """Area surveyed in one stage while cruising: strip width x distance flown."""
    if strip_width_km is None or ground_speed_ms is None:
        raise ValueError("encounter model with a density needs platform strip_width_km and ground_speed_ms")
    if strip_width_km <= 0 or ground_speed_ms <= 0:
        raise ValueError("strip_width_km and ground_speed_ms must be positive")
    return float(strip_width_km) * float(ground_speed_ms) * delta_t * 60.0 / 1000.0


def resolve_density(density, start_month=None):
    """Scalar density, or a 12-entry list / {month: value} mapping selected by start month."""
    if density is None:
        return None
    if isinstance(density, dict):
        if start_month is None:
            raise ValueError("monthly density needs start_month")
        key = start_month if start_month in density else str(start_month)
        if key not in density:
            raise ValueError(f"density has no entry for month {start_month}")
        density = density[key]
    elif not np.isscalar(density):
        vals = np.asarray(density, dtype=float)
        if vals.size != 12:
            raise ValueError("list-valued density must have 12 monthly entries")
        if start_month is None:
            raise ValueError("monthly density needs start_month")
        density = vals[int(start_month) - 1]
    d = float(density)
    if d < 0:
        raise ValueError("density must be non-negative")
    return d


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


class JoA2026WhaleRewardSeries(WhaleRewardSeries):
    """The journal-results profile (``JOA2026_BLOCK_VALUES``): Scott 2001 daytime surface
    proportions with 20:00-06:00 zeroed."""
    def create_series(self, horizon, *, delta_t=15, start_offset_min=0, **kwargs):
        return block_series(JOA2026_BLOCK_VALUES, horizon,
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


class EncounterWhaleSeries(WhaleRewardSeries):
    """Distance-sampling encounter model (see module docstring).

    Keyword options:
        species: card name (``"kogia_spp"``, ``"humpback"``), a dict naming a card with
            overrides (e.g. ``{"name": "kogia_spp", "g0": 1.0}``), or a bare dict with
            ``availability_profile`` and ``g0``. Default ``"kogia_spp"``.
        density_per_km2: location density ``D`` (scalar, 12 monthly values, or
            ``{month: value}``). ``None`` -> focal form ``p = a * g0 * light``.
        platform: ``{"strip_width_km": ..., "ground_speed_ms": ...}``; required with a density.
        light: daylight gate spec (see ``light_gate``); default fixed 06:00-20:00.
        start_month: 1-12, used to pick a monthly density.
        block_hours, delta_t, start_offset_min, solar_cs: geometry / clear-sky inputs.
    """
    def create_series(self, horizon, *, species=None, density_per_km2=None, platform=None,
                      light=None, start_month=None, block_hours=DEFAULT_BLOCK_HOURS,
                      delta_t=15, start_offset_min=0, solar_cs=None, **kwargs):
        card = resolve_species_card(species)
        g0 = float(card["g0"])
        if not 0.0 <= g0 <= 1.0:
            raise ValueError("g0 must lie in [0, 1]")
        a = availability_series(card["availability_profile"], horizon, block_hours=block_hours,
                                delta_t=delta_t, start_offset_min=start_offset_min)
        gate = light_gate(horizon, light, delta_t=delta_t, start_offset_min=start_offset_min,
                          solar_cs=solar_cs)
        density = resolve_density(density_per_km2, start_month)
        if density is None:
            # Focal form: a whale is in view; sighting probability is availability x perception.
            return np.clip(a * g0 * gate, 0.0, 1.0)
        platform = platform or {}
        area = swept_area_km2(platform.get("strip_width_km"), platform.get("ground_speed_ms"), delta_t)
        lam = density * area * a * g0 * gate
        return 1.0 - np.exp(-lam)


class WhaleRewardSeriesFactory:
    """Factory class to create whale reward series."""

    _series_mapping = {
        "sinusoidal": SinusoidalWhaleRewardSeries,
        "constant": ConstantWhaleRewardSeries,
        "JoA2026": JoA2026WhaleRewardSeries,
        "custom": CustomWhaleRewardSeries,
        "encounter": EncounterWhaleSeries,
    }
    # Pre-rename spelling of JoA2026, kept so archived configs still resolve.
    _aliases = {"real": "JoA2026"}

    @classmethod
    def create_series(cls, series_type, horizon, **kwargs):
        """Creates the desired whale reward series.

        Args:
            series_type (str): 'sinusoidal', 'constant', 'JoA2026', 'custom' or 'encounter'.
            horizon (int): Number of time steps to create the series for.
            **kwargs: Series-specific options. Block-based series accept ``delta_t``
                (minutes) and ``start_offset_min`` (minutes after midnight of stage 0);
                'custom' takes ``block_values`` / ``block_hours``; 'encounter' takes the
                options documented on ``EncounterWhaleSeries``.

        Returns:
            numpy.ndarray: The whale reward series.
        """
        if series_type in cls._aliases:
            warnings.warn(
                f"whale_series {series_type!r} was renamed {cls._aliases[series_type]!r}",
                DeprecationWarning, stacklevel=2)
            series_type = cls._aliases[series_type]
        try:
            series_class = cls._series_mapping[series_type]
        except KeyError:
            raise ValueError(f"Unknown whale reward series type: {series_type}")
        return series_class().create_series(horizon, **kwargs)


if __name__ == "__main__":
    # Generate one day's series (96 timesteps of 15 minutes each)
    horizon = 96
    series = WhaleRewardSeriesFactory.create_series('JoA2026', horizon)

    # Convert each timestep to hours of day
    hours = np.arange(horizon) * 0.25  # 0.25 hours per timestep

    plt.figure(figsize=(10, 4))
    plt.plot(hours, series, marker='.')
    plt.xlabel('Hour of Day')
    plt.ylabel('Whale Observation Probability')
    plt.title('JoA2026 Whale Observation Probability Over 24 Hours')
    plt.xticks(np.arange(0, 25, 2))
    plt.xlim(0, 24)
    plt.grid(True)
    plt.tight_layout()
    plt.show()

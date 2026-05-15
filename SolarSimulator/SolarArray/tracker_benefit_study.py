import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import requests
import time
from tqdm import tqdm
from pvlib import location, tracking, irradiance, temperature
from pvlib.pvsystem import pvwatts_dc

# =============================================================================
# Configuration
# =============================================================================
LOCATIONS = {
    'Coast of Japan':      {'lat': 35.0,  'lon': 140.0,  'tz': 'Asia/Tokyo'},
    'Gulf of Mexico':      {'lat': 25.0,  'lon': -90.0,  'tz': 'America/Chicago'},
    'Coast of Hawaii':     {'lat': 21.3,  'lon': -158.0, 'tz': 'Pacific/Honolulu'},
    'Mediterranean Sea':   {'lat': 38.0,  'lon': 15.0,   'tz': 'Europe/Rome'},
    'Coast of Alaska':     {'lat': 58.0,  'lon': -150.0, 'tz': 'America/Anchorage'},
}

YEARS = list(range(2015, 2025))  # 2015-2024, 10 full years

MONTH_LABELS = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

# Panel parameters
PDC0 = 30       # W
GAMMA_PDC = -0.004

# Tracker parameters
MAX_ANGLE = 60  # deg
GCR = 0.06

# Motor parameters
VOLTAGE_SYSTEM = 12.0          # V
MOTOR_STALL_CURRENT = 1.6     # A
MOTOR_OUTPUT_SPEED_RPM = 140
RATIO_EXTERNAL_WORM = 40.0


def simulate_year(lat, lon, tz, year):
    """Run tracker-vs-fixed simulation for one location over a full year.

    Returns a DataFrame with monthly benefit stats (12 rows).
    """
    start_date = f'{year}-01-01'
    end_date = f'{year}-12-31'
    site = location.Location(lat, lon, tz=tz)

    # Fetch full year of weather data (with retry on rate limit)
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": "temperature_2m,wind_speed_10m,wind_direction_10m,"
                  "shortwave_radiation,direct_normal_irradiance,diffuse_radiation",
        "timezone": tz,
    }
    for attempt in range(5):
        response = requests.get(url, params=params)
        if response.status_code == 429:
            wait = 2 ** attempt
            time.sleep(wait)
            continue
        response.raise_for_status()
        break
    else:
        response.raise_for_status()
    hourly = response.json()['hourly']

    times = pd.to_datetime(hourly['time'])
    weather_df = pd.DataFrame({
        'temp_air': hourly['temperature_2m'],
        'wind_speed': hourly['wind_speed_10m'],
        'wind_direction': hourly['wind_direction_10m'],
        'ghi': hourly['shortwave_radiation'],
        'dni': hourly['direct_normal_irradiance'],
        'dhi': hourly['diffuse_radiation'],
    }, index=times)

    # Solar position
    solpos = site.get_solarposition(times)
    dni_extra = irradiance.get_extra_radiation(times)

    wind_direction = weather_df['wind_direction']
    wind_speed = weather_df['wind_speed']
    temp_air = weather_df['temp_air']
    ghi = weather_df['ghi']
    dni = weather_df['dni']
    dhi = weather_df['dhi']

    # --- Tracker simulation ---
    dynamic_axis_azimuth = (wind_direction + 90) % 360

    tracker_data = tracking.singleaxis(
        apparent_zenith=solpos['apparent_zenith'],
        apparent_azimuth=solpos['azimuth'],
        axis_tilt=0,
        axis_azimuth=dynamic_axis_azimuth,
        max_angle=MAX_ANGLE,
        backtrack=True,
        gcr=GCR,
    ).fillna(0)

    poa_tracker = irradiance.get_total_irradiance(
        surface_tilt=tracker_data['surface_tilt'],
        surface_azimuth=tracker_data['surface_azimuth'],
        dni=dni, ghi=ghi, dhi=dhi,
        solar_zenith=solpos['apparent_zenith'],
        solar_azimuth=solpos['azimuth'],
        dni_extra=dni_extra,
        model='haydavies',
    )

    # --- Fixed-tilt simulation ---
    poa_fixed = irradiance.get_total_irradiance(
        surface_tilt=0.0,
        surface_azimuth=wind_direction,
        dni=dni, ghi=ghi, dhi=dhi,
        solar_zenith=solpos['apparent_zenith'],
        solar_azimuth=solpos['azimuth'],
        dni_extra=dni_extra,
        model='haydavies',
    )

    # --- PV power ---
    thermal_params = temperature.TEMPERATURE_MODEL_PARAMETERS['sapm']['open_rack_glass_glass']

    cell_temp_tracker = temperature.sapm_cell(
        poa_global=poa_tracker['poa_global'],
        temp_air=temp_air, wind_speed=wind_speed, **thermal_params,
    )
    dc_power_tracker = pvwatts_dc(
        g_poa_effective=poa_tracker['poa_global'],
        temp_cell=cell_temp_tracker,
        pdc0=PDC0, gamma_pdc=GAMMA_PDC,
    ).fillna(0).clip(upper=30.0)

    cell_temp_fixed = temperature.sapm_cell(
        poa_global=poa_fixed['poa_global'],
        temp_air=temp_air, wind_speed=wind_speed, **thermal_params,
    )
    dc_power_fixed = pvwatts_dc(
        g_poa_effective=poa_fixed['poa_global'],
        temp_cell=cell_temp_fixed,
        pdc0=PDC0, gamma_pdc=GAMMA_PDC,
    ).fillna(0)

    # --- Motor energy (sunlight hours only) ---
    actuation_speed = (MOTOR_OUTPUT_SPEED_RPM / RATIO_EXTERNAL_WORM) * 6.0
    power_watts = MOTOR_STALL_CURRENT * VOLTAGE_SYSTEM

    theta_diff = tracker_data['tracker_theta'].diff().abs().fillna(0)
    move_duration_sec = theta_diff / actuation_speed
    motor_energy_wh = (power_watts * move_duration_sec) / 3600.0

    # Zero out motor energy when sun is below horizon
    sunlight_mask = solpos['apparent_zenith'] < 90
    motor_energy_wh = motor_energy_wh.where(sunlight_mask, 0.0)

    net_power_tracker = dc_power_tracker - motor_energy_wh

    # --- Aggregate by month ---
    month = times.month
    monthly_records = []
    for m in range(1, 13):
        mask = month == m
        fixed_wh = dc_power_fixed[mask].sum()
        tracker_net_wh = net_power_tracker[mask].sum()
        motor_wh = motor_energy_wh[mask].sum()

        if fixed_wh > 0:
            net_gain_pct = ((tracker_net_wh - fixed_wh) / fixed_wh) * 100
        else:
            net_gain_pct = 0.0

        monthly_records.append({
            'month_idx': m - 1,
            'year': year,
            'net_gain_pct': net_gain_pct,
            'total_fixed_wh': fixed_wh,
            'total_tracker_net_wh': tracker_net_wh,
            'motor_energy_wh': motor_wh,
        })

    return monthly_records


def main():
    records = []
    combos = [(name, info, year) for name, info in LOCATIONS.items() for year in YEARS]

    for name, info, year in tqdm(combos, desc='Simulating'):
        monthly = simulate_year(info['lat'], info['lon'], info['tz'], year)
        for rec in monthly:
            rec['location'] = name
            rec['month'] = MONTH_LABELS[rec['month_idx']]
            records.append(rec)

    df = pd.DataFrame(records)

    # Average across years for each location x month
    avg = df.groupby(['location', 'month_idx', 'month'])['net_gain_pct'].mean().reset_index()
    avg.rename(columns={'net_gain_pct': 'mean_benefit'}, inplace=True)

    print("\n--- Mean Tracker Benefit (%) Averaged Over 2015-2024 ---")
    print(avg.pivot(index='location', columns='month', values='mean_benefit')
          .reindex(LOCATIONS.keys())[MONTH_LABELS].to_string())

    # Build benefit matrix for heatmap
    pivot = avg.pivot(index='location', columns='month_idx', values='mean_benefit')
    pivot.columns = MONTH_LABELS
    pivot = pivot.reindex(LOCATIONS.keys())

    # Stats for bar chart: mean and std across all months and years per location
    loc_stats = df.groupby('location')['net_gain_pct'].agg(['mean', 'min', 'max']).reindex(LOCATIONS.keys())

    # =========================================================================
    # Visualization
    # =========================================================================
    sns.set_theme(style='whitegrid')
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 9),
                                    gridspec_kw={'height_ratios': [3, 2]})

    # --- Panel 1: Heatmap ---
    vmax = max(abs(pivot.values.min()), abs(pivot.values.max()))
    sns.heatmap(
        pivot, annot=True, fmt='.1f', center=0,
        cmap='RdYlGn', vmin=-vmax, vmax=vmax,
        linewidths=0.5, linecolor='white',
        cbar_kws={'label': 'Net Tracker Benefit (%)'},
        ax=ax1,
    )
    ax1.set_title('Net Tracker Benefit by Location and Month (%, 2015-2024 Mean)', fontsize=14, fontweight='bold')
    ax1.set_ylabel('')
    ax1.set_xlabel('')

    # --- Panel 2: Bar chart with min/max error bars ---
    loc_order = list(LOCATIONS.keys())
    means = loc_stats['mean']
    mins = loc_stats['min']
    maxs = loc_stats['max']

    x = np.arange(len(loc_order))
    err_low = means - mins
    err_high = maxs - means

    bars = ax2.bar(x, means, color=sns.color_palette('RdYlGn', n_colors=len(loc_order)),
                   edgecolor='black', linewidth=0.5)
    ax2.errorbar(x, means, yerr=[err_low, err_high], fmt='none', ecolor='black',
                 capsize=5, capthick=1.2)
    ax2.set_xticks(x)
    ax2.set_xticklabels(loc_order, rotation=15, ha='right')
    ax2.set_ylabel('Net Tracker Benefit (%)')
    ax2.set_title('Mean Annual Tracker Benefit by Location (min/max range, 2015-2024)', fontsize=14, fontweight='bold')
    ax2.axhline(0, color='black', linewidth=0.8, linestyle='--')

    plt.tight_layout()
    outfile = 'tracker_benefit_locations.png'
    fig.savefig(outfile, dpi=800, bbox_inches='tight')
    print(f"\nFigure saved to {outfile}")
    plt.show()


if __name__ == '__main__':
    main()

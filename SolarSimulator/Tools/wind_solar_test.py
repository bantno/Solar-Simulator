# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import pvlib
# import requests
# from pvlib import location, tracking, irradiance, temperature
# from pvlib.pvsystem import pvwatts_dc

# def simulate_floating_tracker():
#     # -------------------------------------------------------------------------
#     # 1. Setup Location and Time
#     # -------------------------------------------------------------------------
#     # Example: A reservoir in Arizona (high sun)
#     lat, lon = 34.1787, -84.0403
#     tz = 'America/Phoenix' # Use specific timezone for API alignment
#     site = location.Location(lat, lon, tz=tz, name='Floating Array')

#     # Define date for simulation (Past date for historical data)
#     sim_date = '2024-06-21'
#     # sim_end_date = '2024-09-21'
#     sim_end_date = '2024-06-28'
    
    
#     # -------------------------------------------------------------------------
#     # 2. Fetch Real Weather Data from Open-Meteo
#     # -------------------------------------------------------------------------
#     print(f"Fetching weather data for {sim_date} from Open-Meteo API...")
    
#     # Open-Meteo Archive API endpoint
#     url = "https://archive-api.open-meteo.com/v1/archive"
#     params = {
#         "latitude": lat,
#         "longitude": lon,
#         "start_date": sim_date,
#         "end_date": sim_end_date,
#         # Fetch all necessary solar and weather components
#         "hourly": "temperature_2m,wind_speed_10m,wind_direction_10m,shortwave_radiation,direct_normal_irradiance,diffuse_radiation",
#         "timezone": tz
#     }
    
#     response = requests.get(url, params=params)
#     response.raise_for_status()
#     data_json = response.json()
    
#     # Process Hourly Data
#     hourly = data_json['hourly']
    
#     # Open-Meteo returns time as ISO strings. Convert to datetime.
#     times = pd.to_datetime(hourly['time'])
    
#     # Create a DataFrame with the fetched data
#     weather_df = pd.DataFrame({
#         'temp_air': hourly['temperature_2m'],
#         'wind_speed': hourly['wind_speed_10m'],
#         'wind_direction': hourly['wind_direction_10m'],
#         'ghi': hourly['shortwave_radiation'],
#         'dni': hourly['direct_normal_irradiance'],
#         'dhi': hourly['diffuse_radiation']
#     }, index=times)
    
#     # Localize index to match site timezone (API returns local time strings if tz is requested)
#     weather_df.index = weather_df.index.tz_localize(tz, ambiguous='NaT', nonexistent='shift_forward')
    
#     # Use the DataFrame index as our main time series
#     times = weather_df.index

#     # -------------------------------------------------------------------------
#     # 3. Solar Position & Extra Radiation
#     # -------------------------------------------------------------------------
#     print("Calculating solar position...")
#     # Calculate solar position for the exact times returned by API
#     solpos = site.get_solarposition(times)
    
#     # Calculate extraterrestrial radiation (required for Hay-Davies model)
#     dni_extra = irradiance.get_extra_radiation(times)
    
#     # Extract vectors for simulation
#     wind_direction = weather_df['wind_direction']
#     wind_speed = weather_df['wind_speed']
#     temp_air = weather_df['temp_air']
    
#     # Irradiance Inputs
#     ghi = weather_df['ghi']
#     dni = weather_df['dni']
#     dhi = weather_df['dhi']

#     # -------------------------------------------------------------------------
#     # 4. Tracker Simulation (Active Tracking + Weather Vane)
#     # -------------------------------------------------------------------------
#     print("Calculating active single-axis tracking...")

#     # Logic: The array faces the wind. The tracker axis is perpendicular to the wind.
#     # axis_azimuth is 90 degrees offset from wind direction.
#     dynamic_axis_azimuth = (wind_direction + 90) % 360

#     # System Parameters
#     max_angle = 60 # Max rotation of the tracker bar
#     gcr = 0.06      # Ground Coverage Ratio
    
#     tracker_data = tracking.singleaxis(
#         apparent_zenith=solpos['apparent_zenith'],
#         apparent_azimuth=solpos['azimuth'],
#         axis_tilt=0,                  
#         axis_azimuth=dynamic_axis_azimuth, 
#         max_angle=max_angle,
#         backtrack=True,               
#         gcr=gcr
#     )
    
#     # Fill NaNs (usually occur at night when sun is below horizon)
#     tracker_data = tracker_data.fillna(0)

#     poa_tracker = irradiance.get_total_irradiance(
#         surface_tilt=tracker_data['surface_tilt'],
#         surface_azimuth=tracker_data['surface_azimuth'],
#         dni=dni,
#         ghi=ghi,
#         dhi=dhi,
#         solar_zenith=solpos['apparent_zenith'],
#         solar_azimuth=solpos['azimuth'],
#         dni_extra=dni_extra,
#         model='haydavies'
#     )

#     # -------------------------------------------------------------------------
#     # 5. Fixed-Tilt Simulation (Passive Weather Vane Only)
#     # -------------------------------------------------------------------------
#     print("Calculating fixed-tilt weather vane...")

#     # Logic: The barge still rotates so panels face the wind, but tilt is fixed.
#     fixed_tilt_angle = 15.0 
#     fixed_surface_azimuth = wind_direction

#     poa_fixed = irradiance.get_total_irradiance(
#         surface_tilt=fixed_tilt_angle,
#         surface_azimuth=fixed_surface_azimuth,
#         dni=dni,
#         ghi=ghi,
#         dhi=dhi,
#         solar_zenith=solpos['apparent_zenith'],
#         solar_azimuth=solpos['azimuth'],
#         dni_extra=dni_extra,
#         model='haydavies'
#     )

#     # -------------------------------------------------------------------------
#     # 6. Calculate Power (Comparison)
#     # -------------------------------------------------------------------------
#     thermal_params = temperature.TEMPERATURE_MODEL_PARAMETERS['sapm']['open_rack_glass_glass']
#     pdc0 = 30 
#     gamma_pdc = -0.004

#     # --- A. Tracker Power ---
#     cell_temp_tracker = temperature.sapm_cell(
#         poa_global=poa_tracker['poa_global'],
#         temp_air=temp_air,
#         wind_speed=wind_speed,
#         **thermal_params
#     )
#     dc_power_tracker = pvwatts_dc(
#         g_poa_effective=poa_tracker['poa_global'],
#         temp_cell=cell_temp_tracker,
#         pdc0=pdc0,
#         gamma_pdc=gamma_pdc
#     ).fillna(0)

#     # --- B. Fixed Power ---
#     cell_temp_fixed = temperature.sapm_cell(
#         poa_global=poa_fixed['poa_global'],
#         temp_air=temp_air,
#         wind_speed=wind_speed,
#         **thermal_params
#     )
#     dc_power_fixed = pvwatts_dc(
#         g_poa_effective=poa_fixed['poa_global'],
#         temp_cell=cell_temp_fixed,
#         pdc0=pdc0,
#         gamma_pdc=gamma_pdc
#     ).fillna(0)

#     # -------------------------------------------------------------------------
#     # 7. Visualization
#     # -------------------------------------------------------------------------
#     print("Plotting results...")
    
#     fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 15), sharex=True)
    
#     # Plot 1: Orientations
#     ax1.plot(times, wind_direction, label='Wind Dir (Real Data)', color='blue', linewidth=2)
#     ax1.plot(times, dynamic_axis_azimuth, label='Tracker Axis Azimuth', color='red', linestyle='--')
#     ax1.set_ylabel('Degrees (0=N, 90=E)')
#     ax1.set_title(f'Floating Platform Orientation ({sim_date})')
#     ax1.legend()
#     ax1.grid(True)

#     # Plot 2: Tracker Rotation Angle
#     ax2.plot(times, tracker_data['tracker_theta'], label='Tracker Angle', color='purple')
#     ax2.set_ylabel('Degrees')
#     ax2.set_title('Tracker Rotation Angle (Relative to Axis)')
#     ax2.legend()
#     ax2.grid(True)

#     # Plot 3: Power Output Comparison
#     # Data is hourly, so sum() is Wh. Divide by 1000 for kWh.
#     total_e_tracker = dc_power_tracker.sum() / 1000
#     total_e_fixed = dc_power_fixed.sum() / 1000
    
#     if total_e_fixed > 0:
#         gain = ((total_e_tracker - total_e_fixed) / total_e_fixed) * 100
#     else:
#         gain = 0

#     ax3.plot(times, dc_power_tracker, label=f'Tracker ({total_e_tracker:.2f} kWh)', color='green')
#     ax3.plot(times, dc_power_fixed, label=f'Fixed Tilt ({total_e_fixed:.2f} kWh)', color='orange', linestyle='-.')
    
#     ax3.fill_between(times, dc_power_tracker, dc_power_fixed, where=(dc_power_tracker > dc_power_fixed), 
#                      color='green', alpha=0.1, label='Tracker Gain')
    
#     ax3.set_ylabel('Power (Watts per 1kW)')
#     ax3.set_xlabel('Time of Day')
#     ax3.set_title(f'Power Comparison: Tracker Advantage = +{gain:.1f}%')
#     ax3.legend()
#     ax3.grid(True)

#     plt.tight_layout()
#     output_file = 'floating_solar_comparison.png'
#     plt.savefig(output_file)
#     print(f"Simulation complete. Results saved to {output_file}")

#     # Output Sample Data
#     results_df = pd.DataFrame({
#         'Wind_Dir': wind_direction,
#         'GHI': ghi,
#         'Tracker_Power': dc_power_tracker,
#         'Fixed_Power': dc_power_fixed,
#         'Tracker_Theta': tracker_data['tracker_theta']
#     })
    
#     print("\nSample Comparison (12:00 PM - 3:00 PM):")
#     print(results_df.between_time('12:00', '15:00').head())

# if __name__ == "__main__":
#     simulate_floating_tracker()


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pvlib
import requests
from pvlib import location, tracking, irradiance, temperature
from pvlib.pvsystem import pvwatts_dc

def simulate_floating_tracker():
    # -------------------------------------------------------------------------
    # 1. Setup Location and Time
    # -------------------------------------------------------------------------
    lat, lon = 33.4, -112.0
    tz = 'America/Phoenix'
    site = location.Location(lat, lon, tz=tz, name='Floating Array')
    sim_date = '2024-06-21'
    
    # -------------------------------------------------------------------------
    # 2. Fetch Real Weather Data
    # -------------------------------------------------------------------------
    print(f"Fetching weather data for {sim_date} from Open-Meteo API...")
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": sim_date,
        "end_date": sim_date,
        "hourly": "temperature_2m,wind_speed_10m,wind_direction_10m,shortwave_radiation,direct_normal_irradiance,diffuse_radiation",
        "timezone": tz
    }
    
    response = requests.get(url, params=params)
    response.raise_for_status()
    data_json = response.json()
    
    hourly = data_json['hourly']
    times = pd.to_datetime(hourly['time'])
    
    weather_df = pd.DataFrame({
        'temp_air': hourly['temperature_2m'],
        'wind_speed': hourly['wind_speed_10m'],
        'wind_direction': hourly['wind_direction_10m'],
        'ghi': hourly['shortwave_radiation'],
        'dni': hourly['direct_normal_irradiance'],
        'dhi': hourly['diffuse_radiation']
    }, index=times)
    
    weather_df.index = weather_df.index.tz_localize(tz, ambiguous='NaT', nonexistent='shift_forward')
    times = weather_df.index

    # -------------------------------------------------------------------------
    # 3. Solar Position & Extra Radiation
    # -------------------------------------------------------------------------
    print("Calculating solar position...")
    solpos = site.get_solarposition(times)
    dni_extra = irradiance.get_extra_radiation(times)
    
    # Extract vectors
    wind_direction = weather_df['wind_direction']
    wind_speed = weather_df['wind_speed']
    temp_air = weather_df['temp_air']
    ghi = weather_df['ghi']
    dni = weather_df['dni']
    dhi = weather_df['dhi']

    # -------------------------------------------------------------------------
    # 4. Tracker Simulation (Active Tracking + Weather Vane)
    # -------------------------------------------------------------------------
    print("Calculating active single-axis tracking...")
    
    # Weather Vane Logic: Tracker axis is perpendicular to wind (wind hits panel chord-wise)
    dynamic_axis_azimuth = (wind_direction + 90) % 360

    max_angle = 60 
    gcr = 0.4      
    
    tracker_data = tracking.singleaxis(
        apparent_zenith=solpos['apparent_zenith'],
        apparent_azimuth=solpos['azimuth'],
        axis_tilt=0,                  
        axis_azimuth=dynamic_axis_azimuth, 
        max_angle=max_angle,
        backtrack=True,               
        gcr=gcr
    )
    tracker_data = tracker_data.fillna(0)

    # POA Calculation
    poa_tracker = irradiance.get_total_irradiance(
        surface_tilt=tracker_data['surface_tilt'],
        surface_azimuth=tracker_data['surface_azimuth'],
        dni=dni,
        ghi=ghi,
        dhi=dhi,
        solar_zenith=solpos['apparent_zenith'],
        solar_azimuth=solpos['azimuth'],
        dni_extra=dni_extra,
        model='haydavies'
    )

    # -------------------------------------------------------------------------
    # 5. Fixed-Tilt Simulation (Passive Weather Vane Only)
    # -------------------------------------------------------------------------
    print("Calculating fixed-tilt weather vane...")
    fixed_tilt_angle = 0.0 
    fixed_surface_azimuth = wind_direction

    poa_fixed = irradiance.get_total_irradiance(
        surface_tilt=fixed_tilt_angle,
        surface_azimuth=fixed_surface_azimuth,
        dni=dni,
        ghi=ghi,
        dhi=dhi,
        solar_zenith=solpos['apparent_zenith'],
        solar_azimuth=solpos['azimuth'],
        dni_extra=dni_extra,
        model='haydavies'
    )

    # -------------------------------------------------------------------------
    # 6. PV Power Calculation
    # -------------------------------------------------------------------------
    thermal_params = temperature.TEMPERATURE_MODEL_PARAMETERS['sapm']['open_rack_glass_glass']
    pdc0 = 30 # 1kW System
    gamma_pdc = -0.004

    # A. Tracker Generation
    cell_temp_tracker = temperature.sapm_cell(
        poa_global=poa_tracker['poa_global'],
        temp_air=temp_air,
        wind_speed=wind_speed,
        **thermal_params
    )
    dc_power_tracker = pvwatts_dc(
        g_poa_effective=poa_tracker['poa_global'],
        temp_cell=cell_temp_tracker,
        pdc0=pdc0,
        gamma_pdc=gamma_pdc
    ).fillna(0)

    # B. Fixed Generation
    cell_temp_fixed = temperature.sapm_cell(
        poa_global=poa_fixed['poa_global'],
        temp_air=temp_air,
        wind_speed=wind_speed,
        **thermal_params
    )
    dc_power_fixed = pvwatts_dc(
        g_poa_effective=poa_fixed['poa_global'],
        temp_cell=cell_temp_fixed,
        pdc0=pdc0,
        gamma_pdc=gamma_pdc
    ).fillna(0)

    # -------------------------------------------------------------------------
    # 7. Motor Energy Consumption Model (Including Gravity & Inertia)
    # -------------------------------------------------------------------------
    print("Calculating motor energy consumption...")

    # --- Mechanical Parameters ---
    area_array = 1.0       # m^2
    chord_length = 0.25    # meters
    
    # Mass Calculation
    mass_density = 20.0    # kg/m^2 (Panels + Racking)
    total_mass = area_array * mass_density # ~20 kg
    
    # --- GRAVITY / IMBALANCE PARAMETERS ---
    # The distance from the rotation axis to the Center of Mass.
    # If perfectly balanced, this is 0. 
    # We assume 5cm offset due to mounting brackets/glass thickness.
    com_offset_m = 0.05    
    
    # --- Inertial Parameters ---
    # I = (1/12) * mass * width^2 (Approximation for flat plate)
    inertia_array = (1.0/12.0) * total_mass * (chord_length ** 2)
    actuation_speed_deg_per_sec = 2.0 
    omega_rad_s = np.radians(actuation_speed_deg_per_sec)

    # --- Friction Parameters ---
    shaft_diameter_in = 0.5
    shaft_radius_m = (shaft_diameter_in * 0.0254) / 2.0
    
    # 1. Bearing Friction (Dependent on Normal Force/Weight)
    mu_bearing = 0.2 
    normal_force = total_mass * 9.81 
    torque_bearing = normal_force * mu_bearing * shaft_radius_m
    
    # 2. Stuffing Box Friction (Constant Drag)
    torque_stuffing_box = 1.5 
    
    friction_torque_base = torque_bearing + torque_stuffing_box

    # --- Efficiency ---
    # Worm gears are usually non-backdrivable. 
    # This means gravity generally cannot "charge" the battery (regen).
    eff_total = 0.42 

    # --- Torque Calculation Loop ---
    
    # Get angle in radians
    tracker_theta_rad = np.radians(tracker_data['tracker_theta'])
    
    # 1. Aerodynamic Torque (Wind)
    # Torque = Cm * q * Area * Chord
    rho_air = 1.225 
    Cm = 0.05 + 0.1 * np.sin(np.abs(tracker_theta_rad)) 
    torque_wind = Cm * 0.5 * rho_air * (wind_speed ** 2) * area_array * chord_length
    
    # 2. Gravitational Torque (Weight Imbalance)
    # T = m * g * r * sin(theta)
    # As theta increases (tilts away from flat), gravity pulls harder.
    torque_gravity = total_mass * 9.81 * com_offset_m * np.sin(np.abs(tracker_theta_rad))
    
    # 3. Total Static Load Torque
    # We sum the absolute values to be conservative (Worst Case).
    # In reality, gravity might help move the panel "down", but wind might oppose it.
    # For sizing a motor/battery, assume all forces fight you.
    torque_total_static = friction_torque_base + torque_wind + torque_gravity

    # --- Energy Calculation ---
    
    # A. Static Work (Wind + Gravity + Friction)
    theta_diff = tracker_data['tracker_theta'].diff().abs().fillna(0)
    theta_diff_rad = np.radians(theta_diff)
    
    energy_static_joules = (torque_total_static * theta_diff_rad) / eff_total

    # B. Inertial Work (Startup Spike)
    # Kinetic Energy = 0.5 * I * w^2
    kinetic_energy_joules = (0.5 * inertia_array * (omega_rad_s ** 2)) / eff_total
    
    # Only apply inertial cost if the tracker moves
    move_occurred = (theta_diff > 0.001).astype(int) 
    energy_inertial_total = kinetic_energy_joules * move_occurred

    # Total Energy
    total_energy_joules = energy_static_joules + energy_inertial_total
    energy_wh = total_energy_joules / 3600.0

    # Net Power Calculation
    net_power_tracker = dc_power_tracker - energy_wh

    # -------------------------------------------------------------------------
    # 8. Visualization
    # -------------------------------------------------------------------------
    print("Plotting results...")
    
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 16), sharex=True)
    
    # Plot 1: Wind & Torque
    ax1.plot(times, torque_wind, label='Wind Torque Load (Nm)', color='blue', alpha=0.7)
    ax1.plot(times, torque_total_static, label='Total Torque (Nm)', color='black', linestyle='--')
    ax1.set_ylabel('Torque (Nm)')
    ax1.set_title('Mechanical Load on Tracker Motor')
    ax1.legend()
    ax1.grid(True)

    # Plot 2: Consumption vs Generation
    # Use twinx to show consumption on a different scale if needed, 
    # but here we want to see them relative to each other (log scale might be better, but linear is honest)
    ax2.plot(times, dc_power_tracker, label='PV Generation (W)', color='green')
    ax2.plot(times, energy_wh, label='Motor Consumption (Wh/step)', color='red')
    ax2.set_ylabel('Power / Energy')
    ax2.set_title('Generation vs. Actuation Cost')
    ax2.legend()
    ax2.grid(True)

    # Plot 3: Net Energy Comparison
    total_gen_tracker = dc_power_tracker.sum() / 1000
    total_cons_tracker = energy_wh.sum() / 1000
    total_net_tracker = net_power_tracker.sum() / 1000
    total_gen_fixed = dc_power_fixed.sum() / 1000
    
    gain_gross = ((total_gen_tracker - total_gen_fixed) / total_gen_fixed) * 100
    gain_net = ((total_net_tracker - total_gen_fixed) / total_gen_fixed) * 100

    ax3.plot(times, net_power_tracker, label=f'Net Tracker ({total_net_tracker:.2f} kWh)', color='green')
    ax3.plot(times, dc_power_fixed, label=f'Fixed Tilt ({total_gen_fixed:.2f} kWh)', color='orange', linestyle='-.')
    
    ax3.fill_between(times, net_power_tracker, dc_power_fixed, where=(net_power_tracker > dc_power_fixed), 
                     color='green', alpha=0.1, label='Net Benefit')
    
    ax3.set_ylabel('Net Power (W)')
    ax3.set_xlabel('Time of Day')
    ax3.set_title(f'Net Benefit: {gain_net:.1f}% (Gross Gain: {gain_gross:.1f}%, Motor Cost: {total_cons_tracker:.3f} kWh)')
    ax3.legend()
    ax3.grid(True)

    plt.tight_layout()
    output_file = 'floating_solar_simulation_with_motor.png'
    plt.savefig(output_file)
    print(f"Simulation complete. Results saved to {output_file}")

    # Output Sample Data
    results_df = pd.DataFrame({
        'Wind_Speed': wind_speed,
        'Tracker_Theta': tracker_data['tracker_theta'],
        'Torque_Load_Nm': torque_wind,
        'Motor_Energy_Wh': energy_wh,
        'PV_Gen_W': dc_power_tracker
    })
    
    print("\nSample Mechanical Data (12:00 PM - 3:00 PM):")
    print(results_df.between_time('12:00', '15:00').head())
    
    print(f"\n--- Final Summary ---")
    print(f"Total Generation (Tracker): {total_gen_tracker:.4f} kWh")
    print(f"Total Motor Consumption:    {total_cons_tracker:.4f} kWh")
    print(f"Parasitic Load:             {(total_cons_tracker/total_gen_tracker)*100:.2f}%")

if __name__ == "__main__":
    simulate_floating_tracker()
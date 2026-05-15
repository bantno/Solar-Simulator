# # import pandas as pd
# # import numpy as np
# # import matplotlib.pyplot as plt
# # import pvlib
# # import requests
# # from pvlib import location, tracking, irradiance, temperature
# # from pvlib.pvsystem import pvwatts_dc

# # def simulate_floating_tracker():
# #     # -------------------------------------------------------------------------
# #     # 1. Setup Location and Time
# #     # -------------------------------------------------------------------------
# #     # Example: A reservoir in Arizona (high sun)
# #     lat, lon = 34.1787, -84.0403
# #     tz = 'America/Phoenix' # Use specific timezone for API alignment
# #     site = location.Location(lat, lon, tz=tz, name='Floating Array')

# #     # Define date for simulation (Past date for historical data)
# #     sim_date = '2024-06-21'
# #     # sim_end_date = '2024-09-21'
# #     sim_end_date = '2024-06-28'
    
    
# #     # -------------------------------------------------------------------------
# #     # 2. Fetch Real Weather Data from Open-Meteo
# #     # -------------------------------------------------------------------------
# #     print(f"Fetching weather data for {sim_date} from Open-Meteo API...")
    
# #     # Open-Meteo Archive API endpoint
# #     url = "https://archive-api.open-meteo.com/v1/archive"
# #     params = {
# #         "latitude": lat,
# #         "longitude": lon,
# #         "start_date": sim_date,
# #         "end_date": sim_end_date,
# #         # Fetch all necessary solar and weather components
# #         "hourly": "temperature_2m,wind_speed_10m,wind_direction_10m,shortwave_radiation,direct_normal_irradiance,diffuse_radiation",
# #         "timezone": tz
# #     }
    
# #     response = requests.get(url, params=params)
# #     response.raise_for_status()
# #     data_json = response.json()
    
# #     # Process Hourly Data
# #     hourly = data_json['hourly']
    
# #     # Open-Meteo returns time as ISO strings. Convert to datetime.
# #     times = pd.to_datetime(hourly['time'])
    
# #     # Create a DataFrame with the fetched data
# #     weather_df = pd.DataFrame({
# #         'temp_air': hourly['temperature_2m'],
# #         'wind_speed': hourly['wind_speed_10m'],
# #         'wind_direction': hourly['wind_direction_10m'],
# #         'ghi': hourly['shortwave_radiation'],
# #         'dni': hourly['direct_normal_irradiance'],
# #         'dhi': hourly['diffuse_radiation']
# #     }, index=times)
    
# #     # Localize index to match site timezone (API returns local time strings if tz is requested)
# #     weather_df.index = weather_df.index.tz_localize(tz, ambiguous='NaT', nonexistent='shift_forward')
    
# #     # Use the DataFrame index as our main time series
# #     times = weather_df.index

# #     # -------------------------------------------------------------------------
# #     # 3. Solar Position & Extra Radiation
# #     # -------------------------------------------------------------------------
# #     print("Calculating solar position...")
# #     # Calculate solar position for the exact times returned by API
# #     solpos = site.get_solarposition(times)
    
# #     # Calculate extraterrestrial radiation (required for Hay-Davies model)
# #     dni_extra = irradiance.get_extra_radiation(times)
    
# #     # Extract vectors for simulation
# #     wind_direction = weather_df['wind_direction']
# #     wind_speed = weather_df['wind_speed']
# #     temp_air = weather_df['temp_air']
    
# #     # Irradiance Inputs
# #     ghi = weather_df['ghi']
# #     dni = weather_df['dni']
# #     dhi = weather_df['dhi']

# #     # -------------------------------------------------------------------------
# #     # 4. Tracker Simulation (Active Tracking + Weather Vane)
# #     # -------------------------------------------------------------------------
# #     print("Calculating active single-axis tracking...")

# #     # Logic: The array faces the wind. The tracker axis is perpendicular to the wind.
# #     # axis_azimuth is 90 degrees offset from wind direction.
# #     dynamic_axis_azimuth = (wind_direction + 90) % 360

# #     # System Parameters
# #     max_angle = 60 # Max rotation of the tracker bar
# #     gcr = 0.06      # Ground Coverage Ratio
    
# #     tracker_data = tracking.singleaxis(
# #         apparent_zenith=solpos['apparent_zenith'],
# #         apparent_azimuth=solpos['azimuth'],
# #         axis_tilt=0,                  
# #         axis_azimuth=dynamic_axis_azimuth, 
# #         max_angle=max_angle,
# #         backtrack=True,               
# #         gcr=gcr
# #     )
    
# #     # Fill NaNs (usually occur at night when sun is below horizon)
# #     tracker_data = tracker_data.fillna(0)

# #     poa_tracker = irradiance.get_total_irradiance(
# #         surface_tilt=tracker_data['surface_tilt'],
# #         surface_azimuth=tracker_data['surface_azimuth'],
# #         dni=dni,
# #         ghi=ghi,
# #         dhi=dhi,
# #         solar_zenith=solpos['apparent_zenith'],
# #         solar_azimuth=solpos['azimuth'],
# #         dni_extra=dni_extra,
# #         model='haydavies'
# #     )

# #     # -------------------------------------------------------------------------
# #     # 5. Fixed-Tilt Simulation (Passive Weather Vane Only)
# #     # -------------------------------------------------------------------------
# #     print("Calculating fixed-tilt weather vane...")

# #     # Logic: The barge still rotates so panels face the wind, but tilt is fixed.
# #     fixed_tilt_angle = 15.0 
# #     fixed_surface_azimuth = wind_direction

# #     poa_fixed = irradiance.get_total_irradiance(
# #         surface_tilt=fixed_tilt_angle,
# #         surface_azimuth=fixed_surface_azimuth,
# #         dni=dni,
# #         ghi=ghi,
# #         dhi=dhi,
# #         solar_zenith=solpos['apparent_zenith'],
# #         solar_azimuth=solpos['azimuth'],
# #         dni_extra=dni_extra,
# #         model='haydavies'
# #     )

# #     # -------------------------------------------------------------------------
# #     # 6. Calculate Power (Comparison)
# #     # -------------------------------------------------------------------------
# #     thermal_params = temperature.TEMPERATURE_MODEL_PARAMETERS['sapm']['open_rack_glass_glass']
# #     pdc0 = 30 
# #     gamma_pdc = -0.004

# #     # --- A. Tracker Power ---
# #     cell_temp_tracker = temperature.sapm_cell(
# #         poa_global=poa_tracker['poa_global'],
# #         temp_air=temp_air,
# #         wind_speed=wind_speed,
# #         **thermal_params
# #     )
# #     dc_power_tracker = pvwatts_dc(
# #         g_poa_effective=poa_tracker['poa_global'],
# #         temp_cell=cell_temp_tracker,
# #         pdc0=pdc0,
# #         gamma_pdc=gamma_pdc
# #     ).fillna(0)

# #     # --- B. Fixed Power ---
# #     cell_temp_fixed = temperature.sapm_cell(
# #         poa_global=poa_fixed['poa_global'],
# #         temp_air=temp_air,
# #         wind_speed=wind_speed,
# #         **thermal_params
# #     )
# #     dc_power_fixed = pvwatts_dc(
# #         g_poa_effective=poa_fixed['poa_global'],
# #         temp_cell=cell_temp_fixed,
# #         pdc0=pdc0,
# #         gamma_pdc=gamma_pdc
# #     ).fillna(0)

# #     # -------------------------------------------------------------------------
# #     # 7. Visualization
# #     # -------------------------------------------------------------------------
# #     print("Plotting results...")
    
# #     fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 15), sharex=True)
    
# #     # Plot 1: Orientations
# #     ax1.plot(times, wind_direction, label='Wind Dir (Real Data)', color='blue', linewidth=2)
# #     ax1.plot(times, dynamic_axis_azimuth, label='Tracker Axis Azimuth', color='red', linestyle='--')
# #     ax1.set_ylabel('Degrees (0=N, 90=E)')
# #     ax1.set_title(f'Floating Platform Orientation ({sim_date})')
# #     ax1.legend()
# #     ax1.grid(True)

# #     # Plot 2: Tracker Rotation Angle
# #     ax2.plot(times, tracker_data['tracker_theta'], label='Tracker Angle', color='purple')
# #     ax2.set_ylabel('Degrees')
# #     ax2.set_title('Tracker Rotation Angle (Relative to Axis)')
# #     ax2.legend()
# #     ax2.grid(True)

# #     # Plot 3: Power Output Comparison
# #     # Data is hourly, so sum() is Wh. Divide by 1000 for kWh.
# #     total_e_tracker = dc_power_tracker.sum() / 1000
# #     total_e_fixed = dc_power_fixed.sum() / 1000
    
# #     if total_e_fixed > 0:
# #         gain = ((total_e_tracker - total_e_fixed) / total_e_fixed) * 100
# #     else:
# #         gain = 0

# #     ax3.plot(times, dc_power_tracker, label=f'Tracker ({total_e_tracker:.2f} kWh)', color='green')
# #     ax3.plot(times, dc_power_fixed, label=f'Fixed Tilt ({total_e_fixed:.2f} kWh)', color='orange', linestyle='-.')
    
# #     ax3.fill_between(times, dc_power_tracker, dc_power_fixed, where=(dc_power_tracker > dc_power_fixed), 
# #                      color='green', alpha=0.1, label='Tracker Gain')
    
# #     ax3.set_ylabel('Power (Watts per 1kW)')
# #     ax3.set_xlabel('Time of Day')
# #     ax3.set_title(f'Power Comparison: Tracker Advantage = +{gain:.1f}%')
# #     ax3.legend()
# #     ax3.grid(True)

# #     plt.tight_layout()
# #     output_file = 'floating_solar_comparison.png'
# #     plt.savefig(output_file)
# #     print(f"Simulation complete. Results saved to {output_file}")

# #     # Output Sample Data
# #     results_df = pd.DataFrame({
# #         'Wind_Dir': wind_direction,
# #         'GHI': ghi,
# #         'Tracker_Power': dc_power_tracker,
# #         'Fixed_Power': dc_power_fixed,
# #         'Tracker_Theta': tracker_data['tracker_theta']
# #     })
    
# #     print("\nSample Comparison (12:00 PM - 3:00 PM):")
# #     print(results_df.between_time('12:00', '15:00').head())

# # if __name__ == "__main__":
# #     simulate_floating_tracker()


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
#     lat, lon = 33.4, -112.0
#     tz = 'America/Phoenix'
#     site = location.Location(lat, lon, tz=tz, name='Floating Array')
#     sim_date = '2024-06-21'
    
#     # -------------------------------------------------------------------------
#     # 2. Fetch Real Weather Data
#     # -------------------------------------------------------------------------
#     print(f"Fetching weather data for {sim_date} from Open-Meteo API...")
#     url = "https://archive-api.open-meteo.com/v1/archive"
#     params = {
#         "latitude": lat,
#         "longitude": lon,
#         "start_date": sim_date,
#         "end_date": sim_date,
#         "hourly": "temperature_2m,wind_speed_10m,wind_direction_10m,shortwave_radiation,direct_normal_irradiance,diffuse_radiation",
#         "timezone": tz
#     }
    
#     response = requests.get(url, params=params)
#     response.raise_for_status()
#     data_json = response.json()
    
#     hourly = data_json['hourly']
#     times = pd.to_datetime(hourly['time'])
    
#     weather_df = pd.DataFrame({
#         'temp_air': hourly['temperature_2m'],
#         'wind_speed': hourly['wind_speed_10m'],
#         'wind_direction': hourly['wind_direction_10m'],
#         'ghi': hourly['shortwave_radiation'],
#         'dni': hourly['direct_normal_irradiance'],
#         'dhi': hourly['diffuse_radiation']
#     }, index=times)
    
#     weather_df.index = weather_df.index.tz_localize(tz, ambiguous='NaT', nonexistent='shift_forward')
#     times = weather_df.index

#     # -------------------------------------------------------------------------
#     # 3. Solar Position & Extra Radiation
#     # -------------------------------------------------------------------------
#     print("Calculating solar position...")
#     solpos = site.get_solarposition(times)
#     dni_extra = irradiance.get_extra_radiation(times)
    
#     # Extract vectors
#     wind_direction = weather_df['wind_direction']
#     wind_speed = weather_df['wind_speed']
#     temp_air = weather_df['temp_air']
#     ghi = weather_df['ghi']
#     dni = weather_df['dni']
#     dhi = weather_df['dhi']

#     # -------------------------------------------------------------------------
#     # 4. Tracker Simulation (Active Tracking + Weather Vane)
#     # -------------------------------------------------------------------------
#     print("Calculating active single-axis tracking...")
    
#     # Weather Vane Logic: Tracker axis is perpendicular to wind (wind hits panel chord-wise)
#     dynamic_axis_azimuth = (wind_direction + 90) % 360

#     max_angle = 60 
#     gcr = 0.4      
    
#     tracker_data = tracking.singleaxis(
#         apparent_zenith=solpos['apparent_zenith'],
#         apparent_azimuth=solpos['azimuth'],
#         axis_tilt=0,                  
#         axis_azimuth=dynamic_axis_azimuth, 
#         max_angle=max_angle,
#         backtrack=True,               
#         gcr=gcr
#     )
#     tracker_data = tracker_data.fillna(0)

#     # POA Calculation
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
#     fixed_tilt_angle = 0.0 
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
#     # 6. PV Power Calculation
#     # -------------------------------------------------------------------------
#     thermal_params = temperature.TEMPERATURE_MODEL_PARAMETERS['sapm']['open_rack_glass_glass']
#     pdc0 = 30 # 1kW System
#     gamma_pdc = -0.004

#     # A. Tracker Generation
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

#     # B. Fixed Generation
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
#     # 7. Motor Energy Consumption Model (Pololu 3478 Specific)
#     # -------------------------------------------------------------------------
#     print("Calculating motor energy consumption...")

#     # --- Pololu 3478 Specifications ---
#     voltage_system = 12.0          # Volts
#     motor_stall_current = 1.6      # Amps (Max consumption / Worst Case)
#     motor_no_load_current = 0.08   # Amps
#     motor_output_speed_rpm = 140/4   # RPM at gearbox output (No Load)

#     # --- External Gearing (Assumed) ---
#     # The Pololu motor (0.6Nm) cannot drive the stuffing box (1.5Nm) directly.
#     # We assume it drives the Worm Gear mentioned in your first prompt.
#     ratio_external_worm = 40.0
    
#     # Total Speed Calculation
#     # Speed at panel = Motor Speed / External Ratio
#     # 140 RPM / 40 = 3.5 RPM = ~21 degrees/second
#     # This is quite fast. We will assume we throttle it or it runs effectively this fast.
#     actuation_speed_deg_per_sec = (motor_output_speed_rpm / ratio_external_worm) * 6.0 
#     # Note: *6.0 converts RPM to deg/s (360/60) -> 140/40 * 6 = 21 deg/s.

#     # --- User Request: Conservative Current Estimate ---
#     # We fix the current to the Stall Current (1.6A) to create a "Max Limit" budget.
#     # In reality, it will likely run at ~0.5A.
#     current_conservative = motor_stall_current 
    
#     power_conservative_watts = current_conservative * voltage_system # 19.2 Watts

#     # --- Energy Calculation ---
    
#     # 1. Calculate Move Duration
#     # Time = Distance / Speed
#     theta_diff_deg = tracker_data['tracker_theta'].diff().abs().fillna(0)
#     move_duration_sec = theta_diff_deg / actuation_speed_deg_per_sec
    
#     # 2. Calculate Energy (Joules)
#     # Energy = Power (19.2W) * Time
#     energy_joules = power_conservative_watts * move_duration_sec

#     # 3. Inertial Startup (Startup Spike)
#     # Pololu 3478 is small, so rotor inertia is negligible compared to the 
#     # 1.6A "Stall" current we are already assuming for the whole duration.
#     # We will ignore explicit KE calculation because we are already over-estimating 
#     # the run current by ~300%.
    
#     # 4. Convert to Wh
#     energy_wh = energy_joules / 3600.0
    
#     # --- Diagnostics ---
#     daily_energy_wh = energy_wh.sum() / (len(energy_wh)/24.0) # Approx daily avg
#     print(f"--- Pololu 3478 Analysis ---")
#     print(f"Modeled Current Draw: {current_conservative} A (Conservative/Stall)")
#     print(f"Modeled Power Draw:   {power_conservative_watts:.2f} W")
#     print(f"Actuation Speed:      {actuation_speed_deg_per_sec:.1f} deg/s")
#     print(f"Est. Daily Energy:    {daily_energy_wh:.4f} Wh/day")

#     # Net Power
#     net_power_tracker = dc_power_tracker - energy_wh

#     # -------------------------------------------------------------------------
#     # 8. Visualization
#     # -------------------------------------------------------------------------
#     print("Plotting results...")
    
#     fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 16), sharex=True)
    
#     # # Plot 1: Wind & Torque
#     # ax1.plot(times, torque_wind, label='Wind Torque Load (Nm)', color='blue', alpha=0.7)
#     # ax1.plot(times, torque_total_load, label='Total Torque (Nm)', color='black', linestyle='--')
#     # ax1.set_ylabel('Torque (Nm)')
#     # ax1.set_title('Mechanical Load on Tracker Motor')
#     # ax1.legend()
#     # ax1.grid(True)

#     # Plot 2: Consumption vs Generation
#     # Use twinx to show consumption on a different scale if needed, 
#     # but here we want to see them relative to each other (log scale might be better, but linear is honest)
#     ax2.plot(times, dc_power_tracker, label='PV Generation (W)', color='green')
#     ax2.plot(times, energy_wh, label='Motor Consumption (Wh/step)', color='red')
#     ax2.set_ylabel('Power / Energy')
#     ax2.set_title('Generation vs. Actuation Cost')
#     ax2.legend()
#     ax2.grid(True)

#     # Plot 3: Net Energy Comparison
#     total_gen_tracker = dc_power_tracker.sum() / 1000
#     total_cons_tracker = energy_wh.sum() / 1000
#     total_net_tracker = net_power_tracker.sum() / 1000
#     total_gen_fixed = dc_power_fixed.sum() / 1000
    
#     gain_gross = ((total_gen_tracker - total_gen_fixed) / total_gen_fixed) * 100
#     gain_net = ((total_net_tracker - total_gen_fixed) / total_gen_fixed) * 100

#     ax3.plot(times, net_power_tracker, label=f'Net Tracker ({total_net_tracker:.2f} kWh)', color='green')
#     ax3.plot(times, dc_power_fixed, label=f'Fixed Tilt ({total_gen_fixed:.2f} kWh)', color='orange', linestyle='-.')
    
#     ax3.fill_between(times, net_power_tracker, dc_power_fixed, where=(net_power_tracker > dc_power_fixed), 
#                      color='green', alpha=0.1, label='Net Benefit')
    
#     ax3.set_ylabel('Net Power (W)')
#     ax3.set_xlabel('Time of Day')
#     ax3.set_title(f'Net Benefit: {gain_net:.1f}% (Gross Gain: {gain_gross:.1f}%, Motor Cost: {total_cons_tracker:.3f} kWh)')
#     ax3.legend()
#     ax3.grid(True)

#     plt.tight_layout()
#     output_file = 'floating_solar_simulation_with_motor.png'
#     plt.savefig(output_file)
#     print(f"Simulation complete. Results saved to {output_file}")

#     # Output Sample Data
#     results_df = pd.DataFrame({
#         'Wind_Speed': wind_speed,
#         'Tracker_Theta': tracker_data['tracker_theta'],
#         # 'Torque_Load_Nm': torque_wind,
#         'Motor_Energy_Wh': energy_wh,
#         'PV_Gen_W': dc_power_tracker
#     })
    
#     print("\nSample Mechanical Data (12:00 PM - 3:00 PM):")
#     print(results_df.between_time('12:00', '15:00').head())
    
#     print(f"\n--- Final Summary ---")
#     print(f"Total Generation (Tracker): {total_gen_tracker:.4f} kWh")
#     print(f"Total Motor Consumption:    {total_cons_tracker:.4f} kWh")
#     print(f"Parasitic Load:             {(total_cons_tracker/total_gen_tracker)*100:.2f}%")

# # -------------------------------------------------------------------------
#     # 8. Visualization (Updated with Net Components)
#     # -------------------------------------------------------------------------
#     print("Plotting results...")
    
#     # --- Calculations for the new plot ---
#     # 1. Power Difference (Watts): Instantaneous benefit
#     power_diff_watts = net_power_tracker - dc_power_fixed
    
#     # 2. Cumulative Energy Gain (Wh): Total benefit over time
#     # Since interval is 1 hour, sum of Watts = Watt-hours
#     cumulative_energy_gain_wh = power_diff_watts.cumsum()

#     # Increase figure height to accommodate 4 plots
#     fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(10, 20), sharex=True)
    
#     # Plot 1: Tracker vs Wind Angles (Context)
#     ax1.plot(times, wind_direction, label='Wind Direction', color='blue', alpha=0.6)
#     ax1.plot(times, tracker_data['tracker_theta'], label='Tracker Angle', color='purple', linestyle='--')
#     ax1.set_ylabel('Degrees')
#     ax1.set_title('System Dynamics: Wind vs Tracker Response')
#     ax1.legend(loc='upper right')
#     ax1.grid(True, alpha=0.3)

#     # Plot 2: Gross Power vs Consumption
#     ax2.plot(times, dc_power_tracker, label='Tracker Gen (W)', color='green')
#     ax2.plot(times, dc_power_fixed, label='Fixed Gen (W)', color='orange', linestyle='-.')
#     # Plot consumption on secondary axis because it is very small
#     ax2_right = ax2.twinx()
#     ax2_right.fill_between(times, energy_wh, color='red', alpha=0.3, label='Motor Cons (Wh)')
#     ax2_right.set_ylabel('Motor Energy (Wh)', color='red')
#     ax2_right.tick_params(axis='y', labelcolor='red')
#     ax2.set_ylabel('Power Generation (W)')
#     ax2.set_title('Gross Generation & Motor Consumption')
#     ax2.legend(loc='upper left')
#     ax2.grid(True, alpha=0.3)

#     # Plot 3: Net Power Comparison
#     ax3.plot(times, net_power_tracker, label='Net Tracker Power (W)', color='green')
#     ax3.plot(times, dc_power_fixed, label='Fixed Power (W)', color='orange', linestyle='-.')
#     ax3.fill_between(times, net_power_tracker, dc_power_fixed, where=(net_power_tracker > dc_power_fixed), 
#                      color='green', alpha=0.2, label='Net Benefit')
#     ax3.fill_between(times, net_power_tracker, dc_power_fixed, where=(net_power_tracker < dc_power_fixed), 
#                      color='red', alpha=0.2, label='Net Loss')
#     ax3.set_ylabel('Net Power (W)')
#     ax3.set_title('Net Power Comparison (Generation - Motor Load)')
#     ax3.legend(loc='upper right')
#     ax3.grid(True, alpha=0.3)

#     # --- Plot 4: The Requested "Net Components" ---
#     # Left Axis: Instantaneous Power Difference
#     ax4.plot(times, power_diff_watts, label='Instantaneous Power Gain (W)', color='black', linewidth=1)
#     ax4.axhline(0, color='gray', linewidth=0.8) # Zero line for reference
#     ax4.set_ylabel('Power Gain (W)', color='black')
    
#     # Right Axis: Cumulative Energy Gain
#     ax4_right = ax4.twinx()
#     ax4_right.plot(times, cumulative_energy_gain_wh, label='Cumulative Energy Gain (Wh)', color='blue', linewidth=2)
#     ax4_right.set_ylabel('Cumulative Gain (Wh)', color='blue')
#     ax4_right.tick_params(axis='y', labelcolor='blue')
    
#     ax4.set_title('Net Benefit Components: Power Difference & Cumulative Energy')
#     ax4.set_xlabel('Time of Day')
    
#     # Combined Legend for the dual-axis plot
#     lines, labels = ax4.get_legend_handles_labels()
#     lines2, labels2 = ax4_right.get_legend_handles_labels()
#     ax4.legend(lines + lines2, labels + labels2, loc='upper left')
#     ax4.grid(True, alpha=0.3)

#     plt.tight_layout()
#     output_file = 'floating_solar_net_components.png'
#     plt.savefig(output_file)
#     print(f"Simulation complete. Results saved to {output_file}")
    
#     # Print Summary Stats
#     total_gain = cumulative_energy_gain_wh.iloc[-1]
#     print(f"\n--- Final Net Analysis ---")
#     print(f"Total Energy Gain over Period: {total_gain:.4f} Wh")

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
    lat, lon = 25, -90.0
    tz = 'America/Chicago'
    site = location.Location(lat, lon, tz=tz, name='Floating Array')
    sim_date = '2024-06-21'
    sim_end_date = '2024-06-28'
    
    # -------------------------------------------------------------------------
    # 2. Fetch Real Weather Data
    # -------------------------------------------------------------------------
    print(f"Fetching weather data for {sim_date} from Open-Meteo API...")
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": sim_date,
        "end_date": sim_end_date,
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
    
    # weather_df.index = weather_df.index.tz_localize(tz, ambiguous='NaT', nonexistent='shift_forward')
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
    
    # Weather Vane Logic: Tracker axis is perpendicular to wind
    dynamic_axis_azimuth = (wind_direction + 90) % 360

    max_angle = 60 
    gcr = 0.06      
    
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
    pdc0 = 30 
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
    dc_power_tracker = dc_power_tracker.clip(upper=30.0)

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
    # 7. Motor Energy Consumption Model
    # -------------------------------------------------------------------------
    print("Calculating motor energy consumption...")
    voltage_system = 12.0          
    motor_stall_current = 1.6      
    motor_output_speed_rpm = 140   
    ratio_external_worm = 40.0
    
    actuation_speed_deg_per_sec = (motor_output_speed_rpm / ratio_external_worm) * 6.0 
    current_conservative = motor_stall_current 
    power_conservative_watts = current_conservative * voltage_system 

    theta_diff_deg = tracker_data['tracker_theta'].diff().abs().fillna(0)
    move_duration_sec = theta_diff_deg / actuation_speed_deg_per_sec
    energy_joules = power_conservative_watts * move_duration_sec
    energy_wh = energy_joules / 3600.0
    
    # Net Power
    net_power_tracker = dc_power_tracker - energy_wh

    # -------------------------------------------------------------------------
    # 8. Visualization
    # -------------------------------------------------------------------------
    print("Plotting results...")
    
    # --- Calculate Totals for Titles ---
    total_fixed_wh = dc_power_fixed.sum()
    total_net_tracker_wh = net_power_tracker.sum()
    
    # Calculate Percentage Gain (Net vs Fixed)
    if total_fixed_wh > 0:
        net_gain_pct = ((total_net_tracker_wh - total_fixed_wh) / total_fixed_wh) * 100
    else:
        net_gain_pct = 0.0

    # Calculate Differences for Plot 4
    power_diff_watts = net_power_tracker - dc_power_fixed
    cumulative_energy_gain_wh = power_diff_watts.cumsum()

    # fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(10, 20), sharex=True)
    
    # # Plot 1: Tracker vs Wind
    # ax1.plot(times, wind_direction, label='Wind Direction', color='blue', alpha=0.6)
    # ax1.plot(times, tracker_data['tracker_theta'], label='Tracker Angle', color='purple', linestyle='--')
    # ax1.set_ylabel('Degrees')
    # ax1.set_title('System Dynamics: Wind vs Tracker Response')
    # ax1.legend(loc='upper right')
    # ax1.grid(True, alpha=0.3)

    # # Plot 2: Gross Gen vs Motor
    # ax2.plot(times, dc_power_tracker, label='Tracker Gen (W)', color='green')
    # ax2.plot(times, dc_power_fixed, label='Fixed Gen (W)', color='orange', linestyle='-.')
    # ax2_right = ax2.twinx()
    # ax2_right.fill_between(times, energy_wh, color='red', alpha=0.3, label='Motor Cons (Wh)')
    # ax2_right.set_ylabel('Motor Energy (Wh)', color='red')
    # ax2_right.tick_params(axis='y', labelcolor='red')
    # ax2.set_ylabel('Power Generation (W)')
    # ax2.set_title('Gross Generation & Motor Consumption')
    # ax2.legend(loc='upper left')
    # ax2.grid(True, alpha=0.3)

    # # Plot 3: Net Power Comparison (WITH PERCENTAGE IN TITLE)
    # ax3.plot(times, net_power_tracker, label='Net Tracker Power (W)', color='green')
    # ax3.plot(times, dc_power_fixed, label='Fixed Power (W)', color='orange', linestyle='-.')
    # ax3.fill_between(times, net_power_tracker, dc_power_fixed, where=(net_power_tracker > dc_power_fixed), 
    #                  color='green', alpha=0.2, label='Net Benefit')
    # ax3.fill_between(times, net_power_tracker, dc_power_fixed, where=(net_power_tracker < dc_power_fixed), 
    #                  color='red', alpha=0.2, label='Net Loss')
    # ax3.set_ylabel('Net Power (W)')
    
    # # --- TITLE UPDATE HERE ---
    # ax3.set_title(f'Net Power Comparison (Net Gain: {net_gain_pct:.2f}%)')
    
    # ax3.legend(loc='upper right')
    # ax3.grid(True, alpha=0.3)

    # # Plot 4: Components
    # ax4.plot(times, power_diff_watts, label='Instantaneous Gain (W)', color='black', linewidth=1)
    # ax4.axhline(0, color='gray', linewidth=0.8)
    # ax4.set_ylabel('Power Gain (W)', color='black')
    # ax4_right = ax4.twinx()
    # ax4_right.plot(times, cumulative_energy_gain_wh, label='Cumulative Energy Gain (Wh)', color='blue', linewidth=2)
    # ax4_right.set_ylabel('Cumulative Gain (Wh)', color='blue')
    # ax4_right.tick_params(axis='y', labelcolor='blue')
    # ax4.set_title('Net Benefit Components: Power Difference & Cumulative Energy')
    # ax4.set_xlabel('Time of Day')
    # lines, labels = ax4.get_legend_handles_labels()
    # lines2, labels2 = ax4_right.get_legend_handles_labels()
    # ax4.legend(lines + lines2, labels + labels2, loc='upper left')
    # ax4.grid(True, alpha=0.3)

    # plt.tight_layout()
    # output_file = 'floating_solar_net_components.png'
    # plt.savefig(output_file)
    # print(f"Simulation complete. Results saved to {output_file}")
    # print(f"Final Net Gain: {net_gain_pct:.2f}%")

    # fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(8.5, 3), sharex=True)

    # fig, ((ax3, ax4)) = plt.subplots(1, 2, figsize=(8.5, 3), sharex=True,dpi=1024)

    # fig, ((ax4)) = plt.subplots(1, 1, figsize=(8.5, 3), sharex=True,dpi=1024)

    # # Plot 1: Tracker vs Wind (Top Left)
    # ax1.plot(times, wind_direction, label='Wind Direction', color='blue', alpha=0.6)
    # ax1.plot(times, tracker_data['tracker_theta'], label='Tracker Angle', color='purple', linestyle='--')
    # ax1.set_ylabel('Degrees')
    # ax1.set_title('System Dynamics: Wind vs Tracker Response')
    # ax1.legend(loc='upper right')
    # ax1.grid(True, alpha=0.3)

    # # Plot 2: Gross Gen vs Motor (Top Right)
    # ax2.plot(times, dc_power_tracker, label='Tracker Gen (W)', color='green')
    # ax2.plot(times, dc_power_fixed, label='Fixed Gen (W)', color='orange', linestyle='-.')
    # ax2_right = ax2.twinx()
    # ax2_right.fill_between(times, energy_wh, color='red', alpha=0.3, label='Motor Cons (Wh)')
    # ax2_right.set_ylabel('Motor Energy (Wh)', color='red')
    # ax2_right.tick_params(axis='y', labelcolor='red')
    # ax2.set_ylabel('Power Generation (W)')
    # ax2.set_title('Gross Generation & Motor Consumption')
    # ax2.legend(loc='upper left')
    # ax2.grid(True, alpha=0.3)

    # # Plot 3: Net Power Comparison (Bottom Left)
    # ax3.plot(times, net_power_tracker, label='Net Tracker Power (W)', color='green')
    # ax3.plot(times, dc_power_fixed, label='Fixed Power (W)', color='orange', linestyle='-.')
    # ax3.fill_between(times, net_power_tracker, dc_power_fixed, where=(net_power_tracker > dc_power_fixed), 
    #                 color='green', alpha=0.2, label='Net Benefit')
    # ax3.fill_between(times, net_power_tracker, dc_power_fixed, where=(net_power_tracker < dc_power_fixed), 
    #                 color='red', alpha=0.2, label='Net Loss')
    # ax3.set_ylabel('Net Power (W)')
    # ax3.set_title(f'Average Net Power Gain: {net_gain_pct:.2f}%')
    # ax3.legend(loc='upper right')
    # ax3.grid(True, alpha=0.3)
    # ax3.set_xlabel('Time of Day') # Explicitly set x-label for bottom row
    # ax3.tick_params(axis='x', rotation=45)

    # # Plot 4: Components (Bottom Right)
    # ax4.plot(times, power_diff_watts, label='Instantaneous Gain (W)', color='black', linewidth=1)
    # ax4.axhline(0, color='gray', linewidth=0.8)
    # ax4.set_ylabel('Power Gain (W)', color='black')
    # ax4_right = ax4.twinx()
    # ax4_right.plot(times, cumulative_energy_gain_wh, label='Cumulative Energy Gain (Wh)', color='blue', linewidth=2)
    # ax4_right.set_ylabel('Cumulative Gain (Wh)', color='blue')
    # ax4_right.tick_params(axis='y', labelcolor='blue')
    # ax4.set_title('Power Difference & Cumulative Energy')
    # ax4.set_xlabel('Time of Day') # Explicitly set x-label for bottom row
    # ax4.tick_params(axis='x', rotation=45)

    # lines, labels = ax4.get_legend_handles_labels()
    # lines2, labels2 = ax4_right.get_legend_handles_labels()
    # ax4.legend(lines + lines2, labels + labels2, loc='upper left')
    # ax4.grid(True, alpha=0.3)

    # plt.tight_layout()
    # output_file = 'floating_solar_net_components.png'
    # plt.savefig(output_file)
    # print(f"Simulation complete. Results saved to {output_file}")
    # print(f"Final Net Gain: {net_gain_pct:.2f}%")

    
    import matplotlib.dates as mdates
    import seaborn as sns

    # Set a professional style
    sns.set_style("whitegrid")
    
    # Create one landscape figure with two stacked subplots (share x-axis)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 3), sharex=True, 
                                   gridspec_kw={'height_ratios': [1.5, 1]})
    
    # --- Data Prep for Cumulative Plot ---
    cumulative_fixed = dc_power_fixed.cumsum()
    cumulative_tracker = net_power_tracker.cumsum()
# -------------------------------------------------------------------------
    # Subplot 1: Instantaneous Power
    # -------------------------------------------------------------------------
    # Fixed Array (Baseline)
    ax1.plot(times, dc_power_fixed, color='grey', linestyle='--', linewidth=1, 
             label='Fixed (Base)', alpha=0.7)

    # Tracker (Net)
    ax1.plot(times, net_power_tracker, color="#3327d6", linewidth=1.5, 
             label='Tracker (Net)')

    # Shaded Gain Area
    ax1.fill_between(times, dc_power_fixed, net_power_tracker, 
                     where=(net_power_tracker > dc_power_fixed),
                     interpolate=True, color='#2ca02c', alpha=0.3, 
                     label='Gain')

    # Formatting Subplot 1
    # Note: Smaller font sizes (10pt is standard for papers/reports)
    ax1.set_ylabel('Power (W)', fontsize=9, fontweight='bold')
    ax1.set_title(f'Tracker Benefit: +{net_gain_pct:.1f}% Energy Gain', 
                  fontsize=11, fontweight='bold')
    
    # Compact Legend
    # ax1.legend(loc='upper right', fontsize=8, framealpha=0.9, 
            #    borderpad=0.3, handlelength=1.5)
    ax1.legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize=8)
    ax1.margins(x=0)

    # -------------------------------------------------------------------------
    # Subplot 2: Cumulative Energy Comparison
    # -------------------------------------------------------------------------
    # Fixed Cumulative (Baseline)
    ax2.plot(times, cumulative_fixed, color='grey', linestyle='--', linewidth=1, 
            label='Fixed Accum.', alpha=0.7)

    # Tracker Cumulative (Net)
    ax2.plot(times, cumulative_tracker, color='#3327d6', linewidth=1.5, 
            label='Tracker Accum.')

    # FILL THE GAP: This visualizes the "Bonus" Energy
    ax2.fill_between(times, cumulative_fixed, cumulative_tracker, 
                    color='#2ca02c', alpha=0.3, label='Net Gain')

    # Formatting Subplot 2
    ax2.set_ylabel('Energy (Wh)', fontsize=9, fontweight='bold')

    # # Annotation: Show the final delta explicitly
    # final_fixed = cumulative_fixed.iloc[-1]
    # final_tracker = cumulative_tracker.iloc[-1]
    # delta = final_tracker - final_fixed

    # # Place text at the end of the shaded area
    # ax2.text(times[-1], (final_fixed + final_tracker)/2, f" +{delta:.0f} Wh", 
    #         color='#2ca02c', fontweight='bold', fontsize=9, 
    #         ha='right', va='center')

    # --- ANNOTATION CHANGE ---
    final_fixed = cumulative_fixed.iloc[-1]
    final_tracker = cumulative_tracker.iloc[-1]
    delta = final_tracker - final_fixed
    midpoint = (final_fixed + final_tracker) / 2

    # 1. Use 'annotate' instead of 'text' for better control
    # 2. xytext=(5, 0) moves the text 5 "points" (pixels) to the right
    # 3. ha='left' aligns the start of the text to that point
    # 4. annotation_clip=False allows it to render even if it pokes outside the axis lines
    ax2.annotate(f"+{delta:.0f} Wh", 
                xy=(times[-1], midpoint), 
                xytext=(5, 0), textcoords="offset points",
                color='#2ca02c', fontweight='bold', fontsize=9, 
                ha='left', va='center',
                annotation_clip=False)

    # -------------------------------------------------------------------------
    # Global Formatting
    # -------------------------------------------------------------------------
    # Concise Date Format to save horizontal space
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
    ax2.xaxis.set_major_locator(mdates.DayLocator(interval=1))
    ax2.tick_params(axis='x', rotation=0, labelsize=9)
    ax2.tick_params(axis='y', labelsize=8)
    ax1.tick_params(axis='y', labelsize=8)
    
    # Save precisely
    plt.tight_layout(rect=[0, 0, 0.85, 1])
    plt.savefig('tracker_benefit_8x4.png', dpi=800)
    plt.show()


if __name__ == "__main__":
    simulate_floating_tracker()
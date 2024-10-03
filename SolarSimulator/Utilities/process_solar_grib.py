import pygrib
import pandas as pd
import numpy as np
import pytz
from scipy.stats import beta
from tqdm import tqdm
from datetime import datetime, timedelta


def extract_ssrd_data(grib_file_path):
    # Open the GRIB file
    grbs = pygrib.open(grib_file_path)
    
    data_dict = {}
    utc_tz = pytz.utc
    eastern_tz = pytz.timezone('US/Eastern')
    
    # Loop through all messages in the GRIB file
    for grb in tqdm(grbs.select()):
        # Extract only SSRD messages for January
        if grb.parameterName == 'Surface solar radiation downwards' and grb.validDate.month == 1:
            lats, lons = grb.latlons()
            values = grb.values
            
            # Extract the date and time from the GRIB message
            date = grb.validityDate    # YYYYMMDD format
            hour = grb.validityTime    # HHMM format, may need padding
            utc_date = pd.to_datetime(f'{date} {hour:04}', format='%Y%m%d %H%M')  # Convert to datetime
            utc_date = utc_date.tz_localize(utc_tz)
            valid_date = utc_date.astimezone(eastern_tz)

            day_month_hour = (valid_date.month, valid_date.day, valid_date.hour)

            # Store the SSRD data by (latitude, longitude) and day/month/hour in January (across years)
            for i in range(lats.shape[0]):
                for j in range(lons.shape[1]):
                    lat_lon_pair = (lats[i, j], lons[i, j])
                    
                    if lat_lon_pair not in data_dict:
                        data_dict[lat_lon_pair] = {}
                    
                    if day_month_hour not in data_dict[lat_lon_pair]:
                        data_dict[lat_lon_pair][day_month_hour] = []
                    
                    data_dict[lat_lon_pair][day_month_hour].append(values[i, j])  # Add SSRD value for this day/month/hour
    
    grbs.close()
    return data_dict

def normalize_ssrd_data(data_dict):
    normalized_data = {}
    
    for lat_lon_pair, datetime_dict in data_dict.items():
        max_ssrd = max([max(v) for v in datetime_dict.values() if len(v) > 0])  # Global max for this lat-lon pair
        
        for date_time, values in datetime_dict.items():
            if lat_lon_pair not in normalized_data:
                normalized_data[lat_lon_pair] = {}
            
            normalized_values = np.array(values) / max_ssrd  # Normalize the values
            normalized_data[lat_lon_pair][date_time] = normalized_values
    
    return normalized_data

def fit_beta_distributions(normalized_data, epsilon=1e-6):
    beta_params = {}
    
    for lat_lon_pair, datetime_dict in normalized_data.items():
        for date_time, values in datetime_dict.items():
            if len(values) > 0:  # Ensure there is data
                values = np.array(values)
                
                # Check if any of the values are zero, or if the range is too narrow (all values are identical)
                if np.any(values == 0) or np.any(values == 1) or np.ptp(values) == 0:
                    # Assign NaN to both alpha and beta if values are problematic
                    if lat_lon_pair not in beta_params:
                        beta_params[lat_lon_pair] = {}
                    beta_params[lat_lon_pair][date_time] = (np.nan, np.nan)
                else:
                    # Slightly adjust values that are too close to 0 or 1 to avoid boundary issues
                    clipped_values = np.clip(values, epsilon, 1 - epsilon)

                    # Fit the beta distribution using adjusted values
                    try:
                        a, b, loc, scale = beta.fit(clipped_values, floc=0, fscale=1)  # Fix loc and scale to [0,1]
                        if lat_lon_pair not in beta_params:
                            beta_params[lat_lon_pair] = {}
                        
                        beta_params[lat_lon_pair][date_time] = (a, b)  # Store the alpha and beta params
                    except Exception as e:
                        # In case fitting still fails, assign NaN
                        beta_params[lat_lon_pair][date_time] = (np.nan, np.nan)
                        print(f"Fitting failed for {lat_lon_pair} at {date_time}: {e}")

    beta_df = pd.DataFrame.from_dict(beta_params)
    return beta_df

# Main function to process the GRIB file, normalize, and fit distributions
def process_grib_file(grib_file_path):
    ssrd_data = extract_ssrd_data(grib_file_path)  # Step 1: Extract and segment
    normalized_ssrd_data = normalize_ssrd_data(ssrd_data)  # Step 2: Normalize data
    beta_params = fit_beta_distributions(normalized_ssrd_data)  # Step 3: Fit beta distributions
    return beta_params

# Example usage
grib_file_path = r'C:\Users\brian\OneDrive\Documents\Georgia Tech\Research\Whale Plane\SolarSim\Data\GRIB\January\solar_radiation.grib'
beta_params = process_grib_file(grib_file_path)


# Output: Dictionary with (lat, lon) as keys and datetime -> (alpha, beta) parameters as values
# for lat_lon, params in beta_params.items():
#     print(f"Lat-Lon: {lat_lon}")
#     for dt, (alpha, beta_val) in params.items():
#         print(f"  {dt}: Alpha = {alpha}, Beta = {beta_val}")

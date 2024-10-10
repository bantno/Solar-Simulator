import sys
import os
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime, timedelta
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../BaseClasses')))
from uncertainty_base import Uncertainty

# Load TMY Data
tmy_data = pd.read_csv(r"C:\Users\brian\OneDrive\Documents\Georgia Tech\Research\Whale Plane\SolarSim\Data\tmy_data")
tmy_data['datetime'] = pd.to_datetime(tmy_data[['Year', 'Month', 'Day', 'Hour', 'Minute']])
tmy_data.set_index('datetime', inplace=True)

uncertainties = {
    'ghi': 25,
    'dhi': 25,
    'dni': 25
}
summer_months = [6, 7, 8]
tmy_subset = tmy_data[tmy_data.index.month.isin(summer_months)]

start_date = datetime(tmy_data['Year'][0],tmy_data['Month'][0], tmy_data['Day'][0])
time_step = timedelta(minutes=60)

uncertainty_model = Uncertainty(tmy_data, uncertainties, start_date, time_step,0)

# Generate time series data for 30 days
simulation_data = uncertainty_model.generate_time_series(timedelta(days=30))

# Plot the first 7 days of data
uncertainty_model.plot_time_series(simulation_data, days=7)
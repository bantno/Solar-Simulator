import xarray as xr
import pandas as pd

# # Open the GRIB file
# ds = xr.open_dataset(r'C:\Users\brian\OneDrive\Documents\Georgia Tech\Research\Whale Plane\SolarSim\SolarSimulator\Utilities\JanuaryData.grib', engine='cfgrib')

# # Convert to DataFrame and then CSV
# df = ds.to_dataframe()
# df.to_csv('output.csv')

from cfgrib import xarray_store

datasets = xarray_store.open_datasets(r'C:\Users\brian\OneDrive\Documents\Georgia Tech\Research\Whale Plane\SolarSim\SolarSimulator\Utilities\JanuaryData.grib', engine='cfgrib')

i = 0
for d in datasets:
    df = d.to_dataframe()
    df.to_csv(f'output{i}.csv')
    print(i)
    i+=1
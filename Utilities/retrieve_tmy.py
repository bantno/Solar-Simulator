import pvlib.iotools
import pandas as pd

def get_tmy(lat,lon,filename):
    """Saves TMY weather data from the NSRDB to the input filename"""

    # Weather Data Info
    api_key = 'unURaXbAGeMjP8359wy5gyQfWIKq1g1y7hdhUmNo'
    email = 'bepstein8@gatech.edu'
    keys = ['ghi', 'dni', 'dhi', 'temp_air', 'wind_speed', 'wind_direction',
            'albedo']

    psm3, psm3_metadata = pvlib.iotools.get_psm3(lat, lon, api_key,
                                            email, interval=60, names='tmy',
                                            map_variables=True, leap_day=False,
                                            attributes=keys)
    psm3.to_csv(filename)
    return


# Define constant parameters
lat = 29.02291491363789
lon = -90.23223029442693
df = pd.read_csv(r"C:\Users\brian\OneDrive\Documents\Georgia Tech\Research\Whale Plane\Solar Sim\2019TMY.csv")
print(df['ghi'])

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

# df, metadata = pvlib.iotools.read_tmy3(r"C:\path\to\file.csv", map_variables=True)


# Define constant parameters
lat = 28.52291491363789
lon = -90.23223029442693
get_tmy(lat,lon,'test')

# df = pd.read_csv(r"C:\Users\brian\OneDrive\Documents\Georgia Tech\Research\Whale Plane\SolarSim\SolarSimulator\Utilities\test")
# print(df)
# print(df['ghi'])
# df, metadata = pvlib.iotools.read_tmy3(r"C:\Users\brian\OneDrive\Documents\Georgia Tech\Research\Whale Plane\SolarSim\SolarSimulator\Utilities\test", map_variables=True)
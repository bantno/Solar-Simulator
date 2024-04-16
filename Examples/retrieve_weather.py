import pvlib.iotools
import pvlib.pvsystem as pvsystem




latitude = 40.3864
longitude = -104.5512
api_key = 'unURaXbAGeMjP8359wy5gyQfWIKq1g1y7hdhUmNo'
email = 'bepstein8@gatech.edu'

keys = ['ghi', 'dni', 'dhi', 'temp_air', 'wind_speed', 'wind_direction',
        'albedo', 'precipitable_water']

# psm3, psm3_metadata = pvlib.iotools.get_psm3(latitude, longitude, api_key,
#                                              email, interval=, names=2019,
#                                              map_variables=True, leap_day=True,
#                                              attributes=keys)


# # data,metadata = pvlib.iotools.get_psm3(latitude, longitude, api_key, email, names='tmy', interval=60,
# #                         attributes=('air_temperature', 'dhi', 'dni', 'ghi', 'surface_albedo', 'surface_pressure', 'wind_direction', 'wind_speed'),
# #                         leap_day=False,
# #                         full_name='pvlib python',
# #                         affiliation='pvlib python',
# #                         map_variables=None,
# #                         url="https://developer.nrel.gov/api/nsrdb/v2/solar/psm3-tmy-download",
# #                         timeout=30)

# print(psm3)

modules = pvsystem.retrieve_sam('SandiaMod')
modules = modules.sort_values(by='Area',axis=1)
print(modules.iloc[:,10:12])
# print(modules)
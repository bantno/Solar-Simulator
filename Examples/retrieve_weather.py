import pvlib.iotools

latitude = 29.02291491363789
longitude = -90.23223029442693
api_key = 'unURaXbAGeMjP8359wy5gyQfWIKq1g1y7hdhUmNo'
email = 'bepstein8@gatech.edu'

keys = ['ghi', 'dni', 'dhi', 'temp_air', 'wind_speed', 'wind_direction',
        'albedo', 'precipitable_water']

psm3, psm3_metadata = pvlib.iotools.get_psm3(latitude, longitude, api_key,
                                             email, interval=60, names=2019,
                                             map_variables=True, leap_day=True,
                                             attributes=keys)

print(psm3)

# print(psm3)

# modules = pvsystem.retrieve_sam('SandiaMod')
# modules = modules.sort_values(by='Area',axis=1)
# print(modules.iloc[:,10:12])
# print(modules)
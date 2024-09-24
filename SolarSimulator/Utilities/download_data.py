import cdsapi

# # Initialize the CDS API client
# c = cdsapi.Client()

# # Define the data request parameters
# c.retrieve(
#     'reanalysis-era5-single-levels',  # The dataset
#     {
#         'product_type': 'reanalysis',
#         'variable': '2m_temperature',  # Example: temperature at 2m
#         'year': '2022',  # Specify the year
#         'month': '01',  # Specify the month (January)
#         'day': [
#             '01'
#         ],  # Specify the days
#         'time': [
#             '00:00', '01:00', '02:00', '03:00', '04:00', '05:00',
#             '06:00', '07:00', '08:00', '09:00', '10:00', '11:00',
#             '12:00', '13:00', '14:00', '15:00', '16:00', '17:00',
#             '18:00', '19:00', '20:00', '21:00', '22:00', '23:00'
#         ],  # Specify the hours of the day
#         'format': 'grib',  # Choose the format (netcdf or grib)
#     },
#     'era5_temperature_202201.grib'  # The file where the data will be saved
# )

import cdsapi
dataset = "reanalysis-era5-single-levels"
request = {
    'product_type': ['reanalysis'],
    'variable': ['10m_u_component_of_wind', '10m_v_component_of_wind'],
    'year': ['2023'],
    'month': ['08'],
    'day': ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21', '22', '23', '24', '25', '26', '27', '28', '29', '30', '31'],
    'time': ['00:00', '01:00', '02:00', '03:00', '04:00', '05:00', '06:00', '07:00', '08:00', '09:00', '10:00', '11:00', '12:00', '13:00', '14:00', '15:00', '16:00', '17:00', '18:00', '19:00', '20:00', '21:00', '22:00', '23:00'],
    'data_format': 'grib',
    'download_format': 'unarchived',
    'area': [44.59, -97.27, 17.72, -63.98]
}

client = cdsapi.Client()
client.retrieve(dataset, request).download()
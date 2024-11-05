from pvlib import pvsystem, location, modelchain, iotools
import pandas as pd
import matplotlib.pyplot as plt

# Create Location
loc = location.Location(30, 90)

# Create Array
array = pvsystem.Array(
    mount=pvsystem.FixedMount(surface_azimuth=90,surface_tilt=5),
    module_parameters=dict(pdc0=1, gamma_pdc=-0.004, b=0.05),
    temperature_model_parameters=dict(a=-3.56, b=-0.075, deltaT=3))

# Create PVSystem
system = pvsystem.PVSystem(arrays=[array], inverter_parameters=dict(pdc0=3))

# Create modelchain
mc = modelchain.ModelChain(system, loc, spectral_model='no_loss')

# Create Time series
times = pd.date_range('2019-01-01 00:00', '2019-01-01 22:00', freq='15min',
                      tz='Etc/GMT-6')

# Get weather information
weather = loc.get_clearsky(times)

# Run Model
mc.run_model(weather)

# Plot and show results
mc.results.dc.plot()
plt.ylabel('Output Power')
plt.show()

# API KEY 
#eDMo3KDgzlfYpCrx9bY2Y2LglMVxqRBrabyj8D5H

from pvlib import pvsystem, location, modelchain, iotools
import pandas as pd
import matplotlib.pyplot as plt

class DualAxisTrackerMount(pvsystem.AbstractMount):
    def get_orientation(self, solar_zenith, solar_azimuth):
        # no rotation limits, no backtracking
        return {'surface_tilt': solar_zenith, 'surface_azimuth': solar_azimuth}

loc = location.Location(30, 90)

array1 = pvsystem.Array(
    mount=pvsystem.SingleAxisTrackerMount(axis_azimuth=90),
    module_parameters=dict(pdc0=1, gamma_pdc=-0.004, b=0.05),
    temperature_model_parameters=dict(a=-3.56, b=-0.075, deltaT=3),
    surface_type='sea',
    module='klucher')

array2 = pvsystem.Array(
    mount=DualAxisTrackerMount(),
    module_parameters=dict(pdc0=1, gamma_pdc=-0.004, b=0.05),
    temperature_model_parameters=dict(a=-3.56, b=-0.075, deltaT=3))

array3 = pvsystem.Array(
    mount=pvsystem.FixedMount(surface_azimuth=90,surface_tilt=5),
    module_parameters=dict(pdc0=1, gamma_pdc=-0.004, b=0.05),
    temperature_model_parameters=dict(a=-3.56, b=-0.075, deltaT=3))

system1 = pvsystem.PVSystem(arrays=[array1], inverter_parameters=dict(pdc0=3))
system2 = pvsystem.PVSystem(arrays=[array2], inverter_parameters=dict(pdc0=3))
system3 = pvsystem.PVSystem(arrays=[array3], inverter_parameters=dict(pdc0=3))

mc1 = modelchain.ModelChain(system1, loc, spectral_model='no_loss')
mc2 = modelchain.ModelChain(system2, loc, spectral_model='no_loss')
mc3 = modelchain.ModelChain(system3, loc, spectral_model='no_loss')

times = pd.date_range('2019-06-01 00:00', '2019-06-01 22:00', freq='15min',
                      tz='Etc/GMT-6')

weather = loc.get_clearsky(times)

mc1.run_model(weather)
mc2.run_model(weather)
mc3.run_model(weather)

mc1.results.dc.plot()
mc2.results.dc.plot()
mc3.results.dc.plot()

plt.ylabel('Output DC Power [W]')
plt.legend(["Single Axis","Dual Axis","Fixed"])

plt.show()
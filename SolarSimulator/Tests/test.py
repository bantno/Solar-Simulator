import os
import sys
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))
from BaseClasses.expectedValue_base import ExpectedValueTable
from BaseClasses.seaplane_base import Seaplane

lat = 25
lon = -90
tz = "Etc/GMT-5"
pdc0 = 0
gamma = -0.0047
capacity_ah = 50.0
voltage = 22.2
Cdtot = 0.0
Cd0 = 0.02584
S = 0.653
af_mass = 8.8
cruise_speed = 20.0
rho = 1.19
N_PROP = 0.82
N_ESC = 0.9
n_tot = N_ESC * N_PROP
start_date = "2019-07-01"
end_date = "2019-08-02"
dt = 30
num_runs = 10000
visualize = False
save_dir = "."
show = False
use_expected = False

plane = Seaplane(
    lat=30,
    lon=-90,
    tz=0,
    pdc0=0,
    gamma=0,
    cd0=Cd0,
    cs=True,
    tracking=False,
    cdtot=Cdtot,
    n_tot=n_tot,
    S=S,
    af_mass=af_mass,
    voltage=voltage,
    capacity=capacity_ah,
)

expected_solar_data = np.zeros((int(25 / 0.5), 3))
expected_solar_data[:, 0] = 1
expected_solar_data[:, 1] = 1000

expected_wind_data = np.zeros((int(25 / 0.5), 3))
expected_wind_data[:, 0] = 5.0  # loc
expected_wind_data[:, 1] = 20.0  # scale

whale_observation_data = np.zeros((int(25 / 0.5), 1))
whale_observation_data[12:24] = 1.0
mdp_model = ExpectedValueTable(
    plane,
    expected_solar_data,
    expected_wind_data,
    whale_observation_data,
    soc_increment=1,
    timestep_min=dt,
)
mdp_model.generate_ev_table()

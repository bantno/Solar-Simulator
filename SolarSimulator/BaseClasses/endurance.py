from seaplane_base import Seaplane


lat = 30
lon = -90
tz = "Etc/GMT-0"
pdc0 = 0
gamma = -0.0047
Cd0 = 0.02584
cs = False
tracking=False
Cdtot = 0.0
n_tot = 0.738
S = 0.653
af_mass = 0
voltage = 44.4
capacity_ah = 5

plane = Seaplane(
            lat=lat, lon=lon, tz=tz, pdc0=pdc0, gamma=-0.0047,
            cd0=Cd0, cs=True, tracking=False, cdtot=Cdtot,
            n_tot=n_tot, S=S, af_mass=af_mass,
            voltage=voltage, capacity=capacity_ah
        )

capacities = range(0,105,5)
for cap in capacities:
    plane.capacity = cap
    plane.update_plane()
    # print(plane.get_required_power(20,1.2))
    # print(plane.get_endurance(20,1.2))
    # print(plane.required_cruise_power)
    print(plane.weight/9.81)
    print(f"Capacity: {cap}, Lift Coefficient: {plane.get_lift_coefficient(1.2,20)}, Endurance: {plane.get_endurance(20,1.2)}")

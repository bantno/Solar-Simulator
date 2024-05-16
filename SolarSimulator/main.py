import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from BaseClasses.Seaplane_base import Seaplane
import datetime
from Utilities import ParetoFront

def plot_endurance(plane,S,Cd0,af_mass,capacity,rho):
    # Create a figure and two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2,figsize=(10,5))
    ax1.set_title('Endurance vs Forward Flight Speed')
    ax1.set_xlabel('Forward Flight Speed [m/s]')
    ax1.set_ylabel('Endurance [H]')
    ax2.set_title('Required Power vs Forward Flight Speed')
    ax2.set_xlabel("Forward Flight Speed [m/s]")
    ax2.set_ylabel('Required Power [W]')

    # Get endurance and required power

    if isinstance(S,float):
        E = []
        P_req = []
        U = range(5,40)    
        for v in U:
            E.append(plane.get_endurance(v,rho))
            P_req.append(plane.get_required_power(U=v,rho=rho))
        label_1 = "S = {0}".format(plane.S)
        ax1.plot(U, E,label=label_1)
        # Plot Required Power
        ax2.plot(U, P_req,label=label_1)
    else:
        for i in range(0,len(S)):
            E = []
            P_req = []
            U = range(5,40)
            for v in U:
                plane.S = S[i]
                plane.cd0 = Cd0[i]
                plane.weight = weight[i]
                plane.capacity = capacity[i]
                E.append(plane.get_endurance(v,rho))
                P_req.append(plane.get_required_power(U=v,rho=rho))
            # Plot endurance
            label_1 = "S = {0}".format(plane.S)
            ax1.plot(U, E,label=label_1)
            # Plot Required Power
            ax2.plot(U, P_req,label=label_1)

    # Display the plots
    ax1.legend()
    ax2.legend()
    plt.show()
    return

def make_pareto(plane):
    # Define the bounds for each decision variable
    bounds = [(3, 25), (0, 1)]  # Example bounds for a 2D problem

    # Define the objective functions
    def objective_functions(x,plane):
        """
        Define your objective functions here.
        For example, for a two-objective problem:

        """
        days=31
        f1,num_takeoffs = battery_sweep(plane,x[0],days=days,month=6)
        # # TODO: Create an objective function that accounts for the riskiness of taking off
        f2 = num_takeoffs/days

        # f1 = np.sqrt(1+x[0]**2)
        # f2 = np.sqrt((1-x[0])**4)*1.5
        return [100-f1, f2*(1-x[1])]

    # Create an instance of ParetoFront
    pareto_front = ParetoFront.Pareto(objective_functions, bounds)

    # Number of samples for Latin Hypercube Sampling
    n_samples = 250

    # Calculate the Pareto front using LHS
    pf,non_dominated_points = pareto_front.generate_pareto_front(n_samples,plane)

    # Extract the objective values for plotting
    pf = np.array(pf)
    non_dominated_points = np.array(non_dominated_points)

    # Plot the Pareto front
    plt.scatter(pf[:, 0], pf[:, 1], marker='o', color='b', label='Pareto Front (LHS)')
    plt.scatter(non_dominated_points[:, 0], non_dominated_points[:, 1], marker='o', color='grey', label='Non-dominated Points')
    plt.xlabel('Percentage of Daylight Hours on Water [%]')
    plt.ylabel('Takeoffs Per Day')
    plt.title('Pareto Front using Latin Hypercube Sampling')
    plt.legend()
    plt.grid(True)
    plt.show()

def battery_sweep(plane: Seaplane,capacities,year=2019,month=6,day=1,days=1):
    duty_list = []
    if isinstance(capacities,float):
        plane.capacity = capacities
        plane.calculate_weight()
        _, P_solar = plane.calc_collected_energy((year,year),(month,month),(day,day),periods=12*24*days,frequency='5min')
        duty_cycle,_,_,num_takeoffs = plane.simulate_deployment(U,rho,1,.05,P_solar,5)
        return duty_cycle,num_takeoffs
    else:
        for cap in capacities:
            plane.capacity = cap
            plane.calculate_weight()
            _, P_solar = plane.calc_collected_energy((year,year),(month,month),(day,day),periods=12*24*days,frequency='5min')
            duty_cycle,_,_,num_takeoffs = plane.simulate_deployment(U,rho,1,.05,P_solar,5)
            duty_list.append(duty_cycle)

    return duty_list,num_takeoffs

def plot_battery_sweep(cap,duty,filename=-1):
    plt.plot(cap,duty)
    plt.xlabel("Battery Capacity [Ah]")
    plt.ylabel("Duty Cycle [%]")
    plt.title("Battery Capacity Sweep")
    plt.tight_layout()
    if not filename==-1:
        plt.savefig(filename)
    else:
        plt.show()
    

def plot_simulation(plane: Seaplane, year=2019,month=6,day=1,days=1,filename=-1):
    # TODO: Add clearsky vs TMY distinction
    """Simulates and plots the duty cycle for the given plane and times
    
    Parameters:
    -----------
    plane: Seaplane
        Seaplane object which is to be simulated
    year: int
        Year in which simulation is to start
    month: int
        Month in which simulation is to start
    day: int
        Day in which simulation is to start
    days: int
        Number of days to simulate
    """
    times, P_solar = plane.calc_collected_energy((year,year),(month,month),(day,day),periods=12*24*days,frequency='5min')
    duty_cycle,e_h,state,_ = plane.simulate_deployment(U,rho,1,.10,P_solar,5)
    
    # Plot Results
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1,figsize=(10,5))
    ax1.set_title('Battery Charge Level')
    ax1.set_xlabel('Dates')
    ax1.set_ylabel('Battery Charge [%]')
    ax2.set_title('Available Solar Power')
    ax2.set_xlabel('Dates')
    ax2.set_ylabel('Power [W]')
    ax3.set_title('Vehicle State')
    ax3.set_xlabel('Dates')
    ax3.set_ylabel('State')

    ax1.plot(times,e_h)
    ax2.plot(times,P_solar)
    ax3.plot(times,state)

    plt.tight_layout() 
    if not filename==-1:
        plt.savefig(filename)
    else:
        plt.show()
    return duty_cycle





########################################################


# Define constant parameters
lat = 29.02291491363789
lon = -90.23223029442693
# lat = 44.3655
# lon = -68.0818
tz = 'Etc/GMT+6'
pdc0 = 80
gamma = -0.0043

# # Airplane params
# capacity_ah = 10.6
# voltage = 22.2
# Cdtot = 0.02616
# Cd0 = 0.01487
# S = 0.653
# af_mass = 6
# cruise_speed = 20
# rho = 1.1 # air density (dependent on altitude)
# U = cruise_speed
# plane = Seaplane(lat, lon, tz, pdc0,gamma,cd0=Cd0,cs=True ,tracking=False,cdtot = Cdtot,n_tot=.75,S=S,af_mass=af_mass,voltage=voltage,capacity=capacity_ah)


# # List Parameters for different wing area values
# S =      [0.65340, 0.79]
# Cd0 =    [0.01487, 0.015]
# Cdtot =  [0.02572, 0.03]
# af_mass = [af_mass, 149.169/9.81]
# capacity = [capacity_ah, 28.8]

# plot_endurance(plane,S,Cd0,weight,capacity,rho) # TODO: Fix weight estimation

############################################

# Airplane params
capacity_ah = 10.6
voltage = 22.2
Cdtot = 0.02616
Cd0 = 0.01487
S = 0.653
af_mass = 6 #TODO: Read in AF mass from VSPAero, multiply by safety factor
cruise_speed = 20
rho = 1.1 # air density (dependent on altitude)
U = cruise_speed

plane = Seaplane(lat, lon, tz, pdc0,gamma,cd0=Cd0*1.5,cs=True ,tracking=False,cdtot = Cdtot,n_tot=.75,S=S,af_mass=af_mass,voltage=voltage,capacity=capacity_ah)

make_pareto(plane)

'''
year = 2019
month = 1
day = 1
days = 1

cap = np.linspace(1,25,50)
duty,num_takeoffs = battery_sweep(plane,cap,month=month,days=days)
filename = "BatterySweep_1-25.png"
plot_battery_sweep(cap,duty,filename)

data = data = {'Capacity': cap, 'Duty': duty}
df = pd.DataFrame(data)
max_duty = df['Duty'].max()
max_duty_row = df[df['Duty'] == max_duty]
max_cap= max_duty_row['Capacity'].values[0]
print(max_cap,max_duty)

# Run simulation for optimal battery size
plane.capacity = max_cap
# year = 2019
# month = 7
# day = 1
days = 7
filename = "MaxSim.png"
dc = plot_simulation(plane,year,month,day,days,filename)
print(dc)


# # YEARLY DUTY CYCLE ##########################################
# duty = []
# monthly_duty = []
# daily_duty = []
# days = 1
# year = 2019
# for j in range(1,13):
#     if j==12:
#         days_in_month=31
#     else:
#         days_in_month = (datetime.date(year, j % 12 + 1, 1) - datetime.date(year, j, 1)).days
#     for i in range(1, days_in_month+1):
#         times, P_solar = plane.calc_collected_energy((year,year),(j,j),(i,i+1),periods=12*24*days,frequency='5min')
#         duty_cycle,e_h,state,_ = plane.simulate_deployment(U,rho,.100,.15,P_solar,5)
#         duty.append(duty_cycle)
#         daily_duty.append(duty_cycle)
#     monthly_duty.append(np.mean(duty))
#     duty = []

# print(np.mean(daily_duty))

# plt.clf()
# plt.plot(daily_duty)
# plt.title('Daily Duty Cycle')
# plt.xlabel("Day of Year")
# plt.ylabel("Duty Cycle")
# plt.savefig("YearlyDutyCycle.png")
# # plt.show()

# ########################################
'''





# TODO: Make plotting function for this plot
# # Battery Power Required Plot
# plane_cs = Seaplane(lat, lon, tz, pdc0,gamma,cd0=Cd0,cs=True ,tracking=False,cdtot = Cdtot,n_tot=.75,S=S,weight=weight,voltage=voltage,capacity=capacity_ah)
# times_cs, P_solar_cs = plane_cs.calc_collected_energy((2019,2019),(1,1),(1,3))

# plane_w  = Seaplane(lat, lon, tz, pdc0,gamma,cd0=Cd0,cs=False,tracking=False,cdtot = Cdtot,n_tot=.75,S=S,weight=weight,voltage=voltage,capacity=capacity_ah)
# times_w, P_solar_w = plane_w.calc_collected_energy((2019,2019),(1,1),(1,3))

# U = cruise_speed
# P_req_cruise = plane_cs.get_required_power(U,rho)
# P_bat_cs = P_req_cruise-P_solar_cs
# P_bat_w = P_req_cruise-P_solar_w

# plt.figure(figsize=(10, 6))
# plt.plot(times_cs,P_bat_cs,label="Clear Sky")
# plt.plot(times_w,P_bat_w,label="TMY")
# plt.xlabel("Date")
# plt.ylabel("Required Battery Power")
# plt.title("Required Battery Power vs Date")
# plt.legend()
# plt.show()



# Endurance and P_req calculations
# plane = Seaplane(lat, lon, tz, pdc0,gamma,cd0=0.0145,cs=True,tracking=False,cdtot = 0.025,n_tot=.75,S=0.38,weight=4*9.81,voltage=22.2,capacity=28.82)

# List Parameters for different wing area values
# S =      [0.65340, 0.79]
# Cd0 =    [0.01487, 0.015]
# Cdtot =  [0.02572, 0.03]
# weight = [weight, 149.169]

# plot_endurance(plane,S,Cd0,weight,rho)



# TODO: Make endurance contour plot
# lat = [26.0857 , 29.1615 , 32.9148 , 35.3777 , 39.7020 , 42.07658 , 44.66028]
# lon = [-80.0695, -80.8963, -79.7388, -75.4398, -74.0343, -69.81796, -66.9243]
# plane = Seaplane(lat, lon, tz, pdc0,gamma,cd0=0.0145,cs=False,tracking=False,cdtot = 0.025,n_tot=.75,S=0.38,weight=4*9.81,voltage=15.2,capacity=5.3)
# rho = 1.225
# U = 15
# E = np.empty((len(lat),365))

# pdc,power_kWh = plane.calc_tmy_energy(2019)

# days_of_year = np.linspace(1, 365, 365)
# latitude = lat
# endurance = E  

# # Create a meshgrid from the data
# X, Y = np.meshgrid(days_of_year, latitude)

# # Plot the contour
# plt.figure(figsize=(10, 6))
# contour = plt.contourf(X, Y, endurance, cmap='viridis')
# plt.colorbar(contour, label='Endurance')

# # Add labels and title
# plt.xlabel('Days of the Year')
# plt.ylabel('Latitude')
# plt.title('Endurance Contour Plot')

# # Show plot
# plt.show()

# lat = 29.02291491363789
# lon = -90.23223029442693




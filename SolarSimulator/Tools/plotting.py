import os
from datetime import datetime, timedelta

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from tqdm import tqdm

from paretoset import paretoset

from BaseClasses.simulation_base import Simulation
from BaseClasses.seaplane_base import Seaplane
from Utilities import ParetoFront

def day_to_month_day(day_number, year):
    day_number = int(day_number)
    # Create a datetime object for the given year and day number
    date_obj = datetime(year, 1, 1) + timedelta(day_number - 1)
    
    # Extract month and day from the datetime object as numbers
    month = date_obj.month
    day = date_obj.day
    
    return month, day

def plot_solar(sim: Simulation,year:int,month:int,day:int,days:int,filename=""):
    periods = 12*24*days
    freq = '5min'
    times, P_solar = sim.calc_collected_energy((year,year),(month,month),(day,day),periods=periods,frequency=freq)
    start = f"{year}-{month}-{day}"
    times = pd.date_range(start, periods=periods, freq=freq, tz=sim.tz)
    wthr = sim.get_weather(sim.cs,times)
    ghi = wthr['ghi']*sim.plane.S # need to multiply by S

    plot_path = os.path.join("Figures", f"{filename}.png")
    
    fig, ax = plt.subplots(figsize=(10,4))
    ax.plot(times,P_solar,label="Collected solar power")
    ax.plot(times,ghi, label = "Available solar power")

    # plt.title("Solar Energy")
    plt.xlabel("Time")
    plt.ylabel("Power [W]")
    ax.legend(loc='best')
    plt.tight_layout()
    plt.grid(True)
    
    
    if not filename==-1:
        plt.savefig(plot_path)
    else:
        plt.show()
    
    return fig


def plot_endurance(plane:Seaplane,S,Cd0,af_mass,capacity,rho,filename=-1):
    # Create a figure and two subplots
    _, (ax1, ax2) = plt.subplots(1, 2,figsize=(18,5))
    # ax1.set_title('Endurance vs Forward Flight Speed')
    ax1.set_xlabel('Forward Flight Speed [meters/second]')
    ax1.set_ylabel('Endurance [Hours]')
    # ax2.set_title('Required Power vs Forward Flight Speed')
    ax2.set_xlabel("Forward Flight Speed [meters/second]")
    ax2.set_ylabel('Required Power [Watts]')

    # Get endurance and required power

    if isinstance(S,float):
        E = []
        P_req = []
        U = range(10,30)    
        for v in U:
            E.append(plane.get_endurance(v,rho))
            P_req.append(plane.get_required_power(U=v,rho=rho))
        label_1 = f"S = {plane.S}"
        ax1.plot(U, E,label=label_1)
        ax2.plot(U, P_req,label=label_1)
    else:
        for i,s in enumerate(S):
            E = []
            P_req = []
            U = range(5,30)
            for v in U:
                plane.S = s
                plane.cd0 = Cd0[i]
                plane.af_mass = af_mass[i]
                plane.capacity = capacity[i]
                plane.update_plane()
                # plane.weight = af_mass[i]*9.81
                E.append(plane.get_endurance(v,rho))
                P_req.append(plane.get_required_power(U=v,rho=rho))
            # Plot endurance
            label_1 = f"S = {plane.S}"
            ax1.plot(U, E,label=label_1)
            # Plot Required Power
            ax2.plot(U, P_req,label=label_1)

    # Display the plots
    ax1.legend()
    ax2.legend()
    ax1.grid(True)
    ax2.grid(True)
    if not filename==-1:
        plot_path = os.path.join("Figures", f"{filename}.png")
        plt.savefig(plot_path)
    else:
        plt.show()
    return

def battery_sweep(sim: Simulation,capacities,year=2019,month=6,day=1,days=1,U=20,rho=1.1,algo="Greedy"):
    plane = sim.plane
    duty_list = []
    if isinstance(capacities,float):
        plane.capacity = capacities
        plane.update_plane()
        _, P_solar = sim.calc_collected_energy((year,year),(month,month),(day,day),periods=12*24*days,frequency='5min',cs=sim.cs)
        duty_cycle,_,_,num_takeoffs = sim.simulate_deployment(U,rho,1,.1,P_solar,10,algo)
        return duty_cycle,num_takeoffs
    else:
        for cap in tqdm(capacities):
            plane.capacity = cap
            plane.update_plane()
            _, P_solar = sim.calc_collected_energy((year,year),(month,month),(day,day),periods=12*24*days,frequency='5min',cs=sim.cs)
            duty_cycle,_,_,num_takeoffs = sim.simulate_deployment(U,rho,1,.1,P_solar,10,algo)
            duty_list.append(duty_cycle)

    return duty_list,num_takeoffs

def plot_battery_sweep(cap,duty,label = "",filename=-1,fig = -1,title=""):
    plot_path = os.path.join("Figures", f"{filename}.png")
    DOT_SIZE = 20
    if fig == -1:
        fig, ax = plt.subplots(figsize=(8,4))
        ax.scatter(cap,duty,label=label,s=DOT_SIZE)
    elif isinstance(fig,Figure):
        fig.axes[0].scatter(cap,duty,label=label,s=DOT_SIZE)
    
    plt.xlabel("Battery Capacity [Ah]")
    plt.ylabel("Duty Cycle [%]")
    plt.tight_layout()
    plt.grid(True)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    if not filename==-1:
        plt.savefig(plot_path)
    else:
        plt.show()
    
    return fig
    

def run_simulation(sim: Simulation,
                   solar_file,
                   U=20,
                   rho=1.1,
                   algo="MDP"):
    
    """
    Simulates and plots the duty cycle for the given plane, solar data file, cruise speed, air density, and algorithm.
    
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
    plane = sim.plane
    plane.update_plane()
    # times, P_solar = sim.calc_collected_energy((year,year),(month,month),(day,day),periods=6*24*days,frequency='10min',cs=sim.cs)

    start_index = (1, 2, 0)
    end_index = (1, 3, 23)

    # Extract the slice of the DataFrame
    P_solar_actual = pd.read_pickle(solar_file)[(29.25,  -85.0)].loc[start_index:end_index]
    P_solar_expected = pd.read_pickle(r"Data\DISTRIBUTIONS\solar_ev.pkl")[(29.25,  -85.0)].loc[start_index:end_index]
    state_history,solar_power = sim.simulate_deployment(U,
                                                       rho,
                                                       1,
                                                       .05,
                                                       avail_solar_w=P_solar_actual,
                                                       expected_solar_w=P_solar_expected,
                                                       dt=60,
                                                       algo=algo)
    
    times = generate_datetimes(start_index, end_index, timestep=60)
    
    return times, state_history, solar_power

    
def plot_simulation(times,P_solar,state_history,filename=-1, fig = -1, label=""):
    """Plot results of simulation"""
    num_plots = 2
    if fig == -1:
        fig, axes = plt.subplots(num_plots, 1,figsize=(12,6))
    if isinstance(fig,Figure):
        axes = fig.axes
    titles = ["Battery Charge Level", "Collected Solar Power", "Vehicle State"]
    xlabel = "Dates"
    ylabels = ["Battery Charge [%]", "Power [W]", "State"]
    soc = [s[0] for s in state_history]
    data = [soc,P_solar]
    for i in range(np.min([len(axes),num_plots])):
        plot_data(axes[i],times,data[i],titles[i],xlabel,ylabels[i],label)

    plt.tight_layout() 

    if not filename==-1:
        plot_path = os.path.join("Figures", f"{filename}.png")
        plt.savefig(plot_path)
        
    else:
        plt.show()

    return fig
    
def plot_data(ax,x_data,y_data, title:str="", xlabel:str="X Data", ylabel:str="Y Data",label:str=""):
    ax.plot(x_data,y_data,label=label)
    # ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if not label == "":
        ax.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
    ax.grid(True)

def make_pareto(plane: Seaplane,filename:str = "Pareto"):
    # Define the bounds for each decision variable
    bounds = [(3, 25), (1, 1)]  # Example bounds for a 2D problem

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
        return [f1, f2]

    # Create an instance of ParetoFront
    pareto_front = ParetoFront.Pareto(objective_functions, bounds)

    # Number of samples for Latin Hypercube Sampling
    n_samples = 250

    # Calculate the Pareto front using LHS
    samples,pf,non_dominated_points = pareto_front.generate_pareto_front(n_samples,plane)

    # Extract the objective values for plotting
    pf = np.array(pf)
    non_dominated_points = np.array(non_dominated_points)

    # Plot the Pareto front
    # fig, (ax1,ax2) = plt.subplots(1, 2,figsize=(12,6))
    # ax1.scatter(samples.iloc[:,0],samples.iloc[:,1])
    # ax1.set_xlabel("Battery Capacity [Ah]")
    # ax1.set_title("Samples")
    
    # ax2.scatter(non_dominated_points[:, 0], non_dominated_points[:, 1], marker='o', color='grey', label='Non-dominated Points')
    # ax2.scatter(pf[:, 0], pf[:, 1], marker='o', color='b', label='Pareto Front (LHS)')
    # ax2.set_xlabel('Percentage of Daylight Hours on Water [%]')
    # ax2.set_ylabel('Takeoffs Per Day')
    # ax2.set_title('Pareto Front using Latin Hypercube Sampling')

    plt.scatter(non_dominated_points[:, 0], non_dominated_points[:, 1], marker='o', color='grey', label='Non-dominated Points')
    plt.scatter(pf[:, 0], pf[:, 1], marker='o', color='b', label='Pareto Front (LHS)')
    plt.xlabel('Percentage of Daylight Hours on Water [%]')
    plt.ylabel('Takeoffs Per Day')
    plt.title('Pareto Front using Latin Hypercube Sampling')\


        

    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plot_path = os.path.join("Figures", f"{filename}.png")
    plt.savefig(plot_path)
    plt.close()

def make_pareto_classic(plane: Seaplane,bounds,n_samples: int,filename: str = "ParetoFront"):
    n_w = 5
    values = np.zeros((n_w, n_samples))
    samples = np.random.uniform(low=bounds[0], high=bounds[1], size=n_samples)
    i = 0
    
    for  w in tqdm(np.linspace(0,1,n_w)):
        j = 0
        for sample in tqdm(samples):
            f1,f2 = func(sample,plane)
            F = f1*w+f2*(1-w)
            values[i,j] = F
            j+=1
        plt.scatter(samples,values[i,:], label = f'W = {w}')
        i+=1

        
    
    # Add a legend
    plt.legend()

    # Add labels and title
    plt.xlabel('Battery Capacity')
    plt.ylabel('Objective Function Value (F)')
    plt.title('F = DutyCycle*w+NumTakeoff*(1-w)')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.grid(True)

    # Show the plot
    plot_path = os.path.join("Figures", f"{filename}.png")
    plt.savefig(plot_path)

    return values

def func(x,plane):
    """
    Define objective functions here.

    """
    days=31
    f1,num_takeoffs = battery_sweep(plane,x,days=days,month=6)
    # # TODO: Create an objective function that accounts for the riskiness of taking off
    f2 = num_takeoffs/days

    # f1 = np.sqrt(1+x[0]**2)
    # f2 = np.sqrt((1-x[0])**4)*1.5
    return f1, f2

def plot_yearly_dc(plane: Seaplane,
                   year: int,
                   month: int,
                   day: int,
                   days : int,
                   ):
    label = f"{plane.capacity} Ah"
    times,e_h,P_solar,states,dc = run_simulation(plane,year,month,day,days)
    df = pd.DataFrame(e_h,index=times,columns=['battery'])
    df['P_solar'] = P_solar
    df['states'] = states
    df['duty cycle'] = dc

    df_daylight_hours = df.between_time('08:00', '18:00')
    daily_avg = df_daylight_hours['states'].resample('D').sum()/120.0*100

    print("Plane Capacity: {0}, Average Duty Cycle: {1}".format(plane.capacity,np.mean(dc)))

    plt.clf()
    plt.plot(range(0,days),daily_avg)
    plt.scatter(range(0,days),daily_avg,s=7)
    plt.title('Daily Duty Cycle')
    plt.xlabel("Day of Year")
    plt.ylabel("Duty Cycle [%]")
    plt.tight_layout()
    filename = "YearSweep"
    plot_path = os.path.join("Figures", f"{filename}.png")
    plt.savefig(plot_path)

def generate_datetimes(start_index, end_index, timestep):
    """
    Generates a list of datetime objects between start_index and end_index.
    
    :param start_index: Tuple representing (month, day, hour)
    :param end_index: Tuple representing (month, day, hour)
    :param timestep: Time difference between consecutive datetimes (in minutes)
    :return: List of datetime objects
    """
    # Unpack the tuples (month, day, hour)
    start_month, start_day, start_hour = start_index
    end_month, end_day, end_hour = end_index
    
    # Create the starting and ending datetime objects
    start_datetime = datetime(year=2024, month=start_month, day=start_day, hour=start_hour)
    end_datetime = datetime(year=2024, month=end_month, day=end_day, hour=end_hour)

    # List to store the generated datetime objects
    datetime_list = []
    
    # Use a timedelta of `timestep` minutes to increment between start and end
    current_datetime = start_datetime
    while current_datetime <= end_datetime:
        datetime_list.append(current_datetime)
        current_datetime += timedelta(minutes=timestep)

    return datetime_list

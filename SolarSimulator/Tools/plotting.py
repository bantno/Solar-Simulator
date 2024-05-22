import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from BaseClasses.Seaplane_base import Seaplane
import datetime
from Utilities import ParetoFront
import os

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

def battery_sweep(plane: Seaplane,capacities,year=2019,month=6,day=1,days=1,U=20,rho=1.1):
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
    plot_path = os.path.join("Figures", f"{filename}.png")
    plt.plot(cap,duty)
    plt.xlabel("Battery Capacity [Ah]")
    plt.ylabel("Duty Cycle [%]")
    plt.title("Battery Capacity Sweep")
    plt.tight_layout()
    if not filename==-1:
        plt.savefig(plot_path)
    else:
        plt.show()
    

def plot_simulation(plane: Seaplane, year=2019,month=6,day=1,days=1,filename=-1,U=20,rho=1.1):
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
        plot_path = os.path.join("Figures", f"{filename}.png")
        plt.savefig(plot_path)
    else:
        plt.show()
    return duty_cycle
import os
from datetime import datetime, timedelta

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from tqdm import tqdm

from BaseClasses.simulation_base import Simulation
from BaseClasses.seaplane_base import Seaplane


def day_to_month_day(day_number, year):
    day_number = int(day_number)
    # Create a datetime object for the given year and day number
    date_obj = datetime(year, 1, 1) + timedelta(day_number - 1)

    # Extract month and day from the datetime object as numbers
    month = date_obj.month
    day = date_obj.day

    return month, day


def plot_solar(sim: Simulation, year: int, month: int, day: int, days: int, filename=""):
    periods = 12 * 24 * days
    freq = "5min"
    times, P_solar = sim.calc_collected_energy(
        (year, year), (month, month), (day, day), periods=periods, frequency=freq
    )
    start = f"{year}-{month}-{day}"
    times = pd.date_range(start, periods=periods, freq=freq, tz=sim.tz)
    wthr = sim.get_weather(sim.cs, times)
    ghi = wthr["ghi"] * sim.plane.S  # need to multiply by S

    plot_path = os.path.join("Figures", f"{filename}.png")

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(times, P_solar, label="Collected solar power")
    ax.plot(times, ghi, label="Available solar power")

    # plt.title("Solar Energy")
    plt.xlabel("Time")
    plt.ylabel("Power [W]")
    ax.legend(loc="best")
    plt.tight_layout()
    plt.grid(True)

    if not filename == -1:
        plt.savefig(plot_path)
    else:
        plt.show()

    return fig


def plot_endurance(plane: Seaplane, S, Cd0, af_mass, capacity, rho, filename=-1):
    # Create a figure and two subplots
    _, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 5))
    # ax1.set_title('Endurance vs Forward Flight Speed')
    ax1.set_xlabel("Forward Flight Speed [meters/second]")
    ax1.set_ylabel("Endurance [Hours]")
    # ax2.set_title('Required Power vs Forward Flight Speed')
    ax2.set_xlabel("Forward Flight Speed [meters/second]")
    ax2.set_ylabel("Required Power [Watts]")

    # Get endurance and required power

    if isinstance(S, float):
        E = []
        P_req = []
        U = range(10, 30)
        for v in U:
            E.append(plane.get_endurance(v, rho))
            P_req.append(plane.get_required_power(U=v, rho=rho))
        label_1 = f"S = {plane.S}"
        ax1.plot(U, E, label=label_1)
        ax2.plot(U, P_req, label=label_1)
    else:
        for i, s in enumerate(S):
            E = []
            P_req = []
            U = range(5, 30)
            for v in U:
                plane.S = s
                plane.cd0 = Cd0[i]
                plane.af_mass = af_mass[i]
                plane.capacity = capacity[i]
                plane.update_plane()
                # plane.weight = af_mass[i]*9.81
                E.append(plane.get_endurance(v, rho))
                P_req.append(plane.get_required_power(U=v, rho=rho))
            # Plot endurance
            label_1 = f"S = {plane.S}"
            ax1.plot(U, E, label=label_1)
            # Plot Required Power
            ax2.plot(U, P_req, label=label_1)

    # Display the plots
    ax1.legend()
    ax2.legend()
    ax1.grid(True)
    ax2.grid(True)
    if not filename == -1:
        plot_path = os.path.join("Figures", f"{filename}.png")
        plt.savefig(plot_path)
    else:
        plt.show()
    return


def battery_sweep(
    sim: Simulation, capacities, year=2019, month=6, day=1, days=1, U=20, rho=1.1, algo="Greedy"
):
    plane = sim.plane
    duty_list = []
    if isinstance(capacities, float):
        plane.capacity = capacities
        plane.update_plane()
        _, P_solar = sim.calc_collected_energy(
            (year, year),
            (month, month),
            (day, day),
            periods=12 * 24 * days,
            frequency="5min",
            cs=sim.cs,
        )
        duty_cycle, _, _, num_takeoffs = sim.simulate_deployment(U, rho, 1, 0.1, P_solar, 10, algo)
        return duty_cycle, num_takeoffs
    else:
        for cap in tqdm(capacities):
            plane.capacity = cap
            plane.update_plane()
            _, P_solar = sim.calc_collected_energy(
                (year, year),
                (month, month),
                (day, day),
                periods=12 * 24 * days,
                frequency="5min",
                cs=sim.cs,
            )
            duty_cycle, _, _, num_takeoffs = sim.simulate_deployment(
                U, rho, 1, 0.1, P_solar, 10, algo
            )
            duty_list.append(duty_cycle)

    return duty_list, num_takeoffs


def plot_battery_sweep(cap, duty, label="", filename=-1, fig=-1, title=""):
    plot_path = os.path.join("Figures", f"{filename}.png")
    DOT_SIZE = 20
    if fig == -1:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.scatter(cap, duty, label=label, s=DOT_SIZE)
    elif isinstance(fig, Figure):
        fig.axes[0].scatter(cap, duty, label=label, s=DOT_SIZE)

    plt.xlabel("Battery Capacity [Ah]")
    plt.ylabel("Duty Cycle [%]")
    plt.tight_layout()
    plt.grid(True)
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")

    if not filename == -1:
        plt.savefig(plot_path)
    else:
        plt.show()

    return fig


def plot_simulation(times, state_history, P_solar, filename=-1, fig=-1, label=""):
    """Plot results of simulation"""
    num_plots = 2
    if fig == -1:
        fig, axes = plt.subplots(num_plots, 1, figsize=(12, 6))
    if isinstance(fig, Figure):
        axes = fig.axes
    titles = ["Battery Charge Level", "Solar Power", "Vehicle State"]
    xlabel = "Dates"
    ylabels = ["Battery Charge [%]", "Power [W]", "State"]
    soc = [s[0] for s in state_history]
    solar_power = P_solar
    data = [soc, solar_power]
    for i in range(np.min([len(axes), num_plots])):
        plot_data(axes[i], times[0 : len(data[i])], data[i], titles[i], xlabel, ylabels[i], label)

    plt.tight_layout()

    if not filename == -1:
        plot_path = os.path.join("Figures", f"{filename}.png")
        plt.savefig(plot_path)

    else:
        plt.show()

    return fig


def plot_simulation_results(
    times, state_history, expected_solar, actual_solar, filename=-1, fig=-1, label=""
):
    """Plot results of simulation"""
    num_plots = 2
    if fig == -1:
        fig, axes = plt.subplots(num_plots, 1, figsize=(12, 6))
    if isinstance(fig, Figure):
        axes = fig.axes
    titles = ["Battery Charge Level", "Solar Power"]
    xlabel = "Dates"
    ylabels = ["Battery Charge [%]", "Power [W/m\u00B2]"]

    # Ensure all data series are aligned by trimming to the shortest length
    min_length = min(len(times), len(state_history), len(expected_solar), len(actual_solar))
    times = times[:min_length]
    soc = [s[0] for s in state_history[:min_length]]  # Correct indexing to avoid extra dimension
    expected_solar = expected_solar[:min_length]
    actual_solar = actual_solar[:min_length]

    # Debugging output to check data lengths
    print(
        f"Data lengths after trimming: times={len(times)}, soc={len(soc)}, expected_solar={len(expected_solar)}, actual_solar={len(actual_solar)}"
    )

    # Plot state of charge
    plot_data(axes[0], times[: len(soc)], soc, titles[0], xlabel, ylabels[0], label)
    plot_data(
        axes[1],
        times[: len(actual_solar)],
        expected_solar,
        titles[1],
        xlabel,
        ylabels[1],
        label="Expected Solar Power",
    )
    plot_data(
        axes[1],
        times[: len(actual_solar)],
        actual_solar,
        titles[1],
        xlabel,
        ylabels[1],
        label="Actual Solar Power",
    )

    plt.tight_layout()

    if filename != -1:
        plot_path = os.path.join("Figures", f"{filename}.png")
        plt.savefig(plot_path)
    else:
        plt.show()

    return fig


def plot_data(
    ax,
    x_data,
    y_data,
    title: str = "",
    xlabel: str = "X Data",
    ylabel: str = "Y Data",
    label: str = "",
):
    ax.plot(x_data, y_data, label=label)
    # ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if not label == "":
        ax.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
    ax.grid(True)


def func(x, plane):
    """Define objective functions here."""
    days = 31
    f1, num_takeoffs = battery_sweep(plane, x, days=days, month=6)
    # # TODO: Create an objective function that accounts for the riskiness of taking off
    f2 = num_takeoffs / days

    # f1 = np.sqrt(1+x[0]**2)
    # f2 = np.sqrt((1-x[0])**4)*1.5
    return f1, f2


def plot_yearly_dc(
    plane: Seaplane,
    year: int,
    month: int,
    day: int,
    days: int,
):
    label = f"{plane.capacity} Ah"
    times, e_h, P_solar, states, dc = run_simulation(plane, year, month, day, days)
    df = pd.DataFrame(e_h, index=times, columns=["battery"])
    df["P_solar"] = P_solar
    df["states"] = states
    df["duty cycle"] = dc

    df_daylight_hours = df.between_time("08:00", "18:00")
    daily_avg = df_daylight_hours["states"].resample("D").sum() / 120.0 * 100

    print("Plane Capacity: {0}, Average Duty Cycle: {1}".format(plane.capacity, np.mean(dc)))

    plt.clf()
    plt.plot(range(0, days), daily_avg)
    plt.scatter(range(0, days), daily_avg, s=7)
    plt.title("Daily Duty Cycle")
    plt.xlabel("Day of Year")
    plt.ylabel("Duty Cycle [%]")
    plt.tight_layout()
    filename = "YearSweep"
    plot_path = os.path.join("Figures", f"{filename}.png")
    plt.savefig(plot_path)


def generate_datetimes(start_index, end_index, timestep):
    """Generates a list of datetime objects between start_index and end_index.

    Args:
        start_index: Tuple representing (month, day, hour)
        end_index: Tuple representing (month, day, hour)
        timestep: Time difference between consecutive datetimes (in minutes)

    Returns:
        List of datetime objects
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

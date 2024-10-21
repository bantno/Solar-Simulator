from multiprocessing import Pool, Lock, Manager, cpu_count
import pandas as pd
import datetime
import os

from BaseClasses.seaplane_base import Seaplane
from BaseClasses.simulation_base import Simulation
from Tools import plotting

# Simulation Parameters
lat, lon = 29.02291491363789, -90.23223029442693
tz = "Etc/GMT+6"
pdc0, gamma = 0, -0.0047  # Solar parameters
capacity_ah, voltage = 50.0, 22.2  # Battery parameters
Cd0, Cdtot, S, af_mass = 0.02584, 0.0, 0.653, 8.8
cruise_speed, rho = 20.0, 1.19  # Flight parameters
N_PROP, N_ESC = 0.82, 0.9  # Propeller and ESC efficiency
solar_file = r"Data\DISTRIBUTIONS\2022_solar_data.pkl"

start_index = (1, 2, 0)  # Start time for weather data
end_index = (1, 7, 23)  # End time for weather data
U = cruise_speed  # Airspeed
algo = "MDP"  # Algorithm to run

# Initialize the plane and simulation object
plane = Seaplane(
    lat, lon, tz, pdc0, gamma,
    cd0=Cd0 * 1.5, cs=True, tracking=False, 
    cdtot=Cdtot, n_tot=N_PROP * N_ESC, 
    S=S, af_mass=af_mass, voltage=voltage, 
    capacity=capacity_ah
)
sim = Simulation(plane, lat, lon, tz, cs=False)

def run_single_simulation(run_id, capacity, success_prob):
    """Run a single simulation and return the result."""
    try:
        sim.plane.capacity = capacity
        times, states, P_solar, reward = sim.run_simulation(
            solar_file, start_index, end_index, U, rho, algo=algo, success_prob=success_prob
        )
        print(f"Completed Run {run_id} with Reward: {reward}")
        return {"Run": run_id, "Capacity": capacity, "Success_Prob": success_prob, "Reward": round(reward), "Failure Step": len(states)}
    except Exception as e:
        print(f"Error in Run {run_id}: {str(e)}")
        return None

def write_result_to_excel(result, lock, filename):
    """Append a single simulation result to the Excel file."""
    if result is None:
        return

    # Use a lock to ensure only one process writes at a time.
    with lock:
        if not os.path.exists(filename):
            df = pd.DataFrame([result])
            df.to_excel(filename, index=False)
        else:
            existing_df = pd.read_excel(filename)
            new_df = pd.DataFrame([result])
            updated_df = pd.concat([existing_df, new_df], ignore_index=True)
            updated_df.to_excel(filename, index=False)

if __name__ == "__main__":
    num_runs = 4
    capacity = 50
    success_prob = 0.9

    # Create a unique filename for the results.
    filename = f"SimResults_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.xlsx"

    # Use a Manager lock to ensure safe Excel writing.
    manager = Manager()
    lock = manager.Lock()

    with Pool(cpu_count()) as pool:
        for run_id in range(1, num_runs + 1):
            result = pool.apply_async(
                run_single_simulation, (run_id, capacity, success_prob),
                callback=lambda res: write_result_to_excel(res, lock, filename)
            )

        pool.close()
        pool.join()

    print(f"All simulations completed. Results saved to {filename}.")

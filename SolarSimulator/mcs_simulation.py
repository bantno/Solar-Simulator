import os
import datetime
import pandas as pd
import numpy as np
from multiprocessing import Pool, cpu_count, Manager
from tqdm import tqdm
from BaseClasses.seaplane_base import Seaplane
from BaseClasses.simulation_base import Simulation
import signal

# Initialize a global flag to detect keyboard interrupts
shutdown_flag = False

# Simulation Parameters
lat, lon = 29.02291491363789, -90.23223029442693
tz = "Etc/GMT+6"
pdc0, gamma = 0, -0.0047  # Solar parameters
capacity_ah, voltage = 50.0, 22.2  # Battery parameters
Cd0, Cdtot, S, af_mass = 0.02584, 0.0, 0.653, 8.8
cruise_speed, rho = 20.0, 1.19  # Flight parameters
N_PROP, N_ESC = 0.82, 0.9  # Propeller and ESC efficiency
solar_file = r"Data\DISTRIBUTIONS\2022_solar_data.pkl"
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

# Signal handler for graceful shutdown
def handle_interrupt(sig, frame):
    global shutdown_flag
    shutdown_flag = True
    print("Received interrupt. Safely shutting down...")

# Register the signal handler
signal.signal(signal.SIGINT, handle_interrupt)

def run_simulation(capacity, success_prob, run_id):
    """Single simulation function."""
    sim = Simulation(plane, lat, lon, tz, cs=False)  # Initialize sim inside each process
    sim.plane.capacity = capacity
    
    start_index = (1, 2, 0)
    end_index = (1, 30, 23)
    solar_file = r"Data\DISTRIBUTIONS\2022_solar_data.pkl"
    
    _, states, _, reward = sim.run_simulation(solar_file, start_index, end_index,
                                         U, rho, algo='MDP', success_prob=success_prob)
    
    return {"RunID": run_id, "Capacity": capacity, "SuccessProb": success_prob, "Reward": reward, "Failure Step": len(states)}

def run_batch(capacity, success_prob, num_runs, batch_size=7):
    """Run simulations in batches and write results to disk periodically."""
    results = []  # Store results in memory

    try:
        with Pool(cpu_count() - 1) as pool:
            for i in tqdm(range(0, num_runs, batch_size), desc="Simulations"):
                # Create the batch of tasks
                batch = [
                    (capacity, success_prob, run_id)
                    for run_id in range(i, min(i + batch_size, num_runs))
                ]

                # Run the batch in parallel and collect results
                batch_results = pool.starmap(run_simulation, batch)
                results.extend(batch_results)

                # Write to CSV after each batch completes
                append_to_csv(batch_results, filename=filename)

                # Check for shutdown request
                if shutdown_flag:
                    print("Shutdown requested. Writing remaining results to disk...")
                    append_to_csv(results, filename=filename)
                    return

    except KeyboardInterrupt:
        print("KeyboardInterrupt detected. Writing remaining results to disk...")
        append_to_csv(results, filename=filename)
        return

    print("All simulations completed successfully.")

def append_to_csv(data, filename="simulation_results.csv"):
    """Append simulation results to a CSV file."""
    df = pd.DataFrame(data)
    file_exists = os.path.isfile(filename)

    # Append to the file if it exists, otherwise create a new one
    df.to_csv(filename, mode='a', header=not file_exists, index=False)

if __name__ == "__main__":
    # Simulation parameters
    capacity = 50  # Example capacity in Ah
    success_prob = 0.9  # Example success probability
    num_runs = 50  # Total number of simulations to run

    # Create a timestamped results file based on capacity and success probability
    current_time = datetime.datetime.now()
    time_string = current_time.strftime("%Y-%m-%d_%H-%M-%S")
    filename = f"simulation_results_capacity{capacity}_success{int(success_prob*100)}_{time_string}.csv"

    print(f"Starting {num_runs} simulations with capacity={capacity} Ah and success_prob={success_prob}...")

    try:
        run_batch(capacity, success_prob, num_runs)
    except Exception as e:
        print(f"An error occurred: {e}")

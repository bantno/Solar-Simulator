from BaseClasses.simulation_storage import SimulationStorage

# Specify the directory where simulation files are stored.
storage_dir = "simulation_results"

# Create an instance of SimulationStorage pointing to the target directory.
storage = SimulationStorage(storage_dir)

# Load all simulation runs from the directory.
all_simulations = storage.load_all_simulations()

print(f"Loaded {len(all_simulations)} simulation run(s) from '{storage_dir}':\n")

# Iterate through each loaded simulation run.
for sim_run in all_simulations:
    sim_meta = sim_run["simulation_metadata"]
    episodes = sim_run["episodes"]
    print("Simulation Metadata:")
    for key, value in sim_meta.items():
        print(f"  {key}: {value}")
    print(f"Number of Episodes: {len(episodes)}")
    
    # Optionally, print summary info for each episode.
    for idx, ep in enumerate(episodes):
        total_reward = ep.get("total_reward", sum(ep.get("rewards", [])))
        print(f"   Episode {idx}: Total Reward = {total_reward}")
    print("-" * 40)

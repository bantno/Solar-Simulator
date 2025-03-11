from BaseClasses.simulation_storage import SimulationStorage

if __name__ == "__main__":
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
        
        total_rewards = []  # To collect the total reward from each episode
        
        # Process each episode
        for idx, ep in enumerate(episodes):
            # If 'total_reward' is not provided, calculate it as the sum of the rewards list.
            total_reward = ep.get("total_reward", sum(ep.get("rewards", [])))
            total_rewards.append(total_reward)
            # print(f"   Episode {idx}: Total Reward = {total_reward}")
        
        # Calculate and print the average reward for all episodes in this simulation
        if total_rewards:
            avg_reward = sum(total_rewards) / len(total_rewards)
            print(f"Average Reward for this simulation: {avg_reward}")
        else:
            print("No episodes found to calculate average reward.")
        print("-" * 40)

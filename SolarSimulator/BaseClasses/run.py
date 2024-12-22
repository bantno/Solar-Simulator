import yaml
from run_sim import SolarPlaneSimulation

def load_simulations_config(file_path):
    """Load simulation parameters from a YAML file."""
    with open(file_path, "r") as file:
        config = yaml.safe_load(file)
    return config["simulations"]

def run_simulation(params, algorithm):
    """Initialize and run a SolarPlaneSimulation with the given parameters for a specific algorithm."""
    simulation = SolarPlaneSimulation(
        lat=params["latitude"],
        lon=params["longitude"],
        tz="Etc/GMT-0",
        start_date=params["start_date"],
        end_date=params["end_date"],
        dt=params["dt"],
        num_runs=params["num_runs"],
        visualize=False,
        save_dir=params["save_dir"],
        show=False
    )
    print(f"Running simulation for {algorithm} algorithm with parameters:")
    print(params)

    simulation.run(
        capacities=params["capacities"],
        thresholds=params.get("thresholds", []) if algorithm == "Threshold" else [],
        mdp_probs=params.get("mdp_probs", []) if algorithm == "Optimal" else [],
        charge_thresholds=params.get("charge_thresholds", []),
        success_prob=params["success_prob"]
    )

def main():
    config_file = r"Results\Analysis\sim_params.yaml"  # Update path as needed
    simulations = load_simulations_config(config_file)

    for i, params in enumerate(simulations, start=1):
        print(f"Running simulation set {i}/{len(simulations)}...")
        algorithms = params["algorithms"]
        for algorithm in algorithms:
            run_simulation(params, algorithm)

if __name__ == "__main__":
    main()

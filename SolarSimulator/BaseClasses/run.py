import yaml
from run_sim import SolarPlaneSimulation  # Replace with the actual import path

def load_simulations_config(file_path):
    """Load simulation parameters from a YAML file."""
    with open(file_path, "r") as file:
        config = yaml.safe_load(file)
    return config["simulations"]

def run_simulation(params):
    """Initialize and run a SolarPlaneSimulation with the given parameters for a specific algorithm."""
    simulation = SolarPlaneSimulation(
        lat=-30,
        lon=-90,
        tz="Etc/GMT-0",
        start_date=params["start_date"],
        end_date=params["end_date"],
        dt=params["dt"],
        num_runs=params["num_runs"],
        visualize=False,
        save_dir=r".",  # Change this if needed
        show=False
    )
    print(f"Running simulation for with parameters:")
    print(params)

    simulation.run(
        capacities=params["capacities"],
        thresholds=params.get("thresholds", []),
        mdp_probs=params.get("mdp_probs", []),
        charge_thresholds=params.get("charge_thresholds", []),
        success_prob=params["success_prob"]
    )

def main():
    config_file = r"Results\Analysis\simulation_params.yaml"  # Update path as needed
    simulations = load_simulations_config(config_file)

    for i, params in enumerate(simulations, start=1):
        print(f"Running simulation set {i}/{len(simulations)}...")
        run_simulation(params)

if __name__ == "__main__":
    main()

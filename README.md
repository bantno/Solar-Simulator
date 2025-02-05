# Solar-Powered Seaplane Simulation  

## Overview  
This repository contains the simulation framework for modeling the operation of a solar-powered seaplane designed for long-term oceanic monitoring. The simulation incorporates environmental factors such as solar radiation, wind conditions, and whale sighting probabilities to evaluate decision-making strategies for energy management and mission planning.  

## Features  
- **State-Based Decision System**: The seaplane operates in three states—`moored`, `flying`, and `broken`—with transitions influenced by energy availability, environmental conditions, and failure probabilities.  
- **Energy Management**: Models battery charge and discharge dynamics based on solar radiation, energy consumption, and storage capacity.  
- **Environmental Interaction**: Incorporates whale sighting probabilities to determine maneuvering decisions and evaluates the impact of wind and solar conditions.  
- **Failure Modeling**: Uses a probability mass function dependent on state and action to simulate system failures.  
- **Algorithmic Comparisons**: Supports multiple decision-making strategies, including an optimal policy, threshold-based policies, and a greedy approach.  

## File Naming Conventions  
Simulation results are stored using the following naming patterns:  

- **Optimal Policy Results**:  
  ```
  Optimal_Data_cXX_p0.X_XXmin_X-X_X.pkl
  ```
  - `cXX`: Battery capacity (Ah)  
  - `p0.X`: Probability of failure  
  - `XXmin`: Time step  
  - `X-X`: Start and end day of the year  
  - `X`: Number of runs  

- **Threshold Policy Results**:  
  ```
  Threshold_Data_cXX_t0.X_XXmin_X-X_X.pkl
  ```
  - `t0.X`: Threshold-based decision parameter  
  - Other parameters follow the same structure as above  

- **Greedy Policy Results** (special case of threshold with `t0.0`):  
  ```
  Threshold_Data_cXX_t0.0_XXmin_X-X_X.pkl
  ```

## Development Environment  
- **Python Package Management**: Conda  
- **IDE**: VSCode  
- **Version Control**: Git (with `git filter-repo` for large file removal)  
- **Operating System**: Windows  

## Installation  
1. Clone the repository:  
   ```bash
   git clone https://github.com/yourusername/solar-seaplane-simulation.git
   cd solar-seaplane-simulation
   ```  
2. Create and activate a Conda environment:  
   ```bash
   conda create --name seaplane-env python=3.9  
   conda activate seaplane-env  
   ```  
3. Install dependencies:  
   ```bash
   pip install -r requirements.txt  
   ```  

## Usage  
Run the main simulation script:  
```bash
python run_simulation.py --data_dir path/to/data
```  

## License  
[Specify license here, e.g., MIT License]  

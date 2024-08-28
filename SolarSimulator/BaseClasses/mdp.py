import pandas as pd

def create_table(charges, y_states, actions, transitions, rewards):
    data = []
    for charge in charges:
        for y in y_states:
            for sun in [0, 1]:
                for action, transition, reward in zip(actions, transitions[sun], rewards[sun]):
                    next_charge = transition  # Ensure charge stays within 0-100
                    data.append([charge, y, sun, action, next_charge, reward])
    
    df = pd.DataFrame(data, columns=["x (State of Charge) @ t=i", "y @ t=i", "Sun?", "Action @ t=i", "delta_x @ t=i+1", "Reward @ i"])
    return df

# Define the states of charge, y states, actions, transitions, and rewards
charges = list(range(0, 101, 10))  # State of charge from 0 to 100 in increments of 10
y_states = ["flying", "floating"]
actions = ["float", "fly"]
transitions = [
    [-1, -7],  # Transitions for sun=0
    [3, -3]    # Transitions for sun=1
]
rewards = [
    [0, 0],    # Rewards for sun=0
    [0, 10]    # Rewards for sun=1
]

# Create the table
table = create_table(charges, y_states, actions, transitions, rewards)
print(table.head(16))


import pandas as pd

def simulate_multiple_steps(initial_charge, y_states, actions, transitions, rewards, max_steps):
    data = []
    
    for charge in initial_charge:
        for y in y_states:
            for sun in [0, 1]:
                for action, transition, reward in zip(actions, transitions[sun], rewards[sun]):
                    current_charge = charge
                    cumulative_reward = 0
                    current_sun = sun
                    
                    for step in range(max_steps):
                        next_charge = current_charge + transition
                        
                        if next_charge < 0:
                            next_charge = "Battery Fail"
                            data.append([charge, y, current_sun, action, next_charge, cumulative_reward])
                            break
                        
                        cumulative_reward += reward
                        current_charge = next_charge

                        # Record the data for this step, including the sun state
                        data.append([charge, y, current_sun, action, current_charge, cumulative_reward])

                        # If battery fails, stop further steps
                        if current_charge == "Battery Fail":
                            break
    
    df = pd.DataFrame(data, columns=["x @ t_i-1", "y @ t_i-1", "Sun @ t_i-1", "Action", "Final x", "Reward"])
    return df

# Define the states of charge, y states, actions, transitions, and rewards
initial_charge = list(range(0, 101, 10))  # State of charge from 0 to 100 in increments of 10
y_states = ["flying", "floating"]
actions = ["float", "fly"]
transitions = [
    [-10, -70],  # Transitions for sun=0 (e.g., decrease state of charge)
    [30, -30]    # Transitions for sun=1 (e.g., increase or decrease state of charge)
]
rewards = [
    [0, 0],    # Rewards for sun=0
    [0, 10]    # Rewards for sun=1
]
max_steps = 3  # Number of time steps to simulate

# Create the full table with multiple steps and sun state
full_table = simulate_multiple_steps(initial_charge, y_states, actions, transitions, rewards, max_steps)
print(full_table.head(20))  # Print the first 20 rows to see the simulation over time steps
full_table.to_csv('output.csv',index=False)


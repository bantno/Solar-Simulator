import numpy as np
from scipy.optimize import linprog

# Define the MDP parameters
states = [0, 1, 2]  # Example states
actions = [0, 1]  # Example actions
gamma = 0.9  # Discount factor
time_steps = 5  # Define number of time steps to plan over

# Rewards
rewards = [1, 2, 3]

# Transition probabilities
P = {
    0: {0: [0, 1, 0], 1: [0, 0, 1]},  # State 0 transitions
    1: {0: [1, 0, 0], 1: [0, 0, 1]},  # State 1 transitions
    2: {0: [0, 0, 1], 1: [0, 1, 0]},  # State 2 transitions
}

# Coefficients for the linear program
c = [-r for r in rewards] * time_steps  # We maximize, but linprog minimizes

# Create constraints for each state and time step
A = []
b = []

# Create constraints for each state and time step
for t in range(time_steps):
    for s in states:
        constraint = [0] * (len(states) * time_steps)
        for a in actions:
            for next_state in states:
                # Calculate the index for the next state in the current time step
                next_state_index = next_state + t * len(states)
                constraint[next_state_index] += P[s][a][next_state] * gamma
        # Set the constraint for R(s) at time t
        A.append(constraint)
        b.append(rewards[s])  # R(s)

# Convert to numpy arrays
A = np.array(A)
b = np.array(b)

# Solve the linear program
result = linprog(c, A_ub=A, b_ub=b, bounds=[(None, None)] * len(c), method="highs")

# Check the result
if result.success:
    optimal_values = result.x
    print("Optimal value function over time:", -result.fun)  # Negate to get the maximized value
    print("Optimal values for states over time:")

    for t in range(time_steps):
        print(f"Time step {t}:")
        for s in states:
            print(f"  State {s}: {optimal_values[s + t * len(states)]}")
else:
    print("No solution found:", result.message)

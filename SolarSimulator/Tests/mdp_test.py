# Define the states of charge, y states, actions, transitions, and rewards
charges = list(range(0, 101, 10))  # State of charge from 0 to 100 in increments of 10
y_states = ["flying", "floating"]
actions = ["float", "fly"]
transitions = [
    [-10, -70],  # Transitions for sun=0
    [30, -30]    # Transitions for sun=1
]
rewards = [
    [0, 0],    # Rewards for sun=0
    [0, 10]    # Rewards for sun=1
]

# Create the table
table = create_table(charges, y_states, actions, transitions, rewards)
table.to_csv('mdp_table.csv',index=False)
print(table.head(16))
import os
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import pandas as pd


def plot_all_state_and_solar_histories(df):
    plt.figure(figsize=(10, 10))
    
    # Plot State of Charge for each iteration
    plt.subplot(2, 1, 1)
    for i, row in df.iterrows():
        state_history = [state[0] for state in row['StateHistory']]
        plt.plot(state_history, label=f'Iteration {row["Iteration"]}')
    plt.title('State of Charge Over Time')
    plt.xlabel('Time Step')
    plt.ylabel('Charge Level (%)')
    plt.legend()
    plt.grid(True)

    # Plot Solar History for each iteration
    plt.subplot(2, 1, 2)
    for i, row in df.iterrows():
        solar_history = row['SolarHistory']
        plt.plot(solar_history, label=f'Iteration {row["Iteration"]}')
    plt.title('Solar History Over Time')
    plt.xlabel('Time Step')
    plt.ylabel('Solar Power (arbitrary units)')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()


data = pd.read_pickle("Greedy_Data_c35_p0.9.pkl")
data2 = pd.read_pickle("MDP_Data_c35_p0.9.pkl")
comp = pd.concat((data.head(2),data2))
print(comp)
plot_all_state_and_solar_histories(comp)
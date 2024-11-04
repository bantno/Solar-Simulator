import os
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta, timezone
from matplotlib.figure import Figure
import matplotlib.pyplot as plt



def plot_all_state_and_solar_histories(df, start_date, time_step='H'):
    """
    Plots State of Charge and Solar History with datetime x-axis.
    
    Parameters:
    - df: DataFrame containing the data.
    - start_date: The starting date and time as a string (e.g., '2023-01-01 00:00:00').
    - time_step: Frequency string for the time step (e.g., 'H' for hourly, 'D' for daily).
    """
    plt.figure(figsize=(10, 10))
    
    # Plot State of Charge for each iteration
    plt.subplot(2, 1, 1)
    for i, row in df.iterrows():
        state_history = [state[0] for state in row['StateHistory']]
        
        # Generate datetime index for x-axis
        time_index = pd.date_range(start=start_date, periods=len(state_history), freq=time_step)
        plt.plot(time_index, state_history, label=f'Iteration {row["Iteration"]}')
    
    plt.title('State of Charge Over Time')
    plt.xlabel('Datetime')
    plt.ylabel('Charge Level (%)')
    plt.legend()
    plt.grid(True)

    # Plot Solar History for each iteration
    plt.subplot(2, 1, 2)
    for i, row in df.iterrows():
        solar_history = row['SolarHistory']
        
        # Generate datetime index for x-axis
        time_index = pd.date_range(start=start_date, periods=len(solar_history), freq=time_step)
        plt.plot(time_index, solar_history, label=f'Iteration {row["Iteration"]}')
    
    plt.title('Solar History Over Time')
    plt.xlabel('Datetime')
    plt.ylabel('Solar Power (arbitrary units)')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()


data = pd.read_pickle("Greedy_Data_c35_p0.9.pkl")
data2 = pd.read_pickle("MDP_Data_c35_p0.9.pkl")
comp = pd.concat((data.head(1),data2.head(1)))
print(comp)
utc_offset = timezone(timedelta(hours=-6))
plot_all_state_and_solar_histories(comp,pd.to_datetime(datetime(2019,1,2).replace(tzinfo=utc_offset)))
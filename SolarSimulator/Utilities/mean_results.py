import pandas as pd
import matplotlib.pyplot as plt

def calculate_mean_rewards_and_failures(filename):
    # Read the CSV file into a DataFrame
    df = pd.read_pickle(filename)

    # Check if the required columns exist
    if 'Reward' in df.columns and 'LastStep' in df.columns:
        # Calculate mean values
        mean_reward = df['Reward'].mean()
        mean_failure_step = round(df['LastStep'].mean())

        # Print the results
        print(f"Mean Reward: {mean_reward}")
        print(f"Mean Failure Step: {mean_failure_step}")
    else:
        print("The required columns 'Reward' and 'LastStep' are not found in the CSV file.")

def plot_results(filename):
    df = pd.read_pickle(filename)
    if "StateHistory" in df.columns:
        state_history = df["StateHistory"][0]
        soc = [s[0] for s in state_history]
        plt.plot(soc)
        plt.show()
        

if __name__ == "__main__":
    # Specify the path to your CSV file
    filename = 'Greedy_Data.pkl'
    calculate_mean_rewards_and_failures(filename)
    filename = 'MDP_Data.pkl'
    calculate_mean_rewards_and_failures(filename)
    # plot_results(filename)

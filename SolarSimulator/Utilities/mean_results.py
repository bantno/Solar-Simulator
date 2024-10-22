import pandas as pd

def calculate_mean_rewards_and_failures(filename):
    # Read the CSV file into a DataFrame
    df = pd.read_csv(filename)

    # Check if the required columns exist
    if 'Reward' in df.columns and 'Failure Step' in df.columns:
        # Calculate mean values
        mean_reward = df['Reward'].mean()
        mean_failure_step = round(df['Failure Step'].mean())

        # Print the results
        print(f"Mean Reward: {mean_reward}")
        print(f"Mean Failure Step: {mean_failure_step}")
    else:
        print("The required columns 'Reward' and 'Failure Step' are not found in the CSV file.")

if __name__ == "__main__":
    # Specify the path to your CSV file
    filename = 'simulation_results_capacity50_success90_2024-10-21_22-19-21.csv'
    calculate_mean_rewards_and_failures(filename)

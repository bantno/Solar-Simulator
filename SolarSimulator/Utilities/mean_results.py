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
        print(f"Results for {filename}")
        print(f"Mean Reward: {mean_reward}")
        print(f"Mean Failure Step: {mean_failure_step}")
        return mean_reward,mean_failure_step
    else:
        print("The required columns 'Reward' and 'LastStep' are not found in the CSV file.")
        return None,None

def plot_results(filename):
    df = pd.read_pickle(filename)
    if "StateHistory" in df.columns:
        state_history = df["StateHistory"][0]
        soc = [s[0] for s in state_history]
        plt.plot(soc)
        plt.show()
        

if __name__ == "__main__":
    # Specify the path to your CSV file
    greedy_mean_reward_list = []
    greedy_mean_failure_step_list = []
    cap_list = []
    prob_list = []

    mdp_mean_reward_list = []
    mdp_mean_failure_step_list = []

    for i in [25,50,75,100]:
        for p in [0.5,0.75,0.9,1.0]:
            cap_list.append(i)
            prob_list.append(i)
            filename = f"Greedy_Data_c{i}_p{p}.pkl"
            mean_reward,failue_step = calculate_mean_rewards_and_failures(filename)
            greedy_mean_reward_list.append(mean_reward)
            greedy_mean_failure_step_list.append(failue_step)
            filename = f"MDP_Data_c{i}_p{p}.pkl"
            mean_reward,failue_step = calculate_mean_rewards_and_failures(filename)
            mdp_mean_reward_list.append(mean_reward)
            mdp_mean_failure_step_list.append(failue_step)
    
    plt.scatter(cap_list,greedy_mean_reward_list,label="Greedy Reward")
    # plt.scatter(cap_list,greedy_mean_failure_step_list)

    plt.scatter(cap_list,mdp_mean_reward_list,label="MDP Reward")
    # plt.scatter(cap_list,mdp_mean_failure_step_list)
    plt.legend()

    plt.show()

    # plot_results(filename)

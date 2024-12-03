from BaseClasses.plotting_base import SolarChargePlotter,DataProcessor
import datetime

if __name__ == "__main__":
    directory = r"."

<<<<<<< HEAD
    # plt.figure(figsize=(10, 6))
    # df['Actual'].plot(marker='o', label='SSRD=2019 Data')
    # df['Expected'].plot(marker='x', linestyle='--', label='SSRD=Expected Data')

    # plt.xlabel('Capacity')
    # plt.ylabel('Value')
    # plt.title('Percent Improvement of Optimal Algorithm over Threshold Algorithm')
    # plt.legend()
    # plt.grid(True)
    # plt.show()

    dire = r"."
    processor = PickleDataProcessor(directory=dire)  # Use "." for the current directory
    # processor.plot_reward_histogram(directory=dire,bins=30)
    processor.process_files()
    df = processor.get_results_df()
    processor.calculate_percent_improvement(df).to_csv("test.csv")
=======
    # Generate figure
    processor = DataProcessor(directory=directory)  # Use "." for the current directory
>>>>>>> c4cd1f504e38e01057eb167164573c942c708b88
    processor.plot_all_data()
    
    # Calculate percent improvement
    processor.process_files()
    mcs_results = processor.get_results_df()
    processor.calculate_percent_improvement(mcs_results).to_csv("Improvement.csv")

    #
    
    # processor.plot_reward_histogram(directory=dire,bins=30)
    # print(processor.results)
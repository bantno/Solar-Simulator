from BaseClasses.plotting_base import SolarChargePlotter,DataProcessor
import datetime

if __name__ == "__main__":
    directory = r"."

    # Generate figure
    processor = DataProcessor(directory=directory)  # Use "." for the current directory
    processor.plot_all_data()
    
    # Calculate percent improvement
    processor.process_files()
    mcs_results = processor.get_results_df()
    processor.calculate_percent_improvement(mcs_results).to_csv("Improvement.csv")

    #
    
    # processor.plot_reward_histogram(directory=dire,bins=30)
    # print(processor.results)
from BaseClasses.plotting_base import DataProcessor

# dire = r"Results\Analysis"
dire = r"Results\Analysis\prob"
processor = DataProcessor(directory=dire)  # Use "." for the current directory
processor.process_files()
df = processor.get_results_df()
processor.plot_all_data(dire)
# processor.plot_optimal_battery_capacity(df,dire)
# processor.plot_combined_data(dire)

from datetime import datetime, timedelta, timezone
import pandas as pd
from BaseClasses.plotting_base import SolarChargePlotter
from Tools.analyze_results import PickleDataProcessor

dire = r"Results\TakeoffPowerConsumption\6monthRun0.75"
processor = PickleDataProcessor(directory=dire)  # Use "." for the current directory
# processor.plot_reward_histogram(directory=dire,bins=30)
processor.process_files()
df = processor.get_results_df()
processor.calculate_percent_improvement(df).to_csv("test.csv")
processor.plot_all_data()
dt=30
utc_offset = timezone(timedelta(hours=0))
start_date = pd.to_datetime(datetime(2024,3,1).replace(tzinfo=utc_offset))
plotter = SolarChargePlotter(dire,start_date=start_date,time_step=f"{dt}min")
plotter.plot_data()
import os
from datetime import datetime, timedelta, timezone
from BaseClasses.plotting_base import DataProcessor

if __name__ == "__main__":
    directory = r"Results\New-Trans\2500"
    output_dir = r"Figures\Histogram"
    
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    processor = DataProcessor(directory=directory, show=False)
    processor.plot_reward_histogram(directory=directory, bins=50)

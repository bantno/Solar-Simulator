from datetime import datetime, timedelta, timezone
import pandas as pd
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from BaseClasses.plotting_base import StateHistoryPlotter

direct = r"Results\Analysis"
utc_offset = timezone(timedelta(hours=0))
start_date = pd.to_datetime(datetime(2024,1,1).replace(tzinfo=utc_offset))
solar = StateHistoryPlotter(direct,start_date,"60min")
solar.plot_data(direct)
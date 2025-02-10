from datetime import datetime, timedelta, timezone
import pandas as pd
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from BaseClasses.plotting_base import StateHistoryPlotter

# direct = r"Results\Analysis"
direct = r"Data\TEST_CASES\Wind\Meeting-Results\noWind-constantWhale"
utc_offset = timezone(timedelta(hours=-6))
start_date = pd.to_datetime(datetime(2025, 6, 1).replace(tzinfo=utc_offset))
solar = StateHistoryPlotter(direct, start_date, "15min")
solar.plot_data(direct)

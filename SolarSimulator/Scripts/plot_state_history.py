from datetime import datetime, timedelta, timezone
import pandas as pd
import sys
import os
import plotly.graph_objects as go
from BaseClasses.plotting_base import StateHistoryPlotter

# direct = r"Results\Analysis"
direct = r"Results\Analysis"
utc_offset = timezone(timedelta(hours=-6))
start_date = pd.to_datetime(datetime(2025, 1, 1).replace(tzinfo=utc_offset))
solar = StateHistoryPlotter(direct, start_date, "15min")
solar.plot_data(direct, use_plotly=False)

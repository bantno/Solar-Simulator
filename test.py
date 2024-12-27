import pandas as pd
# df = pd.read_pickle(r"Data\SYNTHETIC_DATA\lat0\data_10min_1.pkl")
df = pd.read_pickle(r"Data\EXPECTED_DATA\data_expected_10min.pkl")
import pandas as pd
import plotly.express as px

# Reset index to make the datetime accessible
df = df.reset_index().rename(columns={"index": "datetime"})

# Create an interactive plot
fig = px.line(
    df,
    x="datetime",
    y="expected_solar_rad",
    title="Shortwave Radiation Over Time",
    labels={"datetime": "Time", "shortwave_radiation": "Shortwave Radiation (W/m²)"},
)

# Add range sliders and selectors
fig.update_xaxes(
    rangeslider_visible=True,
    rangeselector=dict(
        buttons=[
            {"count": 1, "label": "1 Hour", "step": "hour", "stepmode": "backward"},
            {"count": 6, "label": "6 Hours", "step": "hour", "stepmode": "backward"},
            {"count": 1, "label": "1 Day", "step": "day", "stepmode": "backward"},
            {"step": "all"},
        ]
    )
)

# Show the plot
fig.show()

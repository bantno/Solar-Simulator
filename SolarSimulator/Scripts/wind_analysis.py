import pandas as pd
import plotly.express as px

# Example DataFrame (replace this with your actual DataFrame)
# df = pd.read_pickle(r"Data\HISTORICAL_DATA\data_60min_lat_30.pkl")
df = pd.read_pickle(r"Data\HISTORICAL_DATA\data_60min_lat_30.pkl")

# Filter data for daylight hours (shortwave_radiation > 0)
daylight_df = df[df['shortwave_radiation'] > 0]

# If no data passes the filter, provide a message and exit
if daylight_df.empty:
    print("No data available for daylight hours (shortwave_radiation > 0).")
else:
    # Extract month from the datetime index
    daylight_df['month'] = daylight_df.index.month
    daylight_df['month_name'] = daylight_df.index.strftime('%B')

    # Group by month and calculate average wind speed
    monthly_avg_wind_speed = (
        daylight_df.groupby(['month', 'month_name'])['wind_speed_10m']
        .mean()
        .reset_index()
        .sort_values(by='month')
    )

    # Create a plotly bar chart
    fig = px.bar(
        monthly_avg_wind_speed,
        x='month_name',
        y='wind_speed_10m',
        title='Average Wind Speed by Month (Daylight Hours)',
        labels={'wind_speed_10m': 'Average Wind Speed (m/s)', 'month_name': 'Month'},
        text='wind_speed_10m'
    )

    # Update layout for better visuals
    fig.update_traces(texttemplate='%{text:.2f}', textposition='outside')
    fig.update_layout(xaxis_title="Month", yaxis_title="Average Wind Speed (m/s)")
    fig.show()

# # Filter data for daylight hours (shortwave_radiation > 0)
# daylight_df = df[df['shortwave_radiation'] > 0]

# # If no data passes the filter, provide a message and exit
# if daylight_df.empty:
#     print("No data available for daylight hours (shortwave_radiation > 0).")
# else:
#     # Determine daylight hours dynamically for each day
#     daylight_df['date_only'] = daylight_df.index.date
#     daylight_hours = daylight_df.groupby('date_only').apply(lambda group: sorted(group.index.hour)).reset_index(name='hours')
    
#     # Map hours to time segments (morning, afternoon, evening) for each day
#     time_of_day_map = {}
#     for _, row in daylight_hours.iterrows():
#         hours = row['hours']
#         if len(hours) < 3:
#             continue  # Skip days with insufficient daylight data
#         segment_size = len(hours) // 3
#         time_of_day_map.update({
#             hour: "Morning" if i < segment_size else ("Afternoon" if i < 2 * segment_size else "Evening")
#             for i, hour in enumerate(hours)
#         })

#     daylight_df['hour'] = daylight_df.index.hour
#     daylight_df['time_of_day'] = daylight_df['hour'].map(time_of_day_map)

#     # Extract month from the datetime index
#     daylight_df['month'] = daylight_df.index.month
#     daylight_df['month_name'] = daylight_df.index.strftime('%B')

#     # Group by month and time of day, calculate average wind speed
#     breakdown = (
#         daylight_df.groupby(['month', 'month_name', 'time_of_day'])['wind_speed_10m']
#         .mean()
#         .reset_index()
#         .sort_values(by=['month', 'time_of_day'])
#     )

#     # Create a plotly bar chart
#     fig = px.bar(
#         breakdown,
#         x='month_name',
#         y='wind_speed_10m',
#         color='time_of_day',
#         barmode='group',
#         title='Average Wind Speed by Month and Time of Day (Daylight Hours)',
#         labels={'wind_speed_10m': 'Average Wind Speed (m/s)', 'month_name': 'Month', 'time_of_day': 'Time of Day'},
#         text='wind_speed_10m'
#     )

#     # Update layout for better visuals
#     fig.update_traces(texttemplate='%{text:.2f}', textposition='outside')
#     fig.update_layout(
#         xaxis_title="Month",
#         yaxis_title="Average Wind Speed (m/s)",
#         legend_title="Time of Day"
#     )
#     fig.show()

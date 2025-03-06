import pandas as pd
import plotly.express as px
import plotly.subplots as sp
import os
import numpy as np


# Directory containing the .pkl files
directory = r"Results\Analysis\Corrected Failure Penalty\1month"

# List all .pkl files in the directory
pkl_files = [f for f in os.listdir(directory) if f.endswith('.pkl')]

# Create a subplot figure for rewards and failures
fig = sp.make_subplots(rows=2, cols=len(pkl_files), subplot_titles=[f"{pkl_file} - Rewards" for pkl_file in pkl_files] + [f"{pkl_file} - Failures" for pkl_file in pkl_files])

for i, pkl_file in enumerate(pkl_files):
    # Load data from .pkl file
    data = pd.read_pickle(os.path.join(directory, pkl_file))
    
    # Extract reward values
    rewards = data.loc["Reward"].values
    
    # Create histogram for rewards
    fig_rewards = px.histogram(x=rewards, nbins=30, labels={"x": "Reward", "y": "Count"})
    
    # Add histogram to subplot
    for trace in fig_rewards.data:
        fig.add_trace(trace, row=1, col=i+1)
    
    # Extract failure types
    failures = data.loc["FailureType"].value_counts().reset_index()
    failures.columns = ["FailureType", "Count"]

    fig_failures = px.bar(
    failures, 
    x="FailureType", 
    y="Count", 
    text_auto=True,  # Show count values on bars
    labels={"Failure Type": "Failure Type", "Count": "Count"}
)

    # Add histogram to subplot
    for trace in fig_failures.data:
        fig.add_trace(trace, row=2, col=i+1)

# Update layout and show figure
fig.update_layout(title_text="Histograms of Rewards and Failures")
fig.show()

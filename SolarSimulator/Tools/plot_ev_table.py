import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def plot_data(x, y, plot_type='line', xlabel=None, ylabel=None, title=None, legend=None, style=None):
    """
    A generic plotting function to create different types of plots.

    Parameters:
    - x (list or array): x-axis data.
    - y (list or array): y-axis data.
    - plot_type (str): Type of plot ('line', 'scatter', 'bar', 'hist', etc.).
    - xlabel (str, optional): Label for the x-axis.
    - ylabel (str, optional): Label for the y-axis.
    - title (str, optional): Title of the plot.
    - legend (list, optional): Labels for the plot legend.
    - style (str, optional): Matplotlib style for the plot.
    """
    # plt.figure(figsize=(8, 4))
    if style:
        plt.style.use(style)
    
    if plot_type == 'line':
        plt.plot(x, y, label=legend)
    elif plot_type == 'scatter':
        plt.scatter(x, y, label=legend)
    elif plot_type == 'bar':
        plt.bar(x, y, label=legend)
    elif plot_type == 'hist':
        plt.hist(y, bins=30, label=legend)
    else:
        print(f"Unsupported plot type: {plot_type}")
        return

    if xlabel:
        plt.xlabel(xlabel)
    if ylabel:
        plt.ylabel(ylabel)
    if title:
        plt.title(title)
    if legend:
        plt.legend()
    plt.grid(True)

    

table = pd.read_csv(r"Data\TEST_CASES\Wind\Meeting-Results\varyWind-varyWhale\ev_table.csv", header=None).to_numpy()
plot_data(range(len(table[100,:])),table[100,:],legend="Moored")
plot_data(range(len(table[100,:])),table[201,:],xlabel="Stage",ylabel="Expected Value",title="Flying vs Moored for Battery Capacity = 100%",legend="Flying")
plt.show()
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go

class PlottingUtils:
    @staticmethod
    def plot_surface(data, capacity=0):
        """
        Plots separate surface plots for the 'moored,' 'flying,' and 'broken' states.

        Parameters:
            data (numpy.ndarray): A 2D array where:
                - Rows 0-100 represent battery percentages for the 'moored' state.
                - Rows 101-201 represent battery percentages for the 'flying' state.
                - Row 202 represents the 'broken' state.
            capacity (int): Battery capacity in Ah (used for the plot titles).
        """
        # Extract data for each state
        moored_data = data[:101, :]
        flying_data = data[101:202, :]
        broken_data = data[202:, :]

        # Generate grids
        time_steps = np.arange(data.shape[1])  # Time steps (x-axis)
        battery_percentages = np.linspace(0, 100, 101)  # Battery percentages (y-axis)

        # Plot for 'moored' state
        fig = plt.figure(figsize=(12, 8))
        ax = plt.subplot(projection="3d")
        x, y = np.meshgrid(time_steps, battery_percentages)
        surf = ax.plot_surface(x, y, moored_data, cmap="viridis", edgecolor="none")
        cbar = plt.colorbar(surf, ax=ax, shrink=0.5, aspect=10)
        cbar.set_label("Expected Value")
        ax.set_title(f"Surface Plot for State: Moored\nBattery Capacity: {capacity} Ah")
        ax.set_xlabel("Stages")
        ax.set_ylabel("State of Charge (%)")
        ax.set_zlabel("Expected Value")
        plt.tight_layout()
        plt.savefig("ev_table_moored.png")

        # Plot for 'flying' state
        fig = plt.figure(figsize=(12, 8))
        ax = plt.subplot(projection="3d")
        surf = ax.plot_surface(x, y, flying_data, cmap="plasma", edgecolor="none")
        cbar = plt.colorbar(surf, ax=ax, shrink=0.5, aspect=10)
        cbar.set_label("Expected Value")
        ax.set_title(f"Surface Plot for State: Flying\nBattery Capacity: {capacity} Ah")
        ax.set_xlabel("Stages")
        ax.set_ylabel("State of Charge (%)")
        ax.set_zlabel("Expected Value")
        plt.tight_layout()
        plt.savefig("ev_table_flying.png")

        # Plot for 'broken' state (single row, flatten data)
        fig = plt.figure(figsize=(12, 8))
        ax = plt.subplot(projection="3d")
        ax.plot(
            np.arange(data.shape[1]),
            [0] * data.shape[1],
            broken_data.flatten(),
            label="Broken State",
            color="red",
        )
        ax.set_title(f"Surface Plot for State: Broken\nBattery Capacity: {capacity} Ah")
        ax.set_xlabel("Stages")
        ax.set_ylabel("State of Charge (%)")
        ax.set_zlabel("Expected Value")
        ax.legend()
        plt.tight_layout()
        plt.savefig("ev_table_broken.png")

    @staticmethod
    def plot_surface_plotly(data, capacity=50, filename="ev_table_combined.html"):
        """
        Plots an interactive 3D surface plot for the 'moored,' 'flying,' and 'broken'
        states using Plotly.

        Parameters:
            data (numpy.ndarray): A 2D array where:
                - Rows 0-100 represent battery percentages for the 'moored' state.
                - Rows 101-201 represent battery percentages for the 'flying' state.
                - Row 202 represents the 'broken' state.
            capacity (int): Battery capacity in Ah (used for the plot title).
        """
        # Extract data for each state
        moored_data = data[:101, :]
        flying_data = data[101:202, :]
        broken_data = data[202, :]  # Single row for broken state

        # Generate grids
        time_steps = np.arange(data.shape[1])  # Time steps (x-axis)
        battery_percentages = np.linspace(0, 100, 101)  # Battery percentages (y-axis)
        x, y = np.meshgrid(time_steps, battery_percentages)

        # Create figure
        fig = go.Figure()

        # Add Moored State surface
        fig.add_trace(
            go.Surface(
                z=moored_data, x=x, y=y, colorscale="Blues", opacity=0.95, name="Moored"
            )
        )

        # Add Flying State surface
        fig.add_trace(
            go.Surface(
                z=flying_data, x=x, y=y, colorscale="Magma", opacity=0.8, name="Flying"
            )
        )

        # Add Broken State line
        fig.add_trace(
            go.Scatter3d(
                x=time_steps,
                y=[0] * len(time_steps),
                z=broken_data,
                mode="lines",
                line=dict(color="red", width=4),
                name="Broken",
            )
        )

        # Update layout
        fig.update_layout(
            title=(
                "Surface Plot for Moored, Flying, and Broken States "
                f"(Battery Capacity: {capacity} Ah)"
            ),
            scene=dict(
                xaxis_title="Stages",
                yaxis_title="State of Charge (%)",
                zaxis_title="Expected Value",
            ),
        )

        # Save plot
        fig.write_html(filename)
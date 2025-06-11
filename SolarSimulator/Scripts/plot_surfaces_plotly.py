import numpy as np
import plotly.graph_objects as go
import os
import re

class EVTablePlotterPlotly:
    @staticmethod
    def plot_surface_overlaid(
        data: np.ndarray,
        capacity: float = 0,
        horizon: int = 0,
        penalty: float = 0,
        outdir: str = "."
    ):
        """
        Overlaid moored/flying/broken surfaces in one 3D plot.
        Saves both HTML and PNG versions.
        """
        # split states
        moored     = data[:101, :]
        flying     = data[101:202, :]
        broken_row = data[202, :]

        # grids
        T      = data.shape[1]
        stages = np.arange(T)
        soc    = np.linspace(0, 100, 101)
        X, Y   = np.meshgrid(stages, soc)
        Zb     = np.tile(broken_row, (101, 1))

        # build figure
        fig = go.Figure()

        # moored
        fig.add_trace(go.Surface(
            x=X, y=Y, z=moored,
            colorscale="Blues",
            opacity=1.0,
            name="Moored"
        ))
        # flying
        fig.add_trace(go.Surface(
            x=X, y=Y, z=flying,
            colorscale="Reds",
            opacity=0.7,
            name="Flying"
        ))
        # broken at SoC=0 plane
        fig.add_trace(go.Surface(
            x=X, y=Y*0, z=Zb,
            colorscale="Inferno",
            opacity=0.9,
            name="Broken"
        ))

        # layout
        fig.update_layout(
            title=f"Overlaid Surfaces: {capacity}Wh | {horizon}h | p={penalty}",
            scene=dict(
                xaxis_title="Stages",
                yaxis_title="SoC (%)",
                zaxis_title="Expected Value"
            ),
            legend=dict(x=0.02, y=0.95)
        )

        # output
        fname_base = f"ev_plotly_overlaid_{int(capacity)}Wh_{horizon}h_{penalty}p"
        html_path = os.path.join(outdir, fname_base + ".html")
        # png_path  = os.path.join(outdir, fname_base + ".png")

        fig.write_html(html_path)
        # requires kaleido installed
        # fig.write_image(png_path, width=1200, height=800)

        print(f"Saved interactive plot to {html_path}")
        # print(f"Saved static export to    {png_path}")


    @staticmethod
    def plot_delta_surface(
        data: np.ndarray,
        capacity: float = 0,
        horizon: int = 0,
        penalty: float = 0,
        outdir: str = "."
    ):
        """
        Single surface Δ = flying − moored.
        """
        moored = data[:101, :]
        flying = data[101:202, :]

        T      = data.shape[1]
        stages = np.arange(T)
        soc    = np.linspace(0, 100, 101)
        X, Y   = np.meshgrid(stages, soc)

        delta = flying - moored
        vmax = np.max(delta)
        fig = go.Figure()
        fig.add_trace(go.Surface(
            x=X, y=Y, z=delta,
            cmin=vmax-2, cmax=vmax,
            colorscale="RdBu",
            colorbar=dict(title="Flying–Moored"),
            name="Δ Surface"
        ))

        fig.update_layout(
            title=f"Δ Surface (Flying−Moored): {capacity}Wh | {horizon}h | p={penalty}",
            scene=dict(
                xaxis_title="Stages",
                yaxis_title="SoC (%)",
                zaxis_title="Δ Expected Value",
                aspectmode='manual',
                aspectratio=dict(x=1, y=1, z=0.2),
            )
        )

        fname_base = f"ev_plotly_delta_{int(capacity)}Wh_{horizon}h_{penalty}p"
        html_path = os.path.join(outdir, fname_base + ".html")
        # png_path  = os.path.join(outdir, fname_base + ".png")

        fig.write_html(html_path)
        # fig.write_image(png_path, width=1200, height=800)

        print(f"Saved interactive delta plot to {html_path}")
        # print(f"Saved static export to       {png_path}")


def parse_filename(filename: str):
    """
    As before: future_value_table_{cap}Wh_{horizon}h_{penalty}p.npy
    """
    base = os.path.basename(filename)
    pattern = (
        r"future_value_table_"
        r"(?P<cap>[\d\.]+)Wh_"
        r"(?P<horizon>\d+)h_"
        r"(?P<pen>[\d\.]+)p\.npy$"
    )
    m = re.match(pattern, base)
    if not m:
        raise ValueError(f"Filename '{base}' does not match.")
    return float(m.group("cap")), int(m.group("horizon")), float(m.group("pen"))


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("path", help=".npy file or directory")
    p.add_argument("--outdir", default=".", help="where to save plots")
    args = p.parse_args()

    # gather files
    if os.path.isdir(args.path):
        files = [os.path.join(args.path, f) for f in os.listdir(args.path) if f.endswith(".npy")]
    else:
        files = [args.path]

    for fp in files:
        try:
            cap, hor, pen = parse_filename(fp)
            data = np.load(fp)
            EVTablePlotterPlotly.plot_surface_overlaid(data, cap, hor, pen, outdir=args.outdir)
            EVTablePlotterPlotly.plot_delta_surface(data, cap, hor, pen, outdir=args.outdir)
        except Exception as e:
            print(f"Skipping {fp}: {e}")

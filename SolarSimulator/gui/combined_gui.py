import sys
import h5py
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
# mpl.rcParams.update({
#     'font.size':           20,   # default text size
#     'axes.titlesize':      22,   # subplot title
#     'axes.labelsize':      20,   # x- and y-labels
#     'xtick.labelsize':     18,   # tick labels
#     'ytick.labelsize':     18,
#     'legend.fontsize':     18,   # legend text
#     'legend.title_fontsize':20,  # legend title
#     'figure.titlesize':    24,   # overall figure title
# })
import math
import matplotlib.dates as mdates
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
import cartopy.crs as ccrs
import cartopy.feature as cfeature

from cycler import cycler

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QLineEdit, QPushButton, QFileDialog, QMessageBox,
    QCheckBox, QTabWidget, QSpinBox, QDateTimeEdit, QListWidget,
    QListWidgetItem, QComboBox, QSizePolicy, QSlider
)
from PyQt5.QtCore import Qt, QDateTime, QDate, QTime

from matplotlib.backends.backend_qt5agg import (
    FigureCanvasQTAgg as FigureCanvas,
    NavigationToolbar2QT as NavToolbar
)


# ------------------------------------------------------------------------------
# 1) Multi‐Simulation Episode Inspector (adapted from plot_states_gui.py)
# ------------------------------------------------------------------------------

def _finish_plot(
    ax,
    title: str = "",
    xlabel: str = "",
    ylabel: str = "",
    legend_outside: bool = True,
    show: bool = True
):
    """Apply standard formatting to axes and optionally show plot.

    Args:
        ax (matplotlib.axes.Axes): The axes to format.
        title (str, optional): Plot title.
        xlabel (str, optional): X-axis label.
        ylabel (str, optional): Y-axis label.
        legend_outside (bool, optional): If True, place legend outside plot area on the right.
        show (bool, optional): If True, call plt.show() to display the plot.
    """
    if title:
        ax.set_title(title)
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)

    if legend_outside:
        ax.legend(
            loc='upper left',
            bbox_to_anchor=(1.02, 1),
            borderaxespad=0
        )
        plt.subplots_adjust(right=0.75)
    else:
        ax.legend()

    ax.grid(True)
    plt.tight_layout()

    if show:
        plt.show()


class InspectorTab(QWidget):
    """Tab for inspecting simulation episodes stored in an HDF5 file.

    Based on MultiSimInspector, but reimplemented as a QWidget.
    """
    def __init__(self):
        super().__init__()
        self.file_path = None
        self.sim_group_names = []
        self.dataset_names = [
            'solar_series',
            'wind_series',
            'whale_series',
            'energy_series',
            'actions',
            'rewards',
        ]
        self.y_axis_labels = [
            r'$G_k$ (Wh)',
            r'$w_k$ (m/s)',
            r'$O_k$',
            r'$E_k$ (Wh)',
            r'$a_k$',
            r'$r_k$',
        ]
        self.style_name = 'seaborn-v0_8-whitegrid'
        self.rcparams = {
            'font.size':       10,
            'axes.titlesize':  12,
            'axes.labelsize':  11,
            'lines.linewidth': 2,
            'figure.dpi':      120,
            'legend.fontsize': 10,
            'legend.frameon':  True,
            'legend.framealpha': 0.9,
            'legend.edgecolor':  'black',
        }
        self.color_cycle = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', "#9fbd67", "#5c3e00e7", "#bd67b1", "#7e1603", "#3108d4"]
        self.toolbar_enabled = True
        self.use_constrained_layout = False
        self.layout_settings = {
            'top':    0.92,
            'bottom': 0.08,
            'left':   0.10,
            'right':  0.95,
            'hspace': 0.3
        }
        # Time & window settings
        self.time_step_min = 15    # minutes per decision stage
        self.window_size = 100     # default window size in stages
        self.vline_refs = []       # references to cursor lines

        # Apply matplotlib style
        plt.style.use(self.style_name)
        plt.rcParams.update(self.rcparams)
        plt.rcParams['axes.prop_cycle'] = cycler('color', self.color_cycle)

        # Build UI
        self._build_ui()

    def _build_ui(self):
        central = QVBoxLayout(self)

        # 1) File picker
        file_layout = QHBoxLayout()
        file_layout.addWidget(QLabel("HDF5:"))
        self.file_line_edit = QLineEdit()
        self.file_line_edit.setReadOnly(True)
        file_layout.addWidget(self.file_line_edit)
        btn_open = QPushButton("Open File…")
        btn_open.clicked.connect(self.open_file)
        file_layout.addWidget(btn_open)
        central.addLayout(file_layout)

        # 2) Simulation selection checklist
        central.addWidget(QLabel("Simulations to plot:"))
        self.sim_list_widget = QListWidget()
        self.sim_list_widget.itemChanged.connect(self.on_sim_selection_changed)
        central.addWidget(self.sim_list_widget)

        # Unselect All button
        btn_unselect_all = QPushButton("Unselect All")
        btn_unselect_all.clicked.connect(self.unselect_all_simulations)
        central.addWidget(btn_unselect_all)


        # 3) Episode selector
        ctl = QHBoxLayout()
        ctl.addWidget(QLabel("Episode:"))
        self.cb_episode = QComboBox()
        self.cb_episode.currentTextChanged.connect(self.update_plot)
        ctl.addWidget(self.cb_episode)
        ctl.addStretch()
        central.addLayout(ctl)

        # NEW: “Open in New Figure” button
        self.btn_new_fig = QPushButton("Open Episode in New Figure")
        self.btn_new_fig.clicked.connect(self.open_episode_in_new_figure)
        central.addWidget(self.btn_new_fig)
        # ——————————————————————————————

        # 4) Create subplots (+1 for cumulative flight-time)
        n_plots = len(self.dataset_names) + 1
        self.fig, self.axes = plt.subplots(
            n_plots, 1,
            sharex=True,
            figsize=(12, 3 * n_plots),
            constrained_layout=self.use_constrained_layout
        )
        if not self.use_constrained_layout:
            self.fig.subplots_adjust(**self.layout_settings)

        # 5) Canvas & toolbar
        self.canvas = FigureCanvas(self.fig)
        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        central.addWidget(self.canvas)
        if self.toolbar_enabled:
            self.toolbar = NavToolbar(self.canvas, self)
            central.addWidget(self.toolbar)

        # 6) Window‐size control
        ctl_window = QHBoxLayout()
        ctl_window.addWidget(QLabel("Window Size (stages):"))
        self.spin_window = QSpinBox()
        self.spin_window.setMinimum(1)
        self.spin_window.setMaximum(self.window_size)
        self.spin_window.setValue(self.window_size)
        self.spin_window.valueChanged.connect(self.on_window_size_change)
        ctl_window.addWidget(self.spin_window)
        ctl_window.addStretch()
        central.addLayout(ctl_window)

        # 7) Slider for panning window
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setMinimum(1)
        self.slider.setMaximum(1)
        self.slider.setValue(1)
        self.slider.setTickPosition(QSlider.TicksBelow)
        self.slider.setTickInterval(self.window_size)
        self.slider.valueChanged.connect(self.on_window_slide)
        central.addWidget(self.slider)

        # 8) Rescale button for cumulative flight‐time axis
        self.btn_rescale = QPushButton("Rescale Flight-Time Axis")
        self.btn_rescale.clicked.connect(self.rescale_cumulative_axis)
        central.addWidget(self.btn_rescale)

        # 9) Cursor slider + spin-box + toggle button
        ctl_line = QHBoxLayout()
        ctl_line.addWidget(QLabel("Cursor Stage:"))
        self.line_slider = QSlider(Qt.Horizontal)
        self.line_slider.setMinimum(1)
        self.line_slider.setMaximum(1)
        self.line_slider.setValue(1)
        self.line_slider.setTickPosition(QSlider.TicksBelow)
        self.line_slider.valueChanged.connect(self.on_line_slide)
        ctl_line.addWidget(self.line_slider)

        self.spin_cursor = QSpinBox()
        self.spin_cursor.setMinimum(1)
        self.spin_cursor.setMaximum(1)
        self.spin_cursor.setValue(1)
        self.spin_cursor.valueChanged.connect(self.on_cursor_spin_change)
        ctl_line.addWidget(self.spin_cursor)

        self.btn_toggle_line = QPushButton("Toggle Cursor Line")
        self.btn_toggle_line.setCheckable(True)
        self.btn_toggle_line.toggled.connect(
            lambda on: self._draw_cursor(self.line_slider.value())
        )
        ctl_line.addWidget(self.btn_toggle_line)
        ctl_line.addStretch()
        central.addLayout(ctl_line)

    def open_episode_in_new_figure(self):
        """Load the current episode and plot it in a standalone Matplotlib window."""
        ep = self.cb_episode.currentText()
        if not (ep and self.file_path and self.sim_group_names):
            return

        # 1) Load data for the selected episode
        loaded = {}
        with h5py.File(self.file_path, 'r') as f:
            for sim in self.sim_group_names:
                try:
                    grp = f[sim]['episodes'][ep]
                except KeyError:
                    continue
                loaded[sim] = {ds: grp[ds][:] for ds in self.dataset_names}
        if not loaded:
            return

        # 2) Create a new figure with one subplot per series + cumulative
        n_plots = len(self.dataset_names) + 1
        fig, axes = plt.subplots(
            n_plots, 1,
            sharex=True,
            figsize=(12, 3 * n_plots),
        )

        # 3) Plot the first three series in black (no labels)
        for idx, (ax, ds, ylabel) in enumerate(zip(axes, self.dataset_names, self.y_axis_labels)):
            use_black = (idx < 3)
            for sim, data in loaded.items():
                y = data[ds]
                x = np.arange(len(y)) * self.time_step_min / (60 * 24)
                if ds == 'energy_series' or idx == 0:
                    y = y / 3600.0
                if np.issubdtype(y.dtype, np.integer) or set(np.unique(y)) <= {0, 1}:
                    ax.step(x, y, where='mid', color='black' if use_black else None)
                else:
                    ax.plot(x, y, color='black' if use_black else None)
            ax.set_ylabel(ylabel)
            ax.grid(True)
            # no titles, no legend here

        # 4) Plot cumulative flight-time with colored lines and labels
        cf_ax = axes[-1]
        for sim, data in loaded.items():
            flags = (data['actions'] != 0).astype(int)
            cum = np.cumsum(flags) * self.time_step_min / 60
            x = np.arange(len(cum)) * self.time_step_min / (60 * 24)
            cf_ax.plot(x, cum, label=self._legend_label(sim))
        cf_ax.set_ylabel('Total Flight (hrs)')
        cf_ax.set_xlabel('Time (days)')
        cf_ax.grid(True)

        # 5) Synchronize x-limits across all subplots
        n_stages = len(next(iter(loaded.values()))['actions'])
        t_max = n_stages * self.time_step_min / (60 * 24)  # in days
        for ax in axes:
            ax.set_xlim(0, t_max)

        # 6) Place legend on the bottom subplot
        handles, labels = cf_ax.get_legend_handles_labels()
        cf_ax.legend(
            handles, labels,
            loc='upper left',
            ncol=min(len(labels), 4),
            frameon=True
        )

        # 7) Adjust layout and display
        fig.tight_layout(rect=[0, 0, 1, 0.96])
        plt.show()

    def unselect_all_simulations(self):
        """Uncheck every item in the simulation list
        and trigger a plot update.
        """
        for i in range(self.sim_list_widget.count()):
            item = self.sim_list_widget.item(i)
            item.setCheckState(Qt.Unchecked)
        self.on_sim_selection_changed()


    def open_file(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select HDF5 File", "", "HDF5 files (*.h5 *.hdf5)"
        )
        if not path:
            return
        self.file_path = path
        self.file_line_edit.setText(path)
        self.load_simulations()
        self.load_episodes()

    def load_simulations(self):
        with h5py.File(self.file_path, "r") as f:
            groups = list(f.keys())
        self.sim_list_widget.clear()
        for sim in groups:
            item = QListWidgetItem(sim)
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            item.setCheckState(Qt.Checked)
            self.sim_list_widget.addItem(item)
        self.on_sim_selection_changed()

    def on_sim_selection_changed(self, _item=None):
        selected = []
        for i in range(self.sim_list_widget.count()):
            itm = self.sim_list_widget.item(i)
            if itm.checkState() == Qt.Checked:
                selected.append(itm.text())
        self.sim_group_names = selected
        if self.cb_episode.currentText():
            self.update_plot(self.cb_episode.currentText())

    def load_episodes(self):
        if not self.sim_group_names:
            return
        with h5py.File(self.file_path, 'r') as f:
            first = self.sim_group_names[0]
            episodes = list(f[first]['episodes'].keys())
        episodes = sorted(episodes, key=lambda s: int(s.split()[-1]))
        self.cb_episode.clear()
        self.cb_episode.addItems(episodes)
        if episodes:
            self.cb_episode.setCurrentText(episodes[0])
            self.update_plot(episodes[0])

    def on_window_size_change(self, new_size):
        self.window_size = new_size
        self.slider.setTickInterval(new_size)
        if hasattr(self, 'current_total_stages'):
            max_val = max(1, self.current_total_stages - self.window_size + 1)
            self.slider.setMaximum(max_val)
            if self.slider.value() > max_val:
                self.slider.setValue(max_val)
            self.on_window_slide(self.slider.value())

    def on_window_slide(self, start_stage):
        end_stage = start_stage + self.window_size - 1
        for ax in self.axes:
            ax.set_xlim(start_stage, end_stage)
        self.update_line_slider_range()
        self._draw_cursor(self.line_slider.value())
        self.canvas.draw()

    def rescale_cumulative_axis(self):
        cf_ax = self.axes[-1]
        x0, x1 = cf_ax.get_xlim()
        lines = cf_ax.get_lines()
        all_y = []
        for line in lines:
            x_data = line.get_xdata()
            y_data = line.get_ydata()
            mask = (x_data >= x0) & (x_data <= x1)
            all_y.append(y_data[mask])
        if all_y:
            ymin = min(y.min() for y in all_y)
            ymax = max(y.max() for y in all_y)
            cf_ax.set_ylim(ymin, ymax)
            self.canvas.draw()

    def _draw_cursor(self, stage):
        for ln in self.vline_refs:
            ln.remove()
        self.vline_refs.clear()
        if self.btn_toggle_line.isChecked():
            for ax in self.axes:
                ln = ax.axvline(stage, linestyle='--', linewidth=1, alpha=0.5)
                self.vline_refs.append(ln)

    def on_cursor_spin_change(self, stage):
        self.line_slider.blockSignals(True)
        self.line_slider.setValue(stage)
        self.line_slider.blockSignals(False)
        self._draw_cursor(stage)
        self.canvas.draw()

    def on_line_slide(self, stage):
        self.spin_cursor.blockSignals(True)
        self.spin_cursor.setValue(stage)
        self.spin_cursor.blockSignals(False)
        self._draw_cursor(stage)
        self.canvas.draw()

    def update_line_slider_range(self):
        start = self.slider.value()
        end = start + self.window_size - 1
        self.line_slider.blockSignals(True)
        self.line_slider.setMinimum(start)
        self.line_slider.setMaximum(end)
        if not (start <= self.line_slider.value() <= end):
            self.line_slider.setValue(start)
        self.line_slider.blockSignals(False)

        self.spin_cursor.blockSignals(True)
        self.spin_cursor.setMinimum(start)
        self.spin_cursor.setMaximum(end)
        if not (start <= self.spin_cursor.value() <= end):
            self.spin_cursor.setValue(start)
        self.spin_cursor.blockSignals(False)

    def update_plot(self, episode_name):
        for ax in self.axes:
            ax.clear()

        loaded = {}
        if not self.file_path or not self.sim_group_names:
            return
        with h5py.File(self.file_path, 'r') as f:
            for sim in self.sim_group_names:
                try:
                    grp = f[sim]['episodes'][episode_name]
                except KeyError:
                    continue
                loaded[sim] = {ds: grp[ds][:] for ds in self.dataset_names}

        if not loaded:
            return

        first = next(iter(loaded.values()))
        total = len(first['actions'])
        self.current_total_stages = total
        self.spin_window.setMaximum(total)
        self.slider.setMaximum(max(1, total - self.window_size + 1))

        for idx, (ax, ds, y_axis_label) in enumerate(zip(self.axes, self.dataset_names, self.y_axis_labels)):
            for jdx, (sim, data) in enumerate(loaded.items()):
                y = data[ds]
                # compute x in days
                x_days = np.arange(len(y)) * self.time_step_min / (60 * 24)
                if np.issubdtype(y.dtype, np.integer) or set(np.unique(y)).issubset({0, 1}):
                    ax.step(x_days, y, where='mid', label=self._legend_label(sim))
                elif idx == 0:
                    ax.plot(x_days, y/3600, color='black')
                elif idx == 3:
                    ax.plot(x_days, y/3600)
                elif idx == 1 or idx == 2:
                    ax.plot(x_days, y, color='black')
                else:
                    ax.plot(x_days, y, label=self._legend_label(sim))
            ax.set_ylabel(y_axis_label)

        # cumulative flight‐time subplot
        cf_ax = self.axes[-1]
        for sim, data in loaded.items():
            flight_flag = (data['actions'] != 0).astype(int)
            cum = np.cumsum(flight_flag) * self.time_step_min / 60  # hours
            x_days = np.arange(len(cum)) * self.time_step_min / (60 * 24)
            cf_ax.plot(x_days, cum, label=sim)
        cf_ax.set_ylabel('Total Flight (hrs)')
        cf_ax.set_xlabel('Time (days)')

        # only one legend, on the last axes, placed outside to the right
        cf_ax.legend(loc='upper left',
                    bbox_to_anchor=(1.02, 1.00),
                    borderaxespad=0,
                    frameon=True)

        # make room on the right for that legend
        self.fig.subplots_adjust(right=0.80)

        self.fig.suptitle(f"Episode {episode_name} across simulations")
        if not self.use_constrained_layout:
            self.fig.subplots_adjust(**self.layout_settings)
        self.canvas.draw()
        self.update_line_slider_range()
        self.fig.tight_layout(pad=2.0)
    @staticmethod
    def _legend_label(sim: str) -> str:
        # 1) Optimal
        if sim.lower().startswith("optimal"):
            return "Optimal Algorithm"

        # 2) Threshold sims use “_t{obs}_w{wind}” in their names
        m = re.search(r"_t(?P<obs>[\d\.]+)_w(?P<wind>[\d\.]+)", sim, re.IGNORECASE)
        if m:
            obs  = m.group("obs")
            wind = m.group("wind")
            return f"Threshold: Obs {obs}, Wind {wind}"

        # 3) Fallback: prettify the algorithm name only
        #    (drops the parameter suffixes)
        algo = sim.split("_")[0]
        return algo.replace("unifiedthresholdcontinuoussimulation",
                             "Unified Threshold Continuous Simulation")\
                   .replace("otheralgoname", "Other Algo Name")\
                   .title()

# ------------------------------------------------------------------------------
# 3) HDF5 Reward Plotter (from h5plotter.py)
# ------------------------------------------------------------------------------

class HDF5RewardPlotter:
    """Non‐GUI class to load a simulation‐result HDF5 file, aggregate summary, and offer various plots:
      • Mean Total Reward by (obs, wind)
      • Mean Failure Step by (obs, wind)
      • Failure Percentage by (obs, wind)
      • Reward vs Capacity, vs Horizon, vs Penalty
      • Failure Percentage / Step vs Penalty
      • Histograms of optimal‐policy rewards / failure steps by penalty
    """
    def __init__(self, file_path):
        self.file_path = file_path
        self.file = None
        self._summary = None
        self.opt_reward = None
        self.opt_failure_step = None
        self.opt_failure_pct = None

    # ========== Helper Methods for Plotting ==========

    def _filter_data(
        self,
        algorithms: list[str] | None = None,
        obs_thresholds: list[float] | None = None,
        wind_thresholds: list[float] | None = None,
        penalties: list[float] | None = None,
        include_optimal: bool = False
    ) -> tuple[pd.DataFrame, pd.DataFrame | None]:
        """Standard filtering applied to summary data.

        Args:
            algorithms (list[str], optional): List of algorithm/sim_type names to include.
            obs_thresholds (list[float], optional): List of observation thresholds to include.
            wind_thresholds (list[float], optional): List of wind thresholds to include.
            penalties (list[float], optional): List of failure penalties to include.
            include_optimal (bool, optional): If True, return filtered optimal data as well.

        Returns:
            tuple[pd.DataFrame, pd.DataFrame | None]: (filtered_main, filtered_optimal) dataframes
        """
        df = self._get_summary()
        df_opt = df[df['sim_type'].str.contains('optimal', case=False, na=False)]
        df_main = df[~df['sim_type'].str.contains('optimal', case=False, na=False)]

        if algorithms:
            df_main = df_main[df_main['sim_type'].isin(algorithms)]
        if obs_thresholds:
            df_main = df_main[df_main['observation_threshold'].isin(obs_thresholds)]
        if wind_thresholds:
            df_main = df_main[df_main['wind_threshold'].isin(wind_thresholds)]
        if penalties:
            df_main = df_main[df_main['failure_penalty'].isin(penalties)]
            df_opt = df_opt[df_opt['failure_penalty'].isin(penalties)]

        if df_main.empty:
            raise ValueError("No data after filtering. Check selections.")

        return df_main, df_opt if include_optimal else (df_main, None)

    def _create_subplot_grid(self, n_items: int, figsize_per_subplot=(5, 4)):
        """Create a square-ish grid of subplots.

        Args:
            n_items (int): Number of subplots needed.
            figsize_per_subplot (tuple, optional): (width, height) per subplot.

        Returns:
            tuple: (fig, axes) where axes is always a 2D array via squeeze=False
        """
        cols = int(np.ceil(np.sqrt(n_items)))
        rows = int(np.ceil(n_items / cols))

        w, h = figsize_per_subplot
        fig, axes = plt.subplots(
            rows, cols,
            figsize=(w * cols, h * rows),
            squeeze=False
        )

        # Delete unused subplots
        for idx in range(n_items, rows * cols):
            fig.delaxes(axes.flatten()[idx])

        return fig, axes

    def _add_optimal_baseline(
        self,
        ax,
        value: float,
        metric_name: str = "Reward",
        **plot_kwargs
    ):
        """Add optimal baseline to plot as a horizontal line.

        Args:
            ax (matplotlib.axes.Axes): The axes to add the baseline to.
            value (float): The optimal value to plot.
            metric_name (str, optional): Name of the metric for the label.
            **plot_kwargs: Additional keyword arguments passed to axhline.
        """
        if value is not None:
            defaults = {
                'linestyle': '--',
                'color': 'black',
                'label': f"Optimal Mean {metric_name} ({value:.3f})"
            }
            defaults.update(plot_kwargs)
            ax.axhline(value, **defaults)

    def _pivot_by_thresholds(
        self,
        df: pd.DataFrame,
        value_column: str
    ) -> pd.DataFrame:
        """Standard pivot: rows=obs_threshold, cols=wind_threshold.

        Args:
            df (pd.DataFrame): DataFrame containing threshold data.
            value_column (str): Column name to use as values in pivot table.

        Returns:
            pd.DataFrame: Pivoted data.
        """
        return df.pivot(
            index='observation_threshold',
            columns='wind_threshold',
            values=value_column
        )

    def _plot_threshold_lines(
        self,
        ax,
        pivot: pd.DataFrame,
        y_label: str = "Wind Threshold",
        **plot_kwargs
    ):
        """Plot a line for each wind threshold column in pivot table.

        Args:
            ax (matplotlib.axes.Axes): The axes to plot on.
            pivot (pd.DataFrame): Pivoted data with wind thresholds as columns.
            y_label (str, optional): Label prefix for legend entries.
            **plot_kwargs: Additional keyword arguments passed to plot.
        """
        for wind in pivot.columns:
            ax.plot(
                pivot.index,
                pivot[wind],
                label=f"{y_label} {wind} m/s",
                **plot_kwargs
            )

    def _create_plot_axes(self, canvas=None, figsize=(8, 6)):
        """Create axes, either on a Qt canvas or a new figure.

        Args:
            canvas (PlotCanvas, optional): Qt canvas to plot on. If None, creates new figure.
            figsize (tuple, optional): Figure size if creating new figure.

        Returns:
            matplotlib.axes.Axes: The axes to plot on.
        """
        if canvas:
            canvas.clear()
            return canvas.get_axes(111)
        else:
            fig, ax = plt.subplots(figsize=figsize)
            return ax

    # ========== End Helper Methods ==========

    def open_file(self):
        if self.file is None:
            self.file = h5py.File(self.file_path, 'r')

    def _load_summary(self):
        self.open_file()
        records = []
        for sim_group in self.file.keys():
            grp = self.file[sim_group]
            sim_type = grp.attrs.get('simulation_type', '')
            obs_t = grp.attrs.get('observation_threshold')
            wind_t = grp.attrs.get('wind_threshold')
            cap = grp.attrs.get('battery_capacity', grp.attrs.get('capacity', np.nan))
            horizon = grp.attrs.get('horizon', np.nan)
            fp = grp.attrs.get('failure_penalty', np.nan)
            loc_id = grp.attrs.get('location_id',np.nan)
            m = re.search(r'lat(?P<lat>[-\d\.]+)_lon(?P<lon>[-\d\.]+)', loc_id)
            if m:
                latitude  = float(m.group('lat'))
                longitude = float(m.group('lon'))
                print(f"Latitude: {latitude}, Longitude: {longitude}")
            else:
                raise ValueError(f"Could not parse coordinates from {loc_id}")
            if (obs_t is None or wind_t is None) and 'optimal' not in sim_type.lower():
                continue
            if obs_t is None:
                obs_t = np.nan
            if wind_t is None:
                wind_t = np.nan



            failure_percentage = grp.attrs.get('failure_percentage', np.nan)
            mean_reward = grp.attrs.get('average_reward', np.nan)
            mean_failure_step = grp.attrs.get('average_failure_step', np.nan)
            mean_total_flight_hours = grp.attrs.get('average_flight_hrs', np.nan)

            if np.isnan(mean_reward):
                rewards = []
                total_eps = 0
                fail_count = 0
                failure_steps = []

                episodes = grp.get('episodes', {})
                for ep in episodes.values():
                    if 'total_reward' in ep:
                        rewards.append(ep['total_reward'][()])
                    if 'failure' in ep and 'failure_step' in ep:
                        total_eps += 1
                        if bool(ep['failure'][()]):
                            fail_count += 1
                            failure_steps.append(ep['failure_step'][()])

                if not rewards and total_eps == 0 and 'optimal' not in sim_type.lower():
                    continue

                mean_reward = np.mean(rewards) if rewards else np.nan
                failure_percentage = (fail_count / total_eps * 100) if total_eps else np.nan
                mean_failure_step = np.mean(failure_steps) if failure_steps else np.nan

            records.append({
                'sim_type': sim_type,
                'observation_threshold': obs_t,
                'wind_threshold': wind_t,
                'battery_capacity': cap,
                'horizon': horizon,
                'failure_penalty': fp,
                'mean_reward': mean_reward,
                'failure_percentage': failure_percentage,
                'mean_failure_step': mean_failure_step,
                'latitude': latitude,
                'longitude': longitude,
                'average_flight_hrs': mean_total_flight_hours,
                'start_time': grp.attrs.get('start_time'),
            })

        df = pd.DataFrame(records)
        self._summary = df
        opt_df = df[df['sim_type'].str.contains('optimal', case=False, na=False)]
        if not opt_df.empty:
            self.opt_reward = opt_df['mean_reward'].mean()
            self.opt_failure_step = opt_df['mean_failure_step'].mean()
            self.opt_failure_pct = opt_df['failure_percentage'].mean()
        print(df)

    def _get_summary(self):
        if self._summary is None:
            self._load_summary()
        return self._summary

    def plot_mean_by_thresholds(
        self,
        algorithms: list[str] | None = None,
        obs_thresholds: list[float] | None = None,
        wind_thresholds: list[float] | None = None,
        canvas=None
    ):
        """Plot mean reward by threshold combinations.

        Args:
            algorithms (list[str], optional): List of algorithm names to include.
            obs_thresholds (list[float], optional): List of observation thresholds to include.
            wind_thresholds (list[float], optional): List of wind thresholds to include.
            canvas (PlotCanvas, optional): Qt canvas to embed plot in. If None, creates popup window.
        """
        df_main, _ = self._filter_data(algorithms, obs_thresholds, wind_thresholds)
        pivot = self._pivot_by_thresholds(df_main, 'mean_reward')

        ax = self._create_plot_axes(canvas, figsize=(8, 6))
        self._plot_threshold_lines(ax, pivot, marker='x')
        self._add_optimal_baseline(ax, self.opt_reward, "Reward")

        _finish_plot(
            ax,
            title="Mean Total Reward by Threshold Combination",
            xlabel="Observation Threshold",
            ylabel="Mean Total Reward",
            show=(canvas is None)
        )

        if canvas:
            canvas.draw()

    def plot_mean_failure_step_by_thresholds(
        self,
        algorithms: list[str] | None = None,
        obs_thresholds: list[float] | None = None,
        wind_thresholds: list[float] | None = None,
        canvas=None
    ):
        """Plot the mean failure step (time to failure) for each
        combination of observation and wind thresholds.

        Args:
            algorithms (list[str], optional): List of sim_type names to include (None = all non-optimal).
            obs_thresholds (list[float], optional): List of observation thresholds to include (None = all).
            wind_thresholds (list[float], optional): List of wind thresholds to include (None = all).
            canvas (PlotCanvas, optional): Qt canvas to embed plot in. If None, creates popup window.
        """
        df_main, _ = self._filter_data(algorithms, obs_thresholds, wind_thresholds)
        pivot = self._pivot_by_thresholds(df_main, 'mean_failure_step')

        ax = self._create_plot_axes(canvas, figsize=(8, 6))
        self._add_optimal_baseline(ax, self.opt_failure_step, "Failure Step")
        self._plot_threshold_lines(ax, pivot, marker='o')

        _finish_plot(
            ax,
            xlabel="Observation Threshold",
            ylabel="Mean Failure Step",
            show=(canvas is None)
        )

        if canvas:
            canvas.draw()


    def plot_failure_percentage_by_thresholds(
        self,
        algorithms: list[str] | None = None,
        obs_thresholds: list[float] | None = None,
        wind_thresholds: list[float] | None = None,
        canvas=None
    ):
        """Plot the failure percentage for each combination of observation and wind thresholds.

        Args:
            algorithms (list[str], optional): List of sim_type names to include (None = all non-optimal).
            obs_thresholds (list[float], optional): List of observation thresholds to include (None = all).
            wind_thresholds (list[float], optional): List of wind thresholds to include (None = all).
            canvas (PlotCanvas, optional): Qt canvas to embed plot in. If None, creates popup window.
        """
        df_main, _ = self._filter_data(algorithms, obs_thresholds, wind_thresholds)
        pivot = self._pivot_by_thresholds(df_main, 'failure_percentage')

        ax = self._create_plot_axes(canvas, figsize=(8, 6))
        self._plot_threshold_lines(ax, pivot, marker='s')
        self._add_optimal_baseline(ax, self.opt_failure_pct, "Failure %", label=f"Optimal Failure % ({self.opt_failure_pct:.1f})" if self.opt_failure_pct else None)

        _finish_plot(
            ax,
            xlabel="Observation Threshold",
            ylabel="Failure Percentage",
            show=(canvas is None)
        )

        if canvas:
            canvas.draw()

    def plot_reward_vs_capacity_by_thresholds(self, canvas=None):
        """Plot mean reward vs battery capacity for each observation threshold.

        Creates a grid of subplots, one for each observation threshold value.
        Each subplot shows how mean reward varies with battery capacity for
        different wind threshold values.

        Args:
            canvas (PlotCanvas, optional): Qt canvas to embed plot in. If None, creates popup window.
        """
        df_main, df_opt = self._filter_data(include_optimal=True)

        obs_vals = sorted(df_main['observation_threshold'].dropna().unique())
        wind_vals = sorted(df_main['wind_threshold'].dropna().unique())

        fig, axes = self._create_subplot_grid(len(obs_vals))

        for idx, obs in enumerate(obs_vals):
            ax = axes[idx // axes.shape[1]][idx % axes.shape[1]]
            subset = df_main[df_main['observation_threshold'] == obs]

            for w in wind_vals:
                series = subset[subset['wind_threshold'] == w]
                if series.empty:
                    continue
                series = series.sort_values('battery_capacity')
                ax.plot(series['battery_capacity'], series['mean_reward'],
                       marker='o', label=f"Wind {w}")

            if df_opt is not None and not df_opt.empty:
                opt_series = df_opt.dropna(subset=['battery_capacity', 'mean_reward'])
                opt_series = opt_series.sort_values('battery_capacity')
                ax.plot(opt_series['battery_capacity'], opt_series['mean_reward'],
                       linestyle='--', marker='s', label='Optimal')

            ax.set_title(f"Obs Threshold = {obs}")
            ax.set_xlabel("Battery Capacity")
            ax.set_ylabel("Mean Total Reward")
            ax.grid(True)
            ax.legend()

        plt.tight_layout()

        if canvas is None:
            plt.show()
        else:
            canvas.draw()

    def plot_reward_vs_horizon_by_thresholds(
        self,
        algorithms: list[str] | None = None,
        obs_thresholds: list[float] | None = None,
        wind_thresholds: list[float] | None = None,
        penalties: list[float] | None = None
    ):
        """Plot Average Reward per Timestep vs Days for each (obs, wind) threshold combination,
        optionally filtering by algorithm, thresholds, and failure penalties.
        """
        # 1) Load summary and split out optimal vs main
        df = self._get_summary()
        df_main = df[~df['sim_type'].str.contains('optimal', case=False, na=False)]
        df_opt  = df[df['sim_type'].str.contains('optimal', case=False, na=False)]

        # 2) Apply filters
        if algorithms:
            df_main = df_main[df_main['sim_type'].isin(algorithms)]
        if obs_thresholds:
            df_main = df_main[df_main['observation_threshold'].isin(obs_thresholds)]
        if wind_thresholds:
            df_main = df_main[df_main['wind_threshold'].isin(wind_thresholds)]
        if penalties:
            df_main = df_main[df_main['failure_penalty'].isin(penalties)]
            df_opt  = df_opt[df_opt['failure_penalty'].isin(penalties)]

        if df_main.empty:
            raise ValueError("No data left after filtering. Check your selections.")

        # 3) Time‐step conversion: assume 15 minutes per step
        minutes_per_step = 15
        steps_per_day     = 24 * 60 / minutes_per_step

        # 4) Identify unique thresholds
        obs_vals  = sorted(df_main['observation_threshold'].dropna().unique())
        wind_vals = sorted(df_main['wind_threshold'].dropna().unique())

        # 5) Prepare subplots grid
        n_cols = int(np.ceil(np.sqrt(len(obs_vals))))
        n_rows = int(np.ceil(len(obs_vals) / n_cols))
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows), squeeze=False)

        # 6) Plot each obs threshold as its own subplot
        for idx, obs in enumerate(obs_vals):
            ax = axes[idx // n_cols][idx % n_cols]
            subset = df_main[df_main['observation_threshold'] == obs]

            for w in wind_vals:
                series = subset[subset['wind_threshold'] == w]
                if series.empty:
                    continue

                # compute average reward per step
                avg_per_step = series['mean_reward'] / series['horizon']

                # convert horizon to days
                days = series['horizon'] / steps_per_day

                # sort by days
                order = days.argsort()
                ax.plot(
                    days.iloc[order],
                    avg_per_step.iloc[order],
                    marker='o',
                    label=f"Wind {w}"
                )

            # overlay optimal-policy baseline if available
            if not df_opt.empty:
                opt_avg = df_opt['mean_reward'] / df_opt['horizon']
                opt_days = df_opt['horizon'] / steps_per_day
                order = opt_days.argsort()
                ax.plot(
                    opt_days.iloc[order],
                    opt_avg.iloc[order],
                    linestyle='--',
                    marker='s',
                    label='Optimal'
                )

            ax.set_title(f"Obs Threshold = {obs}")
            ax.set_xlabel("Days")
            ax.set_ylabel("Average Reward per Timestep")
            ax.grid(True)
            ax.legend()

        # 7) Remove any unused axes
        total_plots = n_rows * n_cols
        for idx in range(len(obs_vals), total_plots):
            fig.delaxes(axes.flatten()[idx])

        plt.tight_layout()
        plt.show()


    def plot_metric_by_location(
        self,
        metric: str = "mean_reward",
        algorithms: list[str] | None = None,
        obs_thresholds: list[float] | None = None,
        wind_thresholds: list[float] | None = None,
        penalties: list[float] | None = None,
        battery_capacity: float = 350,
    ):
        """Line‐plot of <metric> vs latitude for a single battery capacity,
        with one line per (obs, wind) threshold combo plus the optimal policy.

        Args:
            battery_capacity: the capacity (Wh) to plot
            metric: one of the summary columns, e.g. "mean_reward",
                    "failure_percentage" or "mean_failure_step"
            algorithms: list of sim_type names to include (None = all non-optimal)
            obs_thresholds: list of obs thresholds to include (None = all)
            wind_thresholds: list of wind thresholds to include (None = all)
            penalties: list of failure_penalty values to include (None = all)
        """
        df = self._get_summary()

        # split into main and optimal runs
        df_main = df[~df['sim_type'].str.contains('optimal', case=False, na=False)]
        df_opt  = df[df['sim_type'].str.contains('optimal', case=False, na=False)]

        # 1) filter by the specified battery capacity
        df_main = df_main[df_main['battery_capacity'] == battery_capacity]
        df_opt  = df_opt[df_opt['battery_capacity'] == battery_capacity]

        # 2) apply the other filters
        if algorithms:
            df_main = df_main[df_main['sim_type'].isin(algorithms)]
        if obs_thresholds:
            df_main = df_main[df_main['observation_threshold'].isin(obs_thresholds)]
        if wind_thresholds:
            df_main = df_main[df_main['wind_threshold'].isin(wind_thresholds)]
        if penalties:
            df_main = df_main[df_main['failure_penalty'].isin(penalties)]
            df_opt  = df_opt[df_opt['failure_penalty'].isin(penalties)]

        if df_main.empty:
            raise ValueError(f"No data for capacity={battery_capacity} after filtering.")

        # 3) aggregate metric by latitude and threshold combo
        loc_df = (
            df_main
            .groupby(['latitude','observation_threshold','wind_threshold'])[metric]
            .mean()
            .reset_index()
        )
        loc_df['combo_label'] = loc_df.apply(
            lambda r: f"Obs {r['observation_threshold']}, Wind {r['wind_threshold']}",
            axis=1
        )

        # 4) pivot so index=latitude, columns=combo_label
        pivot = (
            loc_df
            .pivot(index='latitude', columns='combo_label', values=metric)
            .sort_index()
        )

        # 5) prepare optimal series (if present)
        opt_series = None
        if not df_opt.empty:
            opt_series = (
                df_opt
                .groupby('latitude')[metric]
                .mean()
                .reindex(pivot.index)
            )

        # 6) plot
        fig, ax = plt.subplots(figsize=(10, 6))
        for combo in pivot.columns:
            ax.plot(
                pivot.index,
                pivot[combo],
                marker='o',
                label=combo
            )

        if opt_series is not None:
            ax.plot(
                pivot.index,
                opt_series.values,
                linestyle='--',
                marker='s',
                label='Optimal Policy'
            )

        ax.set_xlabel("Latitude")
        ax.set_ylabel(metric.replace('_', ' ').title())
        ax.set_title(
            f"{metric.replace('_', ' ').title()} by Latitude\n"
            f"(Capacity = {battery_capacity} Wh)"
        )
        ax.legend(
            loc='upper left',
            bbox_to_anchor=(1.02, 1),
            borderaxespad=0
        )
        ax.grid(True)
        plt.subplots_adjust(right=0.75)
        plt.tight_layout()
        plt.show()

    def plot_metric_by_capacity_by_location(
            self,
            metric: str = "mean_reward",
            algorithms: list[str] | None = None,
            obs_thresholds: list[float] | None = None,
            wind_thresholds: list[float] | None = None,
            penalties: list[float] | None = None,
        ):
        """For each study location:
        1) Show a map of all locations numbered (with matching colors).
        2) Open a separate figure plotting <metric> vs. battery capacity
            for each location (one per figure), without legends.
        3) Finally, emit a standalone legend figure for the metric plots.
        """
        # 1) Load & filter
        df = self._get_summary()
        df_main = df[~df['sim_type'].str.contains('optimal', case=False, na=False)].copy()
        df_opt  = df[df['sim_type'].str.contains('optimal', case=False, na=False)].copy()
        if algorithms:
            df_main = df_main[df_main['sim_type'].isin(algorithms)]
        if obs_thresholds:
            df_main = df_main[df_main['observation_threshold'].isin(obs_thresholds)]
        if wind_thresholds:
            df_main = df_main[df_main['wind_threshold'].isin(wind_thresholds)]
        if penalties:
            df_main = df_main[df_main['failure_penalty'].isin(penalties)]
            df_opt  = df_opt[df_opt['failure_penalty'].isin(penalties)]
        if df_main.empty:
            raise ValueError("No data left after filtering for plotting.")

        # 2) Unique locations
        locs = (
            df_main[['latitude','longitude']]
            .drop_duplicates()
            .sort_values(['latitude','longitude'])
            .to_records(index=False)
        )

        # 3) Determine global combos & color cycle for metric plots
        combos_global = list(df_main[['observation_threshold','wind_threshold']]
                            .drop_duplicates()
                            .to_records(index=False))
        base_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

        # build legend handles/labels for metric combos + optimal
        from matplotlib.lines import Line2D
        legend_handles = []
        legend_labels  = []
        for i, (obs, w) in enumerate(combos_global):
            color = base_colors[i % len(base_colors)]
            legend_handles.append(Line2D([0], [0],
                                        marker='o', linestyle='',
                                        markersize=6, color=color))
            legend_labels.append(f"Obs {obs}, Wind {w}")
        if not df_opt.empty:
            legend_handles.append(Line2D([0], [0],
                                        linestyle='--', marker='s',
                                        markersize=6, color='black'))
            legend_labels.append("Optimal")

        # 4) Plot map of locations with matching colors
        colors_map = [base_colors[i % len(base_colors)] for i in range(len(locs))]
        fig_map = plt.figure(figsize=(10, 6))
        ax_map = fig_map.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
        ax_map.add_feature(cfeature.LAND.with_scale('50m'), facecolor='none', edgecolor='black')
        ax_map.add_feature(cfeature.COASTLINE.with_scale('50m'))
        for idx, (lat, lon) in enumerate(locs, start=1):
            ax_map.scatter(lon, lat, s=80,
                        transform=ccrs.PlateCarree(),
                        color=colors_map[idx-1])
        legend_entries_map = [
            Line2D([0], [0], marker='o', linestyle='',
                markersize=8, color=colors_map[i],
                label=f"({lat:.2f}, {lon:.2f})")
            for i, (lat, lon) in enumerate(locs)
        ]
        ax_map.legend(handles=legend_entries_map,
                    loc='upper left', bbox_to_anchor=(1.02, 1.0),
                    title='Locations (lat,lon)')
        ax_map.set_title("Study Locations")
        ax_map.set_extent([
            min(lon for _, lon in locs) - 5,
            max(lon for _, lon in locs) + 5,
            min(lat for lat, _ in locs) - 5,
            max(lat for lat, _ in locs) + 5
        ], crs=ccrs.PlateCarree())
        plt.tight_layout()
        plt.show()

        # 5) One separate figure per location (no legend)
        for idx, (lat, lon) in enumerate(locs, start=1):
            fig, ax = plt.subplots(figsize=(4, 4))
            sub = df_main[(df_main['latitude'] == lat) & (df_main['longitude'] == lon)]
            for j, (obs, w) in enumerate(combos_global):
                ser = (
                    sub[(sub['observation_threshold'] == obs) &
                        (sub['wind_threshold'] == w)]
                    .sort_values('battery_capacity')
                )
                ax.plot(ser['battery_capacity'],
                        ser[metric],
                        marker='o',
                        color=base_colors[j % len(base_colors)])
            opt_sub = df_opt[(df_opt['latitude'] == lat) &
                            (df_opt['longitude'] == lon)] \
                        .sort_values('battery_capacity')
            if not opt_sub.empty:
                ax.plot(opt_sub['battery_capacity'],
                        opt_sub[metric],
                        linestyle='--',
                        marker='s',
                        color='black')
            ax.set_title(f"Location ({lat:.2f}, {lon:.2f})")
            ax.set_xlabel("Battery Capacity (Wh)")
            ax.set_ylabel(metric.replace('_', ' ').title())
            ax.grid(True)
            plt.tight_layout()
            plt.show()

        # 6) Standalone legend figure
        fig_leg = plt.figure(figsize=(8, 2))
        fig_leg.legend(legend_handles, legend_labels,
                    loc='center', ncol=min(len(legend_labels), 4),
                    frameon=False)
        fig_leg.gca().axis('off')
        plt.tight_layout()
        plt.show()

    def plot_metric_by_start_date(
        self,
        metric: str = "mean_reward",
        algorithms: list[str] | None = None,
        obs_thresholds: list[float] | None = None,
        wind_thresholds: list[float] | None = None,
        penalties: list[float] | None = None,
        battery_capacity: float = 300,
    ):
        """Line‐plot of <metric> vs mission start date for a single battery capacity,
        with one line per (obs, wind) threshold combo plus the optimal policy.

        Supported metrics:
        • "mean_reward"
        • "mean_failure_step"   — plotted as % completed before failure
        • "failure_percentage"
        • "flight_hours_per_day"
        """
        df = self._get_summary()
        df['start_time'] = pd.to_datetime(df['start_time'])

        df_main = df[~df['sim_type'].str.contains('optimal', case=False)]
        df_opt  = df[ df['sim_type'].str.contains('optimal', case=False)]

        # apply filters...
        df_main = df_main[df_main['battery_capacity'] == battery_capacity]
        df_opt  = df_opt [df_opt ['battery_capacity'] == battery_capacity]
        if algorithms:
            df_main = df_main[df_main['sim_type'].isin(algorithms)]
        if obs_thresholds:
            df_main = df_main[df_main['observation_threshold'].isin(obs_thresholds)]
        if wind_thresholds:
            df_main = df_main[df_main['wind_threshold'].isin(wind_thresholds)]
        if penalties:
            df_main = df_main[df_main['failure_penalty'].isin(penalties)]
            df_opt  = df_opt [df_opt ['failure_penalty'].isin(penalties)]
        if df_main.empty:
            raise ValueError(f"No data for capacity={battery_capacity} after filtering.")

        # compute plot_val and ylabel
        minutes_per_step = 15
        steps_per_day    = 24 * 60 / minutes_per_step
        if metric == "mean_failure_step":
            df_main['plot_val'] = 100.0 * df_main['mean_failure_step'] / df_main['horizon']
            ylabel = "% Completed Before Failure"
        elif metric == "flight_hours_per_day":
            df_main['duration_days'] = df_main['horizon'] / steps_per_day
            df_main['plot_val'] = df_main['average_flight_hrs'] / df_main['duration_days']
            ylabel = "Flight Hours per Day"
        else:
            df_main['plot_val'] = df_main[metric]
            ylabel = metric.replace('_',' ').title()

        # prepare pivot
        df_main['start_date'] = df_main['start_time'].dt.normalize()
        df_main['combo'] = df_main.apply(
            lambda r: f"Obs {r['observation_threshold']}, Wind {r['wind_threshold']}",
            axis=1
        )
        pivot = (
            df_main
            .groupby(['start_date','combo'])['plot_val']
            .mean()
            .unstack('combo')
            .sort_index()
        )

        # optimal baseline
        opt_series = None
        if not df_opt.empty:
            df_opt['start_date'] = pd.to_datetime(df_opt['start_time']).dt.normalize()
            if metric == "mean_failure_step":
                df_opt['plot_val'] = 100.0 * df_opt['mean_failure_step'] / df_opt['horizon']
            elif metric == "flight_hours_per_day":
                df_opt['duration_days'] = df_opt['horizon'] / steps_per_day
                df_opt['plot_val'] = df_opt['average_flight_hrs'] / df_opt['duration_days']
            else:
                df_opt['plot_val'] = df_opt[metric]
            opt_series = (
                df_opt
                .groupby('start_date')['plot_val']
                .mean()
                .reindex(pivot.index)
            )

        # plotting
        fig, ax = plt.subplots(figsize=(6,6))
        for combo in pivot.columns:
            ax.plot(
                pivot.index,
                pivot[combo],
                marker='o',
                label=f"Threshold: {combo}"
            )

        if opt_series is not None:
            ax.plot(
                pivot.index,
                opt_series.values,
                linestyle='--',
                marker='s',
                color='k',
                label='Optimal'
            )

        ax.set_xlabel("Mission Start Date")
        ax.set_ylabel(ylabel)
        ax.set_title(f"{ylabel} vs Start Date (Capacity = {battery_capacity} Wh)")
        ax.legend(loc='best')
        ax.grid(True)

        # format the date axis
        ax.xaxis.set_major_locator(mdates.AutoDateLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
        fig.autofmt_xdate()

        plt.tight_layout()
        plt.show()

    def plot_metric_by_duration(
        self,
        metric: str = "mean_reward",
        algorithms: list[str] | None = None,
        obs_thresholds: list[float] | None = None,
        wind_thresholds: list[float] | None = None,
        penalties: list[float] | None = None,
        battery_capacity: float = 300,
    ):
        """Line‐plot of <metric> vs mission duration (days) for a single battery capacity,
        with one line per (obs, wind) threshold combo plus the optimal policy.

        Supported metrics:
          • "mean_reward"                  — plots average total reward
          • "mean_failure_step"            — plots % mission completed before failure
          • "flight_hours_per_day"         — uses stored average_flight_hours
        """
        # 1) Load summary and split main vs optimal
        df = self._get_summary()
        df_main = df[~df['sim_type'].str.contains('optimal', case=False)]
        df_opt  = df[ df['sim_type'].str.contains('optimal', case=False)]

        # 2) Filter by capacity, algorithms, thresholds, penalties
        df_main = df_main[df_main['battery_capacity'] == battery_capacity]
        df_opt  = df_opt [df_opt ['battery_capacity'] == battery_capacity]
        if algorithms:
            df_main = df_main[df_main['sim_type'].isin(algorithms)]
        if obs_thresholds:
            df_main = df_main[df_main['observation_threshold'].isin(obs_thresholds)]
        if wind_thresholds:
            df_main = df_main[df_main['wind_threshold'].isin(wind_thresholds)]
        if penalties:
            df_main = df_main[df_main['failure_penalty'].isin(penalties)]
            df_opt  = df_opt [df_opt ['failure_penalty'].isin(penalties)]
        if df_main.empty:
            raise ValueError(f"No data for capacity={battery_capacity} after filtering.")

        # 3) Constants for duration conversion
        minutes_per_step = 15
        steps_per_day    = 24 * 60 / minutes_per_step

        # Helper to convert failure step to percent
        def to_percent(step, horizon):
            return (step / horizon) * 100.0

        # --- Branch: flight_hours_per_day ---
        if metric == "flight_hours_per_day":
            tmp = df_main[['horizon','observation_threshold','wind_threshold','average_flight_hrs']].copy()
            tmp['duration_days'] = tmp['horizon'] / steps_per_day
            tmp['plot_val'] = tmp['average_flight_hrs'] / tmp['duration_days']
            tmp['combo_label'] = tmp.apply(
                lambda r: f"Obs {r['observation_threshold']}, Wind {r['wind_threshold']}",
                axis=1
            )
            pivot = tmp.pivot(index='duration_days', columns='combo_label', values='plot_val').sort_index()

            # optimal baseline
            opt_series = None
            if not df_opt.empty and 'average_flight_hrs' in df_opt:
                o = df_opt[['horizon','average_flight_hrs']].copy()
                o['duration_days'] = o['horizon'] / steps_per_day
                o['plot_val'] = o['average_flight_hrs'] / o['duration_days']
                opt_series = o.groupby('duration_days')['plot_val'].mean().reindex(pivot.index)

            # plot
            fig, ax = plt.subplots(figsize=(6, 6))
            for combo in pivot.columns:
                ax.plot(pivot.index, pivot[combo], marker='o', label=f"Threshold: {combo}")
            if opt_series is not None:
                ax.plot(opt_series.index, opt_series.values,
                        linestyle='--', marker='s', label='Optimal')
            ax.set_xlabel("Mission Duration (days)")
            ax.set_ylabel("Flight Hours per Day")
            ax.set_title(f"Flight Hours/Day by Mission Duration (Capacity = {battery_capacity} Wh)")
            ax.legend(loc='best')
            ax.grid(True)
            plt.subplots_adjust(right=0.75)
            plt.tight_layout()
            plt.show()
            return

        # --- Fallback: mean_reward or mean_failure_step ---
        tmp = df_main[['horizon','observation_threshold','wind_threshold', metric]].copy()
        tmp['duration_days'] = tmp['horizon'] / steps_per_day

        if metric == "mean_failure_step":
            tmp['plot_val'] = tmp.apply(lambda r: to_percent(r['mean_failure_step'], r['horizon']), axis=1)
            ylabel = "% Mission Completed Before Failure"
            title_metric = "% Completed Before Failure"
        else:
            tmp['plot_val'] = tmp[metric]
            ylabel = metric.replace('_',' ').title()
            title_metric = ylabel

        tmp['combo_label'] = tmp.apply(
            lambda r: f"Obs {r['observation_threshold']}, Wind {r['wind_threshold']}",
            axis=1
        )

        pivot = tmp.pivot(index='duration_days', columns='combo_label', values='plot_val').sort_index()

        # build optimal series
        opt_series = None
        if not df_opt.empty:
            o = df_opt[['horizon', metric]].copy()
            o['duration_days'] = o['horizon'] / steps_per_day
            if metric == "mean_failure_step":
                o['plot_val'] = o.apply(lambda r: to_percent(r['mean_failure_step'], r['horizon']), axis=1)
            else:
                o['plot_val'] = o[metric]
            opt_series = o.groupby('duration_days')['plot_val'].mean().reindex(pivot.index)

        # plot
        fig, ax = plt.subplots(figsize=(6, 6))
        for combo in pivot.columns:
            ax.plot(pivot.index, pivot[combo], marker='o', label=f"Threshold: {combo}")
        if opt_series is not None:
            ax.plot(opt_series.index, opt_series.values,
                    linestyle='--', marker='s', label='Optimal')

        ax.set_xlabel("Mission Duration (days)")
        ax.set_ylabel(ylabel)
        ax.set_title(f"{title_metric} by Mission Duration (Capacity = {battery_capacity} Wh)")
        ax.legend(loc='best')
        ax.grid(True)
        plt.subplots_adjust(right=0.75)
        plt.tight_layout()
        plt.show()

    def plot_reward_vs_penalty(self):
        df = self._get_summary()
        df_main = df[~df['sim_type'].str.contains('optimal', case=False, na=False)]
        df_opt  = df[df['sim_type'].str.contains('optimal', case=False, na=False)]

        if 'failure_penalty' not in df_main.columns:
            raise KeyError("No 'failure_penalty' column found in data")

        main_group = df_main.groupby('failure_penalty')['mean_reward'].mean()
        plt.figure()
        plt.plot(main_group.index, main_group.values, marker='o', label='Mean Reward')

        if not df_opt.empty:
            opt_group = df_opt.groupby('failure_penalty')['mean_reward'].mean()
            plt.plot(opt_group.index, opt_group.values,
                     linestyle='--', marker='s', label='Optimal Reward')

        plt.xlabel("Failure Penalty")
        plt.ylabel("Mean Total Reward")
        plt.title("Mean Total Reward vs Failure Penalty")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    def plot_failure_percentage_by_penalty(self, subplots=True):
        if subplots:
            self._plot_failure_percentage_by_penalty_subplots()
        else:
            self._plot_failure_percentage_by_penalty_single()

    def _plot_failure_percentage_by_penalty_subplots(self):
        df = self._get_summary()
        df_main = df[~df['sim_type'].str.contains('optimal', case=False, na=False)]
        df_opt  = df[df['sim_type'].str.contains('optimal', case=False, na=False)]
        fp_vals = sorted(df_main['failure_penalty'].dropna().unique())
        wind_vals = sorted(df_main['wind_threshold'].dropna().unique())
        n = len(fp_vals)
        cols = int(np.ceil(np.sqrt(n)))
        rows = int(np.ceil(n / cols))
        fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 4*rows), squeeze=False)

        opt_group = None
        if not df_opt.empty:
            opt_group = df_opt.groupby('failure_penalty')['failure_percentage'].mean()

        for idx, fp in enumerate(fp_vals):
            ax = axes[idx//cols][idx%cols]
            subset = df_main[df_main['failure_penalty'] == fp]
            pivot = subset.pivot(
                index='observation_threshold',
                columns='wind_threshold',
                values='failure_percentage'
            )
            for w in pivot.columns:
                ax.plot(pivot.index, pivot[w], marker='x', label=f"Wind {w}")
            if opt_group is not None and fp in opt_group.index:
                ax.axhline(opt_group[fp], linestyle='--', label=f"Optimal ({opt_group[fp]:.1f}%)")
            ax.set_title(f"Penalty = {fp}")
            ax.set_xlabel("Observation Threshold")
            ax.set_ylabel("Failure Percentage (%)")
            ax.grid(True)
            ax.legend()

        for idx in range(n, rows*cols):
            fig.delaxes(axes.flatten()[idx])

        plt.tight_layout()
        plt.show()

    def _plot_failure_percentage_by_penalty_single(self):
        df = self._get_summary()
        df_main = df[~df['sim_type'].str.contains('optimal', case=False, na=False)]
        df_opt  = df[df['sim_type'].str.contains('optimal', case=False, na=False)]

        combos = df_main[['observation_threshold', 'wind_threshold']].drop_duplicates()

        plt.figure()
        for _, combo in combos.iterrows():
            obs = combo['observation_threshold']
            wind = combo['wind_threshold']
            subset = df_main[
                (df_main['observation_threshold'] == obs) &
                (df_main['wind_threshold'] == wind)
            ]
            if subset.empty:
                continue
            series = subset.sort_values('failure_penalty')
            plt.plot(
                series['failure_penalty'],
                series['failure_percentage'],
                marker='o',
                label=f"Obs {obs}, Wind {wind}"
            )

        if not df_opt.empty:
            opt_series = df_opt.groupby('failure_penalty')['failure_percentage'].mean().reset_index()
            plt.plot(
                opt_series['failure_penalty'],
                opt_series['failure_percentage'],
                linestyle='--', marker='s',
                label='Optimal'
            )

        plt.xlabel("Failure Penalty")
        plt.ylabel("Failure Percentage (%)")
        plt.title("Failure Percentage vs Failure Penalty by Threshold Combination")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    def plot_failure_step_by_penalty(self, subplots=True):
        if subplots:
            self._plot_failure_step_by_penalty_subplots()
        else:
            self._plot_failure_step_by_penalty_single()

    def _plot_failure_step_by_penalty_subplots(self):
        df = self._get_summary()
        df_main = df[~df['sim_type'].str.contains('optimal', case=False, na=False)]
        df_opt  = df[df['sim_type'].str.contains('optimal', case=False, na=False)]
        fp_vals = sorted(df_main['failure_penalty'].dropna().unique())
        wind_vals = sorted(df_main['wind_threshold'].dropna().unique())
        n = len(fp_vals)
        cols = int(np.ceil(np.sqrt(n)))
        rows = int(np.ceil(n / cols))
        fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 4*rows), squeeze=False)

        opt_group = None
        if not df_opt.empty:
            opt_group = df_opt.groupby('failure_penalty')['mean_failure_step'].mean()

        for idx, fp in enumerate(fp_vals):
            ax = axes[idx//cols][idx%cols]
            subset = df_main[df_main['failure_penalty'] == fp]
            pivot = subset.pivot(
                index='observation_threshold',
                columns='wind_threshold',
                values='mean_failure_step'
            )
            for w in pivot.columns:
                ax.plot(pivot.index, pivot[w], marker='x', label=f"Wind {w}")
            if opt_group is not None and fp in opt_group.index:
                ax.axhline(opt_group[fp], linestyle='--', label=f"Optimal ({opt_group[fp]:.2f})")
            ax.set_title(f"Penalty = {fp}")
            ax.set_xlabel("Observation Threshold")
            ax.set_ylabel("Mean Failure Step")
            ax.grid(True)
            ax.legend()

        for idx in range(n, rows*cols):
            fig.delaxes(axes.flatten()[idx])

        plt.tight_layout()
        plt.show()

    def _plot_failure_step_by_penalty_single(self):
        df = self._get_summary()
        df_main = df[~df['sim_type'].str.contains('optimal', case=False, na=False)]
        df_opt  = df[df['sim_type'].str.contains('optimal', case=False, na=False)]

        combos = df_main[['observation_threshold', 'wind_threshold']].drop_duplicates()

        plt.figure()
        for _, combo in combos.iterrows():
            obs = combo['observation_threshold']
            wind = combo['wind_threshold']
            subset = df_main[
                (df_main['observation_threshold'] == obs) &
                (df_main['wind_threshold'] == wind)
            ]
            if subset.empty:
                continue
            series = subset.sort_values('failure_penalty')
            plt.plot(
                series['failure_penalty'],
                series['mean_failure_step'],
                marker='o',
                label=f"Obs {obs}, Wind {wind}"
            )

        if not df_opt.empty:
            opt_series = df_opt.groupby('failure_penalty')['mean_failure_step'].mean().reset_index()
            plt.plot(
                opt_series['failure_penalty'],
                opt_series['mean_failure_step'],
                linestyle='--', marker='s',
                label='Optimal'
            )

        plt.xlabel("Failure Penalty")
        plt.ylabel("Mean Failure Step")
        plt.title("Mean Failure Step vs Failure Penalty by Threshold Combination")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    def plot_optimal_reward_distribution_by_penalty(
        self,
        penalties=None,
        max_series=4,
        bins=50,
        subplots=False
    ):
        self.open_file()
        rewards_by_fp = {}
        for sim_group in self.file.keys():
            grp = self.file[sim_group]
            sim_type = grp.attrs.get('simulation_type', '')
            if 'optimal' not in sim_type.lower():
                continue
            fp = grp.attrs.get('failure_penalty', None)
            if fp is None:
                continue
            if penalties is not None and fp not in penalties:
                continue
            episodes = grp.get('episodes', {})
            for ep in episodes.values():
                if 'total_reward' in ep:
                    rewards_by_fp.setdefault(fp, []).append(ep['total_reward'][()])

        fps = sorted(rewards_by_fp.keys())
        fps = (fps if penalties else fps[:max_series])[:max_series]

        if subplots:
            self._plot_opt_reward_dist_subplots(rewards_by_fp, fps, bins)
        else:
            self._plot_opt_reward_dist_overlay(rewards_by_fp, fps, bins)

    def _plot_opt_reward_dist_overlay(self, rewards_by_fp, fps, bins):
        plt.figure()
        for fp in fps:
            data = rewards_by_fp.get(fp, [])
            if not data:
                continue
            plt.hist(data, bins=bins, alpha=0.5, label=f'Penalty {fp}')
        plt.xlabel('Total Reward')
        plt.ylabel('Episode Count')
        plt.title('Reward Distribution for Optimal Policy by Failure Penalty')
        plt.legend()
        plt.tight_layout()
        plt.show()

    def _plot_opt_reward_dist_subplots(self, rewards_by_fp, fps, bins):
        all_vals = [v for fp in fps for v in rewards_by_fp.get(fp, [])]
        if not all_vals:
            return
        xmin, xmax = min(all_vals), max(all_vals)

        n = len(fps)
        fig, axes = plt.subplots(n, 1, figsize=(6, 3 * n), sharex=True)
        if n == 1:
            axes = [axes]

        for idx, fp in enumerate(fps):
            ax = axes[idx]
            data = rewards_by_fp.get(fp, [])
            ax.hist(data, bins=bins, edgecolor='black')
            ax.set_xlim(xmin, xmax)
            ax.set_title(f'Penalty {fp}')
            ax.set_ylabel('Episode Count')
            ax.patch.set_alpha(0.3)
            ax.grid(True)

        axes[-1].set_xlabel('Total Reward')
        plt.tight_layout()
        plt.show()

    def plot_optimal_failure_step_distribution_by_penalty(
        self,
        penalties=None,
        max_series=4,
        bins=50,
        subplots=False
    ):
        self.open_file()
        steps_by_fp = {}
        for sim_group in self.file.keys():
            grp = self.file[sim_group]
            if 'optimal' not in grp.attrs.get('simulation_type', '').lower():
                continue
            fp = grp.attrs.get('failure_penalty', None)
            if fp is None:
                continue
            if penalties is not None and fp not in penalties:
                continue
            episodes = grp.get('episodes', {})
            for ep in episodes.values():
                if 'failure_step' in ep:
                    steps_by_fp.setdefault(fp, []).append(ep['failure_step'][()])

        fps = sorted(steps_by_fp.keys())
        fps = (fps if penalties else fps[:max_series])[:max_series]
        if subplots:
            self._plot_opt_failure_step_subplots(steps_by_fp, fps, bins)
        else:
            self._plot_opt_failure_step_overlay(steps_by_fp, fps, bins)

    def _plot_opt_failure_step_overlay(self, steps_by_fp, fps, bins):
        plt.figure()
        for fp in fps:
            data = steps_by_fp.get(fp, [])
            if not data:
                continue
            plt.hist(data, bins=bins, alpha=0.5, label=f'Penalty {fp}')
        plt.xlabel('Failure Step')
        plt.ylabel('Episode Count')
        plt.title('Failure Step Distribution for Optimal Policy by Penalty')
        plt.legend()
        plt.tight_layout()
        plt.show()

    def _plot_opt_failure_step_subplots(self, steps_by_fp, fps, bins):
        all_vals = [v for fp in fps for v in steps_by_fp.get(fp, [])]
        if not all_vals:
            return
        xmin, xmax = min(all_vals), max(all_vals)

        n = len(fps)
        fig, axes = plt.subplots(n, 1, figsize=(6, 3*n), sharex=True)
        if n == 1:
            axes = [axes]

        for idx, fp in enumerate(fps):
            ax = axes[idx]
            data = steps_by_fp.get(fp, [])
            if data:
                ax.hist(data, bins=bins, edgecolor='black')
            ax.set_xlim(xmin, xmax)
            ax.set_title(f'Penalty {fp}')
            ax.set_ylabel('Episode Count')
            ax.grid(True)

        axes[-1].set_xlabel('Failure Step')


class RewardPlotterTab(QWidget):
    """Tab for selecting an HDF5 file and invoking any of the HDF5RewardPlotter plots.

    Provides a dropdown + buttons for each plot type.
    """
    def __init__(self):
        super().__init__()
        self.file_path = None
        self.plotter = None
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)

        # File picker
        file_layout = QHBoxLayout()
        file_layout.addWidget(QLabel("HDF5:"))
        self.file_line = QLineEdit()
        self.file_line.setReadOnly(True)
        file_layout.addWidget(self.file_line)
        btn_browse = QPushButton("Open File…")
        btn_browse.clicked.connect(self.open_file)
        file_layout.addWidget(btn_browse)
        layout.addLayout(file_layout)

        # Export buttons
        export_layout = QHBoxLayout()
        self.btn_save_csv = QPushButton("Save Summary as CSV")
        self.btn_save_csv.clicked.connect(self.save_summary_csv)
        export_layout.addWidget(self.btn_save_csv)

        self.btn_export_latex = QPushButton("Export LaTeX Table…")
        self.btn_export_latex.clicked.connect(self.export_latex_table)
        export_layout.addWidget(self.btn_export_latex)
        layout.addLayout(export_layout)

        # Dropdown of available plot functions
        self.plot_combo = QComboBox()
        self.plot_combo.addItems([
            "Mean Reward by Thresholds",
            "Mean Failure Step by Thresholds",
            "Failure % by Thresholds",
            "Reward vs Capacity by Thresholds",
            "Reward vs Horizon by Thresholds",
            "Reward vs Penalty",
            "Failure % by Penalty (Subplots)",
            "Failure % by Penalty (Single)",
            "Failure Step by Penalty (Subplots)",
            "Failure Step by Penalty (Single)",
            "Optimal Reward Distribution by Penalty (Overlay)",
            "Optimal Reward Distribution by Penalty (Subplots)",
            "Optimal Failure Step Distribution by Penalty (Overlay)",
            "Optimal Failure Step Distribution by Penalty (Subplots)",
            "Metric by Location",
            "Metric by Duration",
            "Metric by Mission Start Date",
            "Metric by Penalty",
            "Metic vs Capacity by Location",
        ])
        layout.addWidget(QLabel("Select Plot Type:"))
        layout.addWidget(self.plot_combo)

        layout.addWidget(QLabel("Select Algorithms to Plot:"))
        self.algo_list = QListWidget()
        self.algo_list.setSelectionMode(QListWidget.MultiSelection)
        layout.addWidget(self.algo_list)

        # Obs‐threshold list
        layout.addWidget(QLabel("Observation Thresholds:"))
        self.obs_list = QListWidget()
        self.obs_list.setSelectionMode(QListWidget.MultiSelection)
        layout.addWidget(self.obs_list)

        # Wind‐threshold list
        layout.addWidget(QLabel("Wind Thresholds:"))
        self.wind_list = QListWidget()
        self.wind_list.setSelectionMode(QListWidget.MultiSelection)
        layout.addWidget(self.wind_list)

        # Optional: For the “distribution by penalty” charts, allow specifying a comma‐separated list of penalties
        self.penalty_input = QLineEdit()
        self.penalty_input.setPlaceholderText("Penalties (comma-separated, or leave blank)")
        layout.addWidget(self.penalty_input)

        # 1b) Metric selector
        layout.addWidget(QLabel("Select Metric:"))
        self.metric_combo = QComboBox()
        self.metric_combo.addItems([
            "mean_reward",
            "failure_percentage",
            "mean_failure_step",
            "flight_hours_per_day"
        ])
        layout.addWidget(self.metric_combo)

        # “Generate Plot” button
        btn_plot = QPushButton("Generate Plot")
        btn_plot.clicked.connect(self.generate_plot)
        layout.addWidget(btn_plot)

        layout.addStretch()

    def open_file(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select HDF5 File", "", "HDF5 files (*.h5 *.hdf5)"
        )
        if not path:
            return
        self.file_path = path
        self.file_line.setText(path)
        self.plotter = HDF5RewardPlotter(path)
        # populate algorithm list from summary
        df = self.plotter._get_summary()
        df_main = df[~df['sim_type'].str.contains('optimal', case=False, na=False)]
        sims = sorted(
            set(df['sim_type']) - {s for s in df['sim_type'] if 'optimal' in s.lower()}
        )
        self.algo_list.clear()
        for sim in sims:
            item = QListWidgetItem(sim)
            item.setSelected(True)
            self.algo_list.addItem(item)

        obs_vals  = sorted(df_main['observation_threshold'].unique())
        wind_vals = sorted(df_main['wind_threshold'].unique())
        self.obs_list.clear()
        for o in obs_vals:
            item = QListWidgetItem(f"{o:.2f}")
            item.setData(Qt.UserRole, o)
            item.setSelected(True)
            self.obs_list.addItem(item)

        self.wind_list.clear()
        for w in wind_vals:
            item = QListWidgetItem(f"{w:.2f}")
            item.setData(Qt.UserRole, w)
            item.setSelected(True)
            self.wind_list.addItem(item)

    def generate_plot(self):
        if self.plotter is None:
            QMessageBox.critical(self, "Error", "No HDF5 file selected.")
            return

        choice = self.plot_combo.currentText()
        penalties = None
        text = self.penalty_input.text().strip()
        algos = [i.text() for i in self.algo_list.selectedItems()]
        obs   = [i.data(Qt.UserRole) for i in self.obs_list.selectedItems()]
        wind  = [i.data(Qt.UserRole) for i in self.wind_list.selectedItems()]
        if text:
            try:
                penalties = [float(x.strip()) for x in text.split(",") if x.strip()]
            except:
                QMessageBox.critical(self, "Error", "Invalid penalty list.")
                return
        metric = self.metric_combo.currentText()
        try:
            if choice == "Mean Reward by Thresholds":
                self.plotter.plot_mean_by_thresholds(
                    algorithms     = algos,
                    obs_thresholds = obs,
                    wind_thresholds= wind
                )
            elif choice == "Mean Failure Step by Thresholds":
                self.plotter.plot_mean_failure_step_by_thresholds(
                    algorithms     = algos,
                    obs_thresholds = obs,
                    wind_thresholds= wind
                )
            
            elif choice == "Metric by Penalty":
                self.plotter.plot_metric_by_penalty(
                    metric         = metric,
                    algorithms     = algos,
                    obs_thresholds = obs,
                    wind_thresholds= wind,
                    # battery_capacity will default to 300 Wh unless you add a UI for it
                )
            elif choice == "Metric by Location":
                self.plotter.plot_metric_by_location(
                    metric         = metric,
                    algorithms     = algos,
                    obs_thresholds = obs,
                    wind_thresholds= wind,
                    penalties      = penalties
                )

            elif choice == "Metric by Duration":
                self.plotter.plot_metric_by_duration(
                    metric         = metric,
                    algorithms     = algos,
                    obs_thresholds = obs,
                    wind_thresholds= wind,
                    penalties      = penalties
                )

            elif choice == "Metric by Mission Start Date":
                self.plotter.plot_metric_by_start_date(
                    metric         = metric,
                    algorithms     = algos,
                    obs_thresholds = obs,
                    wind_thresholds= wind,
                    penalties      = penalties
                )

            elif choice == "Failure % by Thresholds":
                self.plotter.plot_failure_percentage_by_thresholds(
                    algorithms     = algos,
                    obs_thresholds = obs,
                    wind_thresholds= wind
                )
            elif choice == "Reward vs Capacity by Thresholds":
                self.plotter.plot_reward_vs_capacity_by_thresholds()

            elif choice == "Reward vs Horizon by Thresholds":
                self.plotter.plot_reward_vs_horizon_by_thresholds(
                    algorithms      = algos,
                    obs_thresholds  = obs,
                    wind_thresholds = wind,
                    penalties       = penalties
                )

            elif choice == "Reward vs Penalty":
                self.plotter.plot_reward_vs_penalty()
            elif choice == "Failure % by Penalty (Subplots)":
                self.plotter.plot_failure_percentage_by_penalty(subplots=True)
            elif choice == "Failure % by Penalty (Single)":
                self.plotter.plot_failure_percentage_by_penalty(subplots=False)
            elif choice == "Failure Step by Penalty (Subplots)":
                self.plotter.plot_failure_step_by_penalty(subplots=True)
            elif choice == "Failure Step by Penalty (Single)":
                self.plotter.plot_failure_step_by_penalty(subplots=False)
            elif choice == "Optimal Reward Distribution by Penalty (Overlay)":
                self.plotter.plot_optimal_reward_distribution_by_penalty(
                    penalties=penalties, subplots=False
                )
            elif choice == "Metic vs Capacity by Location":
                self.plotter.plot_metric_by_capacity_by_location(
                    metric         = metric,
                    algorithms     = algos,
                    obs_thresholds = obs,
                    wind_thresholds= wind,
                    penalties      = penalties
                )
            elif choice == "Optimal Reward Distribution by Penalty (Subplots)":
                self.plotter.plot_optimal_reward_distribution_by_penalty(
                    penalties=penalties, subplots=True
                )
            elif choice == "Optimal Failure Step Distribution by Penalty (Overlay)":
                self.plotter.plot_optimal_failure_step_distribution_by_penalty(
                    penalties=penalties, subplots=False
                )
            elif choice == "Optimal Failure Step Distribution by Penalty (Subplots)":
                self.plotter.plot_optimal_failure_step_distribution_by_penalty(
                    penalties=penalties, subplots=True
                )
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Plotting failed:\n{str(e)}")

    def save_summary_csv(self):
        if not getattr(self, "plotter", None):
            QMessageBox.warning(self, "No file selected", "Open an HDF5 file first.")
            return

        try:
            df = self.plotter._get_summary()
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load summary:\n{e}")
            return

        if df is None or df.empty:
            QMessageBox.warning(self, "No data", "No summary data is available to save.")
            return

        # Clean up start_time if it may be bytes from HDF5 attrs
        df_to_save = df.copy()
        if "start_time" in df_to_save.columns:
            def _clean_time(v):
                if isinstance(v, (bytes, bytearray, np.bytes_)):
                    try:
                        v = v.decode()
                    except Exception:
                        pass
                return v
            df_to_save["start_time"] = df_to_save["start_time"].map(_clean_time)
            # Optional: make it ISO if parseable
            df_to_save["start_time"] = pd.to_datetime(df_to_save["start_time"], errors="ignore")

        path, _ = QFileDialog.getSaveFileName(
            self, "Save Summary CSV", "summary.csv", "CSV files (*.csv)"
        )
        if not path:
            return

        try:
            df_to_save.to_csv(path, index=False)
            QMessageBox.information(self, "Saved", f"Summary saved to:\n{path}")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to save CSV:\n{e}")

    def export_latex_table(self):
        if not self.plotter:
            QMessageBox.warning(self, "No file selected", "Open an HDF5 file first.")
            return

        try:
            df = self.plotter._get_summary()
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load summary:\n{e}")
            return

        if df is None or df.empty:
            QMessageBox.warning(self, "No data", "No summary data is available to export.")
            return

        # Clean start_time if bytes
        if "start_time" in df.columns:
            def _clean_time(v):
                if isinstance(v, (bytes, bytearray, np.bytes_)):
                    try:
                        v = v.decode()
                    except Exception:
                        pass
                return v
            df["start_time"] = df["start_time"].map(_clean_time)

        # Select + rename columns for publication-ready table
        col_map = {
            "sim_type": "Sim",
            "battery_capacity": r"$C$ (Ah)",
            "horizon": r"$H$ (days)",
            "threshold": "Obs Thr.",
            "wind_threshold": "Wind Thr.",
            "failure_penalty": r"Penalty $\lambda$",
            "latitude": "Lat (\\textdegree)",
            "longitude": "Lon (\\textdegree)",
            "mean_reward": r"$\bar{R}$",
            "failure_percentage": r"$p_{\\mathrm{fail}}$ (\\%)",
        }

        df_out = df[list(col_map.keys())].rename(columns=col_map)

        # Rounding
        df_out[r"$C$ (Ah)"] = df_out[r"$C$ (Ah)"].round(0).astype("Int64")
        df_out[r"$H$ (days)"] = df_out[r"$H$ (days)"].round(0).astype("Int64")
        df_out["Obs Thr."] = df_out["Obs Thr."].round(2)
        df_out["Wind Thr."] = df_out["Wind Thr."].round(2)
        df_out[r"Penalty $\lambda$"] = df_out[r"Penalty $\lambda$"].round(2)
        df_out["Lat (\\textdegree)"] = df_out["Lat (\\textdegree)"].round(2)
        df_out["Lon (\\textdegree)"] = df_out["Lon (\\textdegree)"].round(2)
        df_out[r"$\bar{R}$"] = df_out[r"$\bar{R}$"].round(3)
        df_out[r"$p_{\\mathrm{fail}}$ (\\%)"] = (100 * df_out[r"$p_{\\mathrm{fail}}$ (\\%)"]).round(1)

        # Sort rows for consistency
        df_out = df_out.sort_values(
            by=["Sim", r"$C$ (Ah)", r"$H$ (days)", "Obs Thr.", "Wind Thr.", r"Penalty $\lambda$",
                "Lat (\\textdegree)", "Lon (\\textdegree)"],
            kind="stable"
        )

        path, _ = QFileDialog.getSaveFileName(
            self, "Export LaTeX Table", "summary_table.tex", "TeX files (*.tex)"
        )
        if not path:
            return

        try:
            latex_str = df_out.to_latex(index=False, escape=False, column_format="l" + "r"*(df_out.shape[1]-1))
            with open(path, "w", encoding="utf-8") as f:
                f.write(latex_str)
            QMessageBox.information(self, "Saved", f"LaTeX table saved to:\n{path}")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to save LaTeX table:\n{e}")

# ------------------------------------------------------------------------------
# 4) Combined Main Window
# ------------------------------------------------------------------------------

class CombinedGUI(QMainWindow):
    """Main window combining:
      • Simulation Runner / Config Creator
      • Multi‐Simulation Episode Inspector
      • HDF5 Reward Plotter
    Each as a top‐level tab.
    """
    def __init__(self):
        super().__init__()
        self.setWindowTitle("All‐In‐One Simulation Toolkit")
        self.resize(1000, 800)

        self.tabs = QTabWidget()
        self.setCentralWidget(self.tabs)

        # Tab 1: Episode Inspector
        inspector_widget = InspectorTab()
        self.tabs.addTab(inspector_widget, "Episode Inspector")

        # Tab 2: Reward Plotter
        reward_widget = RewardPlotterTab()
        self.tabs.addTab(reward_widget, "Reward Plotter")

def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    # Example dark style (optional)
    app.setStyleSheet("""
        QTabWidget::pane {
            border: 1px solid #444;
        }
        QWidget {
            font-size: 11pt;
        }
        QPushButton {
            padding: 5px 10px;
        }
    """)
    window = CombinedGUI()
    window.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()

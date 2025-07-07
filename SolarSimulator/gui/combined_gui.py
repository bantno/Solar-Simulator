import sys
import h5py
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
import matplotlib.dates as mdates

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

def _finish_plot(ax, title):
    # move legend outside on the right
    ax.legend(
        loc='upper left',
        bbox_to_anchor=(1.02, 1),
        borderaxespad=0
    )
    ax.set_title(title)
    ax.grid(True)

    # give room for the legend
    plt.subplots_adjust(right=0.75)
    plt.tight_layout()


class InspectorTab(QWidget):
    """
    Tab for inspecting simulation episodes stored in an HDF5 file.
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

    def unselect_all_simulations(self):
        """
        Uncheck every item in the simulation list
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
        if sim.lower().startswith("optimal"):
            return "Optimal"
        # look for two floats after “Threshold_”
        m = re.search(r"Threshold[_-]([\d\.]+)[_,-]([\d\.]+)", sim, re.IGNORECASE)
        if m:
            wind_th, obs_th = m.groups()
            return f"Threshold ({wind_th}, {obs_th})"
        # fallback
        return sim

# ------------------------------------------------------------------------------
# 3) HDF5 Reward Plotter (from h5plotter.py)
# ------------------------------------------------------------------------------

class HDF5RewardPlotter:
    """
    Non‐GUI class to load a simulation‐result HDF5 file, aggregate summary, and offer various plots:
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
        # opt_df = df[df['sim_type'].str.contains('optimal', case=False, na=False)]
        # # if not opt_df.empty:
        # #     self.opt_reward = opt_df['mean_reward'].mean()
        # #     self.opt_failure_step = opt_df['mean_failure_step'].mean()
        # #     self.opt_failure_pct = opt_df['failure_percentage'].mean()

    def _get_summary(self):
        if self._summary is None:
            self._load_summary()
        return self._summary

    def plot_mean_by_thresholds(
        self,
        algorithms: list[str] | None = None,
        obs_thresholds: list[float] | None = None,
        wind_thresholds: list[float] | None = None
    ):
        df = self._get_summary()
        df_main = df[~df['sim_type'].str.contains('optimal', case=False, na=False)]

        if algorithms:
            df_main = df_main[df_main['sim_type'].isin(algorithms)]

        if obs_thresholds:
            df_main = df_main[df_main['observation_threshold'].isin(obs_thresholds)]
        if wind_thresholds:
            df_main = df_main[df_main['wind_threshold'].isin(wind_thresholds)]

        if df_main.empty:
            raise ValueError("No data left after filtering. Check your selections.")

        pivot = df_main.pivot(
            index='observation_threshold',
            columns='wind_threshold',
            values='mean_reward'
        )

        fig, ax = plt.subplots(figsize=(8, 6))
        for w in pivot.columns:
            ax.plot(pivot.index, pivot[w], marker='x', label=f"Wind Threshold {w} m/s")

        if self.opt_reward is not None:
            ax.axhline(
                self.opt_reward,
                linestyle='--',
                label=f"Optimal Mean Reward ({self.opt_reward:.3f})"
            )

        ax.set_xlabel("Observation Threshold")
        ax.set_ylabel("Mean Total Reward")
        _finish_plot(ax, "Mean Total Reward by Threshold Combination")
        plt.show()

    def plot_mean_failure_step_by_thresholds(
        self,
        algorithms: list[str] | None = None,
        obs_thresholds: list[float] | None = None,
        wind_thresholds: list[float] | None = None
    ):
        """
        Plot the mean failure step (time to failure) for each
        combination of observation and wind thresholds.

        Parameters:
            algorithms: list of sim_type names to include (None = all non-optimal)
            obs_thresholds: list of observation thresholds to include (None = all)
            wind_thresholds: list of wind thresholds to include (None = all)
        """
        # 1) Load and filter out any 'optimal' simulations
        df = self._get_summary()
        df = df[~df['sim_type'].str.contains('optimal', case=False, na=False)]

        # 2) Apply user filters
        if algorithms:
            df = df[df['sim_type'].isin(algorithms)]
        if obs_thresholds:
            df = df[df['observation_threshold'].isin(obs_thresholds)]
        if wind_thresholds:
            df = df[df['wind_threshold'].isin(wind_thresholds)]

        if df.empty:
            raise ValueError("No data left after filtering. Check your selections.")

        # 3) Pivot into a matrix: rows=obs_thresh, cols=wind_thresh
        pivot = df.pivot(
            index='observation_threshold',
            columns='wind_threshold',
            values='mean_failure_step'
        )

        # 4) Create the plot
        fig, ax = plt.subplots(figsize=(8, 6))
        
        # 5) Optimal baseline
        if hasattr(self, 'opt_failure_step') and self.opt_failure_step is not None:
            ax.axhline(
                self.opt_failure_step,
                linestyle='--',
                color = "black",
                label=f"Optimal Mean Failure Step ({self.opt_failure_step:.1f})"
            )

        # 6) Label axes
        ax.set_xlabel("Observation Threshold")
        ax.set_ylabel("Mean Failure Step")
        
        for w in pivot.columns:
            ax.plot(
                pivot.index,
                pivot[w],
                marker='o',
                label=f"Wind Threshold {w} m/s"
            )

        # 7) Move legend outside to the right
        ax.legend(
            loc='upper left',         # anchor point for the legend
            bbox_to_anchor=(1.02, 1), # position just outside the axes
            borderaxespad=0
        )
        plt.subplots_adjust(right=0.75)  # make room on the right
        ax.grid(True)
        plt.tight_layout()
        plt.show()


    def plot_failure_percentage_by_thresholds(
        self,
        algorithms: list[str] | None = None,
        obs_thresholds: list[float] | None = None,
        wind_thresholds: list[float] | None = None
    ):
        """
        Plot the failure percentage for each combination of observation and wind thresholds.

        Parameters:
            algorithms: list of sim_type names to include (None = all non-optimal)
            obs_thresholds: list of observation thresholds to include (None = all)
            wind_thresholds: list of wind thresholds to include (None = all)
        """
        # 1) Load and filter out any 'optimal' simulations
        df = self._get_summary()
        df_opt = df[df['sim_type'].str.contains('optimal', case=False, na=False)]
        df = df[~df['sim_type'].str.contains('optimal', case=False, na=False)]

        # 2) Apply user filters
        if algorithms:
            df = df[df['sim_type'].isin(algorithms)]
        if obs_thresholds:
            df = df[df['observation_threshold'].isin(obs_thresholds)]
        if wind_thresholds:
            df = df[df['wind_threshold'].isin(wind_thresholds)]

        if df.empty:
            raise ValueError("No data left after filtering. Check your selections.")

        # 3) Pivot into a matrix: rows=obs_thresh, cols=wind_thresh
        pivot = df.pivot(
            index='observation_threshold',
            columns='wind_threshold',
            values='failure_percentage'
        )

        # 4) Create the plot
        fig, ax = plt.subplots(figsize=(8, 6))
        for w in pivot.columns:
            ax.plot(
                pivot.index,
                pivot[w],
                marker='s',
                label=f"Wind Threshold {w} m/s"
            )

        # 5) Optional optimal baseline
        failure_percentage = df_opt["failure_percentage"].values[0]
        if df_opt is not None:
            ax.axhline(
                failure_percentage,
                linestyle='--',
                label=f"Optimal Failure % ({failure_percentage:.1f})"
            )

        # 6) Label axes
        ax.set_xlabel("Observation Threshold")
        ax.set_ylabel("Failure Percentage")

        # 7) Move legend outside to the right
        ax.legend(
            loc='upper left',
            bbox_to_anchor=(1.02, 1),
            borderaxespad=0
        )
        plt.subplots_adjust(right=0.75)  # make room on the right
        ax.grid(True)
        plt.tight_layout()
        plt.show()

    def plot_reward_vs_capacity_by_thresholds(self):
        df = self._get_summary()
        df_main = df[~df['sim_type'].str.contains('optimal', case=False, na=False)]
        df_opt = df[df['sim_type'].str.contains('optimal', case=False, na=False)]

        obs_vals = sorted(df_main['observation_threshold'].dropna().unique())
        wind_vals = sorted(df_main['wind_threshold'].dropna().unique())

        n = len(obs_vals)
        cols = int(np.ceil(np.sqrt(n)))
        rows = int(np.ceil(n / cols))
        fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 4*rows), squeeze=False)

        for idx, obs in enumerate(obs_vals):
            ax = axes[idx//cols][idx%cols]
            subset = df_main[df_main['observation_threshold'] == obs]
            for w in wind_vals:
                series = subset[subset['wind_threshold'] == w]
                if series.empty:
                    continue
                series = series.sort_values('battery_capacity')
                ax.plot(series['battery_capacity'], series['mean_reward'], marker='o', label=f"Wind {w}")

            if not df_opt.empty:
                opt_series = df_opt.dropna(subset=['battery_capacity', 'mean_reward'])
                opt_series = opt_series.sort_values('battery_capacity')
                ax.plot(opt_series['battery_capacity'], opt_series['mean_reward'],
                        linestyle='--', marker='s', label='Optimal')

            ax.set_title(f"Obs Threshold = {obs}")
            ax.set_xlabel("Battery Capacity")
            ax.set_ylabel("Mean Total Reward")
            ax.grid(True)
            ax.legend()

        for idx in range(n, rows*cols):
            fig.delaxes(axes.flatten()[idx])

        plt.tight_layout()
        plt.show()

    def plot_reward_vs_horizon_by_thresholds(
        self,
        algorithms: list[str] | None = None,
        obs_thresholds: list[float] | None = None,
        wind_thresholds: list[float] | None = None,
        penalties: list[float] | None = None
    ):
        """
        Plot Average Reward per Timestep vs Days for each (obs, wind) threshold combination,
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
        """
        Line‐plot of <metric> vs latitude for a single battery capacity,
        with one line per (obs, wind) threshold combo plus the optimal policy.

        Parameters:
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

    def plot_metric_by_duration(
        self,
        metric: str = "mean_reward",
        algorithms: list[str] | None = None,
        obs_thresholds: list[float] | None = None,
        wind_thresholds: list[float] | None = None,
        penalties: list[float] | None = None,
        battery_capacity: float = 300,
    ):
        """
        Line‐plot of <metric> vs mission duration (days) for a single battery capacity,
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

    def plot_metric_by_penalty(
        self,
        metric: str = "mean_reward",
        algorithms: list[str] | None = None,
        obs_thresholds: list[float] | None = None,
        wind_thresholds: list[float] | None = None,
        battery_capacity: float = 300,
    ):
        """
        Line‐plot of <metric> vs failure penalty for a single battery capacity,
        with one line per (obs, wind) threshold combo plus the optimal policy.

        Supported metrics:
          • "mean_reward"                  — average total reward
          • "mean_failure_step"            — % mission completed before failure
          • "flight_hours_per_day"         — average flight‐hours per day
        """
        df = self._get_summary()
        # split into threshold and optimal runs
        df_main = df[~df['sim_type'].str.contains('optimal', case=False)]
        df_opt  = df[df['sim_type'].str.contains('optimal', case=False)]

        # 1) filter by capacity
        df_main = df_main[df_main['battery_capacity'] == battery_capacity]
        df_opt  = df_opt [df_opt ['battery_capacity'] == battery_capacity]

        # 2) apply other filters
        if algorithms:
            df_main = df_main[df_main['sim_type'].isin(algorithms)]
        if obs_thresholds:
            df_main = df_main[df_main['observation_threshold'].isin(obs_thresholds)]
            df_opt   = df_opt  [df_opt  ['observation_threshold'].isin(obs_thresholds)]
        if wind_thresholds:
            df_main = df_main[df_main['wind_threshold'].isin(wind_thresholds)]
            df_opt   = df_opt  [df_opt  ['wind_threshold'].isin(wind_thresholds)]

        if df_main.empty:
            raise ValueError(f"No data for capacity={battery_capacity} after filtering.")

        # 3) prepare the temporary DataFrame with plot_val and combo_label
        steps_per_day = 24 * 60 / 15  # for flight_hours_per_day case

        if metric == "flight_hours_per_day":
            tmp = df_main[['failure_penalty','average_flight_hrs','horizon',
                           'observation_threshold','wind_threshold']].copy()
            tmp['duration_days'] = tmp['horizon'] / steps_per_day
            tmp['plot_val'] = tmp['average_flight_hrs'] / tmp['duration_days']
            ylabel = "Flight Hours per Day"
        elif metric == "mean_failure_step":
            tmp = df_main[['failure_penalty','mean_failure_step','horizon',
                           'observation_threshold','wind_threshold']].copy()
            tmp['plot_val'] = tmp.apply(
                lambda r: (r['mean_failure_step'] / r['horizon']) * 100.0,
                axis=1
            )
            ylabel = "% Mission Completed Before Failure"
        else:  # mean_reward or any other raw metric
            tmp = df_main[['failure_penalty', metric,
                           'observation_threshold','wind_threshold']].copy()
            tmp['plot_val'] = tmp[metric]
            ylabel = metric.replace('_', ' ').title()

        tmp['combo_label'] = tmp.apply(
            lambda r: f"Obs {r['observation_threshold']}, Wind {r['wind_threshold']}",
            axis=1
        )

        pivot = tmp.pivot(
            index='failure_penalty',
            columns='combo_label',
            values='plot_val'
        ).sort_index()

        # 4) build the optimal‐policy baseline series (if any)
        opt_series = None
        if not df_opt.empty:
            if metric == "flight_hours_per_day":
                o = df_opt[['failure_penalty','average_flight_hrs','horizon']].copy()
                o['duration_days'] = o['horizon'] / steps_per_day
                o['plot_val'] = o['average_flight_hrs'] / o['duration_days']
            elif metric == "mean_failure_step":
                o = df_opt[['failure_penalty','mean_failure_step','horizon']].copy()
                o['plot_val'] = o.apply(
                    lambda r: (r['mean_failure_step'] / r['horizon']) * 100.0,
                    axis=1
                )
            else:
                o = df_opt[['failure_penalty', metric]].copy()
                o['plot_val'] = o[metric]

            opt_series = o.groupby('failure_penalty')['plot_val'] \
                          .mean().reindex(pivot.index)

        # 5) actually plot
        fig, ax = plt.subplots(figsize=(6, 6))
        for combo in pivot.columns:
            ax.plot(
                pivot.index,
                pivot[combo],
                marker='o',
                label=f"Threshold: {combo}"
            )

        if opt_series is not None:
            ax.plot(
                opt_series.index,
                opt_series.values,
                linestyle='--',
                marker='s',
                label='Optimal'
            )

        ax.set_xlabel("Failure Penalty")
        ax.set_ylabel(ylabel)
        ax.set_title(f"{ylabel} by Failure Penalty (Capacity = {battery_capacity} Wh)")
        ax.legend(loc='best')
        ax.grid(True)
        plt.tight_layout()
        plt.show()

    
    def plot_metric_surface_by_location(
        self,
        metric: str = "mean_reward",
        algorithms: list[str] | None = None,
        obs_thresholds: list[float] | None = None,
        wind_thresholds: list[float] | None = None,
        penalties: list[float] | None = None,
        battery_capacity: float = 350,
    ):
        """
        3D surface plots of <metric> vs (obs_threshold, wind_threshold) for each latitude,
        arranged in rows of up to 3 subplots. Overlays the optimal‐policy as a flat plane.

        Parameters:
            metric:             summary column to plot (e.g. "mean_reward")
            algorithms:         list of non-optimal sim_types to include (None = all)
            obs_thresholds:     list of obs_thresholds to include (None = all)
            wind_thresholds:    list of wind_thresholds to include (None = all)
            penalties:          list of failure_penalty values to include (None = all)
            battery_capacity:   battery capacity (Wh) to restrict to
        """
        # --- 1) Load data & split optimal vs thresholds ---
        df = self._get_summary()
        df_main = df[~df['sim_type'].str.contains('optimal', case=False, na=False)].copy()
        df_opt  = df[ df['sim_type'].str.contains('optimal', case=False, na=False)].copy()

        # --- 2) Apply filters ---
        df_main = df_main[df_main['battery_capacity'] == battery_capacity]
        df_opt  = df_opt [df_opt ['battery_capacity'] == battery_capacity]

        if algorithms:
            df_main = df_main[df_main['sim_type'].isin(algorithms)]

        # thresholds lists
        if obs_thresholds:
            df_main = df_main[df_main['observation_threshold'].isin(obs_thresholds)]
        else:
            obs_thresholds = sorted(df_main['observation_threshold'].unique())

        if wind_thresholds:
            df_main = df_main[df_main['wind_threshold'].isin(wind_thresholds)]
        else:
            wind_thresholds = sorted(df_main['wind_threshold'].unique())

        if penalties:
            df_main = df_main[df_main['failure_penalty'].isin(penalties)]
            df_opt  = df_opt [df_opt ['failure_penalty'].isin(penalties)]

        if df_main.empty:
            raise ValueError("No data after filtering; check your thresholds & filters.")

        # --- 3) Prepare optimal‐policy lookup ---
        latitudes = sorted(df_main['latitude'].unique())
        opt_lookup = {}
        if not df_opt.empty:
            opt_series = (
                df_opt
                .groupby('latitude')[metric]
                .mean()
                .reindex(latitudes)
            )
            opt_lookup = opt_series.to_dict()

        # --- 4) Create subplot grid ---
        n_lat   = len(latitudes)
        n_cols  = min(3, n_lat)
        n_rows  = math.ceil(n_lat / n_cols)
        fig = plt.figure(figsize=(6 * n_cols, 5 * n_rows))

        for idx, lat in enumerate(latitudes):
            ax = fig.add_subplot(n_rows, n_cols, idx + 1, projection='3d')
            df_lat = df_main[df_main['latitude'] == lat]

            # 5) Pivot to obs×wind grid
            loc = (
                df_lat
                .groupby(['observation_threshold','wind_threshold'])[metric]
                .mean()
                .reset_index()
                .pivot(index='observation_threshold',
                    columns='wind_threshold',
                    values=metric)
                .reindex(index=obs_thresholds, columns=wind_thresholds)
            )

            OBS, WIND = np.meshgrid(obs_thresholds, wind_thresholds, indexing='ij')
            Z = loc.values  # shape (len(obs), len(wind))

            # 6) Plot threshold‐policy surface
            surf = ax.plot_surface(
                OBS, WIND, Z,
                cmap='viridis',
                edgecolor='k',
                alpha=0.8,
                linewidth=0.5
            )

            # 7) Overlay optimal‐policy plane if available
            if lat in opt_lookup and not np.isnan(opt_lookup[lat]):
                z0     = opt_lookup[lat]
                Zplane = np.full_like(Z, z0)
                ax.plot_surface(
                    OBS, WIND, Zplane,
                    color='red',
                    alpha=0.3,
                    linewidth=0,
                    label='_nolegend_'
                )

            # 8) Labels & title
            ax.set_title(f"Latitude = {lat}")
            ax.set_xlabel("Obs threshold")
            ax.set_ylabel("Wind threshold")
            ax.set_zlabel(metric.replace('_',' ').title())
            ax.view_init(elev=25, azim=-60)

        # 9) Shared colorbar
        fig.colorbar(surf, ax=fig.get_axes(), shrink=0.6, aspect=20,
                    label=metric.replace('_',' ').title())
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
        plt.tight_layout()
        plt.show()

class RewardPlotterTab(QWidget):
    """
    Tab for selecting an HDF5 file and invoking any of the HDF5RewardPlotter plots.
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

# ------------------------------------------------------------------------------
# 4) Combined Main Window
# ------------------------------------------------------------------------------

class CombinedGUI(QMainWindow):
    """
    Main window combining:
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

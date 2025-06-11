import sys
import os
import multiprocessing
import yaml
import h5py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from cycler import cycler
from numpy.polynomial.legendre import leggauss

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
# 1) Simulation Runner / Config Creator (from gui.py)
# ------------------------------------------------------------------------------

def create_simulation_wrapper(args):
    factory, sim_type, cap, threshold, wind_threshold, save_history, full_history_episodes = args
    return factory.create_simulation(
        sim_type=sim_type,
        cap=cap,
        threshold=threshold,
        wind_threshold=wind_threshold,
        save_states=save_history,
        full_history_episodes=full_history_episodes
    )

class SimulationGUI(QWidget):
    """
    Qt‐based GUI for running simulations and creating YAML config files.
    Internal structure: a QTabWidget with two subtabs (“Run Simulation” and “Create Config”).
    """
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Simulation Runner")
        self.setMinimumSize(650, 450)
        self.init_ui()

    def init_ui(self):
        self.tabs = QTabWidget()
        self.tabs.addTab(self._build_run_tab(), "Run Simulation")
        self.tabs.addTab(self._build_config_tab(), "Create Config")

        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(10)
        main_layout.addWidget(self.tabs)
        self.setLayout(main_layout)

    def _build_run_tab(self):
        run_tab = QWidget()
        layout = QVBoxLayout()
        layout.setSpacing(10)
        layout.setContentsMargins(10, 10, 10, 10)

        # Config file picker
        config_layout = QHBoxLayout()
        self.config_label = QLabel("YAML Config File:")
        self.config_input = QLineEdit()
        self.browse_button = QPushButton("Browse")
        self.browse_button.clicked.connect(self.browse_file)
        config_layout.addWidget(self.config_label)
        config_layout.addWidget(self.config_input)
        config_layout.addWidget(self.browse_button)

        # Options
        self.multiproc_checkbox = QCheckBox("Use Multiprocessing")
        self.save_history_checkbox = QCheckBox("Save Full State Info")
        self.save_history_checkbox.setChecked(False)

        self.full_state_label = QLabel("Number of Full History Episodes:")
        self.full_state_input = QSpinBox()
        self.full_state_input.setRange(1, 100000)
        self.full_state_input.setValue(10)

        self.run_button = QPushButton("Run Simulation")
        self.run_button.clicked.connect(self.run_simulation)

        # Assemble
        layout.addLayout(config_layout)
        layout.addWidget(self.multiproc_checkbox)
        layout.addWidget(self.save_history_checkbox)
        layout.addWidget(self.full_state_label)
        layout.addWidget(self.full_state_input)
        layout.addWidget(self.run_button)

        run_tab.setLayout(layout)
        return run_tab

    def _build_config_tab(self):
        config_tab = QWidget()
        layout = QVBoxLayout()
        layout.setSpacing(10)
        layout.setContentsMargins(10, 10, 10, 10)

        # Output path
        self.output_path_input = QLineEdit()
        browse_out_button = QPushButton("Select Output Path")
        browse_out_button.clicked.connect(self.browse_output)

        # Input fields
        self.battery_input = QLineEdit("100, 200, 300")  # Wh
        self.threshold_input = QLineEdit("0.2, 0.4, 0.6")
        self.wind_input = QLineEdit("5, 10, 15")
        self.lat_input = QLineEdit("30, 32")
        self.lon_input = QLineEdit("-90, -88")
        self.horizons_input = QLineEdit("1000, 2000")
        self.episodes_input = QSpinBox()
        self.episodes_input.setRange(1, 100000)
        self.episodes_input.setValue(3000)

        self.start_date_input = QDateTimeEdit(self)
        self.start_date_input.setCalendarPopup(True)
        self.start_date_input.setDisplayFormat("yyyy-MM-dd HH:mm")
        dt = QDateTime.currentDateTime()
        dt.setTime(QTime(0, 0))
        self.start_date_input.setDateTime(dt)

        self.failure_penalty_input = QLineEdit("5, 10, 15")

        form_layout = QVBoxLayout()
        form_layout.setSpacing(8)
        form_layout.addWidget(QLabel("Start Date & Time:"))
        form_layout.addWidget(self.start_date_input)
        form_layout.addWidget(QLabel("Battery Capacities (Wh, comma-separated):"))
        form_layout.addWidget(self.battery_input)
        form_layout.addWidget(QLabel("Observation Thresholds (comma-separated):"))
        form_layout.addWidget(self.threshold_input)
        form_layout.addWidget(QLabel("Wind Thresholds (m/s, comma-separated):"))
        form_layout.addWidget(self.wind_input)
        form_layout.addWidget(QLabel("Latitudes (comma-separated):"))
        form_layout.addWidget(self.lat_input)
        form_layout.addWidget(QLabel("Longitudes (comma-separated):"))
        form_layout.addWidget(self.lon_input)
        form_layout.addWidget(QLabel("Horizons (time steps, comma-separated):"))
        form_layout.addWidget(self.horizons_input)
        form_layout.addWidget(QLabel("Failure Penalties (comma-separated):"))
        form_layout.addWidget(self.failure_penalty_input)
        form_layout.addWidget(QLabel("Episodes per Simulation:"))
        form_layout.addWidget(self.episodes_input)
        form_layout.addWidget(QLabel("Output YAML File Path:"))
        form_layout.addWidget(self.output_path_input)
        form_layout.addWidget(browse_out_button)

        export_button = QPushButton("Export Config File")
        export_button.clicked.connect(self.export_config)

        layout.addLayout(form_layout)
        layout.addWidget(export_button)
        config_tab.setLayout(layout)
        return config_tab

    def browse_file(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select YAML Config File", "", "YAML Files (*.yaml *.yml)"
        )
        if file_path:
            self.config_input.setText(file_path)

    def browse_output(self):
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save Config File", "config.yaml", "YAML Files (*.yaml *.yml)"
        )
        if file_path:
            self.output_path_input.setText(file_path)

    def run_simulation(self):
        config_path = self.config_input.text().strip()
        if not os.path.exists(config_path):
            QMessageBox.critical(self, "Error", "The selected config file does not exist.")
            return

        use_multiproc = self.multiproc_checkbox.isChecked()
        save_history = self.save_history_checkbox.isChecked()
        full_history_eps = self.full_state_input.value()

        try:
            runner = YAMLSimulationRunner(config_path)
            config = runner.config

            # Build parameter list including all locations and horizons
            param_list = runner._build_param_list()
            job_args = [(*args, save_history, full_history_eps) for args in param_list]

            if use_multiproc:
                with multiprocessing.Pool() as pool:
                    sims = pool.map(create_simulation_wrapper, job_args)
            else:
                sims = [create_simulation_wrapper(arg) for arg in job_args]

            manager = SimulationRunManager(
                episodes_per_simulation=config.get("episodes", full_history_eps),
                storage_dir="simulation_results"
            )
            manager.run_simulations(sims, use_multiprocessing=use_multiproc)

            QMessageBox.information(self, "Success", "Simulations completed successfully.")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Simulation failed with error:\n{str(e)}")

    def export_config(self):
        try:
            batteries = [float(x.strip()) for x in self.battery_input.text().split(",") if x.strip()]
            thresholds = [float(x.strip()) for x in self.threshold_input.text().split(",") if x.strip()]
            winds = [float(x.strip()) for x in self.wind_input.text().split(",") if x.strip()]
            penalties = [float(x.strip()) for x in self.failure_penalty_input.text().split(",") if x.strip()]
            episodes = int(self.episodes_input.value())
            horizons = [int(x.strip()) for x in self.horizons_input.text().split(",") if x.strip()]
            start_dt = self.start_date_input.dateTime().toString(Qt.ISODate)
            out_path = self.output_path_input.text().strip()

            lats = [float(x.strip()) for x in self.lat_input.text().split(",") if x.strip()]
            lons = [float(x.strip()) for x in self.lon_input.text().split(",") if x.strip()]
            if len(lats) != len(lons):
                QMessageBox.critical(self, "Error", "Must supply equal number of latitudes and longitudes.")
                return

            locations = []
            for lat, lon in zip(lats, lons):
                data_path = f"Data/EXPECTED_DATA/data_expected_lat{lat}_lon{lon}_15min.pkl"
                locations.append({
                    "latitude": lat,
                    "longitude": lon,
                    "data_path": data_path
                })

            config = {
                "start_datetime": start_dt,
                "battery_capacities": batteries,
                "threshold_values": thresholds,
                "wind_thresholds": winds,
                "horizons": horizons,
                "failure_penalties": penalties,
                "episodes": episodes,
                "transition_model": "moderate",
                "solar_panel_model": "constant",
                "locations": locations
            }

            with open(out_path, 'w') as f:
                yaml.dump(config, f)

            QMessageBox.information(self, "Success", f"Config file saved to: {out_path}")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to export config:\n{str(e)}")

# ------------------------------------------------------------------------------
# 2) Multi‐Simulation Episode Inspector (adapted from plot_states_gui.py)
# ------------------------------------------------------------------------------

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

        for idx, (ax, ds) in enumerate(zip(self.axes, self.dataset_names)):
            for sim, data in loaded.items():
                y = data[ds]
                x = np.arange(1, len(y) + 1)
                if np.issubdtype(y.dtype, np.integer) or set(np.unique(y)).issubset({0, 1}):
                    ax.step(x, y, where='mid', label=sim)
                else:
                    ax.plot(x, y, label=sim)
            ax.set_ylabel(ds)
            if idx == 0:
                ax.legend(loc='upper right', frameon=True)

        # Cumulative flight-time subplot
        cf_ax = self.axes[-1]
        for sim, data in loaded.items():
            flight_flag = (data['actions'] != 0).astype(int)
            cum = np.cumsum(flight_flag) * self.time_step_min / 60  # convert to hours
            stages = np.arange(1, len(cum) + 1)
            cf_ax.plot(stages, cum, label=sim)
        cf_ax.set_ylabel('Cumulative Flight Time (hours)')
        cf_ax.set_xlabel('Decision Stage')

        self.fig.suptitle(f"Episode {episode_name} across simulations")
        if not self.use_constrained_layout:
            self.fig.subplots_adjust(**self.layout_settings)
        self.canvas.draw()
        self.update_line_slider_range()

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

            if (obs_t is None or wind_t is None) and 'optimal' not in sim_type.lower():
                continue
            if obs_t is None:
                obs_t = np.nan
            if wind_t is None:
                wind_t = np.nan

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
                'mean_failure_step': mean_failure_step
            })

        df = pd.DataFrame(records)
        self._summary = df
        opt_df = df[df['sim_type'].str.contains('optimal', case=False, na=False)]
        if not opt_df.empty:
            self.opt_reward = opt_df['mean_reward'].mean()
            self.opt_failure_step = opt_df['mean_failure_step'].mean()
            self.opt_failure_pct = opt_df['failure_percentage'].mean()

    def _get_summary(self):
        if self._summary is None:
            self._load_summary()
        return self._summary

    def plot_mean_by_thresholds(self):
        df = self._get_summary()
        df_main = df[~df['sim_type'].str.contains('optimal', case=False, na=False)]
        pivot = df_main.pivot(
            index='observation_threshold',
            columns='wind_threshold',
            values='mean_reward'
        )

        plt.figure()
        for w in pivot.columns:
            plt.plot(pivot.index, pivot[w], marker='x', label=f"Wind {w}")

        if self.opt_reward is not None:
            plt.axhline(
                self.opt_reward,
                linestyle='--',
                label=f"Optimal Mean Reward ({self.opt_reward:.3f})"
            )

        plt.xlabel("Observation Threshold")
        plt.ylabel("Mean Total Reward")
        plt.title("Mean Total Reward for each (Obs, Wind) Threshold Combination")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    def plot_mean_failure_step_by_thresholds(self):
        df = self._get_summary()
        df_main = df[~df['sim_type'].str.contains('optimal', case=False, na=False)]
        pivot = df_main.pivot(
            index='observation_threshold',
            columns='wind_threshold',
            values='mean_failure_step'
        )

        plt.figure()
        for w in pivot.columns:
            plt.plot(pivot.index, pivot[w], marker='x', label=f"Wind {w}")

        if self.opt_failure_step is not None:
            plt.axhline(
                self.opt_failure_step,
                linestyle='--',
                label=f"Optimal Mean Failure Step ({self.opt_failure_step:.2f})"
            )

        plt.xlabel("Observation Threshold")
        plt.ylabel("Mean Failure Step")
        plt.title("Mean Failure Step by Threshold Combination")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    def plot_failure_percentage_by_thresholds(self):
        df = self._get_summary()
        df_main = df[~df['sim_type'].str.contains('optimal', case=False, na=False)]
        pivot = df_main.pivot(
            index='observation_threshold',
            columns='wind_threshold',
            values='failure_percentage'
        )

        plt.figure()
        for w in pivot.columns:
            plt.plot(pivot.index, pivot[w], marker='x', label=f"Wind {w}")

        if self.opt_failure_pct is not None:
            plt.axhline(
                self.opt_failure_pct,
                linestyle='--',
                label=f"Optimal Failure % ({self.opt_failure_pct:.1f}%)"
            )

        plt.xlabel("Observation Threshold")
        plt.ylabel("Failure Percentage (%)")
        plt.title("Failure Percentage by Threshold Combination")
        plt.legend()
        plt.grid(True)
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

    def plot_reward_vs_horizon_by_thresholds(self):
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
                series = series.sort_values('horizon')
                ax.plot(series['horizon'], series['mean_reward'], marker='o', label=f"Wind {w}")

            if not df_opt.empty:
                opt_series = df_opt.dropna(subset=['horizon', 'mean_reward'])
                opt_series = opt_series.sort_values('horizon')
                ax.plot(opt_series['horizon'], opt_series['mean_reward'],
                        linestyle='--', marker='s', label='Optimal')

            ax.set_title(f"Obs Threshold = {obs}")
            ax.set_xlabel("Horizon")
            ax.set_ylabel("Mean Total Reward")
            ax.grid(True)
            ax.legend()

        for idx in range(n, rows*cols):
            fig.delaxes(axes.flatten()[idx])

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
            "Optimal Failure Step Distribution by Penalty (Subplots)"
        ])
        layout.addWidget(QLabel("Select Plot Type:"))
        layout.addWidget(self.plot_combo)

        # Optional: For the “distribution by penalty” charts, allow specifying a comma‐separated list of penalties
        self.penalty_input = QLineEdit()
        self.penalty_input.setPlaceholderText("Penalties (comma-separated, or leave blank)")
        layout.addWidget(self.penalty_input)

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

    def generate_plot(self):
        if self.plotter is None:
            QMessageBox.critical(self, "Error", "No HDF5 file selected.")
            return

        choice = self.plot_combo.currentText()
        penalties = None
        text = self.penalty_input.text().strip()
        if text:
            try:
                penalties = [float(x.strip()) for x in text.split(",") if x.strip()]
            except:
                QMessageBox.critical(self, "Error", "Invalid penalty list.")
                return

        try:
            if choice == "Mean Reward by Thresholds":
                self.plotter.plot_mean_by_thresholds()
            elif choice == "Mean Failure Step by Thresholds":
                self.plotter.plot_mean_failure_step_by_thresholds()
            elif choice == "Failure % by Thresholds":
                self.plotter.plot_failure_percentage_by_thresholds()
            elif choice == "Reward vs Capacity by Thresholds":
                self.plotter.plot_reward_vs_capacity_by_thresholds()
            elif choice == "Reward vs Horizon by Thresholds":
                self.plotter.plot_reward_vs_horizon_by_thresholds()
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

        # Tab 1: Simulation Runner / Config
        sim_widget = SimulationGUI()
        self.tabs.addTab(sim_widget, "Simulation")

        # Tab 2: Episode Inspector
        inspector_widget = InspectorTab()
        self.tabs.addTab(inspector_widget, "Episode Inspector")

        # Tab 3: Reward Plotter
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

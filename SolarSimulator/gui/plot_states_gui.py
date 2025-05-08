import sys
import h5py
import numpy as np
import matplotlib.pyplot as plt
from cycler import cycler
from matplotlib.backends.backend_qt5agg import (
    FigureCanvasQTAgg as FigureCanvas,
    NavigationToolbar2QT as NavToolbar
)
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget,
    QVBoxLayout, QHBoxLayout, QLabel,
    QComboBox, QSizePolicy, QSlider,
    QSpinBox, QPushButton, QLineEdit,
    QFileDialog, QListWidget, QListWidgetItem
)
from PyQt5.QtCore import Qt


class MultiSimInspector(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Multi-Simulation Episode Inspector")
        self.resize(1000, 800)

        # ——— USER CONFIG ———
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
        self.style_name = 'seaborn-v0_8-darkgrid'
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
        self.color_cycle = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
        self.toolbar_enabled = True
        self.use_constrained_layout = False
        self.layout_settings = {
            'top':    0.92,
            'bottom': 0.08,
            'left':   0.10,
            'right':  0.95,
            'hspace': 0.3
        }
        # ——— Time & window settings ———
        self.time_step_min = 15    # minutes per decision stage
        self.window_size = 100     # default window size in stages
        self.vline_refs = []       # references to cursor lines

        # apply style & rcParams globally
        plt.style.use(self.style_name)
        plt.rcParams.update(self.rcparams)
        plt.rcParams['axes.prop_cycle'] = cycler('color', self.color_cycle)

        # UI setup
        central = QWidget()
        self.setCentralWidget(central)
        vlay = QVBoxLayout(central)

        # 1) File picker
        file_layout = QHBoxLayout()
        file_layout.addWidget(QLabel("HDF5:"))
        self.file_line_edit = QLineEdit()
        self.file_line_edit.setReadOnly(True)
        file_layout.addWidget(self.file_line_edit)
        btn_open = QPushButton("Open File…")
        btn_open.clicked.connect(self.open_file)
        file_layout.addWidget(btn_open)
        vlay.addLayout(file_layout)

        # 2) Simulation selection checklist
        vlay.addWidget(QLabel("Simulations to plot:"))
        self.sim_list_widget = QListWidget()
        self.sim_list_widget.itemChanged.connect(self.on_sim_selection_changed)
        vlay.addWidget(self.sim_list_widget)

        # 3) Episode selector
        ctl = QHBoxLayout()
        ctl.addWidget(QLabel("Episode:"))
        self.cb_episode = QComboBox()
        self.cb_episode.currentTextChanged.connect(self.update_plot)
        ctl.addWidget(self.cb_episode)
        ctl.addStretch()
        vlay.addLayout(ctl)

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
        vlay.addWidget(self.canvas)
        if self.toolbar_enabled:
            self.toolbar = NavToolbar(self.canvas, self)
            vlay.addWidget(self.toolbar)

        # 6) Window-size control
        ctl_window = QHBoxLayout()
        ctl_window.addWidget(QLabel("Window Size (stages):"))
        self.spin_window = QSpinBox()
        self.spin_window.setMinimum(1)
        self.spin_window.setMaximum(self.window_size)
        self.spin_window.setValue(self.window_size)
        self.spin_window.valueChanged.connect(self.on_window_size_change)
        ctl_window.addWidget(self.spin_window)
        ctl_window.addStretch()
        vlay.addLayout(ctl_window)

        # 7) Slider for panning window
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setMinimum(1)
        self.slider.setMaximum(1)
        self.slider.setValue(1)
        self.slider.setTickPosition(QSlider.TicksBelow)
        self.slider.setTickInterval(self.window_size)
        self.slider.valueChanged.connect(self.on_window_slide)
        vlay.addWidget(self.slider)

        # 8) Rescale button for cumulative flight-time axis
        self.btn_rescale = QPushButton("Rescale Flight-Time Axis")
        self.btn_rescale.clicked.connect(self.rescale_cumulative_axis)
        vlay.addWidget(self.btn_rescale)

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
        vlay.addLayout(ctl_line)

    def open_file(self):
        """Open an HDF5 file and load simulations & episodes."""
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
        """Read top-level groups from the HDF5 and populate check-list."""
        with h5py.File(self.file_path, "r") as f:
            groups = list(f.keys())
        self.sim_list_widget.clear()
        for sim in groups:
            item = QListWidgetItem(sim)
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            item.setCheckState(Qt.Checked)
            self.sim_list_widget.addItem(item)
        # initialize sim_group_names
        self.on_sim_selection_changed()

    def on_sim_selection_changed(self, _item=None):
        """Update self.sim_group_names based on checked items and replot."""
        selected = []
        for i in range(self.sim_list_widget.count()):
            itm = self.sim_list_widget.item(i)
            if itm.checkState() == Qt.Checked:
                selected.append(itm.text())
        self.sim_group_names = selected
        # replot current episode if set
        if self.cb_episode.currentText():
            self.update_plot(self.cb_episode.currentText())

    def load_episodes(self):
        """Populate episode list from the first selected simulation."""
        if not self.sim_group_names:
            return
        with h5py.File(self.file_path, 'r') as f:
            first = self.sim_group_names[0]
            # assume sub-group 'episodes'
            episodes = list(f[first]['episodes'].keys())
        episodes = sorted(episodes, key=lambda s: int(s.split()[-1]))
        self.cb_episode.clear()
        self.cb_episode.addItems(episodes)
        # trigger initial plot
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
        # redraw cursor if visible
        self._draw_cursor(self.line_slider.value())
        self.canvas.draw()

    def rescale_cumulative_axis(self):
        cf_ax = self.axes[-1]
        # find visible window data
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
        # remove old lines
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
        # clear all axes
        for ax in self.axes:
            ax.clear()

        # load data for selected simulations
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

        # update ranges
        first = next(iter(loaded.values()))
        total = len(first['actions'])
        self.current_total_stages = total
        self.spin_window.setMaximum(total)
        self.slider.setMaximum(max(1, total - self.window_size + 1))

        # plot each data series
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

        # cumulative flight-time subplot
        cf_ax = self.axes[-1]
        for sim, data in loaded.items():
            flight_flag = (data['actions'] != 0).astype(int)
            cum = np.cumsum(flight_flag) * self.time_step_min/60  # convert to hours
            stages = np.arange(1, len(cum) + 1)
            cf_ax.plot(stages, cum, label=sim)
        cf_ax.set_ylabel('Cumulative Flight Time (hours)')
        cf_ax.set_xlabel('Decision Stage')

        # finalize
        self.fig.suptitle(f"Episode {episode_name} across simulations")
        if not self.use_constrained_layout:
            self.fig.subplots_adjust(**self.layout_settings)
        self.canvas.draw()
        self.update_line_slider_range()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    w = MultiSimInspector()
    w.show()
    sys.exit(app.exec_())

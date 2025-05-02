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
    QComboBox, QSizePolicy, QSlider, QSpinBox
)
from PyQt5.QtCore import Qt

class MultiSimInspector(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Multi-Simulation Episode Inspector")
        self.resize(1000, 800)

        # ——— USER CONFIG ———
        self.file_path         = r'simulation_results\sim_1000_eps_20250430_022508.h5'
        self.sim_group_names   = [
            'optimalcontinuousanalyticalpolicysimulation_c640',
            'observationthresholdcontinuoussimulation_c640_t0.0_w8.0',
            'observationthresholdcontinuoussimulation_c640_t0.25_w8.0',
        ]
        self.dataset_names     = [
            'solar_series',
            'wind_series',
            'whale_series',
            'energy_series',
            'actions',
            'rewards',
        ]
        self.style_name        = 'seaborn-v0_8-darkgrid'
        self.rcparams          = {
            'font.size':        10,
            'axes.titlesize':   12,
            'axes.labelsize':   11,
            'lines.linewidth':  2,
            'figure.dpi':       120,
            'legend.fontsize':  10,
            'legend.frameon':    True,
            'legend.framealpha': 0.9,
            'legend.edgecolor':  'black',
        }
        self.color_cycle         = ['#1f77b4','#ff7f0e','#2ca02c','#d62728','#9467bd']
        self.toolbar_enabled     = True
        self.use_constrained_layout = False
        self.layout_settings    = {
            'top':    0.92,
            'bottom': 0.08,
            'left':   0.10,
            'right':  0.95,
            'hspace': 0.3
        }
        # ——— Time & window settings ———
        self.time_step_min      = 15    # minutes per decision stage
        self.window_size        = 100   # default window size in stages
        # ——————————————————————

        # apply style & rcParams globally
        plt.style.use(self.style_name)
        plt.rcParams.update(self.rcparams)
        plt.rcParams['axes.prop_cycle'] = cycler('color', self.color_cycle)

        # build episode list
        with h5py.File(self.file_path, 'r') as f:
            eps = set()
            for sim in self.sim_group_names:
                eps.update(f[sim]['episodes'].keys())
            self.episode_list = sorted(eps, key=lambda s: int(s.split()[-1]))

        # UI setup
        central = QWidget()
        self.setCentralWidget(central)
        vlay = QVBoxLayout(central)

        # Episode selector
        ctl = QHBoxLayout()
        ctl.addWidget(QLabel("Episode:"))
        self.cb_episode = QComboBox()
        self.cb_episode.addItems(self.episode_list)
        self.cb_episode.currentTextChanged.connect(self.update_plot)
        ctl.addWidget(self.cb_episode)
        ctl.addStretch()
        vlay.addLayout(ctl)

        # create subplots (+1 for cumulative flight-time)
        n_plots = len(self.dataset_names) + 1
        self.fig, self.axes = plt.subplots(
            n_plots, 1,
            sharex=True,
            figsize=(12, 3 * n_plots),
            constrained_layout=self.use_constrained_layout
        )
        if not self.use_constrained_layout:
            self.fig.subplots_adjust(**self.layout_settings)

        # Canvas & toolbar
        self.canvas = FigureCanvas(self.fig)
        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        vlay.addWidget(self.canvas)
        if self.toolbar_enabled:
            self.toolbar = NavToolbar(self.canvas, self)
            vlay.addWidget(self.toolbar)

        # Window-size control
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

        # Slider for panning window
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setMinimum(1)
        self.slider.setMaximum(1)  # will be set in update_plot
        self.slider.setValue(1)
        self.slider.setTickPosition(QSlider.TicksBelow)
        self.slider.setTickInterval(self.window_size)
        self.slider.valueChanged.connect(self.on_window_slide)
        vlay.addWidget(self.slider)

        # initial draw
        self.update_plot(self.episode_list[0])

    def on_window_size_change(self, new_size):
        """Adjust window size and update slider & view."""
        self.window_size = new_size
        self.slider.setTickInterval(new_size)
        if hasattr(self, 'current_total_stages'):
            max_val = max(1, self.current_total_stages - self.window_size + 1)
            self.slider.setMaximum(max_val)
            # clamp slider value into new range
            if self.slider.value() > max_val:
                self.slider.setValue(max_val)
            # re-draw with updated window
            self.on_window_slide(self.slider.value())

    def on_window_slide(self, start_stage):
        """Pan all subplots to the window [start_stage, start+window_size-1]."""
        end_stage = start_stage + self.window_size - 1
        for ax in self.axes:
            ax.set_xlim(start_stage, end_stage)
        self.canvas.draw()

    def update_plot(self, episode_name):
        # Clear all axes
        for ax in self.axes:
            ax.clear()

        # Load data for this episode
        loaded = {}
        with h5py.File(self.file_path, 'r') as f:
            for sim in self.sim_group_names:
                try:
                    grp = f[sim]['episodes'][episode_name]
                except KeyError:
                    continue
                loaded[sim] = {ds: grp[ds][:] for ds in self.dataset_names}

        if not loaded:
            return

        # Determine total stages and update slider + spinbox ranges
        first_sim = next(iter(loaded.values()))
        total_stages = len(first_sim['actions'])
        self.current_total_stages = total_stages

        # Update spinbox maximum
        self.spin_window.setMaximum(total_stages)

        # Update slider maximum
        max_slider = max(1, total_stages - self.window_size + 1)
        self.slider.setMaximum(max_slider)

        # Plot each series
        for idx, (ax, ds) in enumerate(zip(self.axes, self.dataset_names)):
            for sim, data in loaded.items():
                y = data[ds]
                x = np.arange(1, len(y) + 1)
                if np.issubdtype(y.dtype, np.integer) or set(np.unique(y)).issubset({0,1}):
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
            cum_minutes = np.cumsum(flight_flag) * self.time_step_min
            stages = np.arange(1, len(cum_minutes) + 1)
            cf_ax.plot(stages, cum_minutes, label=sim)
        cf_ax.set_ylabel('Cumulative Flight Time (min)')
        cf_ax.set_xlabel('Decision Stage')

        # Title & redraw
        self.fig.suptitle(f"Episode {episode_name} across simulations")
        if not self.use_constrained_layout:
            self.fig.subplots_adjust(**self.layout_settings)
        self.canvas.draw()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    w = MultiSimInspector()
    w.show()
    sys.exit(app.exec_())

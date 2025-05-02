import sys
import h5py
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from cycler import cycler
from matplotlib.backends.backend_qt5agg import (
    FigureCanvasQTAgg as FigureCanvas,
    NavigationToolbar2QT as NavToolbar
)
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget,
    QVBoxLayout, QHBoxLayout, QLabel,
    QComboBox, QSizePolicy
)

class MultiSimInspector(QMainWindow):
    def __init__(self):
        super().__init__()
        self.time_step_min = 15  # each stage = 15 minutes
        self.setWindowTitle("Multi‐Simulation Episode Inspector")
        self.resize(1000, 800)

        # ——— USER CONFIG ———
        # File & groups
        self.file_path       = r'simulation_results\sim_1000_eps_20250430_022508.h5'
        self.sim_group_names = [
            'optimalcontinuousanalyticalpolicysimulation_c640',
            'observationthresholdcontinuoussimulation_c640_t0.0_w8.0',
            'observationthresholdcontinuoussimulation_c640_t0.25_w8.0',
        ]
        self.dataset_names   = [
            'solar_series',
            'wind_series',
            'whale_series',
            'energy_series',
            'actions',
            'rewards',
        ]

        # Matplotlib style & rcParams
        self.style_name            = 'seaborn-v0_8-darkgrid'  # e.g. 'ggplot', 'Solarize_Light2'
        self.rcparams              = {
            'font.size':       10,
            'axes.titlesize':  12,
            'axes.labelsize':  11,
            'lines.linewidth': 2,
            'figure.dpi':      120,
            'legend.fontsize': 10,
                # Legend background
            'legend.frameon':   True,
            'legend.framealpha': 0.9,      # slightly translucent if you like
            'legend.edgecolor': 'black',
        }

        self.color_cycle           = ['#1f77b4','#ff7f0e','#2ca02c','#d62728','#9467bd']
        self.toolbar_enabled       = True

        # Layout: choose constrained_layout OR manual subplots_adjust
        self.use_constrained_layout = False
        self.layout_settings        = {
            'top': 0.92,
            'bottom': 0.08,
            'left': 0.10,
            'right': 0.95,
            'hspace': 0.3
        }
        # ————————————————

        # apply style & rcParams globally
        plt.style.use(self.style_name)
        plt.rcParams.update(self.rcparams)
        plt.rcParams['axes.prop_cycle'] = cycler('color', self.color_cycle)

        # build episode list (from any sim group that has them)
        with h5py.File(self.file_path, 'r') as f:
            eps = set()
            for sim in self.sim_group_names:
                eps.update(f[sim]['episodes'].keys())
            self.episode_list = sorted(eps, key=lambda s: int(s.split()[-1]))

        # UI setup
        central = QWidget()
        self.setCentralWidget(central)
        vlay = QVBoxLayout(central)

        # episode selector
        ctl = QHBoxLayout()
        ctl.addWidget(QLabel("Episode:"))
        self.cb_episode = QComboBox()
        self.cb_episode.addItems(self.episode_list)
        self.cb_episode.currentTextChanged.connect(self.update_plot)
        ctl.addWidget(self.cb_episode)
        ctl.addStretch()
        vlay.addLayout(ctl)

        # matplotlib canvas + optional toolbar
        n_plots = len(self.dataset_names) + 1
        self.fig, self.axes = plt.subplots(
            n_plots, 1,
            sharex=True,
            figsize=(12, 3 * n_plots),
            constrained_layout=self.use_constrained_layout
        )

        if not self.use_constrained_layout:
            self.fig.subplots_adjust(**self.layout_settings)

        self.canvas = FigureCanvas(self.fig)
        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        vlay.addWidget(self.canvas)

        if self.toolbar_enabled:
            self.toolbar = NavToolbar(self.canvas, self)
            vlay.addWidget(self.toolbar)

        # initial draw
        self.update_plot(self.episode_list[0])

    def update_plot(self, episode_name):
        # clear axes
        for ax in self.axes:
            ax.clear()

        # load data for this episode across all sims
        loaded = {}
        with h5py.File(self.file_path, 'r') as f:
            for sim in self.sim_group_names:
                try:
                    grp = f[sim]['episodes'][episode_name]
                except KeyError:
                    continue
                loaded[sim] = {ds: grp[ds][:] for ds in self.dataset_names}

        # plot
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

        self.axes[-1].set_xlabel("Decision Stage")

        cf_ax = self.axes[-1]
        for sim, data in loaded.items():
            flight_flag = (data['actions'] != 0).astype(int)
            cum_hours = np.cumsum(flight_flag) * self.time_step_min/60.
            stages = np.arange(1, len(cum_hours) + 1)
            cf_ax.plot(stages, cum_hours, label=sim)
        cf_ax.set_ylabel('Cumulative Flight Time (hours)')
        cf_ax.set_xlabel('Decision Stage')
        # cf_ax.legend(loc='upper right', frameon=True)

        self.fig.suptitle(f"Episode {episode_name} across simulations")
        if not self.use_constrained_layout:
            self.fig.subplots_adjust(**self.layout_settings)
        self.canvas.draw()

if __name__ == '__main__':
    print(plt.style.available)
    app = QApplication(sys.argv)
    w = MultiSimInspector()
    w.show()
    sys.exit(app.exec_())

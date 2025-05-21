import sys
import os
import multiprocessing
import yaml

from BaseClasses.run_sim import YAMLSimulationRunner
from BaseClasses.simulation_run_manager import SimulationRunManager
from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QLineEdit, QPushButton,
    QVBoxLayout, QHBoxLayout, QFileDialog, QMessageBox,
    QCheckBox, QTabWidget, QSpinBox, QDateTimeEdit,
)
from PyQt5.QtCore import QDateTime, Qt, QDate, QTime


def create_simulation_wrapper(args):
    """
    Wrapper function to create a simulation using a factory and parameters.

    Parameters:
        args (tuple): A tuple containing:
            - factory: Simulation factory instance
            - sim_type (str): Type of simulation to run
            - cap (float): Battery capacity
            - threshold (float): Observation threshold
            - wind_threshold (float): Wind speed threshold
            - save_history (bool): Flag to save full state info
            - full_history_episodes (int): Number of episodes to save full history

    Returns:
        Simulation: The created simulation instance
    """
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
    Qt-based graphical user interface for running and configuring simulation experiments.

    Provides two tabs:
      - Run Simulation: Select an existing YAML config and execute simulations.
      - Create Config: Build a new YAML config from user inputs.
    """
    def __init__(self):
        """
        Initialize the Simulation GUI window, set style, and build UI.
        """
        super().__init__()
        self.setWindowTitle("Simulation Runner")
        self.setMinimumSize(600, 400)
        self.init_ui()

    def init_ui(self):
        """
        Set up the main layout with tabs for running simulations and creating config files.
        """
        self.tabs = QTabWidget()
        self.tabs.addTab(self._build_run_tab(), "Run Simulation")
        self.tabs.addTab(self._build_config_tab(), "Create Config")

        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(10)
        main_layout.addWidget(self.tabs)
        self.setLayout(main_layout)

    def _build_run_tab(self):
        """
        Construct the 'Run Simulation' tab contents.

        Returns:
            QWidget: The tab widget containing controls to run simulations.
        """
        run_tab = QWidget()
        layout = QVBoxLayout()
        layout.setSpacing(10)
        layout.setContentsMargins(10, 10, 10, 10)

        config_layout = QHBoxLayout()
        self.config_label = QLabel("YAML Config File:")
        self.config_input = QLineEdit()
        self.browse_button = QPushButton("Browse")
        self.browse_button.clicked.connect(self.browse_file)
        config_layout.addWidget(self.config_label)
        config_layout.addWidget(self.config_input)
        config_layout.addWidget(self.browse_button)

        self.multiproc_checkbox = QCheckBox("Use Multiprocessing")
        self.save_history_checkbox = QCheckBox("Save Full State Info")
        self.save_history_checkbox.setChecked(False)

        self.full_state_label = QLabel("Number of Full History Episodes:")
        self.full_state_input = QSpinBox()
        self.full_state_input.setRange(1, 100000)
        self.full_state_input.setValue(10)

        self.run_button = QPushButton("Run Simulation")
        self.run_button.clicked.connect(self.run_simulation)

        layout.addLayout(config_layout)
        layout.addWidget(self.multiproc_checkbox)
        layout.addWidget(self.save_history_checkbox)
        layout.addWidget(self.full_state_label)
        layout.addWidget(self.full_state_input)
        layout.addWidget(self.run_button)
        run_tab.setLayout(layout)
        return run_tab

    def _build_config_tab(self):
        """
        Construct the 'Create Config' tab contents.

        Returns:
            QWidget: The tab widget containing controls to build configuration.
        """
        config_tab = QWidget()
        layout = QVBoxLayout()
        layout.setSpacing(10)
        layout.setContentsMargins(10, 10, 10, 10)

        self.output_path_input = QLineEdit()
        browse_out_button = QPushButton("Select Output Path")
        browse_out_button.clicked.connect(self.browse_output)

        # Inputs for configuration parameters
        self.battery_input = QLineEdit("100, 200, 300")  # Wh
        self.threshold_input = QLineEdit("0.2, 0.4, 0.6")
        self.wind_input = QLineEdit("5, 10, 15")
        self.lat_input = QLineEdit("30, 32")
        self.lon_input = QLineEdit("-90, -88")
        self.horizons_input = QLineEdit("1000, 2000")  # time steps list
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
        """
        Open a file dialog for selecting an existing YAML config file and
        update the config input field.
        """
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select YAML Config File", "", "YAML Files (*.yaml *.yml)"
        )
        if file_path:
            self.config_input.setText(file_path)

    def browse_output(self):
        """
        Open a file dialog for choosing where to save the new YAML config file and
        update the output path input field.
        """
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save Config File", "config.yaml", "YAML Files (*.yaml *.yml)"
        )
        if file_path:
            self.output_path_input.setText(file_path)

    def run_simulation(self):
        """
        Execute simulations based on the selected YAML configuration.

        Validates the config path, then builds simulation parameters,
        creates and runs simulations (optionally in parallel), and stores results.
        """
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

            # build parameter list including all locations and horizons
            param_list = runner._build_param_list()

            # prepare arguments for simulation creation
            job_args = [(*args, save_history, full_history_eps) for args in param_list]

            # create simulations (parallel or serial)
            if use_multiproc:
                with multiprocessing.Pool() as pool:
                    sims = pool.map(create_simulation_wrapper, job_args)
            else:
                sims = [create_simulation_wrapper(arg) for arg in job_args]

            # run and store
            total_episodes = config.get("episodes", full_history_eps)
            manager = SimulationRunManager(
                episodes_per_simulation=total_episodes,
                storage_dir="simulation_results"
            )
            manager.run_simulations(sims, use_multiprocessing=use_multiproc)

            QMessageBox.information(self, "Success", "Simulations completed successfully.")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Simulation failed with error:\n{str(e)}")

    def export_config(self):
        """
        Build a YAML configuration from user inputs and save it to the selected path.

        Parses comma-separated lists and validates input before writing the file.
        """
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


def main():
    """
    Entry point for the Simulation GUI application.

    Initializes QApplication, applies styling, and launches the GUI.
    """
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    app.setStyleSheet("""
        QWidget {
            font-size: 12pt;
            background-color: #121212;
            color: #f0f0f0;
        }
        QLineEdit, QSpinBox, QDateTimeEdit {
            padding: 5px;
            background-color: #1e1e1e;
            border: 1px solid #444;
            border-radius: 4px;
            color: #ffffff;
        }
        QPushButton {
            padding: 6px 12px;
            background-color: #2979FF;
            color: white;
            border: none;
            border-radius: 4px;
        }
        QCheckBox::indicator {
            width: 16px;
            height: 16px;
            border: 1px solid #888;
            background: #121212;
        }

        QCheckBox::indicator:checked {
            background: #2979FF;
            border: 1px solid #888;
        }
        QPushButton:hover {
            background-color: #448AFF;
        }
        QLabel {
            font-weight: bold;
        }
        QTabWidget::pane {
            border: 1px solid #444;
        }
    """)
    gui = SimulationGUI()
    gui.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
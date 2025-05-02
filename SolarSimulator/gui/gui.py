# pylint: disable=no-name-in-module

import sys
import os
import multiprocessing
import yaml

from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QLineEdit, QPushButton,
    QVBoxLayout, QHBoxLayout, QFileDialog, QMessageBox,
    QCheckBox, QTabWidget, QSpinBox, QDateTimeEdit
)
from PyQt5.QtCore import QDateTime, Qt
from BaseClasses.run_sim import YAMLSimulationRunner


def create_simulation_wrapper(args):
    # Unpack including save_history flag and full_history_episodes
    factory, sim_type, cap, threshold, wind_threshold, save_history, full_history_episodes = args
    # Pass save_states and full_history_episodes to factory
    return factory.create_simulation(
        sim_type,
        cap,
        threshold,
        wind_threshold,
        save_states=save_history,
        full_history_episodes=full_history_episodes
    )


class SimulationGUI(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Simulation Runner")
        self.setMinimumSize(600, 400)
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

        # Config file selector
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

        # Spinbox for number of full-history episodes
        self.full_state_label = QLabel("Number of Full History Episodes:")
        self.full_state_input = QSpinBox()
        self.full_state_input.setRange(1, 100000)
        self.full_state_input.setValue(3000)

        # Run button
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
        config_tab = QWidget()
        layout = QVBoxLayout()
        layout.setSpacing(10)
        layout.setContentsMargins(10, 10, 10, 10)

        # Output path selector
        self.output_path_input = QLineEdit()
        browse_out_button = QPushButton("Select Output Path")
        browse_out_button.clicked.connect(self.browse_output)

        # Config inputs
        self.battery_input = QLineEdit("100, 200, 300")
        self.threshold_input = QLineEdit("0.2, 0.4, 0.6")
        self.wind_input = QLineEdit("5, 10, 15")
        self.lat_input = QLineEdit("30")
        self.lon_input = QLineEdit("-90")
        self.horizon_input = QSpinBox()
        self.horizon_input.setRange(1, 100000)
        self.horizon_input.setValue(1000)
        self.episodes_input = QSpinBox()
        self.episodes_input.setRange(1, 100000)
        self.episodes_input.setValue(3000)

        # Start date/time selector
        self.start_date_input = QDateTimeEdit(self)
        self.start_date_input.setCalendarPopup(True)
        self.start_date_input.setDisplayFormat("yyyy-MM-dd HH:mm")
        self.start_date_input.setDateTime(QDateTime.currentDateTime())

        form_layout = QVBoxLayout()
        form_layout.setSpacing(8)
        form_layout.addWidget(QLabel("Start Date & Time:"))
        form_layout.addWidget(self.start_date_input)
        form_layout.addWidget(QLabel("Battery Capacities (Wh):"))
        form_layout.addWidget(self.battery_input)
        form_layout.addWidget(QLabel("Observation Thresholds:"))
        form_layout.addWidget(self.threshold_input)
        form_layout.addWidget(QLabel("Wind Thresholds (m/s):"))
        form_layout.addWidget(self.wind_input)
        form_layout.addWidget(QLabel("Latitude:"))
        form_layout.addWidget(self.lat_input)
        form_layout.addWidget(QLabel("Longitude:"))
        form_layout.addWidget(self.lon_input)
        form_layout.addWidget(QLabel("Horizon (time steps):"))
        form_layout.addWidget(self.horizon_input)
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
            # Load runner and parameters
            runner = YAMLSimulationRunner(config_path)
            param_list = runner._build_parameter_list()

            # Create simulation objects
            if use_multiproc:
                job_args = [
                    (runner.factory, *args, save_history, full_history_eps)
                    for args in param_list
                ]
                with multiprocessing.Pool() as pool:
                    simulations = pool.map(create_simulation_wrapper, job_args)
            else:
                simulations = [
                    runner.factory.create_simulation(
                        *args,
                        save_states=save_history,
                        full_history_episodes=full_history_eps
                    )
                    for args in param_list
                ]

            print(f"Created {len(simulations)} simulation objects.")

            # Determine total episodes from config, not full-history setting
            total_episodes = runner.config.get("episodes", full_history_eps)
            from BaseClasses.simulation_run_manager import SimulationRunManager
            manager = SimulationRunManager(
                episodes_per_simulation=total_episodes,
                storage_dir="simulation_results"
            )
            manager.run_simulations(simulations, use_multiprocessing=use_multiproc)

            QMessageBox.information(self, "Success", "Simulations completed successfully.")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Simulation failed with error:\n{str(e)}")

    def export_config(self):
        try:
            batteries = [float(b.strip()) for b in self.battery_input.text().split(",") if b.strip()]
            thresholds = [float(t.strip()) for t in self.threshold_input.text().split(",") if t.strip()]
            winds = [float(w.strip()) for w in self.wind_input.text().split(",") if w.strip()]
            episodes = int(self.episodes_input.value())
            horizon = int(self.horizon_input.value())
            latitude = float(self.lat_input.text().strip())
            longitude = float(self.lon_input.text().strip())
            start_dt = self.start_date_input.dateTime().toString(Qt.ISODate)
            out_path = self.output_path_input.text().strip()

            config = {
                "start_datetime": start_dt,
                "battery_capacities": batteries,
                "threshold_values": thresholds,
                "wind_thresholds": winds,
                "horizon": horizon,
                "episodes": episodes,
                "transition_model": "moderate",
                "solar_panel_model": "constant",
                "latitude": latitude,
                "longitude": longitude,
                "data_path": f"Data/EXPECTED_DATA/data_expected_lat{latitude}_lon{longitude}_15min.pkl"
            }

            with open(out_path, 'w') as f:
                yaml.dump(config, f)

            QMessageBox.information(self, "Success", f"Config file saved to: {out_path}")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to export config:\n{str(e)}")


def main():
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
            background: #2979FF;        /* matches your button color */
            border: 1px solid #888;  /* slightly lighter */
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

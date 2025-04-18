import sys
import os
import multiprocessing
from functools import partial
from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QLineEdit, QPushButton,
    QVBoxLayout, QHBoxLayout, QFileDialog, QComboBox, QMessageBox, QCheckBox
)
from PyQt5.QtCore import Qt
from BaseClasses.run_sim import YAMLSimulationRunner, SimulationFactory


def create_simulation_wrapper(args):
    factory, sim_type, cap, threshold, wind_threshold = args
    return factory.create_simulation(sim_type, cap, threshold, wind_threshold)


class SimulationGUI(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Simulation Runner")
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()

        # Config file selection
        config_layout = QHBoxLayout()
        self.config_label = QLabel("YAML Config File:")
        self.config_input = QLineEdit()
        self.browse_button = QPushButton("Browse")
        self.browse_button.clicked.connect(self.browse_file)
        config_layout.addWidget(self.config_label)
        config_layout.addWidget(self.config_input)
        config_layout.addWidget(self.browse_button)

        # Multiprocessing option
        self.multiproc_checkbox = QCheckBox("Use Multiprocessing")

        # Run button
        self.run_button = QPushButton("Run Simulation")
        self.run_button.clicked.connect(self.run_simulation)

        layout.addLayout(config_layout)
        layout.addWidget(self.multiproc_checkbox)
        layout.addWidget(self.run_button)

        self.setLayout(layout)

    def browse_file(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Select YAML Config File", "", "YAML Files (*.yaml *.yml)")
        if file_path:
            self.config_input.setText(file_path)

    def run_simulation(self):
        config_path = self.config_input.text().strip()
        if not os.path.exists(config_path):
            QMessageBox.critical(self, "Error", "The selected config file does not exist.")
            return

        use_multiproc = self.multiproc_checkbox.isChecked()
        try:
            runner = YAMLSimulationRunner(config_path)

            param_list = runner._build_parameter_list()
            job_args = [(runner.factory, *args) for args in param_list]

            if use_multiproc:
                with multiprocessing.Pool() as pool:
                    simulations = pool.map(create_simulation_wrapper, job_args)
            else:
                simulations = [runner.factory.create_simulation(*args) for args in param_list]

            print(f"Created {len(simulations)} simulation objects.")
            from BaseClasses.simulation_run_manager import SimulationRunManager
            manager = SimulationRunManager(
                episodes_per_simulation=runner.config.get("episodes", 3000),
                storage_dir="simulation_results"
            )
            manager.run_simulations(simulations, use_multiprocessing=use_multiproc)

            QMessageBox.information(self, "Success", "Simulations completed successfully.")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Simulation failed with error:\n{str(e)}")


def main():
    app = QApplication(sys.argv)
    gui = SimulationGUI()
    gui.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()

"""Simulation validation harness.

A thin, declarative driver around the existing SimulationFactory / SimulationRunManager
stack. One self-describing YAML fully describes an experiment; the CLI executes it and
collects the config copy, HDF5 data, a tidy summary CSV, and figures into a single
per-config, timestamped run directory.

See harness/README.md for the config schema and usage.
"""

HARNESS_VERSION = "0.1.0"

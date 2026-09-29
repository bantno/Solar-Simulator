# chain_vs_iid_sweep_v2

Rerun of the 2026-07-03 chain-vs-i.i.d. wind-persistence sweep (30 configs, 34 historical-weather
cells) with commit 779dc44: corrected solver survival term and seeded, paired crash draws.
Settings are identical to the July run (configs copied from its run directories); only the
data paths were made relative.

## Required data (gitignored, copy into `Data/`)

For each site `lat20.0_lon-159.0`, `lat30.0_lon-90.0`, `lat45.0_lon-100.0`, `lat58.0_lon-161.0`:

- `Data/EXPECTED_DATA/data_expected_<site>_15min.pkl`
- `Data/EXPECTED_DATA/data_expected_<site>_15min_windchain.pkl` (4 quantile bins)
- `Data/EXPECTED_DATA/data_expected_<site>_15min_histcube.pkl`
- the matching historical pickle in `Data/HISTORICAL_DATA/` (`data_20.0_-159.0.pkl`,
  `data_30_-90.pkl`, `data_45.0_-100.0.pkl`, `data_58.0_-161.0.pkl`)

If the windchain or histcube artifacts are present, provisioning builds nothing. If one is
missing it is rebuilt from the historical pickle, and a missing historical pickle is fetched
from Open-Meteo.

## Run (from `SolarSimulator/`, pvlib environment)

    python Scripts/run_chain_sweep.py --configs ../configs/chain_vs_iid_sweep_v2 --out ../results/chain_vs_iid_sweep_v2 --resume
    python Scripts/compare_chain_sweep.py --results ../results/chain_vs_iid_sweep_v2 --manifest ../configs/chain_vs_iid_sweep_v2/chain_vs_iid_sweep_manifest.json
    python Scripts/plot_chain_sweep.py --results ../results/chain_vs_iid_sweep_v2 --manifest ../configs/chain_vs_iid_sweep_v2/chain_vs_iid_sweep_manifest.json

`--workers N` sets the process count (default: CPU count - 1). Total runtime on a 16-thread
desktop was about 1.8 h for the July run; the threshold configs dominate.

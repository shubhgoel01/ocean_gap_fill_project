# ocean_gapfill_mc

Modular Python pipeline for ocean chlorophyll gap filling using:
- temporal compositing
- spatial regridding
- ordered linear interpolation
- full-dataset probabilistic model fitting
- full-dataset Monte Carlo reconstruction
- ensemble uncertainty estimation

## Overview

This project processes daily chlorophyll data and reconstructs values that remain missing after interpolation.

Important design point:
- The selected 4-5 debug cells are used only for reporting, plotting, and inspection.
- Model fitting, Monte Carlo reconstruction, and uncertainty estimation are performed on the whole dataset.

## Pipeline Overview

The main processing flow is:

1. load configuration
2. load daily chlorophyll data from NetCDF
3. convert daily data to 8-day composites
4. regrid data to a 1-degree latitude-longitude grid
5. inspect dataset shape and missing-data coverage
6. apply ordered interpolation:
   - longitude
   - latitude
   - time
7. inspect remaining missing values
8. select a small set of debug cells for reporting only
9. fit probability models for all remaining missing cells in the full dataset
10. extract selected-cell fit summaries for display
11. run Monte Carlo reconstruction for all remaining missing cells in the full dataset
12. build an ensemble of reconstructed datasets
13. compute full-dataset uncertainty maps
14. extract selected-cell Monte Carlo and uncertainty summaries
15. save outputs, plots, summaries, and logs

## Phase Descriptions

`Phase 1: Preprocessing`
- Load the source dataset.
- Convert daily observations into 8-day composites.
- Regrid the data to a common 1-degree grid.
- Inspect dataset shape and NaN percentage.

`Phase 2: Interpolation`
- Apply explicit neighbor-based interpolation in the required order:
  longitude, latitude, then time.
- This reduces missing values before probabilistic reconstruction.

`Phase 3: Full-Dataset Model Fitting`
- Identify all cells still missing after interpolation.
- For each missing cell, gather valid observations from the same spatial location across time.
- Fit candidate models:
  - normal
  - log-normal
  - gamma
  - KDE fallback when needed
- Save whole-dataset fitting summaries and unresolved-cell records.

`Phase 4: Full-Dataset Monte Carlo Reconstruction`
- Use the fitted model for each remaining missing cell.
- Generate `N` reconstructed datasets, where `N` comes from config.
- Preserve all originally available values.
- Only impute cells that remained missing after interpolation.

`Phase 5: Uncertainty Estimation`
- Use the reconstructed ensemble to compute full uncertainty maps:
  - mean
  - standard deviation
  - 5th percentile
  - 95th percentile
- Extract readable uncertainty summaries for selected debug cells.

`Phase 6: Reporting and Visualization`
- Selected debug cells are used only to:
  - show model-fit details
  - show Monte Carlo sample summaries
  - show uncertainty summaries
  - generate compact diagnostic plots
- Whole-dataset summary plots are also generated.

## Selected Debug Cells

Selected debug cells are:
- chosen only from cells still missing after interpolation
- sampled reproducibly using the configured random seed
- not used to limit or drive the main computation

They exist only to make the full-dataset results easier to inspect.

## Outputs

`outputs/logs/`
- pipeline log files

`outputs/summaries/`
- dataset inspection summaries
- interpolation summary
- model-fit summary for the full dataset
- unresolved-cell reports
- Monte Carlo reconstruction summary
- uncertainty summary

`outputs/sampled_cells/`
- selected debug cell coordinates
- selected-cell model-fit summaries
- selected-cell Monte Carlo summaries
- selected-cell uncertainty summaries

`outputs/plots/`
- selected-cell distribution plots
- selected-cell Monte Carlo plots
- selected-cell uncertainty plots
- whole-dataset NaN progression plot
- whole-dataset model-count summary plot

`outputs/reconstructed/`
- reconstructed ensemble datasets
- full uncertainty maps in NetCDF
- full model-fit records

## Assumptions

- Input data can be loaded from NetCDF with `xarray`.
- Chlorophyll values are non-negative.
- Log-normal and gamma fitting use only positive historical observations.
- Monte Carlo reconstruction preserves known values and only fills cells still missing after interpolation.
- Selected debug cells are reporting-only slices of full-dataset results.

## Limitations

- If a remaining missing cell has too little valid history, it may remain unresolved.
- KDE fallback depends on having enough valid and variable observations.
- The current implementation prioritizes readability and traceability over aggressive optimization.
- Whole-dataset Monte Carlo can still be computationally heavy for very large datasets, although the implementation uses a practical memory-aware reconstruction strategy.

## Quick Start

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python scripts/main.py --config configs/default.json
```

## Project Layout

- `configs/`: JSON configuration files
- `data/raw/`: source input files
- `data/processed/`: intermediate processed outputs
- `outputs/`: logs, summaries, plots, sampled-cell reports, reconstructed datasets
- `scripts/main.py`: command-line entrypoint
- `src/ocean_gapfill_mc/`: pipeline source code

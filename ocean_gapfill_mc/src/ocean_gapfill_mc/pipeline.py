"""Pipeline orchestration for the ocean gap-filling workflow."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import random

import numpy as np

from .data_loader import load_chlorophyll_data
from .distribution_fit import (
    extract_selected_fit_results,
    fit_all_missing_cell_distributions,
)
from .inspect_dataset import inspect_phase1_dataset
from .interpolation import apply_ordered_interpolation
from .monte_carlo import (
    extract_selected_monte_carlo_summaries,
    run_full_dataset_monte_carlo,
)
from .plotting import generate_pipeline_plots, save_pipeline_chlorophyll_datasets
from .select_cells import select_monte_carlo_filled_debug_cells
from .spatial_regrid import regrid_to_target_latlon
from .uncertainty import calculate_uncertainty_statistics
from .utils.config import load_config
from .utils.io import ensure_directories
from .utils.logging_utils import configure_logging, get_logger

# This creates a compact mask showing whether a raw observation existed.
def build_raw_observation_support_mask(raw_data):
    """Build a compact mask showing where raw observations existed before gap filling."""
    support_mask = raw_data.notnull().astype(np.uint8)
    support_mask.name = "raw_observation_support"
    return support_mask

# This function takes input the data array and label for the stage, it calculates the total number of cells, number of NaN cells, and percentage of NaN cells.
def summarize_nan_stage(data_array, label: str) -> dict:
    """Return a compact NaN summary for one pipeline stage."""
    total_cells = int(data_array.values.size)
    nan_cells = int(np.isnan(data_array.values).sum())
    nan_percent = float((nan_cells / total_cells) * 100.0) if total_cells else 0.0
    return {
        "label": label,
        "nan_cells": nan_cells,
        "total_cells": total_cells,
        "nan_percent": round(nan_percent, 4),
    }

# this function simply calls summarize_nan_stage to get the NaN summary and then logs it using the provided logger.
# Instead of just computing stats silently, it records them in log files. So later you can inspect stage-by-stage progress.
def log_nan_stage(logger, data_array, label: str) -> dict:
    """Log and return stage-wise NaN counts."""
    summary = summarize_nan_stage(data_array, label)
    logger.info(
        "NaN status after %s: %s/%s cells missing (%.4f%%)",
        label,
        summary["nan_cells"],
        summary["total_cells"],
        summary["nan_percent"],
    )
    return summary


def extract_interpolation_summary(data_array) -> dict:
    """Return the interpolation summary as a dict regardless of attr storage format."""
    raw_summary = data_array.attrs.get("interpolation_summary", {})
    if isinstance(raw_summary, str):
        try:
            return json.loads(raw_summary)
        except json.JSONDecodeError:
            return {"raw_interpolation_summary": raw_summary}
    if isinstance(raw_summary, dict):
        return raw_summary
    return {"raw_interpolation_summary": raw_summary}


def generate_annual_cycle_plots_from_script(config) -> dict[str, str]:
    """Run the script-based regional annual-cycle plotting step."""
    if not config.config_directory:
        logger = get_logger(__name__)
        logger.warning("Skipping annual-cycle plots because config_directory is unavailable.")
        return {}

    script_path = Path(config.config_directory).parent / "scripts" / "plot_annual_cycles.py"
    if not script_path.exists():
        logger = get_logger(__name__)
        logger.warning("Skipping annual-cycle plots because %s was not found.", script_path)
        return {}

    spec = importlib.util.spec_from_file_location("plot_annual_cycles", script_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load annual-cycle plotting script: {script_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.generate_annual_cycle_outputs(config)


def run_pipeline(config_path: Path) -> None:
    """Run the full pipeline using the provided configuration."""
    config = load_config(config_path)                                   # All configurations are loaded
    ensure_directories(config.output_directories())                     # All output directories are created if they don't exist
    configure_logging(config.logs_dir)                                  # Logging is set up to write to the logs directory
    logger = get_logger(__name__)
    random.seed(config.random_seed)                                     # Fixes random_seed, we are using monte carlo that genertaes points randomly, so for reproducibility we set the seed.
    np.random.seed(config.random_seed)

    logger.info("Starting ocean gap-filling pipeline")
    logger.info("Using config: %s", config_path)
    logger.info("Output directory: %s", config.output_directory)

    logger.info("Step 1/13: loading dataset")
    dataset = load_chlorophyll_data(config)                             # load raw data and create mask and calculate nan stats.
    raw_observation_support = build_raw_observation_support_mask(dataset)
    raw_nan_summary = log_nan_stage(logger, dataset, "raw load")

    # Keep a baseline NaN snapshot for final reporting.
    initial_stats = inspect_phase1_dataset(
        dataset,
        label="initial_dataset",
        config=config,
        save_outputs=False,
    )

# phase-2 the data is regridded to a common lat-lon grid (config.target_grid_resolution = 1.0 degree), the support mask is also regridded, and NaN stats are re-calculated and logged.

    logger.info(
        "Step 2/13: regridding to %.3f-degree latitude-longitude grid",
        config.target_grid_resolution,
    )
    regridded, regrid_summary = regrid_to_target_latlon(
        dataset,
        config,
        save_summary=False,
    )
    regridded_support, _ = regrid_to_target_latlon(
        raw_observation_support,
        config,
        save_summary=False,
    )
    regrid_nan_summary = log_nan_stage(logger, regridded, "spatial regridding")

    logger.info("Step 3/13: inspecting dataset after regridding")
    before_stats = inspect_phase1_dataset(
        regridded,
        label="phase1_regridded",
        config=config,
        save_outputs=False,
    )

# Phase-4 the ordered interpolation method is applied to fill in missing values, then the interpolation summary is extracted and NaN stats are re-calculated and logged.

    logger.info("Step 4/13: applying ordered interpolation")
    interpolated = apply_ordered_interpolation(regridded, config, save_summary=False)
    interpolation_summary = extract_interpolation_summary(interpolated)
    interpolation_nan_summary = log_nan_stage(logger, interpolated, "ordered interpolation")

# Compare before interpolation and after interpolation stats, we can see how much NaN percentage has reduced.

    logger.info("Step 5/13: inspecting dataset after interpolation")
    after_stats = inspect_phase1_dataset(
        interpolated,
        label="after_interpolation",
        config=config,
        save_outputs=False,
    )

# Now after interpolation, fit-probability-distributions is identified for all remaining missing (NaN) cells.

    logger.info("Step 6/13: fitting probability models for all remaining missing cells")
    fit_outputs = fit_all_missing_cell_distributions(
        interpolated,
        config,
        save_results=False,
    )
    fitted_models = fit_outputs["fit_results"]
    fit_summary = fit_outputs["summary"]
    unresolved_cells = fit_outputs["unresolved_cells"]

# Now run monte_carlo reconstruction for the full dataset, using the fitted models to stochastically fill in missing values and generate an ensemble of reconstructed datasets. The summary of the Monte Carlo reconstruction process is also generated, and any cells that could not be resolved are identified.

    logger.info("Step 7/13: running Monte Carlo reconstruction for the full dataset")
    monte_carlo_outputs = run_full_dataset_monte_carlo(
        interpolated,
        fitted_models,
        config,
    )
    reconstructed = monte_carlo_outputs["reconstructed_datasets"]
    monte_carlo_summary = monte_carlo_outputs["summary"]
    monte_carlo_unresolved = monte_carlo_outputs["unresolved_cells"]

# A small number of representative cells are selected after reconstruction.

    logger.info("Step 8/13: selecting debug cells filled by Monte Carlo")
    sampled_cells = select_monte_carlo_filled_debug_cells(
        interpolated,
        reconstructed[0],
        config,
    )

# Extracts the fit-results for the selected cells without writing metric tables.

    logger.info("Step 9/13: extracting model-fit details for selected debug cells")
    selected_fit_results = extract_selected_fit_results(
        fitted_models,
        sampled_cells,
        output_dir=None,
    )
    selected_monte_carlo_summaries = extract_selected_monte_carlo_summaries(
        interpolated,
        fitted_models,
        sampled_cells,
        config,
    )

# Now compute uncertainty statistics for all reconstructed dataset.

    logger.info("Step 10/13: computing uncertainty over the reconstructed ensemble")
    uncertainty = calculate_uncertainty_statistics(
        reconstructed,
        regridded_support,
        interpolated,
        config,
        selected_cells=sampled_cells,
    )

# 
# here reconstructed is a list of data-sets that are generated after applying monte-carlo reconstruction.

    logger.info("Step 11/13: inspecting final reconstructed dataset")
    final_reconstructed_stats = inspect_phase1_dataset(
        reconstructed[0],
        label="after_final_monte_carlo_reconstruction",
        config=config,
        save_outputs=False,
    )
    final_reconstruction_nan_summary = log_nan_stage(
        logger,
        reconstructed[0],
        "final reconstruction",
    )

    logger.info("Step 12/13: saving logs and reports")
    dataset_paths = save_pipeline_chlorophyll_datasets(
        raw_data=dataset,
        reconstructed_datasets=reconstructed,
        config=config,
    )
    plot_paths = generate_pipeline_plots(
        raw_data=dataset,
        regridded_data=regridded,
        interpolated_data=interpolated,
        reconstructed_datasets=reconstructed,
        nan_stage_summaries=[
            raw_nan_summary,
            regrid_nan_summary,
            interpolation_nan_summary,
            final_reconstruction_nan_summary,
        ],
        interpolation_summary=interpolation_summary,
        fit_summary=fit_summary,
        selected_fit_results=selected_fit_results,
        selected_uncertainty_summary=uncertainty["selected_cell_summary"],
        config=config,
    )
    annual_cycle_paths = generate_annual_cycle_plots_from_script(config)
    logger.info("Initial dataset baseline summary: %s", initial_stats)
    logger.info("Raw-load NaN summary: %s", raw_nan_summary)
    logger.info("Inspection summary after regridding: %s", before_stats)
    logger.info("Spatial regrid summary: %s", regrid_summary)
    logger.info("Spatial-regrid NaN summary: %s", regrid_nan_summary)
    logger.info("Interpolation summary: %s", interpolation_summary)
    logger.info("Interpolation NaN summary: %s", interpolation_nan_summary)
    logger.info("Selected Monte Carlo-filled debug cells: %s", sampled_cells)
    logger.info("Full-dataset fit summary: %s", fit_summary)
    logger.info("Unresolved missing cell count: %s", len(unresolved_cells))
    logger.info("Selected-cell distribution fitting results: %s", selected_fit_results)
    logger.info("Monte Carlo reconstruction summary: %s", monte_carlo_summary)
    logger.info("Monte Carlo unresolved cell count: %s", len(monte_carlo_unresolved))
    logger.info("Selected-cell Monte Carlo summaries: %s", selected_monte_carlo_summaries)
    logger.info("Uncertainty summary: %s", uncertainty["summary"])
    logger.info("Selected-cell uncertainty summary: %s", uncertainty["selected_cell_summary"])
    logger.info("Inspection summary after interpolation: %s", after_stats)
    logger.info("Inspection summary after final Monte Carlo reconstruction: %s", final_reconstructed_stats)
    logger.info("Final reconstruction NaN summary: %s", final_reconstruction_nan_summary)
    logger.info("Generated NetCDF dataset outputs: %s", dataset_paths)
    logger.info("Generated plot outputs: %s", plot_paths)
    logger.info("Generated annual-cycle outputs: %s", annual_cycle_paths)
    logger.info("Step 13/13: pipeline outputs finalized")
    logger.info("Pipeline finished successfully.")


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser for the main pipeline entrypoint."""
    parser = argparse.ArgumentParser(
        description="Run the ocean chlorophyll gap-filling pipeline."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/default.json"),
        help="Path to the JSON configuration file.",
    )
    return parser


def main() -> int:
    """Run the pipeline from the command line."""
    parser = build_parser()
    args = parser.parse_args()
    run_pipeline(config_path=args.config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

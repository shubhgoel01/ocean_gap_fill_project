"""Monte Carlo reconstruction helpers for full-dataset gap filling."""

from __future__ import annotations

import csv
import json
import logging
from pathlib import Path

import numpy as np
import xarray as xr
from scipy import stats

from .distribution_fit import (
    extract_cell_time_series_samples,
    make_cell_key,
    make_gaussian_kde_bw_method,
)


logger = logging.getLogger(__name__)

# take the dataset after earlier steps => take fitted probability models for remaining missing cells => generate multiple possible values for each missing cell => create multiple reconstructed datasets => save summaries/results if needed
def run_full_dataset_monte_carlo(
    data_array: xr.DataArray,
    fitted_models: list[dict],
    config,
    selected_cells: list[dict] | None = None,
    save_results: bool = True,
) -> dict:
    # Check how many simulations of monte_carlo to do, min=1 (Or how many reconstructed datasets to create) 
    simulation_count = max(1, int(config.monte_carlo_simulations))
    selected_cells = selected_cells or []
    selected_keys = {make_cell_key(cell) for cell in selected_cells}

    # Create 'n' copies of the current data set.
    reconstructed_arrays = initialize_reconstruction_arrays(data_array, simulation_count)
    base_rng = np.random.default_rng(config.random_seed)                                   # Initialize a random_generator

    selected_cell_summaries: list[dict] = []
    unresolved_cells: list[dict] = []
    imputed_cell_count = 0

    # This loop takes one missing cell at a time and process it.
    for index, fit_result in enumerate(fitted_models):
        # make a unique seed for that cell
        cell_seed = int(base_rng.integers(0, 2**32 - 1)) + index

        # Now, suppose simulation count = 10, then this function returns random '10' values for that cell based of the fit-distribution.
        sample_values = None
        if fit_result.get("chosen_model") == "kde":
            sample_values = extract_cell_time_series_samples(data_array, fit_result["cell"])
        sampled_values = simulate_for_cell(fit_result, simulation_count, cell_seed, sample_values) 
        # Tales a dictionary and returns the tuple.
        cell_key = make_cell_key(fit_result["cell"])

        # This takes the generated values and puts them into the right [time, lat, lon] location in each reconstruction array.
        filled = write_samples_into_reconstructions(
            reconstructed_arrays,
            fit_result["cell"],
            sampled_values,
        )
        # Now check if we were actually able to fill atleast one reconstructed_dataset for that cell value, then we mark it as resolved, otherwise we mark it as unresolved.
        if filled:
            imputed_cell_count += 1
        else:
            unresolved_cells.append(build_unresolved_cell_warning(fit_result))
        
        # Now check if current cell is among the selected cell for debugging, then save the detailed summary
        if cell_key in selected_keys:
            selected_cell_summaries.append(summarize_cell_simulation(fit_result, sampled_values))

    reconstructed_datasets = finalize_reconstructed_datasets(data_array, reconstructed_arrays)
    run_summary = build_monte_carlo_run_summary(
        data_array,
        simulation_count,
        imputed_cell_count,
        unresolved_cells,
    )

    # Save selected cell summaries, unresolved warnings, run summary, reconstructed datasets.
    if save_results:
        if selected_cells:
            save_selected_monte_carlo_results(selected_cell_summaries, Path(config.sampled_cells_dir))
        save_unresolved_warnings(unresolved_cells, Path(config.summaries_dir))
        save_monte_carlo_run_summary(run_summary, Path(config.summaries_dir))
        if config.save_reconstructed_datasets:
            save_reconstructed_datasets(reconstructed_datasets, Path(config.reconstructed_dir))

    return {
        "reconstructed_datasets": reconstructed_datasets,
        "selected_cell_summaries": selected_cell_summaries,
        "unresolved_cells": unresolved_cells,
        "summary": run_summary,
    }

# Create numPy-array copies of the dataset. 
def initialize_reconstruction_arrays(
    data_array: xr.DataArray,
    simulation_count: int,
) -> list[np.ndarray]:
    base_values = np.asarray(data_array.values, dtype=float)
    return [base_values.copy() for _ in range(simulation_count)]

# This function generates Monte Carlo samples for one missing cell using the model that was chosen earlier in distribution_fit.py
def simulate_for_cell(
    fit_result: dict,
    simulation_count: int,
    seed: int,
    sample_values: np.ndarray | None = None,
) -> np.ndarray:
    chosen_model = fit_result.get("chosen_model")
    candidate_stats = fit_result.get("candidate_model_statistics", {})
    rng = np.random.default_rng(seed)

    if chosen_model == "normal":
        params = candidate_stats["normal"]["parameters"]
        log_samples = rng.normal(
            loc=params["loc"],
            scale=params["scale"],
            size=simulation_count,
        )
    elif chosen_model == "gamma":
        params = candidate_stats["gamma"]["parameters"]
        log_samples = stats.gamma.rvs(
            a=params["shape"],
            loc=params["loc"],
            scale=params["scale"],
            size=simulation_count,
            random_state=rng,
        )
    elif chosen_model == "kde":
        log_samples = simulate_from_kde(fit_result, simulation_count, rng, sample_values)
    else:
        return np.full(simulation_count, np.nan, dtype=float)

    # Back-transform from log-space to original chlorophyll scale.
    # exp() is always positive so no clipping needed.
    return np.exp(np.asarray(log_samples, dtype=float))

# This function generates Monte Carlo samples for one cell when the chosen model is KDE instead of normal/lognormal/gamma
def simulate_from_kde(
    fit_result: dict,
    simulation_count: int,
    rng: np.random.Generator,
    sample_values: np.ndarray | None,
) -> np.ndarray:
    # KDE returns samples in log-space; caller handles exp() back-transform.
    kde_status = fit_result.get("candidate_model_statistics", {}).get("kde", {})
    if kde_status.get("status") != "ok":
        return np.full(simulation_count, np.nan, dtype=float)

    log_vals = np.asarray(sample_values if sample_values is not None else [], dtype=float)
    log_vals = log_vals[np.isfinite(log_vals)]
    if log_vals.size < 2:
        return np.full(simulation_count, np.nan, dtype=float)

    bandwidth = kde_status.get("bandwidth")
    if bandwidth is None:
        kde = stats.gaussian_kde(log_vals)
    else:
        kde = stats.gaussian_kde(
            log_vals,
            bw_method=make_gaussian_kde_bw_method(log_vals, float(bandwidth)),
        )
    sampled = kde.resample(simulation_count, seed=rng)
    return np.asarray(sampled).reshape(-1)  # log-space; caller does exp()


def enforce_non_negative_samples(samples: np.ndarray) -> np.ndarray:
    """Clip negative sampled chlorophyll values to zero."""
    clipped = np.asarray(samples, dtype=float).copy()
    finite_mask = np.isfinite(clipped)
    clipped[finite_mask] = np.maximum(clipped[finite_mask], 0.0)
    return clipped


def write_samples_into_reconstructions(
    reconstructed_arrays: list[np.ndarray],
    cell: dict,
    sampled_values: np.ndarray,
) -> bool:
    """Write one cell's simulated values into all reconstructed arrays."""
    time_index = int(cell["time_index"])
    lat_index = int(cell["lat_index"])
    lon_index = int(cell["lon_index"])

    wrote_any_value = False
    for simulation_index, array in enumerate(reconstructed_arrays):
        if simulation_index >= len(sampled_values):
            continue

        sampled_value = sampled_values[simulation_index]
        if not np.isfinite(sampled_value):
            continue

        array[time_index, lat_index, lon_index] = float(sampled_value)
        wrote_any_value = True

    return wrote_any_value


def summarize_cell_simulation(fit_result: dict, sampled_values: np.ndarray) -> dict:
    """Build summary statistics for one cell simulation."""
    all_samples = np.asarray(sampled_values, dtype=float)
    finite_samples = all_samples[np.isfinite(all_samples)]

    if finite_samples.size == 0:
        return {
            "cell": fit_result["cell"],
            "chosen_model": fit_result.get("chosen_model"),
            "simulation_count": int(sampled_values.size),
            "sampled_values": [],
            "first_20_samples": [],
            "sample_mean": None,
            "sample_std": None,
            "sample_min": None,
            "sample_max": None,
            "percentiles": {
                "p05": None,
                "p25": None,
                "p50": None,
                "p75": None,
                "p95": None,
            },
        }

    percentile_values = np.percentile(finite_samples, [5, 25, 50, 75, 95])
    return {
        "cell": fit_result["cell"],
        "chosen_model": fit_result.get("chosen_model"),
        "simulation_count": int(sampled_values.size),
        "sampled_values": [float(value) for value in all_samples.tolist()],
        "first_20_samples": [float(value) for value in finite_samples[:20]],
        "sample_mean": float(np.mean(finite_samples)),
        "sample_std": float(np.std(finite_samples, ddof=0)),
        "sample_min": float(np.min(finite_samples)),
        "sample_max": float(np.max(finite_samples)),
        "percentiles": {
            "p05": float(percentile_values[0]),
            "p25": float(percentile_values[1]),
            "p50": float(percentile_values[2]),
            "p75": float(percentile_values[3]),
            "p95": float(percentile_values[4]),
        },
    }


def build_unresolved_cell_warning(fit_result: dict) -> dict:
    """Build a warning record for a cell that could not be Monte Carlo imputed."""
    return {
        "cell": fit_result["cell"],
        "chosen_model": fit_result.get("chosen_model"),
        "fallback_reason": fit_result.get("fallback_reason"),
        "sample_size": fit_result.get("sample_size"),
    }


def finalize_reconstructed_datasets(
    data_array: xr.DataArray,
    reconstructed_arrays: list[np.ndarray],
) -> list[xr.DataArray]:
    """Wrap reconstructed arrays back into xarray DataArrays."""
    reconstructed_datasets: list[xr.DataArray] = []

    for simulation_index, array in enumerate(reconstructed_arrays):
        dataset_copy = data_array.copy(deep=True)
        dataset_copy.values = array
        dataset_copy.attrs = dict(dataset_copy.attrs)
        dataset_copy.attrs["simulation_index"] = simulation_index
        reconstructed_datasets.append(dataset_copy)

    return reconstructed_datasets


def build_monte_carlo_run_summary(
    data_array: xr.DataArray,
    simulation_count: int,
    imputed_cell_count: int,
    unresolved_cells: list[dict],
) -> dict:
    """Build the required whole-run Monte Carlo summary."""
    return {
        "total_reconstructed_datasets": simulation_count,
        "final_dataset_shape": {
            "time": int(data_array.sizes["time"]),
            "lat": int(data_array.sizes["lat"]),
            "lon": int(data_array.sizes["lon"]),
        },
        "cells_imputed_through_monte_carlo": int(imputed_cell_count),
        "unresolved_cell_count": int(len(unresolved_cells)),
        "memory_strategy": (
            "Kept one base array plus N reconstruction arrays in memory and "
            "processed missing cells one at a time. Detailed sampled values "
            "were retained only for selected debug cells."
        ),
    }


def save_selected_monte_carlo_results(results: list[dict], output_dir: Path) -> None:
    """Save selected-cell Monte Carlo summaries derived from full results."""
    output_dir.mkdir(parents=True, exist_ok=True)
    save_monte_carlo_results_json(results, output_dir / "selected_cells_monte_carlo.json")
    save_monte_carlo_results_csv(results, output_dir / "selected_cells_monte_carlo.csv")


def save_monte_carlo_results_json(results: list[dict], output_path: Path) -> Path:
    """Save Monte Carlo summaries as JSON."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)
    return output_path


def save_monte_carlo_results_csv(results: list[dict], output_path: Path) -> Path:
    """Save Monte Carlo summaries as CSV."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "time_index",
        "time_value",
        "lat_index",
        "lat_value",
        "lon_index",
        "lon_value",
        "chosen_model",
        "simulation_count",
        "sample_mean",
        "sample_std",
        "sample_min",
        "sample_max",
        "p05",
        "p25",
        "p50",
        "p75",
        "p95",
        "first_20_samples",
    ]

    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for item in results:
            writer.writerow(flatten_monte_carlo_result(item))

    return output_path


def flatten_monte_carlo_result(result: dict) -> dict:
    """Flatten nested Monte Carlo results for CSV export."""
    cell = result["cell"]
    percentiles = result["percentiles"]
    return {
        "time_index": cell["time_index"],
        "time_value": cell["time_value"],
        "lat_index": cell["lat_index"],
        "lat_value": cell["lat_value"],
        "lon_index": cell["lon_index"],
        "lon_value": cell["lon_value"],
        "chosen_model": result["chosen_model"],
        "simulation_count": result["simulation_count"],
        "sample_mean": result["sample_mean"],
        "sample_std": result["sample_std"],
        "sample_min": result["sample_min"],
        "sample_max": result["sample_max"],
        "p05": percentiles["p05"],
        "p25": percentiles["p25"],
        "p50": percentiles["p50"],
        "p75": percentiles["p75"],
        "p95": percentiles["p95"],
        "first_20_samples": json.dumps(result["first_20_samples"]),
    }


def save_unresolved_warnings(unresolved_cells: list[dict], summaries_dir: Path) -> Path:
    """Save unresolved-cell warnings for Monte Carlo reconstruction."""
    summaries_dir.mkdir(parents=True, exist_ok=True)
    output_path = summaries_dir / "monte_carlo_unresolved_cells.json"
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(unresolved_cells, handle, indent=2)
    return output_path


def save_monte_carlo_run_summary(summary: dict, summaries_dir: Path) -> Path:
    """Save full Monte Carlo run summary to JSON."""
    summaries_dir.mkdir(parents=True, exist_ok=True)
    output_path = summaries_dir / "monte_carlo_reconstruction_summary.json"
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    return output_path


def save_reconstructed_datasets(reconstructed_datasets: list[xr.DataArray], output_dir: Path) -> None:
    """Save each reconstructed dataset as a NetCDF file."""
    output_dir.mkdir(parents=True, exist_ok=True)
    for dataset in reconstructed_datasets:
        simulation_index = int(dataset.attrs.get("simulation_index", 0))
        output_path = output_dir / f"reconstructed_dataset_{simulation_index:03d}.nc"
        dataset.to_netcdf(output_path)


def impute_missing_cells_with_monte_carlo(dataset, fitted_models: list[dict], config):
    """Compatibility wrapper returning only reconstructed datasets."""
    outputs = run_full_dataset_monte_carlo(dataset, fitted_models, config)
    return outputs["reconstructed_datasets"]

from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import xarray as xr
from scipy import stats
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KernelDensity


MIN_PARAMETRIC_SAMPLE_SIZE = 5
MIN_POSITIVE_VARIANCE = 1e-8

# For each missing cell, Take same (lat, long) across all time => Get all known values => Use them to fit the best distribution.
def fit_all_missing_cell_distributions(
    data_array: xr.DataArray,
    config,
    save_results: bool = True,
) -> dict:
    validate_distribution_input(data_array)

    missing_cells = find_missing_cells(data_array)
    grid_fit_cache = {}
    results = []
    for cell in missing_cells:
        grid_key = make_grid_cell_key(cell)
        if grid_key not in grid_fit_cache:
            sample_values = extract_cell_time_series_samples(data_array, cell)
            grid_fit_cache[grid_key] = fit_models_for_cell(
                cell,
                sample_values,
                config.ks_pvalue_threshold,
            )
        results.append(build_cell_fit_result(grid_fit_cache[grid_key], cell))

    summary = summarize_full_fit_results(results)
    unresolved_cells = extract_unresolved_cells(results)

    if save_results:
        save_distribution_results(
            results,
            Path(config.reconstructed_dir) / "full_missing_cell_model_fits.json",
        )

    return {
        "fit_results": results,
        "summary": summary,
        "unresolved_cells": unresolved_cells,
    }

# This function extracts results for the selected cells, cells that are selected to display results.
def extract_selected_fit_results(
    full_results: list[dict],
    selected_cells: list[dict],
    output_dir: Path | None = None,
) -> list[dict]:
    """Extract reporting-only fit results for the selected debug cells."""
    selected_keys = {make_cell_key(cell) for cell in selected_cells}
    filtered = [result for result in full_results if make_cell_key(result["cell"]) in selected_keys]
    return filtered

# Check if data has three dimensions
def validate_distribution_input(data_array: xr.DataArray) -> None:
    required_dims = {"time", "lat", "lon"}
    missing_dims = sorted(required_dims.difference(data_array.dims))
    if missing_dims:
        joined = ", ".join(missing_dims)
        raise ValueError(
            "Distribution fitting requires a DataArray with time, lat, and lon "
            f"dimensions. Missing: {joined}"
        )

# This checks for every value inside the x-array, and checks if it is nan or not, and returns a boolean array of same shape.
# It uses NumPy to locate NaN positions in the 3D array
# Each missing position is then converted into a structured record containing its indices and actual coordinate values, so that it can be processed later for model fitting.
# Example : we have a 3D array that contains NaN value at (time=3, lat=25, lon=50), then it will create a record like : 
# {"time_index": 3, "time_value": "2020-01-04", "lat_index": 25, "lat_value": 12.5, "lon_index": 50, "lon_value": 120.0} 
# and this record will be stored in a list of missing cells, means One missing point → one dictionary.
def find_missing_cells(data_array: xr.DataArray) -> list[dict]:
    missing_indices = np.argwhere(np.isnan(data_array.values))
    return [build_cell_record(data_array, index_triplet) for index_triplet in missing_indices]

# Takes a triplet, extracts values and creates a dictionary with both index and actual coordinate values for time, lat, and lon. 
# Example : input = [3, 25, 80]
# output = {
#   "time_index": 3,
#   "time_value": actual_time,
#   "lat_index": 25,
#   "lat_value": actual_lat,
#   "lon_index": 80,
#   "lon_value": actual_lon
# }
def build_cell_record(data_array: xr.DataArray, index_triplet: np.ndarray) -> dict:
    time_index = int(index_triplet[0])
    lat_index = int(index_triplet[1])
    lon_index = int(index_triplet[2])

    return {
        "time_index": time_index,
        "time_value": str(data_array["time"].values[time_index]),
        "lat_index": lat_index,
        "lat_value": float(data_array["lat"].values[lat_index]),
        "lon_index": lon_index,
        "lon_value": float(data_array["lon"].values[lon_index]),
    }

# takes a cell record (object) and creates a tuple key based on its time, lat, and lon 'indices'. This is because tuples are easy to compare, store in set, match one cell with another.
# input = {
#   "time_index": 3,
#   "lat_index": 25,
#   "lon_index": 80
# }
# Output = (3, 25, 80)
def make_cell_key(cell: dict) -> tuple[int, int, int]:
    return (
        int(cell["time_index"]),
        int(cell["lat_index"]),
        int(cell["lon_index"]),
    )


def make_grid_cell_key(cell: dict) -> tuple[int, int]:
    return (
        int(cell["lat_index"]),
        int(cell["lon_index"]),
    )


def build_cell_fit_result(cached_fit_result: dict, cell: dict) -> dict:
    result = copy.deepcopy(cached_fit_result)
    result["cell"] = {
        "time_index": int(cell["time_index"]),
        "time_value": str(cell["time_value"]),
        "lat_index": int(cell["lat_index"]),
        "lat_value": float(cell["lat_value"]),
        "lon_index": int(cell["lon_index"]),
        "lon_value": float(cell["lon_value"]),
    }
    return result


# This fucntion takes (lon, lat) values of a missing cell and takes values for all time steps, means if some cell at particular (lat, lon) is missing, it extracts all values at that (lat, lon) across all time steps, and then filters out the valid values (non-NaN) to create a sample for distribution fitting. 
# This sample is then used to fit different distributions and find the best fit for that missing cell.
def extract_cell_time_series_samples(data_array: xr.DataArray, cell: dict) -> np.ndarray:
    lat_index = int(cell["lat_index"])
    lon_index = int(cell["lon_index"])

    series = data_array.isel(lat=lat_index, lon=lon_index).values
    series = np.asarray(series, dtype=float)
    return series[np.isfinite(series)]

# for one missing cell => Take its sample values => Fit normal, lognormal, gamma => Check KS test p-value for each fit => Choose best fit that passes the p-value threshold => If no fit passes, mark it as unresolved.
def fit_models_for_cell(cell: dict, sample_values: np.ndarray, pvalue_threshold: float) -> dict:
    sample_values = np.asarray(sample_values, dtype=float)
    sample_values = sample_values[np.isfinite(sample_values)]
    sample_size = int(sample_values.size)

    result = {
        "cell": {
            "time_index": int(cell["time_index"]),
            "time_value": str(cell["time_value"]),
            "lat_index": int(cell["lat_index"]),
            "lat_value": float(cell["lat_value"]),
            "lon_index": int(cell["lon_index"]),
            "lon_value": float(cell["lon_value"]),
        },
        "sample_size": sample_size,
        "candidate_model_statistics": {},
        "chosen_model": None,
        "chosen_p_value": None,
        "fallback_reason": None,
    }

    if sample_size < MIN_PARAMETRIC_SAMPLE_SIZE:
        result["chosen_model"] = "unresolved"
        result["fallback_reason"] = "insufficient_sample_size"
        result["candidate_model_statistics"] = build_insufficient_sample_stats(sample_size)
        return finalize_with_kde_status(result, sample_values)

    normal_stats = fit_normal_distribution(sample_values)
    lognormal_stats = fit_lognormal_distribution(sample_values)
    gamma_stats = fit_gamma_distribution(sample_values)

    result["candidate_model_statistics"] = {
        "normal": normal_stats,
        "lognormal": lognormal_stats,
        "gamma": gamma_stats,
    }

    acceptable_models = [
        ("normal", normal_stats),
        ("lognormal", lognormal_stats),
        ("gamma", gamma_stats),
    ]
    acceptable_models = [
        (name, stats_result)
        for name, stats_result in acceptable_models
        if stats_result["status"] == "ok"
        and stats_result["p_value"] is not None
        and stats_result["p_value"] >= pvalue_threshold
    ]

    if acceptable_models:
        # Select model with highest p-value
        best_name, best_stats = max(acceptable_models, key=lambda item: item[1]["p_value"])
        result["chosen_model"] = best_name
        result["chosen_p_value"] = best_stats["p_value"]
        return result

    # If no model works....
    result["fallback_reason"] = "no_parametric_model_passed_threshold"
    return finalize_with_kde_status(result, sample_values)

# Clean the sample => Check minimum sample size => If sample size is too small, marked as unresolved => Check variance => If values are almost constant, fitting a distribution is meaningless. => estimates the normal distribution parameters => applies the KS test to evaluate the fit
def fit_normal_distribution(sample_values: np.ndarray) -> dict:
    """Fit a normal distribution and evaluate it with the KS test."""
    sample_values = np.asarray(sample_values, dtype=float)
    sample_values = sample_values[np.isfinite(sample_values)]

    if sample_values.size < MIN_PARAMETRIC_SAMPLE_SIZE:
        return failed_model_result("insufficient_sample_size", int(sample_values.size))

    std = float(np.std(sample_values))
    if not np.isfinite(std) or std < MIN_POSITIVE_VARIANCE:
        return failed_model_result("low_variance_sample_values", int(sample_values.size))

    try:
        mean, fitted_std = stats.norm.fit(sample_values)
    except Exception as exc:
        return failed_model_result(f"normal_fit_failed: {exc}", int(sample_values.size))

    if not np.isfinite(fitted_std) or fitted_std <= 0:
        return failed_model_result("invalid_fitted_parameters", int(sample_values.size))

    try:
        ks_statistic, p_value = stats.kstest(sample_values, "norm", args=(mean, fitted_std))
    except Exception as exc:
        return failed_model_result(f"normal_ks_test_failed: {exc}", int(sample_values.size))

    if not np.isfinite(ks_statistic) or not np.isfinite(p_value):
        return failed_model_result("invalid_ks_statistics", int(sample_values.size))

    return {
        "status": "ok",
        "sample_size": int(sample_values.size),
        "parameters": {
            "loc": float(mean),
            "scale": float(fitted_std),
        },
        "ks_statistic": float(ks_statistic),
        "p_value": float(p_value),
    }

# Take only positive values (lognormal requires > 0) => Clean the sample => Check minimum sample size (on positive values) => If too small → fail => Check variance => If values are almost constant → fail => Estimate lognormal parameters (shape, loc, scale) => Validate fitted parameters => Apply KS test to evaluate fit => Return result (success or failure with reason)
def fit_lognormal_distribution(sample_values: np.ndarray) -> dict:
    positive_values = sample_values[sample_values > 0]
    positive_values = np.asarray(positive_values, dtype=float)
    positive_values = positive_values[np.isfinite(positive_values)]

    sample_size = int(positive_values.size)
    if sample_size < MIN_PARAMETRIC_SAMPLE_SIZE:
        return failed_model_result("insufficient_positive_values", sample_size)

    std = float(np.std(positive_values))
    if not np.isfinite(std) or std < MIN_POSITIVE_VARIANCE:
        return failed_model_result("low_variance_positive_values", sample_size)

    try:
        shape, loc, scale = stats.lognorm.fit(positive_values, floc=0.0)
    except Exception as exc:
        return failed_model_result(f"lognormal_fit_failed: {exc}", sample_size)

    if (
        not np.isfinite(shape)
        or not np.isfinite(loc)
        or not np.isfinite(scale)
        or scale <= 0
        or shape <= 0
    ):
        return failed_model_result("invalid_fitted_parameters", sample_size)

    try:
        ks_statistic, p_value = stats.kstest(
            positive_values,
            "lognorm",
            args=(shape, loc, scale),
        )
    except Exception as exc:
        return failed_model_result(f"lognormal_ks_test_failed: {exc}", sample_size)

    if not np.isfinite(ks_statistic) or not np.isfinite(p_value):
        return failed_model_result("invalid_ks_statistics", sample_size)

    return {
        "status": "ok",
        "sample_size": sample_size,
        "parameters": {
            "shape": float(shape),
            "loc": float(loc),
            "scale": float(scale),
        },
        "ks_statistic": float(ks_statistic),
        "p_value": float(p_value),
        "uses_positive_values_only": True,
    }

# Take only positive values (gamma requires > 0) => Clean the sample => Check minimum sample size (on positive values) => If too small → fail => Check variance => If values are almost constant → fail => Estimate gamma parameters (shape, loc, scale) => Validate fitted parameters => Apply KS test to evaluate fit => Return result (success or failure with reason)
def fit_gamma_distribution(sample_values: np.ndarray) -> dict:
    """Fit a gamma distribution using positive values only."""
    positive_values = sample_values[sample_values > 0]
    positive_values = np.asarray(positive_values, dtype=float)
    positive_values = positive_values[np.isfinite(positive_values)]

    sample_size = int(positive_values.size)
    if sample_size < MIN_PARAMETRIC_SAMPLE_SIZE:
        return failed_model_result("insufficient_positive_values", sample_size)

    std = float(np.std(positive_values))
    if not np.isfinite(std) or std < MIN_POSITIVE_VARIANCE:
        return failed_model_result("low_variance_positive_values", sample_size)

    try:
        shape, loc, scale = stats.gamma.fit(positive_values, floc=0.0)
    except Exception as exc:
        return failed_model_result(f"gamma_fit_failed: {exc}", sample_size)

    if (
        not np.isfinite(shape)
        or not np.isfinite(loc)
        or not np.isfinite(scale)
        or scale <= 0
        or shape <= 0
    ):
        return failed_model_result("invalid_fitted_parameters", sample_size)

    try:
        ks_statistic, p_value = stats.kstest(
            positive_values,
            "gamma",
            args=(shape, loc, scale),
        )
    except Exception as exc:
        return failed_model_result(f"gamma_ks_test_failed: {exc}", sample_size)

    if not np.isfinite(ks_statistic) or not np.isfinite(p_value):
        return failed_model_result("invalid_ks_statistics", sample_size)

    return {
        "status": "ok",
        "sample_size": sample_size,
        "parameters": {
            "shape": float(shape),
            "loc": float(loc),
            "scale": float(scale),
        },
        "ks_statistic": float(ks_statistic),
        "p_value": float(p_value),
        "uses_positive_values_only": True,
    }

# Standard format to report failure of a model (Why particular model failed, not for all)
def failed_model_result(reason: str, sample_size: int | None = None) -> dict:
    return {
        "status": "failed",
        "sample_size": sample_size,
        "parameters": None,
        "ks_statistic": None,
        "p_value": None,
        "reason": reason,
    }

# When data is insufficielt, mark all models as failed.
def build_insufficient_sample_stats(sample_size: int | None = None) -> dict:
    return {
        "normal": failed_model_result("insufficient_sample_size", sample_size),
        "lognormal": failed_model_result("insufficient_sample_size", sample_size),
        "gamma": failed_model_result("insufficient_sample_size", sample_size),
    }


# This function summarizes the fitting results by counting how many missing cells were finally assigned to normal, lognormal, gamma, KDE, or remained unresolved.
def summarize_full_fit_results(results: list[dict]) -> dict:
    """Summarize full-dataset fitting coverage and model usage."""
    counts = {
        "normal": 0,
        "lognormal": 0,
        "gamma": 0,
        "kde": 0,
        "unresolved": 0,
    }

    for result in results:
        chosen_model = str(result.get("chosen_model", "unresolved")).lower()
        if chosen_model not in counts:  # If model name is not in counts, then we consider it as unresolved.
            chosen_model = "unresolved"
        counts[chosen_model] += 1

    successfully_modeled = (
        counts["normal"] + counts["lognormal"] + counts["gamma"] + counts["kde"]
    )

    return {
        "total_remaining_missing_cells": len(results),
        "successfully_modeled_cells": successfully_modeled,
        "model_counts": counts,
    }

# Collect only those cells that could not be modeled.
def extract_unresolved_cells(results: list[dict]) -> list[dict]:
    unresolved = []
    for result in results:
        if result.get("chosen_model") != "unresolved":
            continue
        unresolved.append(
            {
                "cell": result["cell"],
                "sample_size": result.get("sample_size"),
                "fallback_reason": result.get("fallback_reason"),
                "candidate_model_statistics": result.get("candidate_model_statistics", {}),
            }
        )
    return unresolved



# This function is used when no parametric model is selected. It checks whether KDE can be used as a fallback. 
# If yes, the chosen model becomes KDE; otherwise the cell is marked as unresolved.
def finalize_with_kde_status(result: dict, sample_values: np.ndarray) -> dict:
    """Attach KDE availability details when parametric fitting is not chosen."""
    kde_status = evaluate_kde_availability(sample_values)
    result["candidate_model_statistics"]["kde"] = kde_status
    if kde_status["status"] == "ok":
        result["chosen_model"] = "kde"
    else:
        result["chosen_model"] = "unresolved"
        if result["fallback_reason"] is None:
            result["fallback_reason"] = kde_status.get("reason", "kde_unavailable")
    return result


# Check whether KDE can be used as fallback or not.
def evaluate_kde_availability(sample_values: np.ndarray) -> dict:
    """Check whether KDE can be used later for sampling."""
    sample_values = np.asarray(sample_values, dtype=float)
    sample_values = sample_values[np.isfinite(sample_values)]
    unique_values = np.unique(sample_values)
    if sample_values.size < 2:
        return {
            "status": "failed",
            "sample_size": int(sample_values.size),
            "reason": "insufficient_sample_size_for_kde",
        }
    if sample_values.size < 5:
        return {
            "status": "failed",
            "sample_size": int(sample_values.size),
            "reason": "insufficient_sample_size_for_5_fold_kde_cv",
        }
    if unique_values.size < 2:
        return {
            "status": "failed",
            "sample_size": int(sample_values.size),
            "reason": "insufficient_value_variation_for_kde",
        }

    try:
        bandwidth = select_kde_bandwidth(sample_values)
        stats.gaussian_kde(sample_values, bw_method=make_gaussian_kde_bw_method(sample_values, bandwidth))
    except Exception as exc:
        return {
            "status": "failed",
            "sample_size": int(sample_values.size),
            "reason": f"kde_fit_failed: {exc}",
        }

    return {
        "status": "ok",
        "sample_size": int(sample_values.size),
        "bandwidth": float(bandwidth),
    }


def select_kde_bandwidth(sample_values: np.ndarray) -> float:
    sample_values = np.asarray(sample_values, dtype=float)
    sample_values = sample_values[np.isfinite(sample_values)].reshape(-1, 1)
    bandwidth_grid = np.logspace(-2, 1, 50)
    search = GridSearchCV(
        KernelDensity(kernel="gaussian"),
        {"bandwidth": bandwidth_grid},
        cv=5,
    )
    search.fit(sample_values)
    return float(search.best_params_["bandwidth"])


def make_gaussian_kde_bw_method(sample_values: np.ndarray, bandwidth: float) -> float:
    sample_values = np.asarray(sample_values, dtype=float)
    sample_values = sample_values[np.isfinite(sample_values)]
    sample_std = float(np.std(sample_values, ddof=1))
    if not np.isfinite(sample_std) or sample_std <= 0:
        raise ValueError("Cannot convert KDE bandwidth for zero-variance sample values.")
    return float(bandwidth) / sample_std


def save_distribution_results(results: list[dict], output_path: Path) -> Path:
    """Serialize per-cell fitting results to JSON."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)
    return output_path



"""Monte Carlo reconstruction helpers for full-dataset gap filling.

Key design decisions
--------------------
* ALL distribution fitting (normal, gamma, KDE) is done in **log-space**
  (log-transformed chlorophyll).  ``simulate_for_cell`` therefore always
  returns log-space samples and the single ``np.exp()`` call at the very end
  of that function converts back to mg m^-3. There is no double-transform.

* ``gamma`` is fitted and sampled in log-space (the log-transformed values are
  passed to ``gamma.fit`` in ``distribution_fit.py``).  The shape/loc/scale
  parameters stored in the fit result therefore describe a gamma distribution
  over log(chl), not over chl itself.  ``gamma.rvs`` is called with those
  parameters and the result is still in log-space before ``np.exp()``.

* After ``np.exp()`` a physical hard-cap of 200 mg m^-3 is applied. Values
  above this are oceanographically impossible and indicate a pathological
  log-space sample.

* ``finalize_reconstructed_datasets`` uses ``xr.DataArray.copy(data=array)``
  instead of the deprecated ``.values =`` assignment, which silently failed in
  newer xarray versions and left NaNs intact.

* With ``simulation_count`` as low as 100 the p05/p95 interval estimates are
  noisy.  The paper uses N=10 000.  A runtime warning is emitted when N<1000.
"""

from __future__ import annotations

import csv
import json
import logging
import warnings
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

# Physical upper bound for chlorophyll-a (mg m^-3).
# Any back-transformed value above this is clamped; it signals an extreme
# log-space draw rather than a real oceanographic signal.
_CHLOROPHYLL_HARD_CAP: float = 200.0

# Minimum recommended simulation count for stable p05/p95 estimates.
_MIN_RECOMMENDED_SIMULATIONS: int = 1_000


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def run_full_dataset_monte_carlo(
    data_array: xr.DataArray,
    fitted_models: list[dict],
    config,
    selected_cells: list[dict] | None = None,
    save_results: bool = True,
) -> dict:
    """Fill every remaining NaN cell with N Monte Carlo draws.

    Parameters
    ----------
    data_array:
        Post-interpolation chlorophyll DataArray (time x lat x lon).
        NaN cells are the ones that need Monte Carlo filling.
    fitted_models:
        Per-cell distribution fit results from ``fit_all_missing_cell_distributions``.
    config:
        Application config.  Must expose ``monte_carlo_simulations``,
        ``random_seed``, ``save_reconstructed_datasets``, and the output
        directory properties.
    selected_cells:
        Optional list of cells for which detailed simulation summaries are kept
        in memory and written to disk.
    save_results:
        Write JSON/CSV/NetCDF outputs when True.

    Returns
    -------
    dict with keys:
        ``reconstructed_datasets`` - list of N xr.DataArray, each gap-free.
        ``selected_cell_summaries`` - detailed stats for debug cells.
        ``unresolved_cells``        - cells that could not be imputed.
        ``summary``                 - run-level statistics.
    """
    simulation_count = max(1, int(config.monte_carlo_simulations))

    if simulation_count < _MIN_RECOMMENDED_SIMULATIONS:
        warnings.warn(
            f"monte_carlo_simulations={simulation_count} is below the recommended "
            f"minimum of {_MIN_RECOMMENDED_SIMULATIONS}.  p05/p95 uncertainty "
            "interval estimates will be noisy.  The paper uses N=10 000.",
            UserWarning,
            stacklevel=2,
        )

    selected_cells = selected_cells or []
    selected_keys = {make_cell_key(cell) for cell in selected_cells}

    # N independent copies of the post-interpolation array (float64).
    reconstructed_arrays = _initialize_reconstruction_arrays(data_array, simulation_count)

    # One master RNG; each cell gets a deterministic child seed derived from it
    # so results are reproducible regardless of how many cells there are.
    base_rng = np.random.default_rng(int(config.random_seed))

    selected_cell_summaries: list[dict] = []
    unresolved_cells: list[dict] = []
    imputed_cell_count = 0

    for index, fit_result in enumerate(fitted_models):
        # Deterministic per-cell seed avoids correlations between cells while
        # still being fully reproducible.
        cell_seed = int(base_rng.integers(0, 2**31 - 1))

        # For KDE we need the raw log-space time series at this grid point.
        log_time_series: np.ndarray | None = None
        if fit_result.get("chosen_model") == "kde":
            log_time_series = extract_cell_time_series_samples(
                data_array, fit_result["cell"]
            )

        # Draw N values in log-space, then back-transform to mg m^-3.
        sampled_values = _simulate_for_cell(
            fit_result, simulation_count, cell_seed, log_time_series
        )

        cell_key = make_cell_key(fit_result["cell"])

        filled = _write_samples_into_reconstructions(
            reconstructed_arrays,
            fit_result["cell"],
            sampled_values,
        )

        if filled:
            imputed_cell_count += 1
        else:
            unresolved_cells.append(_build_unresolved_cell_warning(fit_result))

        if cell_key in selected_keys:
            selected_cell_summaries.append(
                _summarize_cell_simulation(fit_result, sampled_values)
            )

    reconstructed_datasets = _finalize_reconstructed_datasets(
        data_array, reconstructed_arrays
    )
    run_summary = _build_monte_carlo_run_summary(
        data_array,
        simulation_count,
        imputed_cell_count,
        unresolved_cells,
    )

    if save_results:
        if selected_cells:
            save_selected_monte_carlo_results(
                selected_cell_summaries, Path(config.sampled_cells_dir)
            )
        _save_unresolved_warnings(unresolved_cells, Path(config.summaries_dir))
        _save_monte_carlo_run_summary(run_summary, Path(config.summaries_dir))
        if config.save_reconstructed_datasets:
            _save_reconstructed_datasets(
                reconstructed_datasets, Path(config.reconstructed_dir)
            )

    return {
        "reconstructed_datasets": reconstructed_datasets,
        "selected_cell_summaries": selected_cell_summaries,
        "unresolved_cells": unresolved_cells,
        "summary": run_summary,
    }


# ---------------------------------------------------------------------------
# Initialisation helpers
# ---------------------------------------------------------------------------

def _initialize_reconstruction_arrays(
    data_array: xr.DataArray,
    simulation_count: int,
) -> list[np.ndarray]:
    """Create N independent float64 copies of the base array."""
    base = np.asarray(data_array.values, dtype=np.float64)
    return [base.copy() for _ in range(simulation_count)]


# ---------------------------------------------------------------------------
# Per-cell simulation
# ---------------------------------------------------------------------------

def _simulate_for_cell(
    fit_result: dict,
    simulation_count: int,
    seed: int,
    log_time_series: np.ndarray | None = None,
) -> np.ndarray:
    """Draw ``simulation_count`` chlorophyll values (mg m^-3) for one cell.

    All sampling happens in **log-space** (matching how distribution_fit.py
    fitted the models).  A single ``np.exp()`` at the end converts the result
    to mg m^-3, followed by a physical hard-cap.

    Returns
    -------
    np.ndarray of shape (simulation_count,), dtype float64, in mg m^-3.
    NaN entries indicate cells that could not be sampled.
    """
    chosen_model = fit_result.get("chosen_model")
    candidate_stats = fit_result.get("candidate_model_statistics", {})
    rng = np.random.default_rng(seed)

    # ---- draw in log-space -------------------------------------------------
    if chosen_model == "normal":
        params = candidate_stats["normal"]["parameters"]
        log_samples = rng.normal(
            loc=float(params["loc"]),
            scale=float(params["scale"]),
            size=simulation_count,
        )

    elif chosen_model == "gamma":
        # gamma was fitted on log-transformed chlorophyll values, so the
        # shape/loc/scale parameters describe a gamma over log(chl).
        # We sample directly from that gamma; result is still log-space.
        params = candidate_stats["gamma"]["parameters"]
        log_samples = stats.gamma.rvs(
            a=float(params["shape"]),
            loc=float(params["loc"]),
            scale=float(params["scale"]),
            size=simulation_count,
            random_state=rng,
        )

    elif chosen_model == "kde":
        log_samples = _simulate_from_kde(
            fit_result, simulation_count, rng, log_time_series
        )

    else:
        # "unresolved" or unknown model
        return np.full(simulation_count, np.nan, dtype=np.float64)

    log_samples = np.asarray(log_samples, dtype=np.float64)

    # ---- back-transform from log-space to mg m^-3 --------------------------
    # Clip log-samples before exp() to avoid overflow (exp(709) ~ 8e307).
    # log(200) is about 5.3, so anything above ~6 in log-space is already extreme.
    log_samples = np.clip(log_samples, a_min=-10.0, a_max=np.log(_CHLOROPHYLL_HARD_CAP))

    chl_samples = np.exp(log_samples)

    # Physical sanity: must be positive and below hard cap.
    chl_samples = np.where(
        np.isfinite(chl_samples) & (chl_samples > 0),
        np.minimum(chl_samples, _CHLOROPHYLL_HARD_CAP),
        np.nan,
    )

    return chl_samples


def _simulate_from_kde(
    fit_result: dict,
    simulation_count: int,
    rng: np.random.Generator,
    log_time_series: np.ndarray | None,
) -> np.ndarray:
    """Sample ``simulation_count`` log-space values from the fitted KDE.

    Returns log-space values; caller applies ``np.exp()``.
    """
    kde_status = (
        fit_result.get("candidate_model_statistics", {}).get("kde", {})
    )
    if kde_status.get("status") != "ok":
        return np.full(simulation_count, np.nan, dtype=np.float64)

    log_vals = np.asarray(
        log_time_series if log_time_series is not None else [],
        dtype=np.float64,
    )
    log_vals = log_vals[np.isfinite(log_vals)]

    if log_vals.size < 2:
        return np.full(simulation_count, np.nan, dtype=np.float64)

    bandwidth = kde_status.get("bandwidth")
    try:
        if bandwidth is None:
            kde = stats.gaussian_kde(log_vals)
        else:
            bw_method = make_gaussian_kde_bw_method(log_vals, float(bandwidth))
            kde = stats.gaussian_kde(log_vals, bw_method=bw_method)

        # gaussian_kde.resample accepts an np.random.Generator from scipy >= 1.7
        sampled = kde.resample(simulation_count, seed=rng)
        return np.asarray(sampled, dtype=np.float64).reshape(-1)

    except Exception as exc:
        logger.warning("KDE sampling failed for cell: %s", exc)
        return np.full(simulation_count, np.nan, dtype=np.float64)


# ---------------------------------------------------------------------------
# Writing samples into the reconstruction arrays
# ---------------------------------------------------------------------------

def _write_samples_into_reconstructions(
    reconstructed_arrays: list[np.ndarray],
    cell: dict,
    sampled_values: np.ndarray,
) -> bool:
    """Write one cell's N simulated mg m^-3 values into the N arrays.

    Returns True if at least one finite value was written.
    """
    t = int(cell["time_index"])
    r = int(cell["lat_index"])
    c = int(cell["lon_index"])

    wrote_any = False
    n = len(reconstructed_arrays)

    for sim_idx in range(n):
        if sim_idx >= len(sampled_values):
            break
        value = float(sampled_values[sim_idx])
        if not np.isfinite(value) or value <= 0:
            continue
        reconstructed_arrays[sim_idx][t, r, c] = value
        wrote_any = True

    return wrote_any


# ---------------------------------------------------------------------------
# Finalising reconstructed datasets
# ---------------------------------------------------------------------------

def _finalize_reconstructed_datasets(
    data_array: xr.DataArray,
    reconstructed_arrays: list[np.ndarray],
) -> list[xr.DataArray]:
    """Wrap each filled numpy array back into a labelled xr.DataArray.

    Uses ``xr.DataArray.copy(data=...)`` rather than the deprecated
    ``.values = ...`` assignment which silently fails in xarray >= 2023.
    """
    datasets: list[xr.DataArray] = []
    for sim_idx, array in enumerate(reconstructed_arrays):
        # copy(data=...) replaces underlying data while preserving all
        # coordinates, dims, attrs and encoding metadata.
        filled = data_array.copy(data=array.astype(np.float32))
        filled.attrs = dict(data_array.attrs)
        filled.attrs["simulation_index"] = sim_idx
        # netCDF attribute types are limited; avoid Python bools/numpy.bool_
        # which can raise TypeError on save. Store as int (0/1).
        filled.attrs["monte_carlo_filled"] = int(1)
        datasets.append(filled)
    return datasets


# ---------------------------------------------------------------------------
# Summaries and diagnostics
# ---------------------------------------------------------------------------

def _summarize_cell_simulation(
    fit_result: dict,
    sampled_values: np.ndarray,
) -> dict:
    """Build summary statistics for one debug cell's simulation."""
    all_samples = np.asarray(sampled_values, dtype=np.float64)
    finite = all_samples[np.isfinite(all_samples) & (all_samples > 0)]

    base = {
        "cell": fit_result["cell"],
        "chosen_model": fit_result.get("chosen_model"),
        "simulation_count": int(all_samples.size),
        "finite_sample_count": int(finite.size),
    }

    if finite.size == 0:
        return {
            **base,
            "sampled_values": [],
            "first_20_samples": [],
            "sample_mean": None,
            "sample_std": None,
            "sample_min": None,
            "sample_max": None,
            "percentiles": {k: None for k in ("p05", "p25", "p50", "p75", "p95")},
        }

    pcts = np.percentile(finite, [5, 25, 50, 75, 95])
    return {
        **base,
        "sampled_values": all_samples.tolist(),
        "first_20_samples": finite[:20].tolist(),
        "sample_mean": float(np.mean(finite)),
        "sample_std": float(np.std(finite, ddof=0)),
        "sample_min": float(np.min(finite)),
        "sample_max": float(np.max(finite)),
        "percentiles": {
            "p05": float(pcts[0]),
            "p25": float(pcts[1]),
            "p50": float(pcts[2]),
            "p75": float(pcts[3]),
            "p95": float(pcts[4]),
        },
    }


def _build_unresolved_cell_warning(fit_result: dict) -> dict:
    return {
        "cell": fit_result["cell"],
        "chosen_model": fit_result.get("chosen_model"),
        "fallback_reason": fit_result.get("fallback_reason"),
        "sample_size": fit_result.get("sample_size"),
    }


def _build_monte_carlo_run_summary(
    data_array: xr.DataArray,
    simulation_count: int,
    imputed_cell_count: int,
    unresolved_cells: list[dict],
) -> dict:
    return {
        "total_reconstructed_datasets": simulation_count,
        "final_dataset_shape": {
            "time": int(data_array.sizes["time"]),
            "lat": int(data_array.sizes["lat"]),
            "lon": int(data_array.sizes["lon"]),
        },
        "cells_imputed_through_monte_carlo": int(imputed_cell_count),
        "unresolved_cell_count": int(len(unresolved_cells)),
        "chlorophyll_hard_cap_mg_per_m3": _CHLOROPHYLL_HARD_CAP,
        "sampling_space": "log-space with single exp() back-transform",
    }


# ---------------------------------------------------------------------------
# Persistence helpers
# ---------------------------------------------------------------------------

def save_selected_monte_carlo_results(
    results: list[dict], output_dir: Path
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    _save_monte_carlo_results_json(
        results, output_dir / "selected_cells_monte_carlo.json"
    )
    _save_monte_carlo_results_csv(
        results, output_dir / "selected_cells_monte_carlo.csv"
    )


def _save_monte_carlo_results_json(results: list[dict], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2)
    return path


def _save_monte_carlo_results_csv(results: list[dict], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "time_index", "time_value", "lat_index", "lat_value",
        "lon_index", "lon_value", "chosen_model", "simulation_count",
        "finite_sample_count", "sample_mean", "sample_std",
        "sample_min", "sample_max",
        "p05", "p25", "p50", "p75", "p95", "first_20_samples",
    ]
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for item in results:
            writer.writerow(_flatten_monte_carlo_result(item))
    return path


def _flatten_monte_carlo_result(result: dict) -> dict:
    cell = result["cell"]
    pcts = result.get("percentiles", {})
    return {
        "time_index": cell["time_index"],
        "time_value": cell["time_value"],
        "lat_index": cell["lat_index"],
        "lat_value": cell["lat_value"],
        "lon_index": cell["lon_index"],
        "lon_value": cell["lon_value"],
        "chosen_model": result.get("chosen_model"),
        "simulation_count": result.get("simulation_count"),
        "finite_sample_count": result.get("finite_sample_count"),
        "sample_mean": result.get("sample_mean"),
        "sample_std": result.get("sample_std"),
        "sample_min": result.get("sample_min"),
        "sample_max": result.get("sample_max"),
        "p05": pcts.get("p05"),
        "p25": pcts.get("p25"),
        "p50": pcts.get("p50"),
        "p75": pcts.get("p75"),
        "p95": pcts.get("p95"),
        "first_20_samples": json.dumps(result.get("first_20_samples", [])),
    }


def _save_unresolved_warnings(
    unresolved_cells: list[dict], summaries_dir: Path
) -> Path:
    summaries_dir.mkdir(parents=True, exist_ok=True)
    path = summaries_dir / "monte_carlo_unresolved_cells.json"
    with path.open("w", encoding="utf-8") as fh:
        json.dump(unresolved_cells, fh, indent=2)
    return path


def _save_monte_carlo_run_summary(summary: dict, summaries_dir: Path) -> Path:
    summaries_dir.mkdir(parents=True, exist_ok=True)
    path = summaries_dir / "monte_carlo_reconstruction_summary.json"
    with path.open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    return path


def _save_reconstructed_datasets(
    datasets: list[xr.DataArray], output_dir: Path
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for ds in datasets:
        sim_idx = int(ds.attrs.get("simulation_index", 0))
        path = output_dir / f"reconstructed_dataset_{sim_idx:03d}.nc"
        ds.to_netcdf(path)


# ---------------------------------------------------------------------------
# Compatibility shim (used by pipeline.py)
# ---------------------------------------------------------------------------

def impute_missing_cells_with_monte_carlo(
    dataset: xr.DataArray,
    fitted_models: list[dict],
    config,
) -> list[xr.DataArray]:
    """Compatibility wrapper; returns only the reconstructed dataset list."""
    return run_full_dataset_monte_carlo(dataset, fitted_models, config)[
        "reconstructed_datasets"
    ]

"""Uncertainty calculation helpers for reconstructed datasets."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import xarray as xr

from .distribution_fit import make_cell_key


def calculate_uncertainty_statistics(
    reconstructed_datasets: list[xr.DataArray],
    raw_observation_support: xr.DataArray,
    post_interpolation_data: xr.DataArray,
    config,
    selected_cells: list[dict] | None = None,
    save_results: bool = True,
) -> dict:
    """Compute full uncertainty maps and selected-cell summaries.

    Provenance semantics:
    - `raw_observation_support` records where the final working grid/time had
      support from the original raw observations after preprocessing but before
      explicit gap filling.
    - `post_interpolation_data` records the dataset after ordered interpolation.
    - Cells still missing after interpolation are the ones later filled by
      Monte Carlo reconstruction.
    """
    if not reconstructed_datasets:
        return {"status": "no_reconstructed_datasets"}

    selected_cells = selected_cells or []
    stacked = np.stack([dataset.values for dataset in reconstructed_datasets], axis=0)

    mean_map = np.nanmean(stacked, axis=0)
    std_map = np.nanstd(stacked, axis=0, ddof=0)
    lower_percentile = float(config.uncertainty_lower_percentile)
    upper_percentile = float(config.uncertainty_upper_percentile)
    lower_percentile_map = np.nanpercentile(stacked, lower_percentile, axis=0)
    upper_percentile_map = np.nanpercentile(stacked, upper_percentile, axis=0)
    mean_data_array = xr.DataArray(
        mean_map,
        coords=post_interpolation_data.coords,
        dims=post_interpolation_data.dims,
        name="reconstructed_mean",
    )

    final_nan_count = int(np.isnan(mean_map).sum())
    selected_cell_summary = extract_selected_cell_uncertainty_summary(
        mean_map,
        std_map,
        lower_percentile_map,
        upper_percentile_map,
        raw_observation_support,
        post_interpolation_data,
        selected_cells,
        lower_percentile,
        upper_percentile,
    )

    summary = {
        "total_reconstructed_datasets_used": int(stacked.shape[0]),
        "dataset_shape": {
            "time": int(post_interpolation_data.sizes["time"]),
            "lat": int(post_interpolation_data.sizes["lat"]),
            "lon": int(post_interpolation_data.sizes["lon"]),
        },
        "final_nan_count_after_complete_reconstruction": final_nan_count,
        "uncertainty_lower_percentile": lower_percentile,
        "uncertainty_upper_percentile": upper_percentile,
        "behavior_for_raw_observation_supported_values": (
            "Values supported by the original raw observations before explicit gap "
            "filling are preserved in every reconstructed dataset, so their ensemble "
            "mean matches the preserved value and their uncertainty metrics collapse "
            "to deterministic values."
        ),
        "selected_cell_summary_count": len(selected_cell_summary),
    }

    if save_results:
        save_uncertainty_outputs(
            mean_map,
            std_map,
            lower_percentile_map,
            upper_percentile_map,
            post_interpolation_data,
            selected_cell_summary,
            summary,
            config,
        )

    return {
        "summary": summary,
        "selected_cell_summary": selected_cell_summary,
        "mean_map": mean_data_array,
    }


def extract_selected_cell_uncertainty_summary(
    mean_map: np.ndarray,
    std_map: np.ndarray,
    lower_percentile_map: np.ndarray,
    upper_percentile_map: np.ndarray,
    raw_observation_support: xr.DataArray,
    post_interpolation_data: xr.DataArray,
    selected_cells: list[dict],
    lower_percentile: float,
    upper_percentile: float,
) -> list[dict]:
    """Extract readable uncertainty summaries for selected debug cells."""
    results = []
    
    time_axis = post_interpolation_data.get_axis_num("time")
    lat_axis = post_interpolation_data.get_axis_num("lat")
    lon_axis = post_interpolation_data.get_axis_num("lon")

    for cell in selected_cells:
        time_index, lat_index, lon_index = make_cell_key(cell)
        
        idx = [0] * post_interpolation_data.ndim
        idx[time_axis] = time_index
        idx[lat_axis] = lat_index
        idx[lon_axis] = lon_index
        idx = tuple(idx)
        
        results.append(
            {
                "time_index": time_index,
                "time_value": cell["time_value"],
                "lat_index": lat_index,
                "lat_value": cell["lat_value"],
                "lon_index": lon_index,
                "lon_value": cell["lon_value"],
                "mean": float(mean_map[idx]),
                "std": float(std_map[idx]),
                "lower_percentile": float(lower_percentile_map[idx]),
                "upper_percentile": float(upper_percentile_map[idx]),
                "lower_percentile_value": lower_percentile,
                "upper_percentile_value": upper_percentile,
                "provenance_status": determine_cell_status(
                    raw_observation_support,
                    post_interpolation_data,
                    time_index,
                    lat_index,
                    lon_index,
                ),
            }
        )
    return results


def determine_cell_status(
    raw_observation_support: xr.DataArray,
    post_interpolation_data: xr.DataArray,
    time_index: int,
    lat_index: int,
    lon_index: int,
) -> str:
    """Determine whether a cell had raw support, was interpolated, or needed Monte Carlo.

    The label refers to the explicit gap-filling stages on the final working
    grid, not to whether the target-grid cell was a direct raw observation.
    """
    raw_support_value = raw_observation_support.isel(time=time_index, lat=lat_index, lon=lon_index).item()
    post_value = post_interpolation_data.isel(time=time_index, lat=lat_index, lon=lon_index).item()

    if np.isfinite(raw_support_value) and float(raw_support_value) > 0.0:
        return "supported_by_raw_observations"
    if np.isfinite(post_value):
        return "filled_by_interpolation"
    return "filled_by_monte_carlo"


def save_uncertainty_outputs(
    mean_map: np.ndarray,
    std_map: np.ndarray,
    lower_percentile_map: np.ndarray,
    upper_percentile_map: np.ndarray,
    reference_data: xr.DataArray,
    selected_cell_summary: list[dict],
    summary: dict,
    config,
) -> None:
    """Save full uncertainty maps as NetCDF without summary table files."""
    reconstructed_dir = Path(config.reconstructed_dir)

    reconstructed_dir.mkdir(parents=True, exist_ok=True)

    uncertainty_dataset = xr.Dataset(
        data_vars={
            "reconstructed_mean": xr.DataArray(
                mean_map,
                coords=reference_data.coords,
                dims=reference_data.dims,
            ),
            "reconstructed_std": xr.DataArray(
                std_map,
                coords=reference_data.coords,
                dims=reference_data.dims,
            ),
            "reconstructed_lower_percentile": xr.DataArray(
                lower_percentile_map,
                coords=reference_data.coords,
                dims=reference_data.dims,
            ),
            "reconstructed_upper_percentile": xr.DataArray(
                upper_percentile_map,
                coords=reference_data.coords,
                dims=reference_data.dims,
            ),
        }
    )
    uncertainty_dataset.to_netcdf(reconstructed_dir / "uncertainty_maps.nc")

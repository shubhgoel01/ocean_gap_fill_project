"""Spatial regridding utilities for chlorophyll data."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import xarray as xr

# Check that lat and lon dimensions are present => Read target resolution from config => prepare coordinates for interpolatiom => build new target lat and lon axes => interpolate data onto that new grid 
def regrid_to_target_latlon(
    data_array: xr.DataArray,
    config,
    save_summary: bool = True,
) -> tuple[xr.DataArray, dict]:

    validate_spatial_dimensions(data_array)
    resolution = float(config.target_grid_resolution)
    prepared = prepare_spatial_coordinates(data_array)

    # Estimate current resolution (assumes uniform grid)
    lat_res = float(abs(prepared["lat"][1] - prepared["lat"][0]))
    lon_res = float(abs(prepared["lon"][1] - prepared["lon"][0]))

    # Compute coarsening factor (how many small pixels per large cell)
    lat_factor = max(1, int(round(resolution / lat_res)))
    lon_factor = max(1, int(round(resolution / lon_res)))

    # 🔥 CORE FIX: use aggregation instead of interpolation
    regridded = prepared.coarsen(
        lat=lat_factor,
        lon=lon_factor,
        boundary="trim"
    ).mean(skipna=True)

    summary = build_spatial_summary(prepared, regridded)

    if save_summary:
        save_spatial_summary(summary, Path(config.summaries_dir))

    return regridded, summary
# NOTE : Above we are not using weighted mean but uisng normal average because 
# closer pixels should have more importance → weighted mean”  :  Here we estimate a value (interpolation)
# But regridding is not interpolation : Regridding = “What is the average value over this AREA?”



# check if the input data array has 'lat' and 'lon' dimensions, and raise an error if not
def validate_spatial_dimensions(data_array: xr.DataArray) -> None:
    """Ensure the input contains standardized spatial dimensions."""
    required_dims = {"lat", "lon"}
    missing_dims = sorted(required_dims.difference(data_array.dims))
    if missing_dims:
        joined = ", ".join(missing_dims)
        raise ValueError(
            f"Spatial regridding requires 'lat' and 'lon' dimensions. Missing: {joined}"
        )


def prepare_spatial_coordinates(data_array: xr.DataArray) -> xr.DataArray:
    prepared = data_array
    # Keep coordinates increasing without the large copy that xarray.sortby can trigger.
    if prepared["lat"].size > 1 and prepared["lat"].values[0] > prepared["lat"].values[-1]:
        prepared = prepared.isel(lat=slice(None, None, -1))
    if prepared["lon"].size > 1 and prepared["lon"].values[0] > prepared["lon"].values[-1]:
        prepared = prepared.isel(lon=slice(None, None, -1))
    return prepared


# note : here input is co-ordinate values, not chlorophyll data values. So we are building new lat and lon axes based on the min and max of the existing coordinates, and the target resolution. This will create a regular grid that covers the same spatial extent as the original data, but with a specified spacing between points.
def build_target_axis(values: np.ndarray, resolution: float) -> np.ndarray:
    min_value = float(np.nanmin(values))
    max_value = float(np.nanmax(values))

    # From the minimum coordinate value to the maximum coordinate value, create a regular axis with fixed spacing.

    target = np.arange(min_value, max_value + (0.5 * resolution), resolution)
    if target.size == 0:
        return np.array([min_value], dtype=float)

    # Make sure that max-value is present in new axis.
    if target[-1] < max_value:
        target = np.append(target, max_value)

    return target.astype(float)


def build_spatial_summary(original: xr.DataArray, regridded: xr.DataArray) -> dict:
    return {
        "original_spatial_shape": {
            "lat": int(original.sizes["lat"]),
            "lon": int(original.sizes["lon"]),
        },
        "new_spatial_shape": {
            "lat": int(regridded.sizes["lat"]),
            "lon": int(regridded.sizes["lon"]),
        },
        "original_lat_range": {
            "min": float(original["lat"].values.min()),
            "max": float(original["lat"].values.max()),
        },
        "original_lon_range": {
            "min": float(original["lon"].values.min()),
            "max": float(original["lon"].values.max()),
        },
        "target_lat_range": {
            "min": float(regridded["lat"].values.min()),
            "max": float(regridded["lat"].values.max()),
        },
        "target_lon_range": {
            "min": float(regridded["lon"].values.min()),
            "max": float(regridded["lon"].values.max()),
        },
    }


def save_spatial_summary(summary: dict, summaries_dir: Path) -> Path:
    """Save the spatial regridding summary as JSON."""
    summaries_dir.mkdir(parents=True, exist_ok=True)
    output_path = summaries_dir / "spatial_regrid_summary.json"
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    return output_path

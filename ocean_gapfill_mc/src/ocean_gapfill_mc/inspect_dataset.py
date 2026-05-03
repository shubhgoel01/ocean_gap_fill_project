"""Dataset inspection and reporting utilities for Phase 1 outputs."""

from __future__ import annotations

import numpy as np
import xarray as xr


def inspect_phase1_dataset(
    data_array: xr.DataArray,
    label: str,
    config,
    save_outputs: bool = True,
) -> dict:
    """Inspect a processed DataArray and return presentation-ready summaries.

    Parameters
    ----------
    data_array:
        Input chlorophyll data after loading and regridding.
    label:
        Short label used in output filenames, for example
        `phase1_regridded` or `before_interpolation`.
    config:
        Application config object.
    save_outputs:
        Kept for call compatibility. Summary files are no longer written.
    """
    validate_inspection_input(data_array)

    time_size = int(data_array.sizes["time"])
    lat_size = int(data_array.sizes["lat"])
    lon_size = int(data_array.sizes["lon"])

    values = data_array.values
    nan_mask = np.isnan(values)

    total_cells = int(values.size)
    nan_cells = int(nan_mask.sum())
    valid_cells = int(total_cells - nan_cells)
    nan_percent = float((nan_cells / total_cells) * 100.0) if total_cells else 0.0

    time_values = np.asarray(data_array["time"].values)
    nan_percent_per_time = compute_nan_percent_per_time(data_array)

    summary = {
        "label": label,
        "shape": {
            "time": time_size,
            "lat": lat_size,
            "lon": lon_size,
        },
        "total_cells": total_cells,
        "valid_cells": valid_cells,
        "nan_cells": nan_cells,
        "nan_percent": round(nan_percent, 4),
        "nan_percent_per_time": nan_percent_per_time,
    }

    return summary


def validate_inspection_input(data_array: xr.DataArray) -> None:
    """Ensure the input has the expected standardized dimensions."""
    required_dims = {"time", "lat", "lon"}
    missing_dims = sorted(required_dims.difference(data_array.dims))
    if missing_dims:
        joined = ", ".join(missing_dims)
        raise ValueError(
            f"Dataset inspection requires time, lat, and lon dimensions. Missing: {joined}"
        )


def compute_nan_percent_per_time(data_array: xr.DataArray) -> list[dict]:
    """Compute NaN percentage for each time slice."""
    nan_mask = np.isnan(data_array.values)
    nan_percent_values = nan_mask.mean(axis=(1, 2)) * 100.0

    return [
        {
            "time": str(time_value),
            "nan_percent": round(float(nan_percent), 4),
        }
        for time_value, nan_percent in zip(data_array["time"].values, nan_percent_values)
    ]


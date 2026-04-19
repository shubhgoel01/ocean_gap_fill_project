# Its job is to take a chlorophyll DataArray that still has missing values (NaNs) and try to fill some of them using nearby known values.
# It uses simple rules : 
# - along one axis at a time
# - look only at the two immediate neighbors
# - if both neighbors exist, take their average
# - if only one exists, copy that one
# - if neither exists, leave it as NaN

# validate input dimensions => count NaNs before interpolation => apply interpolation in a specific order (lon, lat, time) => count NaNs after interpolation => build summary of results => save summary as JSON and text report
# NOTE : We have used orderd interpolation, this means : first fill along longitude, then use that updated result to fill along latitude, then use that updated result to fill along time.
# NOTE : But inside one single pass, newly filled values are not allowed to influence other fills in that same pass.

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import xarray as xr


def apply_ordered_interpolation(
    data_array: xr.DataArray,
    config,
    save_summary: bool = True,
) -> xr.DataArray:
    validate_interpolation_input(data_array)

    before_nan_count = count_nan_cells(data_array.values)
    working = data_array.copy(deep=True)                # Created a deep copy to make sure original remains unhcanged.
    pass_summaries: list[dict] = []                     # list to store summary of each pass

    # After each pass, it stores: axis name, how many cells filled in that pass, how many NaNs still remain
    for axis_name in ("lon", "lat", "time"):
        working, filled_in_pass = apply_single_axis_pass(working, axis_name)
        pass_summaries.append(
            {
                "axis": axis_name,
                "filled_cells": filled_in_pass,
                "remaining_nan_count": count_nan_cells(working.values),
            }
        )

    after_nan_count = count_nan_cells(working.values)
    filled_cells = int(before_nan_count - after_nan_count)
    total_cells = int(working.values.size)
    remaining_nan_percent = (
        float((after_nan_count / total_cells) * 100.0) if total_cells else 0.0
    )

    summary = {
        "interpolation_order": ["longitude", "latitude", "time"],
        "nan_count_before": before_nan_count,
        "nan_count_after": after_nan_count,
        "filled_cells": filled_cells,
        "remaining_nan_percent": round(remaining_nan_percent, 4),
        "pass_summaries": pass_summaries,
    }

    working.attrs = dict(working.attrs)
    working.attrs["interpolation_summary"] = json.dumps(summary)

    if save_summary:
        save_interpolation_summary(summary, Path(config.summaries_dir))

    return working

# It checks whether those three dimensions are present. If not, raises ValueError with exact missing dimension names
def validate_interpolation_input(data_array: xr.DataArray) -> None:
    """Ensure the input has the expected standardized dimensions."""
    required_dims = {"time", "lat", "lon"}
    missing_dims = sorted(required_dims.difference(data_array.dims))
    if missing_dims:
        joined = ", ".join(missing_dims)
        raise ValueError(
            "Interpolation requires a DataArray with time, lat, and lon "
            f"dimensions. Missing: {joined}"
        )

# This applies interpolation for just one axis.
def apply_single_axis_pass(
    data_array: xr.DataArray,
    axis_name: str,
) -> tuple[xr.DataArray, int]:
    """Apply one neighbor-based interpolation pass along a single axis."""
    axis_index = data_array.get_axis_num(axis_name)
    original_values = np.asarray(data_array.values, dtype=float)
    filled_values, filled_count = interpolate_missing_along_axis(
        original_values,
        axis=axis_index,
    )

    filled_data_array = data_array.copy(deep=True)      # copy original data to new array
    filled_data_array.values = filled_values            # update/replace old values with new interpolated values
    return filled_data_array, filled_count


def interpolate_missing_along_axis(values: np.ndarray, axis: int) -> tuple[np.ndarray, int]:
    
    # here we are creating two copies, 'source' is used only for reading original values, while 'result' is where we write the new interpolated values. This way, we ensure that newly filled values in 'result' do not influence other fills in the same pass.
    source = np.array(values, dtype=float, copy=True)
    result = np.array(values, dtype=float, copy=True)
    filled_count = 0

    # Before performig the interpolation, we move the chosen 'axis' to the last position. 
    # This is because, trhe interpolation logic becomes easier if the axis we are working on is always the last axis.
    # Also, we can now apply the same logic for any axis.
    source_moved = np.moveaxis(source, axis, -1)
    result_moved = np.moveaxis(result, axis, -1)
    trailing_size = source_moved.shape[-1]

    for index in np.ndindex(source_moved.shape[:-1]):
        line = source_moved[index]
        output_line = result_moved[index]

        for position in range(trailing_size):
            if is_valid_value(line[position]):
                continue

            left_value = line[position - 1] if position > 0 else np.nan
            right_value = line[position + 1] if position < trailing_size - 1 else np.nan
            replacement = compute_neighbor_fill_value(left_value, right_value)

            if is_valid_value(replacement):
                output_line[position] = replacement
                filled_count += 1

    # Now we move the axis back to its original position to restore the original shape of the data array.
    restored = np.moveaxis(result_moved, -1, axis)
    return restored, filled_count


def compute_neighbor_fill_value(left_value: float, right_value: float) -> float:
    """Compute the replacement value from immediate neighbors."""
    left_valid = is_valid_value(left_value)
    right_valid = is_valid_value(right_value)

    if left_valid and right_valid:
        return float((left_value + right_value) / 2.0)
    if left_valid:
        return float(left_value)
    if right_valid:
        return float(right_value)
    return np.nan


# Checks weather a value is finite and usable for interpolation. This means it is not NaN or infinite.
def is_valid_value(value: float) -> bool:
    """Return True when a value is finite and usable for interpolation."""
    return bool(np.isfinite(value))


def count_nan_cells(values: np.ndarray) -> int:
    """Count NaN cells in the provided array."""
    return int(np.isnan(values).sum())


def save_interpolation_summary(summary: dict, summaries_dir: Path) -> None:
    """Save interpolation results as JSON and a readable text report."""
    summaries_dir.mkdir(parents=True, exist_ok=True)

    json_path = summaries_dir / "interpolation_summary.json"
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    text_path = summaries_dir / "interpolation_summary.txt"
    lines = [
        "Interpolation Summary",
        "=====================",
        "Interpolation order: longitude -> latitude -> time",
        f"NaN count before interpolation: {summary['nan_count_before']}",
        f"NaN count after interpolation: {summary['nan_count_after']}",
        f"Number of cells filled: {summary['filled_cells']}",
        f"Remaining NaN percentage: {summary['remaining_nan_percent']:.4f}%",
        "",
        "Per-pass results:",
    ]

    for item in summary["pass_summaries"]:
        lines.append(
            f"  {item['axis']}: filled {item['filled_cells']} cells, "
            f"remaining NaNs {item['remaining_nan_count']}"
        )

    with text_path.open("w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")

"""Select reproducible debug cells for reporting and plotting."""

from __future__ import annotations

import csv
import json
from pathlib import Path
import random

import numpy as np
import xarray as xr

# Pick fww missing cells randomly for output and debugging.
# Validate input => Validate input => Find missing cells => Read how many cells to select
def select_debug_cells(
    data_array: xr.DataArray,
    config,
    save_outputs: bool = True,
) -> list[dict]:
    validate_selection_input(data_array)

    missing_indices = np.argwhere(np.isnan(data_array.values))
    requested_count = int(config.sampled_cell_count)

    if missing_indices.size == 0:
        selected_cells: list[dict] = []
    else:
        rng = random.Random(config.random_seed)                         # Creates a random generator with a fixed seed                     
        selection_count = min(requested_count, len(missing_indices))    # Choose min(config.selection_count, number of missing cells that are available in dataset)
        
        # If x = 100, then range(x) = [0,1,2.....99] , Then next select some indices out of these
        chosen_positions = rng.sample(range(len(missing_indices)), selection_count)
        selected_cells = [
            build_cell_record(data_array, missing_indices[position])
            for position in chosen_positions
        ]
        # Now selected cells is an array of dictionries.

    if save_outputs:
        save_selected_cells_json(selected_cells, Path(config.sampled_cells_dir))
        save_selected_cells_csv(selected_cells, Path(config.sampled_cells_dir))

    return selected_cells


def select_monte_carlo_filled_debug_cells(
    post_interpolation_data: xr.DataArray,
    reconstructed_data: xr.DataArray,
    config,
    save_outputs: bool = True,
) -> list[dict]:
    """Select cells that were actually filled by Monte Carlo reconstruction.

    A selected cell must be missing after interpolation and finite in a
    reconstructed output. This avoids choosing cells that interpolation already
    filled or cells that remained unresolved.
    """
    validate_selection_input(post_interpolation_data)
    validate_selection_input(reconstructed_data)

    missing_before_monte_carlo = np.isnan(post_interpolation_data.values)
    finite_after_monte_carlo = np.isfinite(reconstructed_data.values)
    monte_carlo_filled_mask = missing_before_monte_carlo & finite_after_monte_carlo
    filled_indices = np.argwhere(monte_carlo_filled_mask)
    requested_count = int(config.sampled_cell_count)

    if filled_indices.size == 0:
        selected_cells: list[dict] = []
    else:
        rng = random.Random(config.random_seed)
        selection_count = min(requested_count, len(filled_indices))
        chosen_positions = rng.sample(range(len(filled_indices)), selection_count)
        selected_cells = [
            build_cell_record(post_interpolation_data, filled_indices[position])
            for position in chosen_positions
        ]

    if save_outputs:
        save_selected_cells_json(selected_cells, Path(config.sampled_cells_dir))
        save_selected_cells_csv(selected_cells, Path(config.sampled_cells_dir))

    return selected_cells

# Check that input data has all required dimensions: time, lat, lon
def validate_selection_input(data_array: xr.DataArray) -> None:
    required_dims = {"time", "lat", "lon"}
    missing_dims = sorted(required_dims.difference(data_array.dims))
    if missing_dims:
        joined = ", ".join(missing_dims)
        raise ValueError(
            "Debug cell selection requires a DataArray with time, lat, and lon "
            f"dimensions. Missing: {joined}"
        )

# Convert one raw missing-cell index like [time_index, lat_index, lon_index] into structured dictionary.
def build_cell_record(data_array: xr.DataArray, index_triplet: np.ndarray) -> dict:
    time_index = int(index_triplet[0])
    lat_index = int(index_triplet[1])
    lon_index = int(index_triplet[2])

    time_value = data_array["time"].values[time_index]
    lat_value = data_array["lat"].values[lat_index]
    lon_value = data_array["lon"].values[lon_index]

    return {
        "time_index": time_index,
        "time_value": str(time_value),
        "lat_index": lat_index,
        "lat_value": float(lat_value),
        "lon_index": lon_index,
        "lon_value": float(lon_value),
    }


def save_selected_cells_json(selected_cells: list[dict], output_dir: Path) -> Path:
    """Save selected debug cells as JSON."""
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "selected_debug_cells.json"
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(selected_cells, handle, indent=2)
    return output_path

# Saves selected debug cells into a CSV file so these can be viewed easily in tabular format.
def save_selected_cells_csv(selected_cells: list[dict], output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "selected_debug_cells.csv"
    fieldnames = [
        "time_index",
        "time_value",
        "lat_index",
        "lat_value",
        "lon_index",
        "lon_value",
    ]

    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in selected_cells:
            writer.writerow(row)

    return output_path

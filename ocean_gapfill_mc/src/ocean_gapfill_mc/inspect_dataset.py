"""Dataset inspection and reporting utilities for Phase 1 outputs."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr


def inspect_phase1_dataset(
    data_array: xr.DataArray,
    label: str,
    config,
    save_outputs: bool = True,
    save_plot: bool | None = None,
) -> dict:
    """Inspect a processed DataArray and save presentation-ready summaries.

    Parameters
    ----------
    data_array:
        Input chlorophyll data after compositing and regridding.
    label:
        Short label used in output filenames, for example
        `phase1_regridded` or `before_interpolation`.
    config:
        Application config object.
    save_outputs:
        If True, save JSON and text summaries.
    save_plot:
        If True, save a NaN percentage over time plot. If None, it follows
        `config.save_plots`.
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

    if save_outputs:
        save_inspection_json(summary, Path(config.summaries_dir), label)
        save_inspection_report(summary, Path(config.summaries_dir), label)

    should_save_plot = config.save_plots if save_plot is None else save_plot
    if should_save_plot:
        plot_nan_percent_over_time(
            time_values,
            nan_percent_per_time,
            Path(config.plots_dir),
            label,
        )

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
    results: list[dict] = []

    for time_value in data_array["time"].values:
        time_slice = data_array.sel(time=time_value)
        slice_values = time_slice.values
        total_cells = int(slice_values.size)
        nan_cells = int(np.isnan(slice_values).sum())
        nan_percent = float((nan_cells / total_cells) * 100.0) if total_cells else 0.0

        results.append(
            {
                "time": str(time_value),
                "nan_percent": round(nan_percent, 4),
            }
        )

    return results


def save_inspection_json(summary: dict, summaries_dir: Path, label: str) -> Path:
    """Save the inspection summary as JSON."""
    summaries_dir.mkdir(parents=True, exist_ok=True)
    output_path = summaries_dir / f"{label}_inspection_summary.json"
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    return output_path


def save_inspection_report(summary: dict, summaries_dir: Path, label: str) -> Path:
    """Save a readable text report for quick review."""
    summaries_dir.mkdir(parents=True, exist_ok=True)
    output_path = summaries_dir / f"{label}_inspection_report.txt"

    lines = [
        "Dataset Inspection Report",
        "=========================",
        f"Label: {summary['label']}",
        "",
        "Shape",
        f"  Time: {summary['shape']['time']}",
        f"  Latitude: {summary['shape']['lat']}",
        f"  Longitude: {summary['shape']['lon']}",
        "",
        "Cell Counts",
        f"  Total cells: {summary['total_cells']}",
        f"  Valid cells: {summary['valid_cells']}",
        f"  NaN cells: {summary['nan_cells']}",
        f"  NaN percentage: {summary['nan_percent']:.4f}%",
        "",
        "NaN Percentage Per Time Slice",
    ]

    for item in summary["nan_percent_per_time"]:
        lines.append(f"  {item['time']}: {item['nan_percent']:.4f}%")

    with output_path.open("w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")

    return output_path


def plot_nan_percent_over_time(
    time_values: np.ndarray,
    nan_percent_per_time: list[dict],
    plots_dir: Path,
    label: str,
) -> Path:
    """Plot NaN percentage over time using true datetime-like coordinates."""
    plots_dir.mkdir(parents=True, exist_ok=True)
    output_path = plots_dir / f"{label}_nan_percent_over_time.png"

    nan_values = [item["nan_percent"] for item in nan_percent_per_time]

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(time_values, nan_values, color="#1f6f8b", linewidth=1.8)
    ax.set_title("NaN Percentage Over Time")
    ax.set_xlabel("Time")
    ax.set_ylabel("NaN Percentage (%)")
    ax.grid(True, alpha=0.3)
    fig.autofmt_xdate()
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)

    return output_path

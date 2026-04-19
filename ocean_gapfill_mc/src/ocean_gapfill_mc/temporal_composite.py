# This file takes daily data with a time dimension and converts it into larger time blocks like 3-day, 8-day, etc. composites.
# checks that time dimension exists => reads composite settings from config => groups data into time windows using resample() => aggregates each window using mean => creates and saves summary/output


from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import xarray as xr


def create_temporal_composites(
    data_array: xr.DataArray,
    config,
    aggregation_method: str = "mean",
    save_output: bool = True,
    save_summary: bool = True,
) -> tuple[xr.DataArray, dict]:

    validate_time_dimension(data_array)                                 # This checks that input actually has a time dimension.

    window_days = int(config.composite_window_size)                     # read window_size from config and convert to int.
    
    # groups data, if window_days=8, then it groups data into 8-day blocks. So if we have daily data, it will create groups of 8 days each. 
    # note below line does not merge data for now, it just creates windows of eg. 3-days. These windows are then passed to aggregate_resampled_data() which applies the aggregation method (eg. mean) to each window to create the final composite.
    resampled = data_array.resample(time=f"{window_days}D")             
    composite = aggregate_resampled_data(
        resampled,
        aggregation_method,
        min_valid_fraction=float(config.composite_min_valid_fraction),
        window_days=window_days,
    )

    # Creates a meta-data dictionery describing : Original number of time steps, composite number of time steps, composite window size in days, minimum valid fraction used for compositing, minimum valid observations required for a composite, original time coverage (start and end), composite time coverage (start and end).
    summary = build_temporal_summary(
        data_array,
        composite,
        min_valid_fraction=float(config.composite_min_valid_fraction),
        window_days=window_days,
    )

    if save_output:
        save_composite_dataset(composite, Path("data/processed"))
    if save_summary:
        save_temporal_summary(summary, Path(config.summaries_dir))

    return composite, summary


def validate_time_dimension(data_array: xr.DataArray) -> None:
    """Ensure the input DataArray includes a time dimension."""
    if "time" not in data_array.dims:
        raise ValueError(
            "Temporal compositing requires a DataArray with a 'time' dimension."
        )


def aggregate_resampled_data(   
    resampled: xr.core.resample.DataArrayResample,
    aggregation_method: str,
    min_valid_fraction: float,
    window_days: int,
) -> xr.DataArray:
    """Apply the requested aggregation method to resampled data."""
    if aggregation_method == "mean":
        # To ensure that we only compute the mean for windows that have enough valid data, we calculate the minimum number of valid observations required based on the window size and the minimum valid fraction. For example, if we have an 8-day window and a minimum valid fraction of 0.6, then we require at least 5 valid observations (8 * 0.6 = 4.8, rounded up to 5) in that window to compute the mean. If a window has fewer than 5 valid observations, the composite value for that window will be set to NaN.
        # this ensures there are still some missing fields. 
        minimum_valid_observations = max(1, int(np.ceil(window_days * min_valid_fraction))) 
        # below line actually merges the data for each window using mean, but note 'skipna=true' means ignore NaNs while averaging.            
        composite = resampled.mean(skipna=True)
        # This counts how many non-NaN values exist in each window. If the count of valid observations in a window is less than the minimum required, we set the composite value for that window to NaN.
        valid_counts = resampled.count(dim="time")
        # Below returns the composite DataArray applying a filter, keep composite value only where enough valid observations exist, otherwise set it to NaN
        return composite.where(valid_counts >= minimum_valid_observations)
    raise ValueError(
        f"Unsupported aggregation method: {aggregation_method}. "
        "Currently supported methods: mean"
    )


# This function creates a meta-data dictionery describing : Original number of time steps, composite number of time steps, composite window size in days, minimum valid fraction used for compositing, minimum valid observations required for a composite, original time coverage (start and end), composite time coverage (start and end).
def build_temporal_summary(
    original: xr.DataArray,
    composite: xr.DataArray,
    min_valid_fraction: float,
    window_days: int,
) -> dict:
    """Build a compact summary of the temporal aggregation."""
    return {
        "original_time_steps": int(original.sizes["time"]),
        "composite_time_steps": int(composite.sizes["time"]),
        "composite_window_size_days": int(window_days),
        "composite_min_valid_fraction": float(min_valid_fraction),
        "composite_min_valid_observations": max(
            1,
            int(np.ceil(window_days * min_valid_fraction)),
        ),
        "original_time_coverage": {
            "start": str(original["time"].values.min()),
            "end": str(original["time"].values.max()),
        },
        "composite_time_coverage": {
            "start": str(composite["time"].values.min()),
            "end": str(composite["time"].values.max()),
        },
    }


def save_composite_dataset(composite: xr.DataArray, processed_dir: Path) -> Path:
    """Save the temporal composite as a NetCDF file in the processed folder."""
    processed_dir.mkdir(parents=True, exist_ok=True)
    output_path = processed_dir / "chlorophyll_8day_composite.nc"
    composite.to_netcdf(output_path)
    return output_path


def save_temporal_summary(summary: dict, summaries_dir: Path) -> Path:
    """Save the compositing summary as JSON."""
    summaries_dir.mkdir(parents=True, exist_ok=True)
    output_path = summaries_dir / "temporal_composite_summary.json"
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    return output_path

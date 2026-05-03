"""ESA OC-CCI chlorophyll data loading utilities."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr


ESA_OC_CCI_PATTERN = "ESACCI-OC-L3S-CHLOR_A-MERGED-8D_DAILY_4km_GEO_PML_*.nc"
DEFAULT_CHLOROPHYLL_VARIABLE = "chlor_a"


def load_chlorophyll_data(config) -> xr.DataArray:
    preprocessed_path = getattr(config, "preprocessed_data_file", None)
    if preprocessed_path:
        return load_preprocessed_chlorophyll_data(config)

    return load_raw_chlorophyll_data(config)


def load_raw_chlorophyll_data(config) -> xr.DataArray:
    """Load raw multi-file chlorophyll data and crop it to the study domain."""
    input_dir = Path(config.input_directory)
    if not input_dir.exists():
        raise FileNotFoundError(f"Input NetCDF directory not found: {input_dir}")
    if not input_dir.is_dir():
        raise NotADirectoryError(f"Input NetCDF path is not a directory: {input_dir}")

    file_pattern = str(input_dir / ESA_OC_CCI_PATTERN)

    try:
        dataset = xr.open_mfdataset(
            file_pattern,
            combine="by_coords",
            data_vars="minimal",
            coords="minimal",
            compat="override",
            chunks={"time": 1},
        )
    except Exception as exc:
        raise ValueError(f"Failed to open ESA OC-CCI NetCDF files: {file_pattern}") from exc

    variable_name = getattr(config, "variable_name", DEFAULT_CHLOROPHYLL_VARIABLE)
    if variable_name not in dataset.data_vars:
        available = ", ".join(dataset.data_vars) or "none"
        raise ValueError(
            "Chlorophyll variable "
            f"'{variable_name}' was not found in {input_dir}. "
            f"Available variables: {available}"
        )

    chlorophyll = dataset[variable_name]
    chlorophyll = ensure_datetime_time(chlorophyll)
    chlorophyll = crop_to_study_area(chlorophyll, config)
    validate_required_dimensions(chlorophyll)
    print_basic_metadata(chlorophyll, input_dir)
    return chlorophyll


def load_preprocessed_chlorophyll_data(config) -> xr.DataArray:
    """Load a previously merged and cropped chlorophyll NetCDF file."""
    preprocessed_path = Path(config.preprocessed_data_file)
    if not preprocessed_path.exists():
        raise FileNotFoundError(
            f"Preprocessed NetCDF file not found: {preprocessed_path}. "
            "Run scripts/preprocess_data.py first or remove preprocessed_data_file "
            "from the config to load raw files directly."
        )
    if not preprocessed_path.is_file():
        raise ValueError(f"Preprocessed NetCDF path is not a file: {preprocessed_path}")

    try:
        dataset = xr.open_dataset(preprocessed_path)
    except Exception as exc:
        raise ValueError(f"Failed to open preprocessed NetCDF file: {preprocessed_path}") from exc

    variable_name = getattr(config, "variable_name", DEFAULT_CHLOROPHYLL_VARIABLE)
    if variable_name not in dataset.data_vars:
        available = ", ".join(dataset.data_vars) or "none"
        raise ValueError(
            "Chlorophyll variable "
            f"'{variable_name}' was not found in {preprocessed_path}. "
            f"Available variables: {available}"
        )

    chlorophyll = dataset[variable_name]
    chlorophyll = ensure_datetime_time(chlorophyll)
    validate_required_dimensions(chlorophyll)
    print_basic_metadata(chlorophyll, preprocessed_path)
    return chlorophyll


def crop_to_study_area(data_array: xr.DataArray, config) -> xr.DataArray:
    """Crop a DataArray to the configured study-area domain."""
    bounds = get_study_area_bounds(config)
    cropped = data_array.sel(
        lat=coordinate_order_aware_slice(
            data_array["lat"],
            bounds["latitude_min"],
            bounds["latitude_max"],
        ),
        lon=coordinate_order_aware_slice(
            data_array["lon"],
            bounds["longitude_min"],
            bounds["longitude_max"],
        ),
    )
    print(
        "Cropped to study area "
        f"lat {bounds['latitude_min']} to {bounds['latitude_max']}, "
        f"lon {bounds['longitude_min']} to {bounds['longitude_max']}; "
        f"shape: {dict(cropped.sizes)}"
    )
    return cropped


def crop_to_tropical_indian_ocean(data_array: xr.DataArray) -> xr.DataArray:
    """Crop a DataArray to the default tropical Indian Ocean domain."""
    class DefaultConfig:
        study_area_bounds = default_study_area_bounds()

    return crop_to_study_area(data_array, DefaultConfig())


def get_study_area_bounds(config) -> dict[str, float]:
    bounds = getattr(config, "study_area_bounds", None) or default_study_area_bounds()
    return {
        "latitude_min": float(bounds["latitude_min"]),
        "latitude_max": float(bounds["latitude_max"]),
        "longitude_min": float(bounds["longitude_min"]),
        "longitude_max": float(bounds["longitude_max"]),
    }


def default_study_area_bounds() -> dict[str, float]:
    return {
        "latitude_min": -30.0,
        "latitude_max": 30.0,
        "longitude_min": 40.0,
        "longitude_max": 120.0,
    }


def coordinate_order_aware_slice(coordinate: xr.DataArray, lower: float, upper: float) -> slice:
    values = np.asarray(coordinate.values, dtype=float)
    if values.size == 0 or values[0] <= values[-1]:
        return slice(lower, upper)
    return slice(upper, lower)


# The input files already carry the desired temporal product. This only standardizes
# the time coordinate so later stages can rely on datetime values.
def ensure_datetime_time(data_array: xr.DataArray) -> xr.DataArray:
    if "time" not in data_array.coords:
        raise ValueError("Missing required time coordinate after dimension standardization.")

    try:
        parsed_time = pd.to_datetime(data_array["time"].values)
    except Exception as exc:
        raise ValueError("Could not parse the time coordinate as datetime values.") from exc

    return data_array.assign_coords(time=parsed_time)

# this now validates that the dataArray contains all three required dimensions.
def validate_required_dimensions(data_array: xr.DataArray) -> None:
    """Validate that the standardized DataArray has time, lat, and lon."""
    required_dims = {"time", "lat", "lon"}
    missing_dims = sorted(required_dims.difference(data_array.dims))
    if missing_dims:
        joined = ", ".join(missing_dims)
        raise ValueError(
            f"Missing required dimensions after standardization: {joined}. "
            f"Found dimensions: {tuple(data_array.dims)}"
        )

# This prints a compact summary of the loaded data. Display :  Which file is used, which variable was extracted, dataset shape, latitude range, longitude range, time range.
def print_basic_metadata(data_array: xr.DataArray, input_path: Path) -> None:
    """Print a compact metadata summary for the loaded chlorophyll field."""
    lat_values = data_array["lat"].values
    lon_values = data_array["lon"].values
    time_values = data_array["time"].values

    print("Loaded chlorophyll dataset")
    print(f"  File: {input_path}")
    print(f"  Variable: {data_array.name}")
    print(f"  Dims: {dict(data_array.sizes)}")
    print(f"  Latitude range: {lat_values.min()} to {lat_values.max()}")
    print(f"  Longitude range: {lon_values.min()} to {lon_values.max()}")
    print(f"  Time range: {time_values.min()} to {time_values.max()}")

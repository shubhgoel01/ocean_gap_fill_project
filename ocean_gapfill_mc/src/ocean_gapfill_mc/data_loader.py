"""NetCDF data loading utilities for chlorophyll datasets.

This module focuses on one job: opening a NetCDF file with xarray,
as the NetCDF file may contain many variables, picks the required chlorophyll variable, cleans its dimension names, validates it, and returns that single variable as a DataArray.
"""

from __future__ import annotations

from pathlib import Path 

import pandas as pd 
import xarray as xr

# read input path from config => check file exists => open netcdf with xarray => check that required chlorophyll variable exists => extract that variable => standardize dimension names => ensure time is datetime => validate required dimensions => print metadata summary => return DataArray

def load_chlorophyll_data(config) -> xr.DataArray:

    input_path = Path(config.input_file)
    if not input_path.exists():
        raise FileNotFoundError(f"Input NetCDF file not found: {input_path}")

    try:
        dataset = xr.open_dataset(input_path)
    except Exception as exc:
        raise ValueError(f"Failed to open NetCDF file: {input_path}") from exc

    if config.variable_name not in dataset.data_vars:
        available = ", ".join(dataset.data_vars) or "none"
        raise ValueError(
            "Chlorophyll variable "
            f"'{config.variable_name}' was not found in {input_path}. "
            f"Available variables: {available}"
        )

    chlorophyll = dataset[config.variable_name]
    chlorophyll = standardize_dimension_names(chlorophyll, config)
    chlorophyll = ensure_datetime_time(chlorophyll)
    validate_required_dimensions(chlorophyll)
    print_basic_metadata(chlorophyll, input_path)
    return chlorophyll


# This function converts whatever dimension names your raw file uses into the standard names: time, lat, lon which are later used in rest of the pipeline. It checks if any dimension exist with the name stated in the config and then compares its name with the standard name. If they are different, it renames that dimension to the standard name. 
# So from now onwards, the rest of the pipeline uses statndard dimension names : time, lat, lon. 
def standardize_dimension_names(data_array: xr.DataArray, config) -> xr.DataArray:
    rename_map: dict[str, str] = {}

    if config.time_dim in data_array.dims and config.time_dim != "time":
        rename_map[config.time_dim] = "time"
    if config.latitude_dim in data_array.dims and config.latitude_dim != "lat":
        rename_map[config.latitude_dim] = "lat"
    if config.longitude_dim in data_array.dims and config.longitude_dim != "lon":
        rename_map[config.longitude_dim] = "lon"

    if rename_map:
        data_array = data_array.rename(rename_map)

    coord_rename_map: dict[str, str] = {}
    if config.time_dim in data_array.coords and config.time_dim != "time":
        coord_rename_map[config.time_dim] = "time"
    if config.latitude_dim in data_array.coords and config.latitude_dim != "lat":
        coord_rename_map[config.latitude_dim] = "lat"
    if config.longitude_dim in data_array.coords and config.longitude_dim != "lon":
        coord_rename_map[config.longitude_dim] = "lon"

    if coord_rename_map:
        data_array = data_array.rename(coord_rename_map)

    return data_array


# because raw data is downloaded from NASA website on daily basis and then merged, it becomes important time coordinate exists and is converted into proper datetime format.
# It takes the raw time values and asks pandas to convert them into datetime objects, hence now time format is also standardized.
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

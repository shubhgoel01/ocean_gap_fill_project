"""ESA OC-CCI chlorophyll data loading utilities."""

from __future__ import annotations

from pathlib import Path
import math

import numpy as np
import pandas as pd
import xarray as xr


ESA_OC_CCI_PATTERN = "ESACCI-OC-L3S-CHLOR_A-MERGED-8D_DAILY_4km_GEO_PML_*.nc"
DEFAULT_RAW_FILE_PATTERNS = (
    ESA_OC_CCI_PATTERN,
    "*.nc4",
    "*.nc",
)
DEFAULT_CHLOROPHYLL_VARIABLE = "chlor_a"


def load_chlorophyll_data(config) -> xr.DataArray:
    preprocessed_path = getattr(config, "preprocessed_data_file", None)
    if preprocessed_path:
        return load_preprocessed_chlorophyll_data(config)

    return load_raw_chlorophyll_data(config)


def load_raw_chlorophyll_data(config, inspection_callback=None) -> xr.DataArray:
    """Load raw multi-file chlorophyll data and crop it to the study domain."""
    input_dir = Path(config.input_directory)
    if not input_dir.exists():
        raise FileNotFoundError(f"Input NetCDF directory not found: {input_dir}")
    if not input_dir.is_dir():
        raise NotADirectoryError(f"Input NetCDF path is not a directory: {input_dir}")

    print(f"[preprocess] Finding raw NetCDF files in {input_dir}")
    raw_files = find_raw_netcdf_files(input_dir, config)

    print(f"[preprocess] Opening {len(raw_files)} raw NetCDF file(s)")
    try:
        dataset = xr.open_mfdataset(
            [str(path) for path in raw_files],
            combine="by_coords",
            data_vars="minimal",
            coords="minimal",
            compat="override",
            chunks={"time": 1},
        )
    except Exception as exc:
        raise ValueError(
            f"Failed to open raw NetCDF files from {input_dir}. "
            f"Matched {len(raw_files)} file(s)."
        ) from exc

    variable_name = getattr(config, "variable_name", DEFAULT_CHLOROPHYLL_VARIABLE)
    if variable_name not in dataset.data_vars:
        available = ", ".join(dataset.data_vars) or "none"
        raise ValueError(
            "Chlorophyll variable "
            f"'{variable_name}' was not found in {input_dir}. "
            f"Available variables: {available}"
        )

    print(f"[preprocess] Selecting variable '{variable_name}'")
    chlorophyll = standardize_dimension_names(dataset[variable_name], config)
    print("[preprocess] Standardizing time coordinate")
    chlorophyll = ensure_datetime_time(chlorophyll)
    if inspection_callback is not None:
        inspection_callback(chlorophyll, "raw_load")
    print("[preprocess] Applying study-area crop setting")
    chlorophyll = maybe_crop_to_study_area(chlorophyll, config)
    if inspection_callback is not None:
        inspection_callback(chlorophyll, "study_area_crop")
    print("[preprocess] Applying 8-day compositing setting")
    chlorophyll = maybe_apply_8day_compositing(chlorophyll, config)
    if inspection_callback is not None:
        inspection_callback(chlorophyll, "8day_compositing")
    print("[preprocess] Validating required dimensions")
    validate_required_dimensions(chlorophyll)
    print_basic_metadata(chlorophyll, input_dir)
    return chlorophyll


def find_raw_netcdf_files(input_dir: Path, config) -> list[Path]:
    """Return sorted raw NetCDF files using configured patterns plus safe defaults."""
    configured_patterns = getattr(config, "raw_file_pattern", None)
    if configured_patterns is None:
        patterns = list(DEFAULT_RAW_FILE_PATTERNS)
    elif isinstance(configured_patterns, str):
        patterns = [configured_patterns]
    else:
        patterns = list(configured_patterns)

    matched_files: list[Path] = []
    seen_paths: set[Path] = set()
    for pattern in patterns:
        for path in sorted(input_dir.glob(pattern)):
            if path.is_file() and path not in seen_paths:
                matched_files.append(path)
                seen_paths.add(path)

    if matched_files:
        print(
            f"Found {len(matched_files)} raw NetCDF file(s) using pattern(s): "
            f"{', '.join(patterns)}"
        )
        return matched_files

    default_matches: list[Path] = []
    for pattern in DEFAULT_RAW_FILE_PATTERNS:
        for path in sorted(input_dir.glob(pattern)):
            if path.is_file() and path not in default_matches:
                default_matches.append(path)
    if default_matches:
        print(
            f"Configured raw_file_pattern did not match files; "
            f"falling back to {len(default_matches)} default NetCDF file(s)."
        )
        return default_matches

    tried_patterns = ", ".join(patterns + list(DEFAULT_RAW_FILE_PATTERNS))
    raise FileNotFoundError(
        f"No raw NetCDF files found in {input_dir}. Tried pattern(s): {tried_patterns}"
    )


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

    chlorophyll = standardize_dimension_names(dataset[variable_name], config)
    chlorophyll = ensure_datetime_time(chlorophyll)
    chlorophyll = maybe_apply_8day_compositing(chlorophyll, config)
    validate_required_dimensions(chlorophyll)
    print_basic_metadata(chlorophyll, preprocessed_path)
    return chlorophyll


def standardize_dimension_names(
    data: xr.DataArray | xr.Dataset,
    config,
) -> xr.DataArray | xr.Dataset:
    """Rename configured time/latitude/longitude dimensions to pipeline names."""
    configured_to_standard = {
        getattr(config, "time_dim", "time"): "time",
        getattr(config, "latitude_dim", "lat"): "lat",
        getattr(config, "longitude_dim", "lon"): "lon",
    }
    rename_map = {}
    for configured_name, standard_name in configured_to_standard.items():
        if configured_name == standard_name:
            continue
        if configured_name in data.dims or configured_name in data.coords:
            rename_map[configured_name] = standard_name

    if not rename_map:
        return data

    return data.rename(rename_map)


def maybe_crop_to_study_area(data_array: xr.DataArray, config) -> xr.DataArray:
    """Apply the configured study-area crop when enabled."""
    if not getattr(config, "enable_study_area_crop", True):
        print("Skipping study-area crop because enable_study_area_crop is false.")
        return data_array

    return crop_to_study_area(data_array, config)


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


def maybe_apply_8day_compositing(
    data: xr.DataArray | xr.Dataset,
    config,
) -> xr.DataArray | xr.Dataset:
    """Apply consecutive-window compositing when enabled in the config."""
    if not getattr(config, "enable_8day_compositing", False):
        print("Skipping 8-day compositing because enable_8day_compositing is false.")
        return data
    if "composite_window_size" in data.attrs:
        print("Skipping 8-day compositing because the data already has composite metadata.")
        return data

    return composite_consecutive_windows(
        data,
        time_dim="time",
        window_size=int(getattr(config, "composite_window_size", 8)),
        min_valid_fraction=float(getattr(config, "composite_min_valid_fraction", 0.6)),
    )


def composite_consecutive_windows(
    data: xr.DataArray | xr.Dataset,
    time_dim: str = "time",
    window_size: int = 8,
    min_valid_fraction: float = 0.6,
) -> xr.DataArray | xr.Dataset:
    """Average consecutive time windows within each calendar year."""
    if time_dim not in data.dims:
        raise ValueError(
            f"Cannot composite data without '{time_dim}' dimension. "
            f"Found dimensions: {tuple(data.dims)}"
        )
    if window_size <= 0:
        raise ValueError("window_size must be a positive integer.")
    if not 0.0 < min_valid_fraction <= 1.0:
        raise ValueError("min_valid_fraction must be greater than 0 and at most 1.")

    time_size = int(data.sizes[time_dim])
    if time_size == 0:
        return data

    min_valid_count = int(math.ceil(window_size * min_valid_fraction))
    time_values = pd.to_datetime(data[time_dim].values)
    years = pd.Index(time_values.year).unique()
    yearly_composites = []
    for year in years:
        year_mask = time_values.year == int(year)
        year_data = data.isel({time_dim: year_mask})
        yearly_composites.append(
            composite_consecutive_block(
                year_data,
                time_dim=time_dim,
                window_size=window_size,
                min_valid_count=min_valid_count,
            )
        )

    composite = xr.concat(yearly_composites, dim=time_dim).sortby(time_dim)
    composite.attrs.update(data.attrs)
    composite.attrs["composite_window_size"] = window_size
    composite.attrs["composite_min_valid_fraction"] = min_valid_fraction
    composite.attrs["composite_min_valid_count"] = min_valid_count
    composite.attrs["composite_time_label"] = "first day in each same-year consecutive window"
    composite.attrs["composite_year_boundary"] = "windows reset at each calendar year"

    print(
        f"Applied {window_size}-day compositing within each calendar year with minimum "
        f"{min_valid_count}/{window_size} valid observations; "
        f"time steps: {time_size} -> {composite.sizes[time_dim]}"
    )
    return composite


def composite_consecutive_block(
    data: xr.DataArray | xr.Dataset,
    time_dim: str,
    window_size: int,
    min_valid_count: int,
) -> xr.DataArray | xr.Dataset:
    """Average one same-year block of consecutive time windows."""
    time_size = int(data.sizes[time_dim])
    window_ids = xr.DataArray(
        np.arange(time_size) // window_size,
        dims=time_dim,
        coords={time_dim: data[time_dim]},
        name="composite_window",
    )

    valid_counts = data.notnull().groupby(window_ids).sum(dim=time_dim)
    composite = data.groupby(window_ids).mean(dim=time_dim, skipna=True)
    composite = composite.where(valid_counts >= min_valid_count)

    time_values = pd.to_datetime(data[time_dim].values)
    composite_times = [
        time_values[start : min(start + window_size, time_size)][0]
        for start in range(0, time_size, window_size)
    ]
    composite = composite.rename({"composite_window": time_dim})
    composite = composite.assign_coords({time_dim: composite_times})
    return composite


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

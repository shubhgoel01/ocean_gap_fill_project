from __future__ import annotations

import re
from pathlib import Path

import pandas as pd
import xarray as xr


def combine_daily_nasa_files(
    input_dir: str | Path = "data/raw",
    output_file: str | Path = "data/raw/nasa_combined_subset.nc",
) -> Path:
    """
    Combine daily NASA OceanColor NetCDF files into one time-stacked dataset.

    Notes:
    - Each input file is a single daily 2D map with dims (lat, lon).
    - The date is extracted from the filename, then added as a time dimension.
    - Only the 'chlor_a' variable is kept.
    - No spatial subsetting is applied (assumes data already pre-cropped).
    """
    input_dir = Path(input_dir)
    output_file = Path(output_file)

    files = sorted(input_dir.glob("AQUA_MODIS.*.nc"))
    if not files:
        raise FileNotFoundError(f"No matching .nc files found in {input_dir}")

    datasets: list[xr.Dataset] = []

    for file in files:
        ds = xr.open_dataset(file)[["chlor_a"]]

        match = re.search(r"\.(\d{8})\.", file.name)
        if not match:
            raise ValueError(f"Could not extract date from filename: {file.name}")

        date = pd.to_datetime(match.group(1), format="%Y%m%d")
        ds = ds.expand_dims(time=[date])

        datasets.append(ds)

    combined = xr.concat(datasets, dim="time")

    # Handle fill values if xarray did not already decode them to NaN
    fill_value = combined["chlor_a"].attrs.get("_FillValue")
    if fill_value is not None:
        combined["chlor_a"] = combined["chlor_a"].where(
            combined["chlor_a"] != fill_value
        )

    # Chlorophyll should not be <= 0 for this product in normal use
    combined["chlor_a"] = combined["chlor_a"].where(
        combined["chlor_a"] > 0
    )

    output_file.parent.mkdir(parents=True, exist_ok=True)
    combined.to_netcdf(output_file)

    print("Combined dataset saved")
    print(f"  Output: {output_file}")
    print(f"  Shape: {dict(combined.sizes)}")
    print(f"  Time range: {combined.time.values[0]} to {combined.time.values[-1]}")

    return output_file


if __name__ == "__main__":
    combine_daily_nasa_files()
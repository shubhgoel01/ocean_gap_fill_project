"""Temporary script to crop single-day target NetCDF files for one year.

This file is intentionally standalone so it can be deleted safely later.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd
import xarray as xr


def load_config(config_path: Path) -> dict:
    config_path = config_path.expanduser().resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    config["_config_dir"] = config_path.parent
    return config


def resolve_config_path(path_value: str, config_dir: Path) -> Path:
    path = Path(path_value).expanduser()
    if path.is_absolute():
        return path.resolve()
    return (config_dir / path).resolve()


def coordinate_order_aware_slice(values, lower: float, upper: float) -> slice:
    if len(values) == 0 or float(values[0]) <= float(values[-1]):
        return slice(lower, upper)
    return slice(upper, lower)


def filename_contains_year(filename: str, year: int) -> bool:
    return re.search(rf"(?<!\d){year}(?!\d)", filename) is not None


def file_belongs_to_year(path: Path, time_dim: str, year: int) -> bool:
    try:
        with xr.open_dataset(path) as dataset:
            if time_dim in dataset.coords:
                times = pd.to_datetime(dataset[time_dim].values)
                return bool(len(times) and int(times[0].year) == year)
    except Exception as exc:
        print(f"[crop-temp] Warning: could not read time from {path.name}: {exc}")

    return filename_contains_year(path.name, year)


def crop_one_file(source_path: Path, destination_path: Path, config: dict) -> None:
    lat_name = config["latitude_dim"]
    lon_name = config["longitude_dim"]
    bounds = config["study_area_bounds"]

    with xr.open_dataset(source_path) as dataset:
        missing = [name for name in (lat_name, lon_name) if name not in dataset.coords]
        if missing:
            raise ValueError(f"{source_path.name} is missing coordinate(s): {', '.join(missing)}")

        cropped = dataset.sel(
            {
                lat_name: coordinate_order_aware_slice(
                    dataset[lat_name].values,
                    float(bounds["latitude_min"]),
                    float(bounds["latitude_max"]),
                ),
                lon_name: coordinate_order_aware_slice(
                    dataset[lon_name].values,
                    float(bounds["longitude_min"]),
                    float(bounds["longitude_max"]),
                ),
            }
        )

        if int(cropped.sizes.get(lat_name, 0)) == 0 or int(cropped.sizes.get(lon_name, 0)) == 0:
            raise ValueError(f"Crop produced an empty grid for {source_path.name}")

        cropped.load()

    destination_path.parent.mkdir(parents=True, exist_ok=True)
    cropped.to_netcdf(destination_path)
    cropped.close()


def crop_year(
    config_path: Path,
    year: int,
    output_dir: Path | None,
    pattern: str,
    overwrite: bool,
) -> Path:
    config = load_config(config_path)
    config_dir = config["_config_dir"]
    input_dir = resolve_config_path(config["input_directory"], config_dir)

    if output_dir is None:
        output_dir = config_dir / f"../data/target_data_cropped/{year}"
    elif not output_dir.is_absolute():
        output_dir = config_dir / output_dir
    output_dir = output_dir.resolve()

    files = sorted(path for path in input_dir.glob(pattern) if path.is_file())
    year_files = [path for path in files if file_belongs_to_year(path, config["time_dim"], year)]

    if not year_files:
        raise FileNotFoundError(f"No {year} files matched '{pattern}' in {input_dir}")

    bounds = config["study_area_bounds"]
    print(f"[crop-temp] Input: {input_dir}")
    print(f"[crop-temp] Output: {output_dir}")
    print(
        "[crop-temp] Bounds: "
        f"lat {bounds['latitude_min']} to {bounds['latitude_max']}, "
        f"lon {bounds['longitude_min']} to {bounds['longitude_max']}"
    )
    print(f"[crop-temp] Cropping {len(year_files)} file(s) for {year}")

    output_dir.mkdir(parents=True, exist_ok=True)
    for index, source_path in enumerate(year_files, start=1):
        destination_path = output_dir / source_path.name
        if destination_path.exists() and not overwrite:
            raise FileExistsError(
                f"Output already exists: {destination_path}. Use --overwrite to replace it."
            )

        crop_one_file(source_path, destination_path, config)
        print(f"[crop-temp] Saved {index}/{len(year_files)}: {destination_path.name}")

    return output_dir


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Temporarily crop target NetCDF files for one year using config bounds."
    )
    parser.add_argument("--config", type=Path, default=Path("configs/default.json"))
    parser.add_argument("--year", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--pattern", default="*.nc")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    output_dir = crop_year(args.config, args.year, args.output_dir, args.pattern, args.overwrite)
    print(f"[crop-temp] Done: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

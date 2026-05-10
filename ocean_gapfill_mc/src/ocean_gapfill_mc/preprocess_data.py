"""Create a reusable merged and cropped chlorophyll NetCDF file."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from .data_loader import get_study_area_bounds, load_raw_chlorophyll_data
from .spatial_regrid import regrid_to_target_latlon
from .utils.config import load_config


def preprocess_data(config_path: Path, output_path: Path | None = None, overwrite: bool = False) -> Path:
    """Load raw NetCDF files, crop the study domain, and save one prepared file."""
    log_preprocessing_step(f"Loading configuration from {config_path}")
    config = load_config(config_path)
    resolved_output = _resolve_output_path(config, output_path)
    inspection_logger = PreprocessingInspectionLogger(config)
    log_preprocessing_step(f"Preparing output path: {resolved_output}")

    if resolved_output.exists() and not overwrite:
        raise FileExistsError(
            f"Preprocessed file already exists: {resolved_output}. "
            "Use --overwrite to replace it."
        )

    resolved_output.parent.mkdir(parents=True, exist_ok=True)

    log_preprocessing_step("Loading raw chlorophyll data")
    chlorophyll = load_raw_chlorophyll_data(
        config,
        inspection_callback=inspection_logger.record_stage,
    )
    bounds = get_study_area_bounds(config)
    crop_note = (
        "cropped to the configured study area: "
        f"lat {bounds['latitude_min']} to {bounds['latitude_max']}, "
        f"lon {bounds['longitude_min']} to {bounds['longitude_max']}"
        if config.enable_study_area_crop
        else "study-area crop skipped"
    )
    composite_note = (
        f"{config.composite_window_size}-day compositing applied with "
        f"minimum valid fraction {config.composite_min_valid_fraction}"
        if config.enable_8day_compositing
        else "8-day compositing skipped"
    )
    if config.enable_regridding:
        log_preprocessing_step(
            f"Regridding to {config.target_grid_resolution}-degree latitude-longitude grid"
        )
        chlorophyll, regrid_summary = regrid_to_target_latlon(
            chlorophyll,
            config,
            save_summary=False,
        )
        inspection_logger.record_stage(chlorophyll, "spatial_regridding")
        regrid_note = f"regridded to {config.target_grid_resolution}-degree lat-lon resolution"
    else:
        log_preprocessing_step("Skipping spatial regridding because enable_regridding is false")
        regrid_summary = {"status": "skipped", "reason": "enable_regridding is false"}
        inspection_logger.record_stage(chlorophyll, "spatial_regridding_skipped")
        regrid_note = "spatial regridding skipped"
    log_preprocessing_step("Attaching preprocessing metadata")
    chlorophyll.attrs["preprocessing_regridded"] = str(bool(config.enable_regridding)).lower()
    if config.enable_regridding:
        chlorophyll.attrs["preprocessing_target_grid_resolution"] = float(
            config.target_grid_resolution
        )
    chlorophyll.attrs["preprocessing_regrid_summary"] = json.dumps(regrid_summary)
    chlorophyll.attrs["preprocessing_note"] = (
        f"Merged source files, {crop_note}, {composite_note}, "
        f"and {regrid_note}."
    )
    chlorophyll.attrs["source_input_directory"] = str(config.input_directory)

    log_preprocessing_step(f"Saving preprocessed NetCDF to {resolved_output}")
    dataset = chlorophyll.to_dataset(name=config.variable_name)
    dataset.to_netcdf(resolved_output)
    inspection_logger.save()
    print(f"Saved preprocessed chlorophyll data to: {resolved_output}")
    return resolved_output


class PreprocessingInspectionLogger:
    """Collect and save NaN inspection summaries for preprocessing stages."""

    def __init__(self, config) -> None:
        self.output_dir = Path(config.pre_processing_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.log_path = self.output_dir / "preprocessing_inspection.log"
        self.csv_path = self.output_dir / "preprocessing_nan_summary.csv"
        self.rows: list[dict] = []
        self._write_header()

    def _write_header(self) -> None:
        message = "Preprocessing inspection started"
        self.log_path.write_text(f"{message}\n", encoding="utf-8")
        print(f"[preprocess] {message}")
        print(f"[preprocess] Inspection log: {self.log_path}")
        print(f"[preprocess] Inspection CSV: {self.csv_path}")

    def record_stage(self, data_array, stage: str) -> dict:
        summary = summarize_nan_stage(data_array, stage)
        self.rows.append(summary)
        message = (
            f"NaN status after {stage}: "
            f"{summary['nan_cells']}/{summary['total_cells']} cells missing "
            f"({summary['nan_percent']:.4f}%)"
        )
        print(f"[preprocess] {message}")
        with self.log_path.open("a", encoding="utf-8") as handle:
            handle.write(message + "\n")
        return summary

    def save(self) -> None:
        fieldnames = [
            "stage",
            "time_steps",
            "lat_cells",
            "lon_cells",
            "total_cells",
            "nan_cells",
            "valid_cells",
            "nan_percent",
        ]
        with self.csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self.rows)
        message = f"Saved preprocessing inspection CSV to {self.csv_path}"
        print(f"[preprocess] {message}")
        with self.log_path.open("a", encoding="utf-8") as handle:
            handle.write(message + "\n")


def summarize_nan_stage(data_array, stage: str) -> dict:
    """Return NaN counts and dimensions for one preprocessing stage."""
    total_cells = int(np.prod([int(size) for size in data_array.sizes.values()]))
    nan_cells = compute_nan_count(data_array)
    valid_cells = total_cells - nan_cells
    nan_percent = float((nan_cells / total_cells) * 100.0) if total_cells else 0.0
    return {
        "stage": stage,
        "time_steps": int(data_array.sizes.get("time", 0)),
        "lat_cells": int(data_array.sizes.get("lat", 0)),
        "lon_cells": int(data_array.sizes.get("lon", 0)),
        "total_cells": total_cells,
        "nan_cells": nan_cells,
        "valid_cells": valid_cells,
        "nan_percent": round(nan_percent, 4),
    }


def compute_nan_count(data_array) -> int:
    """Compute NaN count for eager or dask-backed xarray data."""
    nan_count = data_array.isnull().sum()
    if hasattr(nan_count, "compute"):
        nan_count = nan_count.compute()
    return int(nan_count.item())


def log_preprocessing_step(message: str) -> None:
    """Print a visible preprocessing progress message."""
    print(f"[preprocess] {message}")


def _resolve_output_path(config, output_path: Path | None) -> Path:
    if output_path is not None:
        candidate = Path(output_path).expanduser()
        if candidate.is_absolute():
            return candidate.resolve()
        return (Path(config.config_directory) / candidate).resolve()

    if config.preprocessed_data_file is not None:
        return Path(config.preprocessed_data_file)

    return (Path(config.config_directory) / "../data/processed/tropical_indian_ocean_chlor_a.nc").resolve()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build one merged and tropical-Indian-Ocean-cropped chlorophyll NetCDF file."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/default.json"),
        help="Path to the JSON configuration file.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "Output NetCDF path. Defaults to preprocessed_data_file from the config, "
            "or data/processed/tropical_indian_ocean_chlor_a.nc."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace the output file if it already exists.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    preprocess_data(args.config, args.output, args.overwrite)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

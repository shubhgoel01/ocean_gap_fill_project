"""Create a reusable merged and cropped chlorophyll NetCDF file."""

from __future__ import annotations

import argparse
from pathlib import Path

from .data_loader import get_study_area_bounds, load_raw_chlorophyll_data
from .utils.config import load_config


def preprocess_data(config_path: Path, output_path: Path | None = None, overwrite: bool = False) -> Path:
    """Load raw NetCDF files, crop the study domain, and save one prepared file."""
    config = load_config(config_path)
    resolved_output = _resolve_output_path(config, output_path)

    if resolved_output.exists() and not overwrite:
        raise FileExistsError(
            f"Preprocessed file already exists: {resolved_output}. "
            "Use --overwrite to replace it."
        )

    resolved_output.parent.mkdir(parents=True, exist_ok=True)

    chlorophyll = load_raw_chlorophyll_data(config)
    bounds = get_study_area_bounds(config)
    chlorophyll.attrs["preprocessing_note"] = (
        "Merged source files and cropped to the configured study area: "
        f"lat {bounds['latitude_min']} to {bounds['latitude_max']}, "
        f"lon {bounds['longitude_min']} to {bounds['longitude_max']}."
    )
    chlorophyll.attrs["source_input_directory"] = str(config.input_directory)

    dataset = chlorophyll.to_dataset(name=config.variable_name)
    dataset.to_netcdf(resolved_output)
    print(f"Saved preprocessed chlorophyll data to: {resolved_output}")
    return resolved_output


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

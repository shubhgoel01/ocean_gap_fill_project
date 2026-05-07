"""Compare reconstructed chlorophyll against online gap-free filled data."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ocean_gapfill_mc.utils.config import load_config
from ocean_gapfill_mc.data_loader import load_chlorophyll_data
from ocean_gapfill_mc.fill_stage import (
    FILL_STAGE_FILENAME,
    FILL_STAGE_INTERPOLATION,
    FILL_STAGE_MONTE_CARLO,
    load_fill_stage_map,
)
from ocean_gapfill_mc.interpolation import apply_ordered_interpolation
from ocean_gapfill_mc.spatial_regrid import regrid_to_target_latlon


DEFAULT_RECONSTRUCTED_MEAN_FILENAME = (
    "final_reconstructed_ensemble_mean_chlorophyll_by_time.nc"
)
DEFAULT_UNCERTAINTY_FILENAME = "uncertainty_maps.nc"
STANDARD_TIME_NAME = "time"
STANDARD_LAT_NAME = "lat"
STANDARD_LON_NAME = "lon"

CHECK_REGIONS = {
    "pacific_ocean": {
        "display_name": "Pacific Ocean",
        "lat_min": -30.0,
        "lat_max": 30.0,
        "lon_min": 120.0,
        "lon_max": 147.0,
    },
    "bay_of_bengal": {
        "display_name": "Bay of Bengal",
        "lat_min": 8.0,
        "lat_max": 16.0,
        "lon_min": 85.0,
        "lon_max": 95.0,
    },
}


def generate_reconstructed_filled_comparison(config) -> dict[str, str]:
    """Generate a year-specific comparison plot and CSV."""
    year = get_comparison_year(config)
    output_dir = Path(config.filled_data_comparison_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    reconstructed_path = resolve_reconstructed_mean_path(config)
    filled_paths = find_filled_data_files(config)

    print(f"[filled-comparison] comparison year: {year}")
    print(f"[filled-comparison] reconstructed mean: {reconstructed_path}")
    print(f"[filled-comparison] filled data files: {len(filled_paths)} file(s)")
    for filled_path in filled_paths:
        print(f"[filled-comparison] filled data: {filled_path}")

    reconstructed_ds = xr.open_dataset(reconstructed_path)
    filled_ds = xr.open_mfdataset(
        [str(path) for path in filled_paths],
        combine="by_coords",
        data_vars="minimal",
        coords="minimal",
        compat="override",
        chunks={"time": 1},
    )

    try:
        reconstructed_series = compute_year_spatial_mean_series(
            reconstructed_ds,
            year,
            preferred_variable=config.variable_name,
            label="pipeline_reconstructed",
        )
        target_series = compute_year_spatial_mean_series(
            filled_ds,
            None,
            preferred_variable=config.filled_variable_name or config.variable_name,
            label="online_gap_free",
        )
        frame = pd.concat([reconstructed_series, target_series], axis=1).sort_index()
        frame.index.name = "time"
        if frame.empty:
            raise ValueError(f"No overlapping comparison data found for year {year}.")

        csv_path = output_dir / f"reconstructed_vs_gapfree_{year}.csv"
        plot_path = output_dir / f"reconstructed_vs_gapfree_{year}.png"
        frame.to_csv(csv_path)
        plot_comparison(frame, year, "Full Available Domain", plot_path)
        outputs = {
            "plot": str(plot_path),
            "csv": str(csv_path),
        }

        outputs.update(
            generate_region_comparisons(
                reconstructed_ds,
                filled_ds,
                year,
                config,
                output_dir,
            )
        )
        outputs.update(
            generate_filled_cell_validation(
                reconstructed_ds,
                filled_ds,
                year,
                config,
                output_dir,
            )
        )
    finally:
        reconstructed_ds.close()
        filled_ds.close()

    print(f"[filled-comparison] saved {plot_path}")
    print(f"[filled-comparison] saved {csv_path}")
    return outputs


def generate_region_comparisons(
    reconstructed_ds: xr.Dataset,
    filled_ds: xr.Dataset,
    year: int,
    config,
    output_dir: Path,
) -> dict[str, str]:
    """Generate extra comparison plots for fixed diagnostic regions."""
    outputs: dict[str, str] = {}
    for region_key, region in {"pacific_ocean": CHECK_REGIONS["pacific_ocean"]}.items():
        try:
            reconstructed_series = compute_year_spatial_mean_series(
                reconstructed_ds,
                year,
                preferred_variable=config.variable_name,
                label="pipeline_reconstructed",
                region=region,
            )
            filled_series = compute_year_spatial_mean_series(
                filled_ds,
                None,
                preferred_variable=config.filled_variable_name or config.variable_name,
                label="online_gap_free",
                region=region,
            )
        except ValueError as exc:
            print(f"[filled-comparison] skipping {region['display_name']}: {exc}")
            continue

        frame = pd.concat([reconstructed_series, filled_series], axis=1).sort_index()
        frame.index.name = "time"
        if frame.empty:
            print(f"[filled-comparison] skipping {region['display_name']}: no data")
            continue

        csv_path = output_dir / f"reconstructed_vs_gapfree_{year}_{region_key}.csv"
        plot_path = output_dir / f"reconstructed_vs_gapfree_{year}_{region_key}.png"
        frame.to_csv(csv_path)
        plot_comparison(frame, year, region["display_name"], plot_path)
        print(f"[filled-comparison] saved {plot_path}")
        print(f"[filled-comparison] saved {csv_path}")
        outputs[f"{region_key}_plot"] = str(plot_path)
        outputs[f"{region_key}_csv"] = str(csv_path)

    return outputs


def get_comparison_year(config) -> int:
    if config.comparison_year is None:
        raise ValueError("comparison_year must be set in the config.")
    return int(config.comparison_year)


def resolve_reconstructed_mean_path(config) -> Path:
    if config.reconstructed_mean_data_file is not None:
        path = Path(config.reconstructed_mean_data_file)
    else:
        path = Path(config.datasets_dir) / DEFAULT_RECONSTRUCTED_MEAN_FILENAME
    if not path.exists():
        raise FileNotFoundError(
            f"Reconstructed mean data file not found: {path}. "
            "Run the main pipeline first or set reconstructed_mean_data_file in the config."
        )
    return path


def find_filled_data_files(config) -> list[Path]:
    directory = Path(config.filled_data_directory)
    if not directory.exists():
        raise FileNotFoundError(
            f"Filled data directory not found: {directory}. "
            "Create it or set filled_data_directory in the config."
        )
    if not directory.is_dir():
        raise NotADirectoryError(f"Filled data path is not a directory: {directory}")

    patterns = normalize_patterns(config.filled_data_pattern)
    matches: list[Path] = []
    seen: set[Path] = set()
    for pattern in patterns:
        for path in sorted(directory.glob(pattern)):
            if path.is_file() and path not in seen:
                matches.append(path)
                seen.add(path)

    if not matches:
        raise FileNotFoundError(
            f"No filled-data NetCDF files found in {directory}. "
            f"Tried pattern(s): {', '.join(patterns)}"
        )
    return matches


def normalize_patterns(patterns) -> list[str]:
    if isinstance(patterns, str):
        return [patterns]
    return list(patterns)


def generate_filled_cell_validation(
    reconstructed_ds: xr.Dataset,
    filled_ds: xr.Dataset,
    year: int,
    config,
    output_dir: Path,
) -> dict[str, str]:
    """Validate only cells filled by interpolation and/or Monte Carlo."""
    reconstructed = prepare_comparison_data_array(
        reconstructed_ds,
        preferred_variable=config.variable_name,
        year=year,
        label="pipeline_reconstructed",
    )
    target = prepare_comparison_data_array(
        filled_ds,
        preferred_variable=config.filled_variable_name or config.variable_name,
        year=year,
        label="online_gap_free",
    )
    fill_stage = load_fill_stage_map_for_validation(config, year)
    if fill_stage is None:
        regridded, interpolated = build_pipeline_stage_arrays_for_validation(config, year)
        reconstructed, target, regridded, interpolated = xr.align(
            reconstructed,
            target,
            regridded,
            interpolated,
            join="inner",
        )
        interpolation_mask = np.isnan(regridded) & np.isfinite(interpolated)
        monte_carlo_mask = np.isnan(interpolated) & np.isfinite(reconstructed)
    else:
        reconstructed, target, fill_stage = xr.align(
            reconstructed,
            target,
            fill_stage,
            join="inner",
        )
        interpolation_mask = fill_stage == FILL_STAGE_INTERPOLATION
        monte_carlo_mask = fill_stage == FILL_STAGE_MONTE_CARLO

    if reconstructed.size == 0 or target.size == 0:
        raise ValueError("No overlapping time-lat-lon points found for filled-cell validation.")

    uncertainty = load_uncertainty_bounds(config, year)
    if uncertainty is not None:
        lower, upper = uncertainty
        lower, upper, reconstructed, target, interpolation_mask, monte_carlo_mask = xr.align(
            lower,
            upper,
            reconstructed,
            target,
            interpolation_mask,
            monte_carlo_mask,
            join="inner",
        )
    all_filled_mask = interpolation_mask | monte_carlo_mask
    rows = [
        calculate_filled_group_metrics(
            "interpolation_filled",
            interpolation_mask,
            reconstructed,
            target,
            uncertainty,
        ),
        calculate_filled_group_metrics(
            "monte_carlo_filled",
            monte_carlo_mask,
            reconstructed,
            target,
            uncertainty,
        ),
        calculate_filled_group_metrics(
            "all_filled",
            all_filled_mask,
            reconstructed,
            target,
            uncertainty,
        ),
    ]

    metrics_path = output_dir / f"filled_cell_validation_metrics_{year}.csv"
    pd.DataFrame(rows).to_csv(metrics_path, index=False)

    for row in rows:
        print(
            "[filled-cell-validation] "
            f"{row['comparison_group']}: "
            f"count={int(row['valid_comparison_count']):,}, "
            f"MSE={row['mse']:.6g}, "
            f"RMSE={row['rmse']:.6g}, "
            f"inside p05-p95={format_optional_percent(row['percent_inside_p05_p95'])}"
        )
    print(f"[filled-cell-validation] saved {metrics_path}")

    return {"filled_cell_validation_metrics": str(metrics_path)}


def load_fill_stage_map_for_validation(config, year: int) -> xr.DataArray | None:
    """Load the pipeline-saved fill-stage map when available."""
    fill_stage_path = Path(config.reconstructed_dir) / FILL_STAGE_FILENAME
    if not fill_stage_path.exists():
        print(
            "[filled-cell-validation] fill-stage map not found; "
            "rebuilding pipeline stage masks"
        )
        return None

    fill_stage = load_fill_stage_map(fill_stage_path)
    prepared = prepare_comparison_data_array(
        fill_stage.to_dataset(name=fill_stage.name or "fill_stage"),
        preferred_variable=fill_stage.name or "fill_stage",
        year=year,
        label="fill_stage",
    )
    print(f"[filled-cell-validation] fill-stage map: {fill_stage_path}")
    return prepared


def build_pipeline_stage_arrays_for_validation(config, year: int) -> tuple[xr.DataArray, xr.DataArray]:
    """Rebuild regridded and interpolation-stage arrays to identify filled cells."""
    print("[filled-cell-validation] rebuilding pipeline stage masks")
    loaded = load_chlorophyll_data(config)
    if is_preprocessing_regridded(loaded):
        regridded = loaded
    else:
        regridded, _ = regrid_to_target_latlon(loaded, config, save_summary=False)
    interpolated = apply_ordered_interpolation(regridded, config, save_summary=False)
    return (
        normalize_pipeline_stage_array(regridded, year, "regridded"),
        normalize_pipeline_stage_array(interpolated, year, "interpolated"),
    )


def is_preprocessing_regridded(data_array: xr.DataArray) -> bool:
    value = data_array.attrs.get("preprocessing_regridded", False)
    if isinstance(value, str):
        return value.strip().lower() in {"true", "1", "yes"}
    return bool(value)


def normalize_pipeline_stage_array(data_array: xr.DataArray, year: int, label: str) -> xr.DataArray:
    dataset = data_array.to_dataset(name=data_array.name or label)
    return prepare_comparison_data_array(dataset, data_array.name or label, year, label)


def load_uncertainty_bounds(config, year: int) -> tuple[xr.DataArray, xr.DataArray] | None:
    uncertainty_path = Path(config.reconstructed_dir) / DEFAULT_UNCERTAINTY_FILENAME
    if not uncertainty_path.exists():
        print(f"[filled-cell-validation] uncertainty file not found: {uncertainty_path}")
        return None

    uncertainty_ds = xr.open_dataset(uncertainty_path)
    try:
        lower_name = detect_uncertainty_variable(
            uncertainty_ds,
            ["reconstructed_lower_percentile", "p05", "lower_percentile"],
        )
        upper_name = detect_uncertainty_variable(
            uncertainty_ds,
            ["reconstructed_upper_percentile", "p95", "upper_percentile"],
        )
        lower = prepare_uncertainty_data_array(uncertainty_ds[lower_name], year).load()
        upper = prepare_uncertainty_data_array(uncertainty_ds[upper_name], year).load()
        print(f"[filled-cell-validation] uncertainty maps: {uncertainty_path}")
        return lower, upper
    finally:
        uncertainty_ds.close()


def calculate_filled_group_metrics(
    group_name: str,
    group_mask: xr.DataArray,
    reconstructed: xr.DataArray,
    target: xr.DataArray,
    uncertainty: tuple[xr.DataArray, xr.DataArray] | None,
) -> dict[str, float | int | str]:
    valid_mask = group_mask & np.isfinite(reconstructed) & np.isfinite(target)
    valid_count = int(valid_mask.sum().compute().item())
    base_row = {
        "comparison_group": group_name,
        "valid_comparison_count": valid_count,
        "mse": np.nan,
        "rmse": np.nan,
        "mae": np.nan,
        "bias": np.nan,
        "percent_inside_p05_p95": np.nan,
        "average_p05_p95_width": np.nan,
        "p05_p95_comparison_count": 0,
    }
    if valid_count == 0:
        return base_row

    error = (reconstructed - target).where(valid_mask)
    error_values = np.asarray(error.values, dtype=float)
    finite_error = error_values[np.isfinite(error_values)]
    base_row.update(
        {
            "mse": float(np.mean(finite_error ** 2)),
            "rmse": float(np.sqrt(np.mean(finite_error ** 2))),
            "mae": float(np.mean(np.abs(finite_error))),
            "bias": float(np.mean(finite_error)),
        }
    )

    if uncertainty is not None:
        lower, upper = uncertainty
        lower, upper, aligned_target, aligned_valid_mask = xr.align(
            lower,
            upper,
            target,
            valid_mask,
            join="inner",
        )
        uncertainty_mask = aligned_valid_mask & np.isfinite(lower) & np.isfinite(upper)
        uncertainty_count = int(uncertainty_mask.sum().compute().item())
        if uncertainty_count > 0:
            inside = ((lower <= aligned_target) & (aligned_target <= upper)).where(uncertainty_mask)
            width = (upper - lower).where(uncertainty_mask)
            base_row.update(
                {
                    "percent_inside_p05_p95": calculate_coverage_percentage(inside, uncertainty_count),
                    "average_p05_p95_width": float(width.mean(skipna=True).compute().item()),
                    "p05_p95_comparison_count": uncertainty_count,
                }
            )

    return base_row


def format_optional_percent(value: float) -> str:
    if not np.isfinite(value):
        return "not available"
    return f"{float(value):.2f}%"


def prepare_comparison_data_array(
    dataset: xr.Dataset,
    preferred_variable: str | None,
    year: int,
    label: str,
) -> xr.DataArray:
    """Return a chlorophyll DataArray with standard time/lat/lon names."""
    variable = detect_variable(dataset, preferred_variable)
    lat_name = detect_coord_name(dataset, ["lat", "latitude", "y"])
    lon_name = detect_coord_name(dataset, ["lon", "longitude", "x"])
    time_name = detect_coord_name(dataset, ["time", "date"])
    data_array = dataset[variable]
    if time_name not in data_array.dims:
        raise ValueError(f"{label} variable '{variable}' has no time dimension.")

    rename_map = {}
    if time_name != STANDARD_TIME_NAME:
        rename_map[time_name] = STANDARD_TIME_NAME
    if lat_name != STANDARD_LAT_NAME:
        rename_map[lat_name] = STANDARD_LAT_NAME
    if lon_name != STANDARD_LON_NAME:
        rename_map[lon_name] = STANDARD_LON_NAME
    if rename_map:
        data_array = data_array.rename(rename_map)

    time_values = pd.to_datetime(data_array[STANDARD_TIME_NAME].values)
    year_mask = time_values.year == int(year)
    if not np.any(year_mask):
        raise ValueError(f"{label} has no data for year {year}.")
    data_array = data_array.isel({STANDARD_TIME_NAME: year_mask})

    if STANDARD_LON_NAME in data_array.coords:
        lon_values = np.asarray(data_array[STANDARD_LON_NAME].values, dtype=float)
        if np.nanmin(lon_values) < 0.0:
            data_array = data_array.assign_coords(
                {STANDARD_LON_NAME: data_array[STANDARD_LON_NAME] % 360.0}
            )
        data_array = data_array.assign_coords(
            {STANDARD_LON_NAME: np.round(np.asarray(data_array[STANDARD_LON_NAME].values, dtype=float), 4)}
        )
    if STANDARD_LAT_NAME in data_array.coords:
        data_array = data_array.assign_coords(
            {STANDARD_LAT_NAME: np.round(np.asarray(data_array[STANDARD_LAT_NAME].values, dtype=float), 4)}
        )

    return data_array.sortby([STANDARD_TIME_NAME, STANDARD_LAT_NAME, STANDARD_LON_NAME])


def calculate_spatial_accuracy_metrics(
    error: xr.DataArray,
    absolute_error: xr.DataArray,
    reconstructed: xr.DataArray,
    target: xr.DataArray,
    valid_mask: xr.DataArray,
) -> dict[str, float | int]:
    """Calculate compact scalar metrics over all valid time-lat-lon points."""
    valid_count = int(valid_mask.sum().compute().item())
    error_values = np.asarray(error.where(valid_mask).values, dtype=float)
    absolute_error_values = np.asarray(absolute_error.where(valid_mask).values, dtype=float)
    reconstructed_values = np.asarray(reconstructed.where(valid_mask).values, dtype=float)
    target_values = np.asarray(target.where(valid_mask).values, dtype=float)

    finite_error = error_values[np.isfinite(error_values)]
    finite_absolute_error = absolute_error_values[np.isfinite(absolute_error_values)]
    finite_pair_mask = np.isfinite(reconstructed_values) & np.isfinite(target_values)
    paired_reconstructed = reconstructed_values[finite_pair_mask]
    paired_target = target_values[finite_pair_mask]
    if paired_reconstructed.size >= 2:
        correlation = float(np.corrcoef(paired_reconstructed, paired_target)[0, 1])
    else:
        correlation = np.nan

    return {
        "valid_comparison_count": valid_count,
        "mae": float(np.mean(finite_absolute_error)),
        "rmse": float(np.sqrt(np.mean(finite_error ** 2))),
        "bias": float(np.mean(finite_error)),
        "pearson_correlation": correlation,
    }


def generate_simple_uncertainty_coverage(
    target: xr.DataArray,
    valid_mask: xr.DataArray,
    year: int,
    config,
    output_dir: Path,
) -> dict[str, str]:
    """Check whether target values fall inside p05-p95 uncertainty bounds."""
    uncertainty_path = Path(config.reconstructed_dir) / DEFAULT_UNCERTAINTY_FILENAME
    if not uncertainty_path.exists():
        print(f"[spatial-validation] uncertainty file not found, skipping coverage: {uncertainty_path}")
        return {}

    print(f"[spatial-validation] uncertainty maps: {uncertainty_path}")
    uncertainty_ds = xr.open_dataset(uncertainty_path)
    try:
        lower_name = detect_uncertainty_variable(
            uncertainty_ds,
            ["reconstructed_lower_percentile", "p05", "lower_percentile"],
        )
        upper_name = detect_uncertainty_variable(
            uncertainty_ds,
            ["reconstructed_upper_percentile", "p95", "upper_percentile"],
        )
        lower = prepare_uncertainty_data_array(uncertainty_ds[lower_name], year)
        upper = prepare_uncertainty_data_array(uncertainty_ds[upper_name], year)
        lower, upper, aligned_target, aligned_mask = xr.align(
            lower,
            upper,
            target,
            valid_mask,
            join="inner",
        )
        coverage_valid_mask = aligned_mask & np.isfinite(lower) & np.isfinite(upper)
        coverage_valid_count = int(coverage_valid_mask.sum().compute().item())
        if coverage_valid_count == 0:
            print("[spatial-validation] no valid p05-p95 coverage points found, skipping coverage")
            return {}

        covered = ((lower <= aligned_target) & (aligned_target <= upper)).where(coverage_valid_mask)
        interval_width = (upper - lower).where(coverage_valid_mask)
        stochastic_mask = coverage_valid_mask & (interval_width > 0.0)
        stochastic_count = int(stochastic_mask.sum().compute().item())
        coverage_percentage = calculate_coverage_percentage(covered, coverage_valid_count)
        average_interval_width = float(interval_width.mean(skipna=True).compute().item())
        if stochastic_count > 0:
            stochastic_coverage_percentage = calculate_coverage_percentage(
                covered.where(stochastic_mask),
                stochastic_count,
            )
            stochastic_average_interval_width = float(
                interval_width.where(stochastic_mask).mean(skipna=True).compute().item()
            )
            coverage_frequency = covered.where(stochastic_mask).mean(
                dim=STANDARD_TIME_NAME,
                skipna=True,
            ) * 100.0
        else:
            stochastic_coverage_percentage = np.nan
            stochastic_average_interval_width = np.nan
            coverage_frequency = covered.mean(dim=STANDARD_TIME_NAME, skipna=True) * 100.0

        coverage_path = output_dir / f"simple_uncertainty_coverage_{year}.csv"
        coverage_map_path = output_dir / f"simple_spatial_coverage_map_{year}.png"
        pd.DataFrame(
            [item for item in [
                {
                    "comparison_group": "all_valid_points",
                    "valid_coverage_count": coverage_valid_count,
                    "coverage_percentage": coverage_percentage,
                    "average_interval_width": average_interval_width,
                    "lower_bound_variable": lower_name,
                    "upper_bound_variable": upper_name,
                },
                {
                    "comparison_group": "nonzero_uncertainty_points",
                    "valid_coverage_count": stochastic_count,
                    "coverage_percentage": stochastic_coverage_percentage,
                    "average_interval_width": stochastic_average_interval_width,
                    "lower_bound_variable": lower_name,
                    "upper_bound_variable": upper_name,
                } if stochastic_count > 0 else None,
            ] if item is not None]
        ).to_csv(coverage_path, index=False)
        plot_spatial_map(
            coverage_frequency,
            title=f"p05-p95 Coverage Frequency: {year}",
            colorbar_label="Target inside interval (% of time)",
            output_path=coverage_map_path,
            cmap="magma",
            vmin=0.0,
            vmax=100.0,
        )

        print(
            "[spatial-validation] "
            f"all-point p05-p95 coverage={coverage_percentage:.2f}%, "
            f"average interval width={average_interval_width:.6g}"
        )
        if stochastic_count > 0:
            print(
                "[spatial-validation] "
                f"nonzero-uncertainty coverage={stochastic_coverage_percentage:.2f}%, "
                f"count={stochastic_count:,}, "
                f"average interval width={stochastic_average_interval_width:.6g}"
            )
        print(f"[spatial-validation] saved {coverage_path}")
        print(f"[spatial-validation] saved {coverage_map_path}")
        return {
            "simple_uncertainty_coverage": str(coverage_path),
            "simple_spatial_coverage_map": str(coverage_map_path),
        }
    finally:
        uncertainty_ds.close()


def calculate_coverage_percentage(covered: xr.DataArray, valid_count: int) -> float:
    if valid_count == 0:
        return np.nan
    return float(covered.sum(skipna=True).compute().item() / valid_count * 100.0)


def prepare_uncertainty_data_array(data_array: xr.DataArray, year: int) -> xr.DataArray:
    """Normalize uncertainty map coordinates to match validation data."""
    dataset = data_array.to_dataset(name=data_array.name or "uncertainty")
    prepared = prepare_comparison_data_array(
        dataset,
        preferred_variable=data_array.name,
        year=year,
        label="uncertainty",
    )
    return prepared


def detect_uncertainty_variable(dataset: xr.Dataset, candidates: list[str]) -> str:
    lower_lookup = {name.lower(): name for name in dataset.data_vars}
    for candidate in candidates:
        if candidate.lower() in lower_lookup:
            return lower_lookup[candidate.lower()]
    raise ValueError(
        f"Could not detect uncertainty variable from {list(dataset.data_vars)}. "
        f"Tried {candidates}."
    )


def plot_spatial_map(
    field: xr.DataArray,
    title: str,
    colorbar_label: str,
    output_path: Path,
    cmap: str,
    vmin: float | None = None,
    vmax: float | None = None,
) -> Path:
    """Plot one simple lat-lon map."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lon = np.asarray(field[STANDARD_LON_NAME].values, dtype=float)
    lat = np.asarray(field[STANDARD_LAT_NAME].values, dtype=float)
    values = np.asarray(field.values, dtype=float)

    fig, ax = plt.subplots(figsize=(9, 5), constrained_layout=True)
    mesh = ax.pcolormesh(
        lon,
        lat,
        np.ma.masked_invalid(values),
        shading="auto",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
    )
    colorbar = fig.colorbar(mesh, ax=ax, orientation="horizontal", pad=0.08)
    colorbar.set_label(colorbar_label)
    ax.set_title(title)
    ax.set_xlabel("Longitude ($^\\circ$E)")
    ax.set_ylabel("Latitude ($^\\circ$N)")
    ax.grid(True, linewidth=0.35, color="white", alpha=0.6)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path


def compute_year_spatial_mean_series(
    dataset: xr.Dataset,
    year: int | None,
    preferred_variable: str | None,
    label: str,
    region: dict | None = None,
) -> pd.Series:
    variable = detect_variable(dataset, preferred_variable)
    lat_name = detect_coord_name(dataset, ["lat", "latitude", "y"])
    lon_name = detect_coord_name(dataset, ["lon", "longitude", "x"])
    time_name = detect_coord_name(dataset, ["time", "date"])

    data_array = dataset[variable]
    if time_name not in data_array.dims:
        raise ValueError(f"{label} variable '{variable}' has no time dimension.")

    time_values = pd.to_datetime(data_array[time_name].values)
    if year is None:
        subset = data_array
    else:
        year_mask = time_values.year == int(year)
        if not np.any(year_mask):
            raise ValueError(f"{label} has no data for year {year}.")
        subset = data_array.isel({time_name: year_mask})
    if region is not None:
        subset = subset_region(subset, lat_name, lon_name, region, label)
    spatial_dims = [dim for dim in (lat_name, lon_name) if dim in subset.dims]
    mean_series = subset.mean(dim=spatial_dims, skipna=True)
    return pd.Series(
        np.asarray(mean_series.values, dtype=float),
        index=pd.to_datetime(mean_series[time_name].values),
        name=label,
    )


def subset_region(
    data_array: xr.DataArray,
    lat_name: str,
    lon_name: str,
    region: dict,
    label: str,
) -> xr.DataArray:
    """Subset a data array to a diagnostic lat-lon box."""
    lat_values = data_array[lat_name]
    lon_values = data_array[lon_name]
    lat_mask = (lat_values >= float(region["lat_min"])) & (
        lat_values <= float(region["lat_max"])
    )
    lon_360 = lon_values % 360.0
    lon_min = float(region["lon_min"]) % 360.0
    lon_max = float(region["lon_max"]) % 360.0
    if lon_min <= lon_max:
        lon_mask = (lon_360 >= lon_min) & (lon_360 <= lon_max)
    else:
        lon_mask = (lon_360 >= lon_min) | (lon_360 <= lon_max)

    subset = data_array.where(lat_mask & lon_mask, drop=True)
    if subset.sizes.get(lat_name, 0) == 0 or subset.sizes.get(lon_name, 0) == 0:
        raise ValueError(
            f"{label} has no cells inside {region['display_name']} "
            f"({region['lat_min']} to {region['lat_max']} lat, "
            f"{region['lon_min']} to {region['lon_max']} lon)."
        )
    return subset


def detect_variable(dataset: xr.Dataset, preferred_variable: str | None) -> str:
    if preferred_variable and preferred_variable in dataset.data_vars:
        return preferred_variable
    for candidate in ("chlor_a", "chlorophyll", "chl", "CHL"):
        if candidate in dataset.data_vars:
            return candidate
    if len(dataset.data_vars) == 1:
        return next(iter(dataset.data_vars))
    raise ValueError(f"Could not detect chlorophyll variable from {list(dataset.data_vars)}")


def detect_coord_name(dataset: xr.Dataset, candidates: list[str]) -> str:
    available = list(dataset.coords) + list(dataset.dims)
    lower_lookup = {name.lower(): name for name in available}
    for candidate in candidates:
        if candidate.lower() in lower_lookup:
            return lower_lookup[candidate.lower()]
    raise ValueError(f"Could not detect coordinate from candidates {candidates}")


def plot_comparison(frame: pd.DataFrame, year: int, region_name: str, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 5), constrained_layout=True)
    ax.plot(
        frame.index,
        frame["pipeline_reconstructed"],
        color="#1f77b4",
        linewidth=2.0,
        marker="o",
        markersize=3.5,
        label="Pipeline reconstructed",
    )
    ax.plot(
        frame.index,
        frame["online_gap_free"],
        color="#d62728",
        linewidth=2.0,
        marker="s",
        markersize=3.5,
        label="Online gap-free filled data",
    )
    ax.set_title(
        f"Reconstructed vs Online Gap-Free Filled Chlorophyll: {year}\n{region_name}"
    )
    ax.set_xlabel("Date")
    ax.set_ylabel("Spatial mean chlorophyll-a concentration (mg m$^{-3}$)")
    ax.grid(True, linewidth=0.4, alpha=0.35)
    ax.legend()
    fig.autofmt_xdate()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare reconstructed mean chlorophyll with online gap-free filled data."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/default.json"),
        help="Path to the JSON configuration file.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    config = load_config(args.config)
    outputs = generate_reconstructed_filled_comparison(config)
    print(f"[filled-comparison] completed with outputs: {outputs}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

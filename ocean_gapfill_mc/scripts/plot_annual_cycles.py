"""Plot regional annual chlorophyll-a cycles for paper-style diagnostics."""

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


REGIONS = {
    "arabian_sea": {
        "display_name": "Arabian Sea",
        "lon_min": 60.0,
        "lon_max": 70.0,
        "lat_min": 8.0,
        "lat_max": 16.0,
    },
    "bay_of_bengal": {
        "display_name": "Bay of Bengal",
        "lon_min": 85.0,
        "lon_max": 95.0,
        "lat_min": 8.0,
        "lat_max": 16.0,
    },
}


def find_existing_mean_datasets(config) -> dict[str, Path | None]:
    """Find reusable time-varying mean and climatology datasets in outputs."""
    datasets_dir = Path(config.datasets_dir)
    reconstructed_dir = Path(config.reconstructed_dir)

    paths = {
        "raw_mean": first_existing(
            [
                datasets_dir / "satellite_derived_raw_mean_chlorophyll.nc",
                datasets_dir / "raw_mean_chlorophyll.nc",
            ]
        ),
        "reconstructed_mean_by_time": first_existing(
            [
                datasets_dir / "final_reconstructed_ensemble_mean_chlorophyll_by_time.nc",
                reconstructed_dir / "reconstructed_mean_by_time.nc",
            ]
        ),
        "reconstructed_mean_2d": first_existing(
            [
                datasets_dir / "final_reconstructed_mean_chlorophyll.nc",
                datasets_dir / "reconstructed_mean_chlorophyll.nc",
            ]
        ),
        "climatology": first_matching(
            datasets_dir,
            [
                "*climatolog*.nc",
                "*annual_cycle*.nc",
                "*long_term*reconstructed*.nc",
            ],
        ),
    }

    for name, path in paths.items():
        if path is None:
            print(f"[annual-cycle] {name}: not found")
        else:
            print(f"[annual-cycle] {name}: found {path}")
    return paths


def first_existing(paths: list[Path]) -> Path | None:
    for path in paths:
        if path.exists():
            return path
    return None


def first_matching(directory: Path, patterns: list[str]) -> Path | None:
    if not directory.exists():
        return None
    for pattern in patterns:
        matches = sorted(directory.glob(pattern))
        if matches:
            return matches[0]
    return None


def load_dataset(path: Path) -> xr.Dataset:
    """Load one NetCDF dataset."""
    print(f"[annual-cycle] loading {path}")
    return xr.open_dataset(path)


def detect_variable_and_coords(
    dataset: xr.Dataset,
    preferred_variable: str | None = None,
) -> tuple[str, str, str, str | None]:
    """Detect data variable plus latitude, longitude, and optional time coordinate."""
    variable = detect_variable(dataset, preferred_variable)
    lat_name = detect_coord_name(dataset, ["lat", "latitude", "y"])
    lon_name = detect_coord_name(dataset, ["lon", "longitude", "x"])
    time_name = detect_coord_name(dataset, ["time", "date"], required=False)
    return variable, lat_name, lon_name, time_name


def detect_variable(dataset: xr.Dataset, preferred_variable: str | None) -> str:
    if preferred_variable and preferred_variable in dataset.data_vars:
        return preferred_variable
    if len(dataset.data_vars) == 1:
        return next(iter(dataset.data_vars))
    for candidate in ("chlor_a", "chlorophyll", "chl", "CHL"):
        if candidate in dataset.data_vars:
            return candidate
    raise ValueError(f"Could not detect chlorophyll variable from {list(dataset.data_vars)}")


def detect_coord_name(
    dataset: xr.Dataset,
    candidates: list[str],
    required: bool = True,
) -> str | None:
    available = list(dataset.coords) + list(dataset.dims)
    lower_lookup = {name.lower(): name for name in available}
    for candidate in candidates:
        if candidate.lower() in lower_lookup:
            return lower_lookup[candidate.lower()]
    if required:
        raise ValueError(f"Could not detect coordinate from candidates {candidates}")
    return None


def subset_region(
    data_array: xr.DataArray,
    lat_name: str,
    lon_name: str,
    region: dict,
) -> xr.DataArray:
    """Subset data to region bounds, handling increasing/decreasing coordinates."""
    lon_adjusted = normalize_longitude_bounds(
        data_array[lon_name],
        float(region["lon_min"]),
        float(region["lon_max"]),
    )
    lat_slice = coordinate_slice(data_array[lat_name], float(region["lat_min"]), float(region["lat_max"]))
    lon_slice = coordinate_slice(data_array[lon_name], lon_adjusted[0], lon_adjusted[1])
    return data_array.sel({lat_name: lat_slice, lon_name: lon_slice})


def normalize_longitude_bounds(lon_coord: xr.DataArray, lon_min: float, lon_max: float) -> tuple[float, float]:
    lon_values = np.asarray(lon_coord.values, dtype=float)
    if np.nanmin(lon_values) >= 0.0 and lon_min < 0.0:
        return lon_min % 360.0, lon_max % 360.0
    if np.nanmax(lon_values) <= 180.0 and lon_max > 180.0:
        return ((lon_min + 180.0) % 360.0) - 180.0, ((lon_max + 180.0) % 360.0) - 180.0
    return lon_min, lon_max


def coordinate_slice(coord: xr.DataArray, lower: float, upper: float) -> slice:
    values = np.asarray(coord.values, dtype=float)
    if values[0] <= values[-1]:
        return slice(lower, upper)
    return slice(upper, lower)


def compute_spatial_mean_timeseries(
    dataset: xr.Dataset,
    region: dict,
    preferred_variable: str | None = None,
) -> pd.Series | None:
    """Subset a dataset and compute spatial mean for each time step."""
    variable, lat_name, lon_name, time_name = detect_variable_and_coords(dataset, preferred_variable)
    data_array = dataset[variable]
    if time_name is None or time_name not in data_array.dims:
        return None

    subset = subset_region(data_array, lat_name, lon_name, region)
    spatial_dims = [dim for dim in (lat_name, lon_name) if dim in subset.dims]
    series = subset.mean(dim=spatial_dims, skipna=True)
    time_values = pd.to_datetime(series[time_name].values)
    values = np.asarray(series.values, dtype=float)
    return pd.Series(values, index=time_values, name=variable)


def compute_reconstructed_mean_if_needed(config, found_paths: dict[str, Path | None]) -> xr.Dataset:
    """Load existing reconstructed mean-by-time or compute it from member files."""
    existing_path = found_paths.get("reconstructed_mean_by_time")
    if existing_path is not None:
        print("[annual-cycle] reusing existing mean reconstructed chlorophyll by time")
        return load_dataset(existing_path)

    member_paths = sorted(Path(config.reconstructed_dir).glob("reconstructed_dataset_*.nc"))
    if not member_paths:
        raise FileNotFoundError("No reconstructed mean-by-time dataset or reconstructed member files found.")

    print(f"[annual-cycle] computing reconstructed mean from {len(member_paths)} member files")
    arrays = []
    variable_name = None
    for path in member_paths:
        ds = load_dataset(path)
        variable, _, _, _ = detect_variable_and_coords(ds, config.variable_name)
        variable_name = variable
        arrays.append(ds[variable].load())
        ds.close()

    mean_array = xr.concat(arrays, dim="member").mean(dim="member", skipna=True)
    mean_array.name = variable_name or config.variable_name
    return mean_array.to_dataset(name=mean_array.name)


def load_raw_time_dataset(config, found_paths: dict[str, Path | None]) -> xr.Dataset | None:
    """Load time-varying satellite/raw chlorophyll data for the raw annual cycle."""
    raw_mean_path = found_paths.get("raw_mean")
    if raw_mean_path is not None:
        with load_dataset(raw_mean_path) as raw_mean_ds:
            _, _, _, time_name = detect_variable_and_coords(raw_mean_ds, config.variable_name)
            if time_name is not None:
                print("[annual-cycle] reusing existing time-varying raw mean dataset")
                return load_dataset(raw_mean_path)
        print("[annual-cycle] raw mean dataset is 2D, so it cannot supply an annual cycle")

    if config.preprocessed_data_file and Path(config.preprocessed_data_file).exists():
        print("[annual-cycle] using configured preprocessed raw satellite dataset for Satellite-Chl")
        return load_dataset(Path(config.preprocessed_data_file))

    print("[annual-cycle] no time-varying raw satellite dataset available")
    return None


def load_climatology_dataset(config, found_paths: dict[str, Path | None]) -> xr.Dataset | None:
    climatology_path = found_paths.get("climatology")
    if climatology_path is None:
        print("[annual-cycle] climatological annual cycle dataset not available")
        return None
    print("[annual-cycle] reusing climatological annual cycle dataset")
    return load_dataset(climatology_path)


def to_annual_cycle(series: pd.Series | None) -> pd.Series | None:
    """Convert a time series into an annual 8-day cycle by day-of-year."""
    if series is None or series.empty:
        return None
    clean = series.sort_index()
    day_of_year = clean.index.dayofyear
    annual = clean.groupby(day_of_year).mean()
    annual.index = pd.Index(annual.index.astype(int), name="day_of_year")
    return annual


def annual_cycle_to_dates(series: pd.Series) -> pd.DatetimeIndex:
    return pd.to_datetime(["2001-01-01"]) + pd.to_timedelta(series.index.to_numpy() - 1, unit="D")


def annual_cycle_dataframe(
    satellite: pd.Series | None,
    reconstructed: pd.Series | None,
    climatology: pd.Series | None,
) -> pd.DataFrame:
    pieces = {}
    if satellite is not None:
        pieces["Satellite-Chl"] = satellite
    if reconstructed is not None:
        pieces["Reconstructed-Chl"] = reconstructed
    if climatology is not None:
        pieces["Climatology / Long-term Reconstructed-Chl"] = climatology
    frame = pd.DataFrame(pieces).sort_index()
    frame.insert(0, "date_label", [format_day_label(day) for day in frame.index])
    return frame


def format_day_label(day_of_year: int) -> str:
    date = pd.Timestamp("2001-01-01") + pd.Timedelta(days=int(day_of_year) - 1)
    return date.strftime("%d-%b")


def plot_region_annual_cycle(
    region_key: str,
    region: dict,
    frame: pd.DataFrame,
    output_path: Path,
) -> Path:
    """Plot one regional annual cycle."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 5), constrained_layout=True)
    x_values = pd.Timestamp("2001-01-01") + pd.to_timedelta(frame.index.to_numpy() - 1, unit="D")

    styles = {
        "Satellite-Chl": {"color": "#1f77b4", "linestyle": "-", "marker": "o"},
        "Reconstructed-Chl": {"color": "#d62728", "linestyle": "-", "marker": "s"},
        "Climatology / Long-term Reconstructed-Chl": {"color": "#2ca02c", "linestyle": "--", "marker": None},
    }
    for column, style in styles.items():
        if column not in frame:
            continue
        ax.plot(x_values, frame[column].values, linewidth=2.0, markersize=3.5, label=column, **style)

    ax.set_title(
        f"Annual Chlorophyll-a Cycle: {region['display_name']}\n"
        f"{region['lat_min']:g}-{region['lat_max']:g}N, {region['lon_min']:g}-{region['lon_max']:g}E"
    )
    ax.set_xlabel("8-day composite date")
    ax.set_ylabel("Chlorophyll-a concentration (mg m$^{-3}$)")
    monthly_ticks = pd.date_range("2001-01-01", "2001-12-01", freq="MS")
    ax.set_xticks(monthly_ticks)
    ax.set_xticklabels([date.strftime("%d-%b") for date in monthly_ticks], rotation=35, ha="right")
    ax.grid(True, linewidth=0.4, alpha=0.35)
    ax.legend()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    print(f"[annual-cycle] saved {output_path}")
    return output_path


def plot_combined_regions(
    region_frames: dict[str, pd.DataFrame],
    output_path: Path,
) -> Path:
    """Plot Arabian Sea and Bay of Bengal annual cycles in one figure."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=False, constrained_layout=True)

    for ax, (region_key, frame) in zip(axes, region_frames.items()):
        region = REGIONS[region_key]
        x_values = pd.Timestamp("2001-01-01") + pd.to_timedelta(frame.index.to_numpy() - 1, unit="D")
        for column, color in [
            ("Satellite-Chl", "#1f77b4"),
            ("Reconstructed-Chl", "#d62728"),
            ("Climatology / Long-term Reconstructed-Chl", "#2ca02c"),
        ]:
            if column in frame:
                linestyle = "--" if "Climatology" in column else "-"
                ax.plot(x_values, frame[column].values, linewidth=2.0, color=color, linestyle=linestyle, label=column)
        ax.set_title(region["display_name"])
        ax.set_xlabel("8-day composite date")
        ax.set_ylabel("Chlorophyll-a concentration (mg m$^{-3}$)")
        monthly_ticks = pd.date_range("2001-01-01", "2001-12-01", freq="2MS")
        ax.set_xticks(monthly_ticks)
        ax.set_xticklabels([date.strftime("%d-%b") for date in monthly_ticks], rotation=35, ha="right")
        ax.grid(True, linewidth=0.4, alpha=0.35)
        ax.legend(fontsize=8)

    fig.suptitle("Regional Annual Chlorophyll-a Cycles")
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    print(f"[annual-cycle] saved {output_path}")
    return output_path


def save_region_csv(frame: pd.DataFrame, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_path, index_label="day_of_year")
    print(f"[annual-cycle] saved {output_path}")
    return output_path


def generate_annual_cycle_outputs(config) -> dict[str, str]:
    """Generate annual cycle plots and CSV summaries for configured outputs."""
    found_paths = find_existing_mean_datasets(config)
    raw_ds = load_raw_time_dataset(config, found_paths)
    reconstructed_ds = compute_reconstructed_mean_if_needed(config, found_paths)
    climatology_ds = load_climatology_dataset(config, found_paths)

    outputs: dict[str, str] = {}
    region_frames: dict[str, pd.DataFrame] = {}
    annual_cycle_dir = Path(config.annual_cycle_dir)

    try:
        for region_key, region in REGIONS.items():
            satellite_series = (
                compute_spatial_mean_timeseries(raw_ds, region, config.variable_name)
                if raw_ds is not None
                else None
            )
            reconstructed_series = compute_spatial_mean_timeseries(
                reconstructed_ds,
                region,
                config.variable_name,
            )
            climatology_series = (
                compute_spatial_mean_timeseries(climatology_ds, region, config.variable_name)
                if climatology_ds is not None
                else None
            )

            frame = annual_cycle_dataframe(
                to_annual_cycle(satellite_series),
                to_annual_cycle(reconstructed_series),
                to_annual_cycle(climatology_series),
            )
            if frame.empty:
                print(f"[annual-cycle] no time-varying data for {region['display_name']}, skipping")
                continue

            region_frames[region_key] = frame
            plot_path = annual_cycle_dir / f"annual_cycle_{region_key}.png"
            csv_path = annual_cycle_dir / f"annual_cycle_{region_key}.csv"
            outputs[f"{region_key}_plot"] = str(plot_region_annual_cycle(region_key, region, frame, plot_path))
            outputs[f"{region_key}_csv"] = str(save_region_csv(frame, csv_path))

        if region_frames:
            combined_path = annual_cycle_dir / "annual_cycle_regions_comparison.png"
            outputs["regions_comparison_plot"] = str(plot_combined_regions(region_frames, combined_path))
    finally:
        for dataset in (raw_ds, reconstructed_ds, climatology_ds):
            if dataset is not None:
                dataset.close()

    return outputs


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Plot regional annual chlorophyll-a cycles.")
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
    outputs = generate_annual_cycle_outputs(config)
    print(f"[annual-cycle] completed with outputs: {outputs}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

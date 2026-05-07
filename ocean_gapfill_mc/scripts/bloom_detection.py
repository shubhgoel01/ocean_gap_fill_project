"""Detect bloom initiation and peak from regional annual chlorophyll-a cycles."""

from __future__ import annotations

import argparse
import json
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


def load_timeseries(csv_path: Path) -> pd.DataFrame:
    """Load annual-cycle CSV, detect time and reconstructed chlorophyll columns."""
    print(f"[bloom] loading {csv_path}")
    frame = pd.read_csv(csv_path)
    time_values = detect_time_values(frame)
    chl_column = detect_chlorophyll_column(frame)

    output = pd.DataFrame(
        {
            "time": time_values,
            "reconstructed_chl": pd.to_numeric(frame[chl_column], errors="coerce"),
        }
    )
    output = output.dropna(subset=["time"]).sort_values("time").reset_index(drop=True)
    print(f"[bloom] using chlorophyll column: {chl_column}")
    return output


def detect_time_values(frame: pd.DataFrame) -> pd.Series:
    """Detect or construct datetime values from annual-cycle columns."""
    lower_map = {column.lower(): column for column in frame.columns}

    for candidate in ("time", "date", "datetime"):
        if candidate in lower_map:
            return pd.to_datetime(frame[lower_map[candidate]], errors="coerce")

    if "day_of_year" in lower_map:
        day_values = pd.to_numeric(frame[lower_map["day_of_year"]], errors="coerce")
        return pd.Timestamp("2001-01-01") + pd.to_timedelta(day_values - 1, unit="D")

    if "date_label" in lower_map:
        return pd.to_datetime("2001-" + frame[lower_map["date_label"]].astype(str), format="%Y-%d-%b", errors="coerce")

    raise ValueError("Could not detect a time/date/day_of_year column.")


def detect_chlorophyll_column(frame: pd.DataFrame) -> str:
    """Detect reconstructed chlorophyll column without hardcoding exact spelling."""
    columns = list(frame.columns)
    normalized = {normalize_column_name(column): column for column in columns}

    for key, original in normalized.items():
        if "reconstructed" in key and "chl" in key:
            return original
    for key, original in normalized.items():
        if "reconstructed" in key and "chlorophyll" in key:
            return original
    for key, original in normalized.items():
        if key in {"reconstructedchl", "reconstructedchlorophyll"}:
            return original
    raise ValueError(f"Could not detect reconstructed chlorophyll column from {columns}")


def normalize_column_name(name: str) -> str:
    return "".join(character for character in name.lower() if character.isalnum())


def select_bloom_year(config, raw_series: pd.Series, reconstructed_series: pd.Series) -> int:
    """Choose configured bloom year, or the latest year shared by both inputs."""
    if config.bloom_detection_year is not None:
        return int(config.bloom_detection_year)

    raw_years = set(raw_series.index.year)
    reconstructed_years = set(reconstructed_series.index.year)
    shared_years = sorted(raw_years.intersection(reconstructed_years))
    if not shared_years:
        raise ValueError("Raw and reconstructed time series have no overlapping years.")
    selected_year = int(shared_years[-1])
    print(f"[bloom] bloom_detection_year not set; using latest shared year {selected_year}")
    return selected_year


def filter_series_to_year(series: pd.Series, year: int) -> pd.Series:
    """Return only values for one calendar year."""
    filtered = series[series.index.year == int(year)].sort_index()
    if filtered.empty:
        raise ValueError(f"No values found for year {year}.")
    return filtered


def smooth_signal(series: pd.Series, window: int = 3) -> pd.Series:
    """Interpolate NaNs and smooth the chlorophyll signal."""
    numeric = pd.to_numeric(series, errors="coerce")
    interpolated = numeric.interpolate(limit_direction="both")
    return interpolated.rolling(window=int(window), center=True, min_periods=1).mean()


def compute_threshold(
    smoothed: pd.Series,
    method: str = "median_multiplier",
    multiplier: float = 1.05,
    percentile: float = 60.0,
    log: bool = True,
) -> float:
    """Compute bloom threshold from the smoothed signal."""
    values = smoothed.dropna().to_numpy(dtype=float)
    if values.size == 0:
        raise ValueError("Cannot compute bloom threshold because the smoothed signal has no finite values.")

    if method == "percentile":
        threshold = float(np.nanpercentile(values, float(percentile)))
    elif method == "median_multiplier":
        threshold = float(np.nanmedian(values) * float(multiplier))
    else:
        raise ValueError(f"Unsupported bloom threshold method: {method}")

    if log:
        print(f"[bloom] threshold={threshold:.6g} using method={method}")
    return threshold


def detect_bloom_initiation(smoothed: pd.Series, threshold: float, peak_index: int | None) -> int | None:
    """Detect bloom start as the nearest upward threshold crossing left of peak."""
    if peak_index is None:
        print("[bloom] warning: cannot detect initiation because peak was not detected")
        return None

    values = smoothed.to_numpy(dtype=float)
    candidate_index = None
    for index in range(0, int(peak_index) + 1):
        current_value = values[index]
        if not np.isfinite(current_value):
            continue
        previous_value = values[index - 1] if index > 0 else np.nan
        crossed_upward = current_value >= threshold and (
            index == 0 or not np.isfinite(previous_value) or previous_value < threshold
        )
        if crossed_upward:
            candidate_index = index

    if candidate_index is None:
        print("[bloom] warning: no upward threshold crossing found to the left of the bloom peak")
    return candidate_index


def detect_bloom_peak(smoothed: pd.Series) -> int | None:
    """Detect global maximum smoothed chlorophyll."""
    if smoothed.dropna().empty:
        print("[bloom] warning: no finite smoothed values available for peak detection")
        return None
    values = smoothed.to_numpy(dtype=float)
    return int(np.nanargmax(values))


def detect_bloom_events(smoothed: pd.Series, threshold: float) -> dict:
    """Detect bloom peak first, then initiation on the rising limb before peak."""
    peak_index = detect_bloom_peak(smoothed)
    initiation_index = detect_bloom_initiation(smoothed, threshold, peak_index)
    return {
        "initiation_index": initiation_index,
        "peak_index": peak_index,
    }


def load_region_timeseries_from_dataset(dataset_path: Path, region: dict, config) -> pd.Series:
    """Load one NetCDF and compute regional spatial mean for every time step."""
    print(f"[bloom] loading time series from {dataset_path}")
    with xr.open_dataset(dataset_path) as dataset:
        variable = detect_dataset_variable(dataset, config.variable_name)
        lat_name = detect_dataset_coord(dataset, ["lat", "latitude", "y"])
        lon_name = detect_dataset_coord(dataset, ["lon", "longitude", "x"])
        time_name = detect_dataset_coord(dataset, ["time", "date"])
        data_array = subset_region(dataset[variable], lat_name, lon_name, region)
        spatial_mean = data_array.mean(dim=[lat_name, lon_name], skipna=True)
        time_values = pd.to_datetime(spatial_mean[time_name].values)
        values = np.asarray(spatial_mean.values, dtype=float)
    return pd.Series(values, index=time_values).sort_index()


def find_reconstructed_mean_by_time(config) -> Path:
    """Find already-computed reconstructed ensemble-mean by time."""
    candidates = [
        Path(config.datasets_dir) / "final_reconstructed_ensemble_mean_chlorophyll_by_time.nc",
        Path(config.reconstructed_dir) / "reconstructed_mean_by_time.nc",
    ]
    for path in candidates:
        if path.exists():
            print(f"[bloom] using reconstructed mean dataset: {path}")
            return path
    raise FileNotFoundError(f"Could not find reconstructed mean-by-time dataset: {candidates}")


def find_raw_satellite_dataset(config) -> Path:
    """Find time-varying raw satellite chlorophyll dataset."""
    if config.preprocessed_data_file and Path(config.preprocessed_data_file).exists():
        print(f"[bloom] using raw satellite dataset: {config.preprocessed_data_file}")
        return Path(config.preprocessed_data_file)
    raise FileNotFoundError("No configured preprocessed raw satellite dataset found.")


def build_metrics(
    region_key: str,
    region: dict,
    reconstructed_series: pd.Series,
    smoothed: pd.Series,
    threshold: float,
    initiation_index: int | None,
    peak_index: int | None,
    ensemble_uncertainty: dict,
    year: int,
    config,
) -> dict:
    """Build serializable bloom detection metrics."""
    region_name = region["display_name"]
    metrics = {
        "region": region_name,
        "year": int(year),
        "input_signal": "reconstructed_chl",
        "smoothing_window": int(config.bloom_smoothing_window),
        "threshold_method": config.bloom_threshold_method,
        "threshold_multiplier": float(config.bloom_threshold_multiplier),
        "threshold_percentile": float(config.bloom_threshold_percentile),
        "threshold_value": threshold,
        "bloom_initiation_date": None,
        "bloom_initiation_value": None,
        "bloom_peak_date": None,
        "bloom_peak_value": None,
        "ensemble_uncertainty": ensemble_uncertainty,
        "status": "ok",
    }

    if initiation_index is None:
        metrics["status"] = "no_bloom_initiation_detected"
        return metrics

    metrics["bloom_initiation_date"] = reconstructed_series.index[initiation_index].strftime("%d-%b")
    metrics["bloom_initiation_value"] = float(smoothed.iloc[initiation_index])

    if peak_index is None:
        metrics["status"] = "no_peak_detected_after_initiation"
        return metrics

    metrics["bloom_peak_date"] = reconstructed_series.index[peak_index].strftime("%d-%b")
    metrics["bloom_peak_value"] = float(smoothed.iloc[peak_index])
    print(
        f"[bloom] {region_name}: initiation={metrics['bloom_initiation_date']}, "
        f"peak={metrics['bloom_peak_date']}, peak_value={metrics['bloom_peak_value']:.6g}"
    )
    return metrics


def plot_bloom_detection(
    raw_series: pd.Series,
    reconstructed_series: pd.Series,
    smoothed: pd.Series,
    threshold: float,
    initiation_index: int | None,
    peak_index: int | None,
    ensemble_uncertainty: dict,
    region: dict,
    year: int,
    output_path: Path,
) -> Path:
    """Plot paper-style phenology markers and timing uncertainty."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    region_name = region["display_name"]

    fig, ax = plt.subplots(figsize=(7.2, 4.6), constrained_layout=True)
    original_color = "#00cfe8"
    reconstructed_color = "#ff1493"
    ax.plot(
        raw_series.index,
        raw_series.values,
        color=original_color,
        linewidth=2.2,
        label="original chl data",
    )
    ax.plot(
        reconstructed_series.index,
        reconstructed_series.values,
        color=reconstructed_color,
        linewidth=2.2,
        linestyle=":",
        label="mean gap-filled chl data",
    )

    if initiation_index is not None:
        initiation_time = reconstructed_series.index[initiation_index]
        initiation_value = reconstructed_series.iloc[initiation_index]
        ax.scatter(
            [initiation_time],
            [initiation_value],
            color=reconstructed_color,
            marker="o",
            s=38,
            zorder=5,
        )
        draw_event_uncertainty_bar(
            ax,
            ensemble_uncertainty.get("initiation_day_range"),
            initiation_value,
            reconstructed_color,
            year,
        )
    if peak_index is not None:
        peak_time = reconstructed_series.index[peak_index]
        peak_value = reconstructed_series.iloc[peak_index]
        ax.scatter(
            [peak_time],
            [peak_value],
            color=reconstructed_color,
            marker="s",
            s=38,
            zorder=5,
        )
        draw_event_uncertainty_bar(
            ax,
            ensemble_uncertainty.get("peak_day_range"),
            peak_value,
            reconstructed_color,
            year,
        )

    ax.set_title(f"Marine Phytoplankton Phenology for year {year}\nBloom Initiation and peak")
    ax.set_xlabel("Time (8-day composites)")
    ax.set_ylabel("Chlorophyll Concentration (mg/m3)")
    tick_values = reconstructed_series.index[::4]
    ax.set_xticks(tick_values)
    ax.set_xticklabels([date.strftime("%d-%b") for date in tick_values], rotation=90, ha="center")
    ax.grid(True, linewidth=0.5, alpha=0.3)
    ax.legend(loc="upper right", frameon=True, fontsize=8)

    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    print(f"[bloom] saved {output_path}")
    return output_path


def draw_event_uncertainty_bar(ax, day_range: dict | None, y_value: float, color: str, year: int) -> None:
    """Draw horizontal timing uncertainty bar with vertical end caps."""
    if not day_range or day_range.get("start_day_of_year") is None or day_range.get("end_day_of_year") is None:
        return
    start = day_of_year_to_timestamp(day_range["start_day_of_year"], year)
    end = day_of_year_to_timestamp(day_range["end_day_of_year"], year)
    if start == end:
        start = start - pd.Timedelta(days=4)
        end = end + pd.Timedelta(days=4)

    ylim = ax.get_ylim()
    cap_half_height = max((ylim[1] - ylim[0]) * 0.035, 0.005)
    ax.hlines(y_value, start, end, colors=color, linewidth=1.6, zorder=4)
    ax.vlines([start, end], y_value - cap_half_height, y_value + cap_half_height, colors=color, linewidth=1.6, zorder=4)


def shade_event_range(ax, day_range: dict | None, color: str, label: str) -> None:
    """Add an uncertainty date range as a translucent vertical band."""
    if not day_range or day_range.get("start_day_of_year") is None or day_range.get("end_day_of_year") is None:
        return
    year = int(day_range.get("year", 2001))
    start = day_of_year_to_timestamp(day_range["start_day_of_year"], year)
    end = day_of_year_to_timestamp(day_range["end_day_of_year"], year)
    if start == end:
        start = start - pd.Timedelta(days=4)
        end = end + pd.Timedelta(days=4)
    ax.axvspan(start, end, color=color, alpha=0.14, label=label)


def day_of_year_to_timestamp(day_of_year: int, year: int = 2001) -> pd.Timestamp:
    return pd.Timestamp(f"{int(year)}-01-01") + pd.Timedelta(days=int(day_of_year) - 1)


def find_annual_cycle_csv(config, region_key: str) -> Path:
    """Find annual-cycle CSV in current objective folder or legacy summaries folder."""
    candidates = [
        Path(config.annual_cycle_dir) / f"annual_cycle_{region_key}.csv",
        Path(config.output_directory) / "summaries" / f"annual_cycle_{region_key}.csv",
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(f"Could not find annual cycle CSV for {region_key}: {candidates}")


def compute_ensemble_bloom_uncertainty(region: dict, year: int, config) -> dict:
    """Run bloom detection for every reconstructed member and summarize date ranges."""
    member_paths = sorted(Path(config.reconstructed_dir).glob("reconstructed_dataset_*.nc"))
    if not member_paths:
        print("[bloom] warning: no reconstructed member files found for bloom uncertainty")
        return empty_ensemble_uncertainty()

    initiation_days = []
    peak_days = []
    event_count = 0
    for member_path in member_paths:
        try:
            member_series = load_member_region_year_timeseries(member_path, region, year, config)
            smoothed = smooth_signal(member_series, window=int(config.bloom_smoothing_window))
            threshold = compute_threshold(
                smoothed,
                method=config.bloom_threshold_method,
                multiplier=float(config.bloom_threshold_multiplier),
                percentile=float(config.bloom_threshold_percentile),
                log=False,
            )
            events = detect_bloom_events(smoothed, threshold)
            initiation_index = events["initiation_index"]
            peak_index = events["peak_index"]
            if initiation_index is None or peak_index is None:
                continue
            initiation_days.append(int(member_series.index[initiation_index].dayofyear))
            peak_days.append(int(member_series.index[peak_index].dayofyear))
            event_count += 1
        except Exception as exc:
            print(f"[bloom] warning: skipped {member_path.name}: {exc}")

    uncertainty = {
        "member_count": len(member_paths),
        "valid_event_count": event_count,
        "initiation_day_range": summarize_day_range(initiation_days, year),
        "peak_day_range": summarize_day_range(peak_days, year),
    }
    print(
        "[bloom] ensemble uncertainty: "
        f"initiation={uncertainty['initiation_day_range']}, "
        f"peak={uncertainty['peak_day_range']}"
    )
    return uncertainty


def load_member_region_year_timeseries(member_path: Path, region: dict, year: int, config) -> pd.Series:
    """Load one reconstructed member and compute regional means for one year."""
    with xr.open_dataset(member_path) as dataset:
        variable = detect_dataset_variable(dataset, config.variable_name)
        lat_name = detect_dataset_coord(dataset, ["lat", "latitude", "y"])
        lon_name = detect_dataset_coord(dataset, ["lon", "longitude", "x"])
        time_name = detect_dataset_coord(dataset, ["time", "date"])
        data_array = subset_region(dataset[variable], lat_name, lon_name, region)
        spatial_mean = data_array.mean(dim=[lat_name, lon_name], skipna=True)
        time_values = pd.to_datetime(spatial_mean[time_name].values)
        values = np.asarray(spatial_mean.values, dtype=float)

    series = pd.Series(values, index=time_values).sort_index()
    return filter_series_to_year(series, year)


def detect_dataset_variable(dataset: xr.Dataset, preferred_variable: str) -> str:
    if preferred_variable in dataset.data_vars:
        return preferred_variable
    if len(dataset.data_vars) == 1:
        return next(iter(dataset.data_vars))
    for candidate in ("chlor_a", "chlorophyll", "chl"):
        if candidate in dataset.data_vars:
            return candidate
    raise ValueError(f"Could not detect variable from {list(dataset.data_vars)}")


def detect_dataset_coord(dataset: xr.Dataset, candidates: list[str]) -> str:
    available = list(dataset.coords) + list(dataset.dims)
    lookup = {name.lower(): name for name in available}
    for candidate in candidates:
        if candidate.lower() in lookup:
            return lookup[candidate.lower()]
    raise ValueError(f"Could not detect coordinate from candidates {candidates}")


def subset_region(data_array: xr.DataArray, lat_name: str, lon_name: str, region: dict) -> xr.DataArray:
    lat_slice = coordinate_slice(data_array[lat_name], float(region["lat_min"]), float(region["lat_max"]))
    lon_slice = coordinate_slice(data_array[lon_name], float(region["lon_min"]), float(region["lon_max"]))
    return data_array.sel({lat_name: lat_slice, lon_name: lon_slice})


def coordinate_slice(coord: xr.DataArray, lower: float, upper: float) -> slice:
    values = np.asarray(coord.values, dtype=float)
    if values[0] <= values[-1]:
        return slice(lower, upper)
    return slice(upper, lower)


def summarize_day_range(days: list[int], year: int) -> dict:
    if not days:
        return {
            "year": int(year),
            "start_day_of_year": None,
            "end_day_of_year": None,
            "start_date": None,
            "end_date": None,
        }
    start_day = int(np.nanmin(days))
    end_day = int(np.nanmax(days))
    return {
        "year": int(year),
        "start_day_of_year": start_day,
        "end_day_of_year": end_day,
        "start_date": format_day_of_year(start_day, year),
        "end_date": format_day_of_year(end_day, year),
    }


def empty_ensemble_uncertainty() -> dict:
    return {
        "member_count": 0,
        "valid_event_count": 0,
        "initiation_day_range": summarize_day_range([], 2001),
        "peak_day_range": summarize_day_range([], 2001),
    }


def format_day_of_year(day_of_year: int, year: int = 2001) -> str:
    return day_of_year_to_timestamp(day_of_year, year).strftime("%d-%b")


def save_metrics(metrics: dict, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)
    print(f"[bloom] saved {output_path}")
    return output_path


def run_region(region_key: str, region: dict, config) -> dict[str, str]:
    raw_path = find_raw_satellite_dataset(config)
    reconstructed_path = find_reconstructed_mean_by_time(config)
    raw_all_years = load_region_timeseries_from_dataset(raw_path, region, config)
    reconstructed_all_years = load_region_timeseries_from_dataset(reconstructed_path, region, config)
    year = select_bloom_year(config, raw_all_years, reconstructed_all_years)
    raw_series = filter_series_to_year(raw_all_years, year)
    reconstructed_series = filter_series_to_year(reconstructed_all_years, year)
    smoothed = smooth_signal(reconstructed_series, window=int(config.bloom_smoothing_window))
    threshold = compute_threshold(
        smoothed,
        method=config.bloom_threshold_method,
        multiplier=float(config.bloom_threshold_multiplier),
        percentile=float(config.bloom_threshold_percentile),
    )
    events = detect_bloom_events(smoothed, threshold)
    initiation_index = events["initiation_index"]
    peak_index = events["peak_index"]
    ensemble_uncertainty = compute_ensemble_bloom_uncertainty(region, year, config)
    metrics = build_metrics(
        region_key,
        region,
        reconstructed_series,
        smoothed,
        threshold,
        initiation_index,
        peak_index,
        ensemble_uncertainty,
        year,
        config,
    )

    bloom_dir = Path(config.bloom_dir)
    metrics_path = save_metrics(metrics, bloom_dir / f"bloom_metrics_{region_key}.json")
    plot_path = plot_bloom_detection(
        raw_series,
        reconstructed_series,
        smoothed,
        threshold,
        initiation_index,
        peak_index,
        ensemble_uncertainty,
        region,
        year,
        bloom_dir / f"bloom_detection_{region_key}.png",
    )
    return {
        "metrics": str(metrics_path),
        "plot": str(plot_path),
    }


def generate_bloom_outputs(config) -> dict[str, dict[str, str]]:
    """Generate bloom metrics and plots for all configured regions."""
    outputs = {}
    for region_key, region in REGIONS.items():
        outputs[region_key] = run_region(region_key, region, config)
    return outputs


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Detect bloom initiation and peak from annual-cycle CSVs.")
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
    outputs = generate_bloom_outputs(config)
    print(f"[bloom] completed with outputs: {outputs}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

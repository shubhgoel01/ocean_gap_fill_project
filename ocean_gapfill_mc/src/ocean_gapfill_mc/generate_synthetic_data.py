"""
Synthetic chlorophyll dataset generator with realistic high-resolution grid
(~4 km equivalent) and structured variability suitable for testing
Monte Carlo gap-filling pipelines.

Key features:
- Higher spatial resolution to simulate ~4km grid behaviour
- Seasonal temporal signal
- Spatial gradients
- Random missing values
- Structured cloud-like missing regions
- Ensures enough time samples for reliable distribution fitting
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr


# -------------------------------------------------
# Main generator
# -------------------------------------------------

def generate_synthetic_chlorophyll_dataset(
    output_path: str | Path = "data/raw/synthetic_data.nc",
    random_seed: int = 42,
    time_steps: int = 365 * 2,  # 2 years daily data
    lat_size: int = 180,        # higher spatial resolution
    lon_size: int = 360,
    random_missing_fraction: float = 0.15,
) -> xr.Dataset:
    """
    Generate synthetic chlorophyll dataset that behaves similar to satellite data.

    Structure simulates:
    - spatial smooth gradients
    - seasonal cycles
    - realistic positive skew
    - structured missing regions (cloud cover)
    - random missing noise
    """

    rng = np.random.default_rng(random_seed)

    # time dimension
    time = pd.date_range("2003-01-01", periods=time_steps, freq="D")

    # spatial grid (~0.66° resolution approximates finer grid behaviour)
    lat = np.linspace(-60.0, 60.0, lat_size)
    lon = np.linspace(0.0, 359.0, lon_size)

    seasonal_component = build_seasonal_component(time_steps)
    spatial_component = build_spatial_component(lat, lon)

    # log-normal noise produces realistic positive skew
    noise = rng.lognormal(mean=0.0, sigma=0.25, size=(time_steps, lat_size, lon_size))

    chlorophyll = (
        seasonal_component[:, None, None]
        + spatial_component[None, :, :]
    ) * noise

    chlorophyll = np.maximum(chlorophyll, 0.01)

    chlorophyll = apply_random_missing_values(
        chlorophyll,
        rng,
        missing_fraction=random_missing_fraction,
    )

    chlorophyll = apply_structured_missing_regions(chlorophyll, rng)

    dataset = xr.Dataset(
        data_vars={
            "chlor_a": (("time", "lat", "lon"), chlorophyll),
        },
        coords={
            "time": time,
            "lat": lat,
            "lon": lon,
        },
        attrs={
            "title": "Synthetic chlorophyll dataset (high resolution)",
            "description": (
                "High-resolution synthetic chlorophyll dataset with realistic"
                " spatial gradients, seasonal variability, and structured"
                " missing values for Monte Carlo pipeline testing."
            ),
        },
    )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    dataset.to_netcdf(output_path)

    print_synthetic_dataset_summary(dataset, output_path)

    return dataset


# -------------------------------------------------
# Components
# -------------------------------------------------

def build_seasonal_component(time_steps: int) -> np.ndarray:
    """
    Create seasonal chlorophyll cycle.

    Mimics phytoplankton bloom pattern:
    gradual increase -> peak -> decline
    """

    phase = np.linspace(0.0, 4.0 * np.pi, time_steps, endpoint=False)

    base_level = 0.8

    seasonal_signal = (
        base_level
        + 0.6 * np.sin(phase)
        + 0.2 * np.sin(2 * phase)
    )

    return seasonal_signal



def build_spatial_component(lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    """
    Create spatial chlorophyll variability.

    Simulates:
    - higher productivity near equator
    - gradual longitudinal variation
    - mild spatial gradients
    """

    lat_radians = np.deg2rad(lat)
    lon_radians = np.deg2rad(lon)

    equatorial_boost = 1.2 * np.cos(lat_radians)[:, None]

    longitudinal_pattern = 0.8 + 0.3 * np.sin(2 * lon_radians)[None, :]

    gradient = 0.2 * (lat[:, None] / np.max(np.abs(lat)))

    return equatorial_boost + longitudinal_pattern + gradient


# -------------------------------------------------
# Missing value simulation
# -------------------------------------------------

def apply_random_missing_values(
    values: np.ndarray,
    rng: np.random.Generator,
    missing_fraction: float,
) -> np.ndarray:
    """
    Apply random scattered missing values.
    """

    result = values.copy()

    missing_mask = rng.random(result.shape) < missing_fraction

    result[missing_mask] = np.nan

    return result



def apply_structured_missing_regions(
    values: np.ndarray,
    rng: np.random.Generator,
    cloud_event_count: int = 25,
) -> np.ndarray:
    """
    Create cloud-like spatial gaps persisting over time.

    These produce realistic missing clusters needed to test
    interpolation and Monte Carlo behaviour.
    """

    result = values.copy()

    time_steps, lat_size, lon_size = result.shape

    for _ in range(cloud_event_count):

        start_time_index = int(rng.integers(0, time_steps - 10))

        duration = int(rng.integers(5, 20))

        lat_center = int(rng.integers(0, lat_size))
        lon_center = int(rng.integers(0, lon_size))

        lat_radius = int(rng.integers(6, max(8, lat_size // 8)))
        lon_radius = int(rng.integers(8, max(10, lon_size // 8)))

        lat_start = max(0, lat_center - lat_radius)
        lat_end = min(lat_size, lat_center + lat_radius)

        lon_start = max(0, lon_center - lon_radius)
        lon_end = min(lon_size, lon_center + lon_radius)

        time_end = min(time_steps, start_time_index + duration)

        result[
            start_time_index:time_end,
            lat_start:lat_end,
            lon_start:lon_end,
        ] = np.nan

    return result


# -------------------------------------------------
# Summary
# -------------------------------------------------

def print_synthetic_dataset_summary(dataset: xr.Dataset, output_path: Path) -> None:

    data_array = dataset["chlor_a"]

    total_cells = int(data_array.size)

    missing_cells = int(np.isnan(data_array.values).sum())

    missing_percent = (
        (missing_cells / total_cells) * 100.0
        if total_cells
        else 0.0
    )

    print("Synthetic dataset created")
    print(f"file: {output_path}")
    print(f"shape: {dict(data_array.sizes)}")
    print(f"missing: {missing_percent:.2f}%")


# -------------------------------------------------
# Run
# -------------------------------------------------

def main() -> None:

    generate_synthetic_chlorophyll_dataset()


if __name__ == "__main__":

    main()

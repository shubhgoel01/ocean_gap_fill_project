"""Fill-stage provenance helpers."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import xarray as xr


FILL_STAGE_FILENAME = "fill_stage_map.nc"
FILL_STAGE_VARIABLE = "fill_stage"

FILL_STAGE_UNRESOLVED = 0
FILL_STAGE_OBSERVED = 1
FILL_STAGE_INTERPOLATION = 2
FILL_STAGE_MONTE_CARLO = 3

FILL_STAGE_MEANINGS = {
    FILL_STAGE_UNRESOLVED: "unresolved_or_missing",
    FILL_STAGE_OBSERVED: "supported_by_observations",
    FILL_STAGE_INTERPOLATION: "filled_by_interpolation",
    FILL_STAGE_MONTE_CARLO: "filled_by_monte_carlo",
}


def build_fill_stage_map(
    regridded: xr.DataArray,
    interpolated: xr.DataArray,
    reconstructed: xr.DataArray,
) -> xr.DataArray:
    """Build a compact provenance map for the final working grid."""
    regridded, interpolated, reconstructed = xr.align(
        regridded,
        interpolated,
        reconstructed,
        join="exact",
    )

    stage_values = np.full(reconstructed.shape, FILL_STAGE_UNRESOLVED, dtype=np.uint8)
    regridded_values = np.asarray(regridded.values)
    interpolated_values = np.asarray(interpolated.values)
    reconstructed_values = np.asarray(reconstructed.values)

    observed_mask = np.isfinite(regridded_values)
    interpolation_mask = ~observed_mask & np.isfinite(interpolated_values)
    monte_carlo_mask = ~np.isfinite(interpolated_values) & np.isfinite(reconstructed_values)

    stage_values[observed_mask] = FILL_STAGE_OBSERVED
    stage_values[interpolation_mask] = FILL_STAGE_INTERPOLATION
    stage_values[monte_carlo_mask] = FILL_STAGE_MONTE_CARLO

    fill_stage = xr.DataArray(
        stage_values,
        coords=reconstructed.coords,
        dims=reconstructed.dims,
        name=FILL_STAGE_VARIABLE,
    )
    fill_stage.attrs.update(
        {
            "long_name": "Gap-fill provenance stage",
            "description": (
                "Integer code identifying how each value on the final working "
                "time-lat-lon grid was obtained."
            ),
            "flag_values": np.array(sorted(FILL_STAGE_MEANINGS), dtype=np.uint8),
            "flag_meanings": " ".join(
                FILL_STAGE_MEANINGS[code] for code in sorted(FILL_STAGE_MEANINGS)
            ),
            "code_0": FILL_STAGE_MEANINGS[FILL_STAGE_UNRESOLVED],
            "code_1": FILL_STAGE_MEANINGS[FILL_STAGE_OBSERVED],
            "code_2": FILL_STAGE_MEANINGS[FILL_STAGE_INTERPOLATION],
            "code_3": FILL_STAGE_MEANINGS[FILL_STAGE_MONTE_CARLO],
        }
    )
    return fill_stage


def save_fill_stage_map(
    regridded: xr.DataArray,
    interpolated: xr.DataArray,
    reconstructed: xr.DataArray,
    output_dir: str | Path,
) -> Path:
    """Save fill-stage provenance as a NetCDF dataset."""
    output_path = Path(output_dir) / FILL_STAGE_FILENAME
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fill_stage = build_fill_stage_map(regridded, interpolated, reconstructed)
    fill_stage.to_dataset(name=FILL_STAGE_VARIABLE).to_netcdf(output_path)
    return output_path


def load_fill_stage_map(path: str | Path) -> xr.DataArray:
    """Load a saved fill-stage map."""
    dataset = xr.open_dataset(path)
    try:
        if FILL_STAGE_VARIABLE not in dataset:
            raise ValueError(f"{path} does not contain variable {FILL_STAGE_VARIABLE!r}.")
        return dataset[FILL_STAGE_VARIABLE].load()
    finally:
        dataset.close()

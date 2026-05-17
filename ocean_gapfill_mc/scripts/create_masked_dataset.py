"""
Creates a masked dataset by applying the spatial/temporal gaps from f2 onto f1.
This is used to prepare data for validation testing.
"""

import argparse
from pathlib import Path
import sys

import numpy as np
import xarray as xr

# Add src/ to the path to import from the project
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ocean_gapfill_mc.utils.config import load_config
from ocean_gapfill_mc.utils.logging_utils import configure_logging, get_logger

def create_masked_dataset(config_path: Path):
    """Load config, mask f1 with f2, and save the result."""
    config = load_config(config_path)
    
    # We use the pipeline's logging configuration, but just log to console
    configure_logging(config.logs_dir, enable_file_logging=False)
    logger = get_logger(__name__)

    f1_path = config.masking_f1_path
    f2_path = config.masking_f2_path
    out_path = config.masking_output_path

    if not f1_path or not f2_path or not out_path:
        logger.error("Missing masking paths in configuration. Please ensure 'masking_f1_path', 'masking_f2_path', and 'masking_output_path' are set in default.json.")
        sys.exit(1)

    logger.info("Loading validation datasets...")
    logger.info(f"Target file (to be masked) [f1]: {f1_path}")
    logger.info(f"Reference file (contains gaps) [f2]: {f2_path}")
    
    if not Path(f1_path).exists():
        logger.error(f"f1 file does not exist: {f1_path}")
        sys.exit(1)
        
    if not Path(f2_path).exists():
        logger.error(f"f2 file does not exist: {f2_path}")
        sys.exit(1)

    with xr.open_dataset(f1_path) as ds1, xr.open_dataset(f2_path) as ds2:
        logger.info("Applying mask from f2 onto f1...")
        if len(ds1.data_vars) != 1 or len(ds2.data_vars) != 1:
            logger.error("Expected each masking dataset to contain exactly one data variable.")
            sys.exit(1)

        target_var = next(iter(ds1.data_vars))
        reference_var = next(iter(ds2.data_vars))
        target = ds1[target_var]
        reference = ds2[reference_var]

        if target.dims != reference.dims or target.shape != reference.shape:
            logger.error(
                "Target and reference variables must have matching dims and shape. "
                f"Got {target_var}{target.dims}{target.shape} and "
                f"{reference_var}{reference.dims}{reference.shape}."
            )
            sys.exit(1)

        # Apply the mask by position. The preprocessed files can have tiny
        # coordinate precision differences, so label alignment would drop cells.
        mask = xr.DataArray(
            np.isfinite(reference.values),
            dims=target.dims,
            coords={dim: target.coords[dim] for dim in target.dims},
        )
        masked_ds1 = ds1.where(mask)

        logger.info(f"Saving masked dataset to: {out_path}")
        
        # Ensure the output parent directory exists
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)

        encoding = {coord: {"_FillValue": None} for coord in masked_ds1.coords}
        masked_ds1.to_netcdf(out_path, encoding=encoding)
        logger.info("Masked dataset generated and saved successfully!")


def main():
    parser = argparse.ArgumentParser(description="Create a masked dataset for validation.")
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "default.json",
        help="Path to the JSON configuration file.",
    )
    args = parser.parse_args()
    create_masked_dataset(args.config)


if __name__ == "__main__":
    main()

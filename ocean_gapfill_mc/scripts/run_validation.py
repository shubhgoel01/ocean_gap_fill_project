"""
Final validation pipeline for the ocean gap-filling project.
Calculates MSE, RMSE, MAE, Bias, and coverage across true missing data,
handling interpolated vs. stochastic (Monte Carlo) cells appropriately.
"""

import argparse
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import xarray as xr

# Add src/ to the path to import from the project
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ocean_gapfill_mc.utils.config import load_config
from ocean_gapfill_mc.utils.logging_utils import configure_logging, get_logger

def calculate_metrics(target_array: np.ndarray, pred_array: np.ndarray) -> dict:
    """Calculate core regression metrics."""
    if len(target_array) == 0:
        return {"count": 0, "mse": np.nan, "rmse": np.nan, "mae": np.nan, "bias": np.nan}
    
    error = pred_array - target_array
    mse = np.mean(error ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(error))
    bias = np.mean(error)
    
    return {
        "count": len(target_array),
        "mse": float(mse),
        "rmse": float(rmse),
        "mae": float(mae),
        "bias": float(bias)
    }

def run_validation(config_path: Path):
    """Run the validation comparing the reconstructed pipeline output against the raw target."""
    config = load_config(config_path)
    
    configure_logging(config.logs_dir, enable_file_logging=False)
    logger = get_logger(__name__)

    # Validate paths exist in config
    target_path = config.validation_target_data_path
    input_path = config.validation_input_data_path
    uncertainty_path = config.uncertainty_dataset_path
    year = config.validation_year

    if not all([target_path, input_path, uncertainty_path, year]):
        logger.error("Missing validation config fields. Please set 'validation_target_data_path', 'validation_input_data_path', 'uncertainty_dataset_path', and 'validation_year'.")
        sys.exit(1)

    logger.info(f"Running validation for year: {year}")
    logger.info(f"Target (Gap-free Truth): {target_path}")
    logger.info(f"Input (Masked Gaps): {input_path}")
    logger.info(f"Pipeline Output: {uncertainty_path}")

    # Load datasets
    try:
        ds_target = xr.open_dataset(target_path)
        ds_input = xr.open_dataset(input_path)
        ds_uncert = xr.open_dataset(uncertainty_path)
    except Exception as e:
        logger.error(f"Failed to load datasets: {e}")
        sys.exit(1)

    # Filter by configured year
    time_dim = config.time_dim
    ds_target = ds_target.sel({time_dim: ds_target[time_dim].dt.year == year})
    ds_input = ds_input.sel({time_dim: ds_input[time_dim].dt.year == year})
    ds_uncert = ds_uncert.sel({time_dim: ds_uncert[time_dim].dt.year == year})

    var_name = config.variable_name

    # Extract 3D numpy arrays for fast boolean masking
    arr_target = ds_target[var_name].values
    arr_input = ds_input[var_name].values
    
    arr_mean = ds_uncert["reconstructed_mean"].values
    arr_lower = ds_uncert["reconstructed_lower_percentile"].values
    arr_upper = ds_uncert["reconstructed_upper_percentile"].values

    # Determine original missing cells that have a valid ground truth to compare against
    missing_mask = np.isnan(arr_input) & np.isfinite(arr_target)
    
    if not np.any(missing_mask):
        logger.warning("No originally missing cells with valid ground-truth target found! Check your masking logic.")
        sys.exit(0)

    # Within the missing cells, classify them by how the pipeline handled them
    # 1. Unresolved: Pipeline output is still NaN
    unresolved_mask = missing_mask & np.isnan(arr_mean)
    
    # 2. Interpolated: Pipeline output is valid, but deterministic (width == 0)
    valid_filled_mask = missing_mask & np.isfinite(arr_mean)
    width = arr_upper - arr_lower
    interpolated_mask = valid_filled_mask & (width == 0)
    
    # 3. Monte Carlo: Pipeline output is valid, and stochastic (width > 0)
    monte_carlo_mask = valid_filled_mask & (width > 0)

    logger.info(f"Total True Missing Cells Evaluated: {np.sum(missing_mask):,}")
    logger.info(f" - Filled by Interpolation: {np.sum(interpolated_mask):,}")
    logger.info(f" - Filled by Monte Carlo: {np.sum(monte_carlo_mask):,}")
    logger.info(f" - Left Unresolved: {np.sum(unresolved_mask):,}")

    results = []

    # Helper function to evaluate a specific group
    def evaluate_group(name: str, mask: np.ndarray):
        if not np.any(mask):
            return
        
        target_vals = arr_target[mask]
        pred_vals = arr_mean[mask]
        lower_vals = arr_lower[mask]
        upper_vals = arr_upper[mask]
        
        metrics = calculate_metrics(target_vals, pred_vals)
        
        # Coverage calculation: what % of true targets fall in the p05-p95 range?
        covered = (target_vals >= lower_vals) & (target_vals <= upper_vals)
        coverage_pct = np.mean(covered) * 100
        
        metrics["group"] = name
        metrics["coverage_percent"] = float(coverage_pct)
        results.append(metrics)

    # Evaluate Groups
    evaluate_group("All Filled Cells", valid_filled_mask)
    evaluate_group("Interpolation Cells", interpolated_mask)
    evaluate_group("Monte Carlo Cells", monte_carlo_mask)

    # Save Results
    df_results = pd.DataFrame(results)
    
    # Reorder columns
    cols = ["group", "count", "mse", "rmse", "mae", "bias", "coverage_percent"]
    df_results = df_results[cols]
    
    out_dir = Path(config.validation_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / f"validation_metrics_{year}.csv"
    
    df_results.to_csv(out_csv, index=False)
    logger.info(f"Validation metrics saved to: {out_csv}")
    
    # Print clean summary to console
    print("\n--- Final Validation Metrics ---")
    print(df_results.to_string(index=False))
    print("--------------------------------\n")


def main():
    parser = argparse.ArgumentParser(description="Run the final gap-filling validation.")
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "default.json",
        help="Path to the JSON configuration file.",
    )
    args = parser.parse_args()
    run_validation(args.config)


if __name__ == "__main__":
    main()

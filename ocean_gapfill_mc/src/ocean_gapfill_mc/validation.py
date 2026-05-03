"""Validation helpers using t1-t2 realistic masking patterns."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import xarray as xr

def prepare_timestep_pair_validation_mask(data_array: xr.DataArray, config) -> dict:
    """Create and apply validation masks before interpolation starts."""
    output_dir = Path(config.validation_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(int(config.validation_random_seed))
    pairs = generate_validation_pairs(data_array, config, rng)
    masked_data = data_array.copy(deep=True)
    masked_values = np.asarray(masked_data.values, dtype=float).copy()
    pair_records = []

    for pair_index, pair in enumerate(pairs, start=1):
        record = build_validation_mask_for_pair(
            data_array,
            pair["t1"],
            pair["t2"],
            pair["season"],
            pair_index,
            float(config.validation_mask_percent),
            rng,
        )
        final_mask = record["final_mask"]
        masked_values[record["t1"]][final_mask] = np.nan
        pair_records.append(record)

    masked_data.values = masked_values
    setup_summary = summarize_validation_mask_setup(pair_records)
    write_mask_setup_report(
        setup_summary,
        output_dir / "summary" / "mask_setup.txt",
    )
    return {
        "masked_data": masked_data,
        "pair_records": pair_records,
        "summary": setup_summary,
    }


def evaluate_timestep_pair_validation(
    reconstructed_datasets: list[xr.DataArray],
    interpolated_data: xr.DataArray,
    validation_setup: dict | None,
    config,
) -> dict:
    """Evaluate pre-interpolation validation masks after normal reconstruction."""
    if not validation_setup:
        return {"pairs": [], "summary": summarize_validation_results([])}

    pair_results = []
    for record in validation_setup["pair_records"]:
        result = evaluate_precomputed_validation_pair(
            reconstructed_datasets,
            interpolated_data,
            record,
        )
        pair_results.append(result)
        pair_dir = Path(config.validation_dir) / f"pair{result['pair_index']}"
        write_pair_report(result, pair_dir / "report.txt")
        write_validation_points_report(result, pair_dir / "validated_points.txt")
        print(format_pair_console_line(result))

    summary = summarize_validation_results(pair_results)
    write_summary_report(summary, Path(config.validation_dir) / "summary" / "summary.txt")
    print(format_summary_console_block(summary))
    return {"pairs": pair_results, "summary": summary}


def build_validation_mask_for_pair(
    data_array: xr.DataArray,
    t1: int,
    t2: int,
    season: str,
    pair_index: int,
    mask_percent: float,
    rng: np.random.Generator,
) -> dict:
    """Build one realistic validation mask without running reconstruction."""
    data_t1 = np.asarray(data_array.isel(time=t1).values, dtype=float)
    data_t2 = np.asarray(data_array.isel(time=t2).values, dtype=float)

    mask_t1 = np.isnan(data_t1)
    mask_t2 = np.isnan(data_t2)
    valid_t1 = ~mask_t1
    candidate_mask = mask_t2 & valid_t1
    final_mask = build_controlled_validation_mask(valid_t1, candidate_mask, mask_percent, rng)

    return {
        "pair_index": int(pair_index),
        "season": season,
        "t1": int(t1),
        "t2": int(t2),
        "t1_time": str(data_array["time"].values[t1]),
        "t2_time": str(data_array["time"].values[t2]),
        "hidden_points": int(np.sum(final_mask)),
        "total_valid_t1": int(np.sum(valid_t1)),
        "candidate_points": int(np.sum(candidate_mask)),
        "true_values": data_t1[final_mask],
        "final_mask": final_mask,
    }


def evaluate_precomputed_validation_pair(
    reconstructed_datasets: list[xr.DataArray],
    interpolated_data: xr.DataArray,
    record: dict,
) -> dict:
    """Evaluate one precomputed validation mask using the normal pipeline output."""
    stage_counts = compute_validation_fill_stage_counts(
        reconstructed_datasets,
        interpolated_data,
        record,
    )
    if not reconstructed_datasets:
        metrics = compute_validation_metrics(
            record["true_values"],
            np.full_like(record["true_values"], np.nan, dtype=float),
            np.full_like(record["true_values"], np.nan, dtype=float),
            np.full_like(record["true_values"], np.nan, dtype=float),
        )
        return {**record_without_arrays(record), **stage_counts, **metrics}

    t1 = int(record["t1"])
    final_mask = record["final_mask"]
    stacked = np.stack([dataset.values for dataset in reconstructed_datasets], axis=0)
    mean_map = np.nanmean(stacked, axis=0)
    p05_map = np.nanpercentile(stacked, 5, axis=0)
    p95_map = np.nanpercentile(stacked, 95, axis=0)

    pred_vals = mean_map[t1][final_mask]
    p05_vals = p05_map[t1][final_mask]
    p95_vals = p95_map[t1][final_mask]
    metrics = compute_validation_metrics(record["true_values"], pred_vals, p05_vals, p95_vals)
    point_records = build_validation_point_records(
        record,
        pred_vals,
        p05_vals,
        p95_vals,
        interpolated_data,
        reconstructed_datasets,
    )
    return {**record_without_arrays(record), **stage_counts, **metrics, "point_records": point_records}


def build_validation_point_records(
    record: dict,
    pred_vals: np.ndarray,
    p05_vals: np.ndarray,
    p95_vals: np.ndarray,
    interpolated_data: xr.DataArray,
    reconstructed_datasets: list[xr.DataArray],
) -> list[dict]:
    """Build per-hidden-point validation records for inspection."""
    t1 = int(record["t1"])
    row_indices, col_indices = np.where(record["final_mask"])
    true_vals = np.asarray(record["true_values"], dtype=float)
    pred_vals = np.asarray(pred_vals, dtype=float)
    p05_vals = np.asarray(p05_vals, dtype=float)
    p95_vals = np.asarray(p95_vals, dtype=float)
    interpolated_vals = np.asarray(interpolated_data.values[t1][record["final_mask"]], dtype=float)
    if reconstructed_datasets:
        reconstructed_vals = np.asarray(
            reconstructed_datasets[0].values[t1][record["final_mask"]],
            dtype=float,
        )
    else:
        reconstructed_vals = np.full_like(true_vals, np.nan, dtype=float)

    lat_values = np.asarray(interpolated_data["lat"].values, dtype=float)
    lon_values = np.asarray(interpolated_data["lon"].values, dtype=float)

    point_records = []
    for index, (lat_index, lon_index) in enumerate(zip(row_indices, col_indices), start=1):
        true_value = true_vals[index - 1]
        estimated_value = pred_vals[index - 1]
        error = estimated_value - true_value if np.isfinite(estimated_value) else np.nan
        point_records.append(
            {
                "point_index": int(index),
                "time_index": t1,
                "time_value": record["t1_time"],
                "lat_index": int(lat_index),
                "lat_value": float(lat_values[lat_index]),
                "lon_index": int(lon_index),
                "lon_value": float(lon_values[lon_index]),
                "fill_stage": determine_validation_point_stage(
                    interpolated_vals[index - 1],
                    reconstructed_vals[index - 1],
                ),
                "actual_value": float(true_value),
                "estimated_value": float(estimated_value) if np.isfinite(estimated_value) else None,
                "error": float(error) if np.isfinite(error) else None,
                "absolute_error": float(abs(error)) if np.isfinite(error) else None,
                "p05": float(p05_vals[index - 1]) if np.isfinite(p05_vals[index - 1]) else None,
                "p95": float(p95_vals[index - 1]) if np.isfinite(p95_vals[index - 1]) else None,
                "covered": bool(
                    np.isfinite(true_value)
                    and np.isfinite(p05_vals[index - 1])
                    and np.isfinite(p95_vals[index - 1])
                    and p05_vals[index - 1] <= true_value <= p95_vals[index - 1]
                ),
            }
        )
    return point_records


def determine_validation_point_stage(interpolated_value: float, reconstructed_value: float) -> str:
    if np.isfinite(interpolated_value):
        return "filled_by_interpolation"
    if np.isfinite(reconstructed_value):
        return "filled_by_monte_carlo"
    return "unresolved_after_reconstruction"


def compute_validation_fill_stage_counts(
    reconstructed_datasets: list[xr.DataArray],
    interpolated_data: xr.DataArray,
    record: dict,
) -> dict:
    """Count whether hidden validation pixels were filled by interpolation or Monte Carlo."""
    t1 = int(record["t1"])
    final_mask = record["final_mask"]
    interpolated_vals = np.asarray(interpolated_data.values[t1][final_mask], dtype=float)
    interpolation_filled = np.isfinite(interpolated_vals)

    if reconstructed_datasets:
        reconstructed_vals = np.asarray(reconstructed_datasets[0].values[t1][final_mask], dtype=float)
        reconstructed_filled = np.isfinite(reconstructed_vals)
    else:
        reconstructed_filled = np.zeros_like(interpolation_filled, dtype=bool)

    monte_carlo_filled = (~interpolation_filled) & reconstructed_filled
    unresolved = (~interpolation_filled) & (~reconstructed_filled)
    return {
        "filled_by_interpolation": int(np.sum(interpolation_filled)),
        "filled_by_monte_carlo": int(np.sum(monte_carlo_filled)),
        "unresolved_after_reconstruction": int(np.sum(unresolved)),
    }


def record_without_arrays(record: dict) -> dict:
    return {
        key: value
        for key, value in record.items()
        if key not in {"final_mask", "true_values"}
    }


def generate_validation_pairs(data_array: xr.DataArray, config, rng: np.random.Generator) -> list[dict]:
    """Generate validation pairs from configured seasons or all time indices."""
    if int(data_array.sizes["time"]) < 2:
        return []
    season_timesteps = normalize_season_timesteps(data_array, config.validation_season_timesteps)
    season_names = sorted(season_timesteps)
    pairs = []
    for season in build_balanced_validation_season_sequence(
        season_names,
        int(config.validation_num_pairs),
        rng,
    ):
        timesteps = np.asarray(season_timesteps[season], dtype=int)
        t1, t2 = rng.choice(timesteps, size=2, replace=False)
        pairs.append({"season": season, "t1": int(t1), "t2": int(t2)})
    return pairs


def build_balanced_validation_season_sequence(
    season_names: list[str],
    pair_count: int,
    rng: np.random.Generator,
) -> list[str]:
    """Spread validation pairs as evenly as possible across seasons."""
    if pair_count <= 0 or not season_names:
        return []

    base_count = pair_count // len(season_names)
    remainder = pair_count % len(season_names)
    season_sequence = []
    for season in season_names:
        season_sequence.extend([season] * base_count)

    if remainder:
        extra_seasons = list(rng.choice(season_names, size=remainder, replace=False))
        season_sequence.extend(str(season) for season in extra_seasons)

    rng.shuffle(season_sequence)
    return [str(season) for season in season_sequence]


def normalize_season_timesteps(
    data_array: xr.DataArray,
    configured_seasons: dict[str, list[int]] | None,
) -> dict[str, list[int]]:
    """Return valid seasonal timestep groups for this dataset."""
    time_count = int(data_array.sizes["time"])
    if configured_seasons:
        normalized = {}
        for season, timesteps in configured_seasons.items():
            valid_steps = sorted({int(step) for step in timesteps if 0 <= int(step) < time_count})
            if len(valid_steps) >= 2:
                normalized[str(season)] = valid_steps
        if normalized:
            return normalized
    return {"all": list(range(time_count))}


def build_controlled_validation_mask(
    valid_t1: np.ndarray,
    candidate_mask: np.ndarray,
    mask_percent: float,
    rng: np.random.Generator,
) -> np.ndarray:
    total_valid = int(np.sum(valid_t1))
    max_to_hide = int(mask_percent * total_valid)
    candidate_indices = np.argwhere(candidate_mask)
    final_mask = np.zeros_like(candidate_mask, dtype=bool)

    if max_to_hide <= 0 or candidate_indices.size == 0:
        return final_mask
    if len(candidate_indices) <= max_to_hide:
        final_mask[candidate_mask] = True
        return final_mask

    chosen_positions = rng.choice(len(candidate_indices), size=max_to_hide, replace=False)
    chosen_indices = candidate_indices[chosen_positions]
    final_mask[chosen_indices[:, 0], chosen_indices[:, 1]] = True
    return final_mask


def compute_validation_metrics(
    true_vals: np.ndarray,
    pred_vals: np.ndarray,
    p05_vals: np.ndarray,
    p95_vals: np.ndarray,
) -> dict:
    """Compute metrics, excluding hidden points with NaN predictions or intervals."""
    true_vals = np.asarray(true_vals, dtype=float)
    pred_vals = np.asarray(pred_vals, dtype=float)
    p05_vals = np.asarray(p05_vals, dtype=float)
    p95_vals = np.asarray(p95_vals, dtype=float)

    total_points = int(true_vals.size)
    evaluable_mask = (
        np.isfinite(true_vals)
        & np.isfinite(pred_vals)
        & np.isfinite(p05_vals)
        & np.isfinite(p95_vals)
    )
    evaluated_points = int(np.sum(evaluable_mask))
    nan_prediction_points = int(total_points - evaluated_points)

    if evaluated_points == 0:
        return {
            "mae": None,
            "rmse": None,
            "covered_count": 0,
            "total_points": total_points,
            "evaluated_points": 0,
            "nan_prediction_points": nan_prediction_points,
            "coverage_percent": None,
            "avg_width": None,
        }

    true_eval = true_vals[evaluable_mask]
    pred_eval = pred_vals[evaluable_mask]
    p05_eval = p05_vals[evaluable_mask]
    p95_eval = p95_vals[evaluable_mask]
    errors = pred_eval - true_eval
    inside_range = (true_eval >= p05_eval) & (true_eval <= p95_eval)
    interval_width = p95_eval - p05_eval

    return {
        "mae": float(np.mean(np.abs(errors))),
        "rmse": float(np.sqrt(np.mean(errors**2))),
        "covered_count": int(np.sum(inside_range)),
        "total_points": total_points,
        "evaluated_points": evaluated_points,
        "nan_prediction_points": nan_prediction_points,
        "coverage_percent": float((np.sum(inside_range) / evaluated_points) * 100.0),
        "avg_width": float(np.mean(interval_width)),
    }


def summarize_validation_results(pair_results: list[dict]) -> dict:
    mae_values = finite_metric_values(pair_results, "mae")
    rmse_values = finite_metric_values(pair_results, "rmse")
    coverage_values = finite_metric_values(pair_results, "coverage_percent")
    width_values = finite_metric_values(pair_results, "avg_width")
    total_covered = int(sum(result["covered_count"] for result in pair_results))
    total_evaluated = int(sum(result["evaluated_points"] for result in pair_results))
    total_interpolation_filled = int(
        sum(result.get("filled_by_interpolation", 0) for result in pair_results)
    )
    total_monte_carlo_filled = int(
        sum(result.get("filled_by_monte_carlo", 0) for result in pair_results)
    )
    total_unresolved_after_reconstruction = int(
        sum(result.get("unresolved_after_reconstruction", 0) for result in pair_results)
    )
    return {
        "pair_count": len(pair_results),
        "average_mae": mean_or_none(mae_values),
        "average_rmse": mean_or_none(rmse_values),
        "average_coverage_percent": mean_or_none(coverage_values),
        "overall_coverage_percent": (
            float((total_covered / total_evaluated) * 100.0) if total_evaluated else None
        ),
        "average_uncertainty_width": mean_or_none(width_values),
        "total_covered_points": total_covered,
        "total_evaluated_points": total_evaluated,
        "total_hidden_points": int(sum(result["total_points"] for result in pair_results)),
        "total_nan_prediction_points": int(
            sum(result["nan_prediction_points"] for result in pair_results)
        ),
        "total_filled_by_interpolation": total_interpolation_filled,
        "total_filled_by_monte_carlo": total_monte_carlo_filled,
        "total_unresolved_after_reconstruction": total_unresolved_after_reconstruction,
    }


def summarize_validation_mask_setup(pair_records: list[dict]) -> dict:
    return {
        "pair_count": int(len(pair_records)),
        "total_hidden_points": int(sum(record["hidden_points"] for record in pair_records)),
        "total_candidate_points": int(sum(record["candidate_points"] for record in pair_records)),
        "total_valid_t1_points": int(sum(record["total_valid_t1"] for record in pair_records)),
        "pairs": [
            {
                "pair_index": record["pair_index"],
                "season": record["season"],
                "t1": record["t1"],
                "t2": record["t2"],
                "hidden_points": record["hidden_points"],
                "candidate_points": record["candidate_points"],
                "total_valid_t1": record["total_valid_t1"],
            }
            for record in pair_records
        ],
    }


def finite_metric_values(results: list[dict], key: str) -> list[float]:
    values = []
    for result in results:
        value = result.get(key)
        if value is not None and np.isfinite(value):
            values.append(float(value))
    return values


def mean_or_none(values: list[float]) -> float | None:
    return float(np.mean(values)) if values else None


def write_pair_report(result: dict, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        f"Validation Pair {result['pair_index']}",
        "=================",
        f"Season: {result['season']}",
        f"t1: {result['t1']} ({result['t1_time']})",
        f"t2: {result['t2']} ({result['t2_time']})",
        f"Hidden Points: {result['hidden_points']}",
        f"Filled by interpolation: {result.get('filled_by_interpolation', 0)}",
        f"Filled by Monte Carlo: {result.get('filled_by_monte_carlo', 0)}",
        f"Unresolved after reconstruction: {result.get('unresolved_after_reconstruction', 0)}",
        f"Evaluated Points: {result['evaluated_points']}",
        f"Masked but NaN after reconstruction: {result['nan_prediction_points']}",
        f"MAE: {format_metric(result['mae'])} mg m^-3",
        f"RMSE: {format_metric(result['rmse'])} mg m^-3",
        f"Coverage: {format_metric(result['coverage_percent'])}%",
        f"Covered Count: {result['covered_count']}",
        f"Avg Uncertainty Width: {format_metric(result['avg_width'])} mg m^-3",
    ]
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output_path


def write_validation_points_report(result: dict, output_path: Path) -> Path:
    """Write actual vs estimated values for every hidden validation point."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    point_records = result.get("point_records", [])
    lines = [
        f"Validation Pair {result['pair_index']} Points",
        "========================",
        "Units: chlorophyll-a concentration, mg m^-3",
        "",
    ]
    if not point_records:
        lines.append("No hidden validation points were available.")
        output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return output_path

    header = (
        "point_index,time_index,time_value,lat_index,lat_value,lon_index,lon_value,"
        "fill_stage,actual_value,estimated_value,error,absolute_error,p05,p95,covered"
    )
    lines.append(header)
    for point in point_records:
        lines.append(
            ",".join(
                [
                    str(point["point_index"]),
                    str(point["time_index"]),
                    str(point["time_value"]),
                    str(point["lat_index"]),
                    format_float(point["lat_value"]),
                    str(point["lon_index"]),
                    format_float(point["lon_value"]),
                    str(point["fill_stage"]),
                    format_float(point["actual_value"]),
                    format_optional_float(point["estimated_value"]),
                    format_optional_float(point["error"]),
                    format_optional_float(point["absolute_error"]),
                    format_optional_float(point["p05"]),
                    format_optional_float(point["p95"]),
                    str(point["covered"]),
                ]
            )
        )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output_path


def write_summary_report(summary: dict, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "Validation Summary",
        "==================",
        f"Pairs: {summary['pair_count']}",
        f"Average MAE: {format_metric(summary['average_mae'])} mg m^-3",
        f"Average RMSE: {format_metric(summary['average_rmse'])} mg m^-3",
        f"Average Coverage: {format_metric(summary['average_coverage_percent'])}%",
        f"Overall Coverage: {format_metric(summary['overall_coverage_percent'])}%",
        f"Average Uncertainty Width: {format_metric(summary['average_uncertainty_width'])} mg m^-3",
        f"Total Evaluated Points: {summary['total_evaluated_points']}",
        f"Total Hidden Points: {summary['total_hidden_points']}",
        f"Total Masked but NaN after reconstruction: {summary['total_nan_prediction_points']}",
        f"Total Filled by Interpolation: {summary['total_filled_by_interpolation']}",
        f"Total Filled by Monte Carlo: {summary['total_filled_by_monte_carlo']}",
        f"Total Unresolved After Reconstruction: {summary['total_unresolved_after_reconstruction']}",
    ]
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output_path


def write_mask_setup_report(summary: dict, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "Validation Mask Setup",
        "=====================",
        f"Pairs: {summary['pair_count']}",
        f"Total Hidden Points: {summary['total_hidden_points']}",
        f"Total Candidate Points: {summary['total_candidate_points']}",
        f"Total Valid t1 Points Across Pairs: {summary['total_valid_t1_points']}",
        "",
        "Pairs",
    ]
    for pair in summary["pairs"]:
        lines.append(
            "Pair {pair_index}: season={season}, t1={t1}, t2={t2}, "
            "hidden={hidden_points}, candidates={candidate_points}, valid_t1={total_valid_t1}".format(
                **pair
            )
        )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output_path


def format_pair_console_line(result: dict) -> str:
    return (
        f"Validation pair {result['pair_index']}: "
        f"Hidden Points={result['hidden_points']}, "
        f"Interpolation={result.get('filled_by_interpolation', 0)}, "
        f"Monte Carlo={result.get('filled_by_monte_carlo', 0)}, "
        f"MAE={format_metric(result['mae'])}, "
        f"RMSE={format_metric(result['rmse'])}, "
        f"Coverage={format_metric(result['coverage_percent'])}%, "
        f"Avg Uncertainty Width={format_metric(result['avg_width'])}, "
        f"NaN predictions={result['nan_prediction_points']}"
    )


def format_summary_console_block(summary: dict) -> str:
    return "\n".join(
        [
            "Validation final summary:",
            f"Average MAE: {format_metric(summary['average_mae'])}",
            f"Average RMSE: {format_metric(summary['average_rmse'])}",
            f"Average Coverage (%): {format_metric(summary['average_coverage_percent'])}",
            f"Overall Coverage (%): {format_metric(summary['overall_coverage_percent'])}",
            f"Average Uncertainty Width: {format_metric(summary['average_uncertainty_width'])}",
            f"Total Evaluated Points: {summary['total_evaluated_points']}",
            f"Total Filled by Interpolation: {summary['total_filled_by_interpolation']}",
            f"Total Filled by Monte Carlo: {summary['total_filled_by_monte_carlo']}",
        ]
    )


def format_metric(value: float | None) -> str:
    return "NA" if value is None else f"{float(value):.6f}"


def format_float(value: float) -> str:
    return f"{float(value):.10g}"


def format_optional_float(value: float | None) -> str:
    return "" if value is None else format_float(value)

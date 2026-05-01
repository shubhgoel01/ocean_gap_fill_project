"""Plotting helpers for summaries and diagnostics."""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from scipy import stats

from .distribution_fit import extract_cell_time_series_samples

try:
    import cartopy.crs as ccrs
except ImportError:  # pragma: no cover - optional dependency
    ccrs = None

# This file generates visual diagnostics for each cell to compare real observed data with fitted statistical models and to summarize key results of the pipeline.
# In this we generate 
#   1. Histogram using real chlorophyll values 
#   2. Curve of fitted model using its parameters 
# And then we can compare both visulaly and validate if the estimated fitted model correctly defines/represents the data or not.
def plot_selected_cell_distributions(
    data_array,
    fitted_results: list[dict],
    config,
) -> list[str]:
    
    # From config file, check if to generate plots or not, if not then exit early.
    if not config.save_plots:
        return []

    # Now create the folder.
    output_dir = Path(config.plots_dir) / "distributions"
    output_dir.mkdir(parents=True, exist_ok=True)   

    saved_paths: list[str] = []
    for fit_result in fitted_results:
        output_path = create_distribution_plot(data_array, fit_result, output_dir)
        if output_path is not None:
            saved_paths.append(str(output_path))

    return saved_paths


def create_distribution_plot(data_array, fit_result: dict, output_dir: Path) -> Path | None:
    """Create a histogram and fitted density plot for one selected cell."""
    cell = fit_result["cell"]
    sample_values = extract_cell_time_series_samples(data_array, cell)
    sample_values = np.asarray(sample_values, dtype=float)

    if sample_values.size == 0:
        return None

    x_values = build_density_x_values(sample_values)
    y_values = evaluate_density_curve(sample_values, fit_result, x_values)
    if y_values is None:
        return None

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(sample_values, bins="auto", density=True)
    ax.plot(x_values, y_values)
    ax.set_xlabel("Chlorophyll")
    ax.set_ylabel("Density")
    ax.set_title(build_distribution_plot_title(fit_result))
    fig.tight_layout()

    output_path = output_dir / build_distribution_plot_filename(cell)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def build_density_x_values(sample_values: np.ndarray) -> np.ndarray:
    """Build a plotting grid over the observed sample range."""
    min_value = float(np.min(sample_values))
    max_value = float(np.max(sample_values))

    if np.isclose(min_value, max_value):
        padding = max(abs(min_value) * 0.1, 1e-6)
        min_value -= padding
        max_value += padding

    return np.linspace(min_value, max_value, 200)


def evaluate_density_curve(
    sample_values: np.ndarray,
    fit_result: dict,
    x_values: np.ndarray,
) -> np.ndarray | None:
    """Evaluate the selected fitted density on the plotting grid."""
    chosen_model = fit_result.get("chosen_model")
    candidate_stats = fit_result.get("candidate_model_statistics", {})

    if chosen_model == "normal":
        parameters = candidate_stats["normal"]["parameters"]
        return stats.norm.pdf(x_values, loc=parameters["loc"], scale=parameters["scale"])

    if chosen_model == "lognormal":
        parameters = candidate_stats["lognormal"]["parameters"]
        return stats.lognorm.pdf(
            x_values,
            s=parameters["shape"],
            loc=parameters["loc"],
            scale=parameters["scale"],
        )

    if chosen_model == "gamma":
        parameters = candidate_stats["gamma"]["parameters"]
        return stats.gamma.pdf(
            x_values,
            a=parameters["shape"],
            loc=parameters["loc"],
            scale=parameters["scale"],
        )

    if chosen_model == "kde":
        kde_status = candidate_stats.get("kde", {})
        if kde_status.get("status") != "ok":
            return None
        kde = stats.gaussian_kde(sample_values)
        return kde.evaluate(x_values)

    return None


def build_distribution_plot_title(fit_result: dict) -> str:
    """Build the requested title for one selected cell plot."""
    cell = fit_result["cell"]
    return (
        f"Time: {cell['time_value']} | "
        f"Lat: {cell['lat_value']}, Lon: {cell['lon_value']} | "
        f"Model: {fit_result['chosen_model']} | "
        f"n={fit_result['sample_size']}"
    )


def build_distribution_plot_filename(cell: dict) -> str:
    """Build a stable filename for one selected cell plot."""
    return (
        f"distribution_t{cell['time_index']}_"
        f"y{cell['lat_index']}_x{cell['lon_index']}.png"
    )


def create_quicklook_plots(
    *,
    fitted_results: list[dict],
    selected_mc_summaries: list[dict],
    uncertainty_stats: dict,
    phase_nan_stats: list[dict],
    fit_summary: dict,
    raw_data,
    interpolated_data,
    final_data,
    config,
) -> list[str]:
    """Create simple readable plots from full-dataset results."""
    if not config.save_plots:
        return []

    saved_paths: list[str] = []
    saved_paths.extend(plot_selected_cell_monte_carlo_summaries(selected_mc_summaries, config))
    saved_paths.extend(
        plot_selected_cell_uncertainty_summaries(
            uncertainty_stats.get("selected_cell_summary", []),
            config,
        )
    )

    progression_path = plot_nan_progression_across_phases(phase_nan_stats, config)
    if progression_path is not None:
        saved_paths.append(str(progression_path))

    model_count_path = plot_model_type_counts(fit_summary, config)
    if model_count_path is not None:
        saved_paths.append(str(model_count_path))

    saved_paths.extend(
        plot_availability_maps(
            raw_data=raw_data,
            interpolated_data=interpolated_data,
            final_data=final_data,
            config=config,
        )
    )

    return saved_paths


def compute_valid_fraction(data: xr.DataArray) -> xr.DataArray:
    """Return the fraction of valid observations per pixel across time."""
    if "time" not in data.dims:
        raise ValueError("Availability maps require a 'time' dimension.")

    valid_fraction = data.notnull().mean(dim="time")
    valid_fraction.name = "valid_fraction"
    return valid_fraction


def plot_availability_map(data, title: str, output_path: Path) -> Path:
    """Plot a 2D availability-like map with optional cartopy coastlines."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    map_data = compute_valid_fraction(data) if "time" in data.dims else data
    map_data = map_data.astype(float)

    figure_kwargs = {"figsize": (9, 5)}
    subplot_kwargs = {}
    if ccrs is not None:
        subplot_kwargs["projection"] = ccrs.PlateCarree()

    fig, ax = plt.subplots(
        subplot_kw=subplot_kwargs or None,
        **figure_kwargs,
    )

    plot_kwargs = {
        "ax": ax,
        "x": "lon",
        "y": "lat",
        "cmap": "viridis",
        "vmin": 0.0,
        "vmax": 1.0,
        "add_colorbar": True,
        "cbar_kwargs": {"label": "Valid Data Fraction"},
    }
    if ccrs is not None:
        plot_kwargs["transform"] = ccrs.PlateCarree()

    map_data.plot(**plot_kwargs)

    if ccrs is not None:
        ax.coastlines(linewidth=0.7)

    ax.set_title(title)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def plot_availability_maps(
    *,
    raw_data: xr.DataArray,
    interpolated_data: xr.DataArray,
    final_data: xr.DataArray,
    config,
) -> list[str]:
    """Create spatial availability maps for the major pipeline stages."""
    output_dir = Path(config.plots_dir) / "availability_maps"
    output_dir.mkdir(parents=True, exist_ok=True)

    raw_valid_fraction = compute_valid_fraction(raw_data)
    final_valid_fraction = compute_valid_fraction(final_data)
    improvement_map = (final_valid_fraction - raw_valid_fraction).clip(min=0.0, max=1.0)
    improvement_map.name = "availability_improvement"

    plot_specs = [
        (raw_data, "Raw Data Availability", output_dir / "raw_data_availability.png"),
        (interpolated_data, "After Interpolation", output_dir / "after_interpolation.png"),
        (final_data, "After Reconstruction", output_dir / "after_reconstruction.png"),
        (improvement_map, "Improvement Map", output_dir / "improvement_map.png"),
    ]

    saved_paths: list[str] = []
    for data, title, output_path in plot_specs:
        saved_paths.append(str(plot_availability_map(data, title, output_path)))

    return saved_paths


def plot_selected_cell_monte_carlo_summaries(
    selected_mc_summaries: list[dict],
    config,
) -> list[str]:
    """Plot Monte Carlo histograms with mean/median/percentile lines."""
    output_dir = Path(config.plots_dir) / "monte_carlo"
    output_dir.mkdir(parents=True, exist_ok=True)

    saved_paths: list[str] = []
    for summary in selected_mc_summaries:
        samples = np.asarray(summary.get("sampled_values", []), dtype=float)
        if samples.size == 0:
            continue

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.hist(samples, bins="auto")

        if summary.get("sample_mean") is not None:
            ax.axvline(summary["sample_mean"], linestyle="-", label="Mean")
        if summary["percentiles"].get("p50") is not None:
            ax.axvline(summary["percentiles"]["p50"], linestyle="--", label="Median")
        if summary["percentiles"].get("p05") is not None:
            ax.axvline(summary["percentiles"]["p05"], linestyle=":", label="P05")
        if summary["percentiles"].get("p95") is not None:
            ax.axvline(summary["percentiles"]["p95"], linestyle=":", label="P95")

        cell = summary["cell"]
        ax.set_title(
            f"Monte Carlo: {cell['time_value']} | "
            f"Lat {cell['lat_value']}, Lon {cell['lon_value']}"
        )
        ax.set_xlabel("Sampled chlorophyll")
        ax.set_ylabel("Count")
        ax.legend()
        fig.tight_layout()

        output_path = output_dir / (
            f"mc_t{cell['time_index']}_y{cell['lat_index']}_x{cell['lon_index']}.png"
        )
        fig.savefig(output_path, dpi=150)
        plt.close(fig)
        saved_paths.append(str(output_path))

    return saved_paths


def plot_selected_cell_uncertainty_summaries(
    selected_uncertainty: list[dict],
    config,
) -> list[str]:
    """Plot error-bar style uncertainty summaries for selected cells."""
    output_dir = Path(config.plots_dir) / "uncertainty"
    output_dir.mkdir(parents=True, exist_ok=True)

    saved_paths: list[str] = []
    for item in selected_uncertainty:
        mean_value = item.get("mean")
        p05 = item.get("p05")
        p95 = item.get("p95")
        if mean_value is None or p05 is None or p95 is None:
            continue

        fig, ax = plt.subplots(figsize=(7, 3))
        lower = mean_value - p05
        upper = p95 - mean_value
        ax.errorbar([0], [mean_value], yerr=[[lower], [upper]], fmt="o", capsize=5)
        ax.set_xlim(-1, 1)
        ax.set_xticks([0])
        ax.set_xticklabels([item["provenance_status"]])
        ax.set_ylabel("Chlorophyll")
        ax.set_title(
            f"Uncertainty: {item['time_value']} | "
            f"Lat {item['lat_value']}, Lon {item['lon_value']}"
        )
        fig.tight_layout()

        output_path = output_dir / (
            f"uncertainty_t{item['time_index']}_y{item['lat_index']}_x{item['lon_index']}.png"
        )
        fig.savefig(output_path, dpi=150)
        plt.close(fig)
        saved_paths.append(str(output_path))

    return saved_paths


def plot_nan_progression_across_phases(
    phase_nan_stats: list[dict],
    config,
) -> Path | None:
    """Plot NaN percentage progression across major processing phases."""
    if not phase_nan_stats:
        return None

    output_dir = Path(config.plots_dir) / "summaries"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "nan_progression_across_phases.png"

    labels = [item["label"] for item in phase_nan_stats]
    nan_values = [item["nan_percent"] for item in phase_nan_stats]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(labels, nan_values, marker="o")
    ax.set_ylabel("NaN Percentage (%)")
    ax.set_title("NaN Percentage Progression Across Phases")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def plot_model_type_counts(fit_summary: dict, config) -> Path | None:
    """Plot model-type counts across all modeled missing cells."""
    model_counts = fit_summary.get("model_counts", {})
    if not model_counts:
        return None

    output_dir = Path(config.plots_dir) / "summaries"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "model_type_counts.png"

    labels = list(model_counts.keys())
    values = [model_counts[label] for label in labels]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(labels, values)
    ax.set_ylabel("Cell Count")
    ax.set_title("Model Types Used Across Missing Cells")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path

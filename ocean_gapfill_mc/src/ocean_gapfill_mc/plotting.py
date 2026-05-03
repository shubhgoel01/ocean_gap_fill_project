"""Plot generation for the ocean gap-filling pipeline."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib import ticker
import numpy as np
from scipy import stats
import xarray as xr

from .distribution_fit import (
    extract_cell_time_series_samples,
    make_cell_key,
    make_gaussian_kde_bw_method,
)

try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
except ImportError:  # pragma: no cover - cartopy is optional
    ccrs = None
    cfeature = None


MAP_COLORMAP = "jet"
BAR_COLORMAP = "turbo"
CHLOROPHYLL_LABEL = "Chlorophyll-a concentration (mg m$^{-3}$)"


def generate_pipeline_plots(
    raw_data: xr.DataArray,
    regridded_data: xr.DataArray,
    interpolated_data: xr.DataArray,
    reconstructed_datasets: list[xr.DataArray],
    nan_stage_summaries: list[dict],
    interpolation_summary: dict,
    fit_summary: dict,
    config,
) -> dict[str, str]:
    """Create all standard pipeline plots and return their output paths."""
    output_dir = Path(config.plots_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    final_mean_reconstruction = build_reconstruction_mean(reconstructed_datasets)
    shared_chlorophyll_vmax = compute_shared_chlorophyll_vmax(
        raw_data,
        final_mean_reconstruction,
    )
    saved_paths = {
        "missing_percentage_raw": str(
            plot_missing_percentage_map(
                raw_data,
                "Raw Data Missing Percentage",
                output_dir / "missing_percentage_raw_data.png",
            )
        ),
        "missing_percentage_after_interpolation": str(
            plot_missing_percentage_map(
                interpolated_data,
                "After Interpolation Missing Percentage",
                output_dir / "missing_percentage_after_interpolation.png",
            )
        ),
        "missing_percentage_final_mean_reconstruction": str(
            plot_missing_percentage_map(
                final_mean_reconstruction,
                "Final Mean Reconstruction Missing Percentage",
                output_dir / "missing_percentage_final_mean_reconstruction.png",
            )
        ),
        "missing_cells_by_stage": str(
            plot_missing_cells_by_stage(
                nan_stage_summaries,
                output_dir / "missing_cells_by_stage.png",
            )
        ),
        "interpolation_contribution": str(
            plot_interpolation_contribution(
                interpolation_summary,
                output_dir / "interpolation_contribution.png",
            )
        ),
        "probability_model_counts": str(
            plot_probability_model_counts(
                fit_summary,
                output_dir / "probability_model_counts.png",
            )
        ),
        "raw_chlorophyll_mean": str(
            plot_chlorophyll_mean_map(
                raw_data,
                "Satellite-Derived Chlorophyll",
                output_dir / "satellite_derived_raw_mean_chlorophyll.png",
                vmax=shared_chlorophyll_vmax,
            )
        ),
        "final_reconstructed_mean_chlorophyll": str(
            plot_chlorophyll_mean_map(
                final_mean_reconstruction,
                "Reconstructed Chlorophyll",
                output_dir / "final_reconstructed_mean_chlorophyll.png",
                vmax=shared_chlorophyll_vmax,
            )
        ),
    }

    # Touch the regridded argument deliberately: the line chart uses its summary,
    # while keeping the data available here makes this function's contract explicit.
    _ = regridded_data
    return saved_paths


def save_pipeline_chlorophyll_datasets(
    raw_data: xr.DataArray,
    reconstructed_datasets: list[xr.DataArray],
    config,
) -> dict[str, str]:
    """Save NetCDF datasets for the chlorophyll maps and reconstruction mean."""
    output_dir = Path(config.datasets_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    final_reconstruction_ensemble_mean = build_reconstruction_mean(reconstructed_datasets)
    satellite_raw_mean = build_time_mean_chlorophyll(
        raw_data,
        long_name="Satellite-derived mean chlorophyll concentration",
    )
    final_reconstructed_mean = build_time_mean_chlorophyll(
        final_reconstruction_ensemble_mean,
        long_name="Final reconstructed mean chlorophyll concentration",
    )
    final_reconstruction_ensemble_mean = prepare_chlorophyll_for_netcdf(
        final_reconstruction_ensemble_mean,
        long_name="Final reconstructed ensemble-mean chlorophyll concentration by time",
    )

    saved_paths = {
        "satellite_derived_raw_mean_chlorophyll": str(
            save_data_array_as_dataset(
                satellite_raw_mean,
                output_dir / "satellite_derived_raw_mean_chlorophyll.nc",
            )
        ),
        "final_reconstructed_mean_chlorophyll": str(
            save_data_array_as_dataset(
                final_reconstructed_mean,
                output_dir / "final_reconstructed_mean_chlorophyll.nc",
            )
        ),
        "final_reconstructed_ensemble_mean_chlorophyll_by_time": str(
            save_data_array_as_dataset(
                final_reconstruction_ensemble_mean,
                output_dir / "final_reconstructed_ensemble_mean_chlorophyll_by_time.nc",
            )
        ),
    }
    return saved_paths


def build_reconstruction_mean(reconstructed_datasets: list[xr.DataArray]) -> xr.DataArray:
    """Average Monte Carlo reconstructions into one mean reconstructed field."""
    if not reconstructed_datasets:
        raise ValueError("At least one reconstructed dataset is required for plotting.")

    stacked = xr.concat(reconstructed_datasets, dim="simulation")
    mean_reconstruction = stacked.mean(dim="simulation", skipna=True)
    mean_reconstruction.name = reconstructed_datasets[0].name
    return mean_reconstruction


def build_time_mean_chlorophyll(data_array: xr.DataArray, long_name: str) -> xr.DataArray:
    """Build the exact 2D time-mean chlorophyll field used by map plots."""
    mean_chlorophyll = data_array.mean(dim="time", skipna=True)
    return prepare_chlorophyll_for_netcdf(mean_chlorophyll, long_name)


def prepare_chlorophyll_for_netcdf(data_array: xr.DataArray, long_name: str) -> xr.DataArray:
    """Attach Panoply-friendly names and metadata to a chlorophyll DataArray."""
    prepared = data_array.copy()
    prepared.name = data_array.name or "chlor_a"
    prepared.attrs = dict(prepared.attrs)
    prepared.attrs["long_name"] = long_name
    prepared.attrs["units"] = "mg m-3"
    prepared.attrs["standard_name"] = "mass_concentration_of_chlorophyll_a_in_sea_water"
    return prepared


def save_data_array_as_dataset(data_array: xr.DataArray, output_path: Path) -> Path:
    """Save one DataArray to a NetCDF file as a named Dataset variable."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    data_array.to_dataset(name=data_array.name or "chlor_a").to_netcdf(output_path)
    return output_path


def generate_selected_cell_fit_plots(
    data_array: xr.DataArray,
    selected_fit_results: list[dict],
    config,
) -> dict[str, list[str]]:
    """Create fit-diagnostic plots for selected Monte Carlo-filled cells."""
    saved_paths: dict[str, list[str]] = {
        "histogram_pdf": [],
        "qq": [],
        "cdf": [],
        "metadata": [],
    }
    for plot_index, fit_result in enumerate(selected_fit_results, start=1):
        model_name = str(fit_result.get("chosen_model", "unresolved")).lower()
        if model_name == "unresolved":
            continue

        sample_values = extract_plot_sample_values(data_array, fit_result)
        if sample_values.size < 2:
            continue

        cell_dir = build_selected_cell_plot_dir(config, plot_index)
        cell_dir.mkdir(parents=True, exist_ok=True)
        saved_paths["histogram_pdf"].append(
            str(
                plot_selected_cell_histogram_pdf(
                    sample_values,
                    fit_result,
                    cell_dir / "histogram_best_fit_pdf.png",
                )
            )
        )
        qq_path = plot_selected_cell_qq(
            sample_values,
            fit_result,
            cell_dir / "qq_plot.png",
        )
        if qq_path is not None:
            saved_paths["qq"].append(str(qq_path))
        saved_paths["cdf"].append(
            str(
                plot_selected_cell_cdf_comparison(
                    sample_values,
                    fit_result,
                    cell_dir / "cdf_comparison.png",
                )
            )
        )
        saved_paths["metadata"].append(
            str(write_selected_cell_metadata(cell_dir, fit_result, sample_values))
        )

    return saved_paths


def generate_selected_cell_uncertainty_plot(
    reconstructed_datasets: list[xr.DataArray],
    selected_fit_results: list[dict],
    config,
) -> dict[str, str]:
    """Plot p05-p95 uncertainty intervals and ensemble mean for each selected cell."""
    if not reconstructed_datasets or not selected_fit_results:
        return {}

    records = build_selected_cell_uncertainty_records(reconstructed_datasets, selected_fit_results)
    if not records:
        return {}

    saved_paths: dict[str, str] = {}
    for plot_index, record in enumerate(records, start=1):
        cell_dir = build_selected_cell_plot_dir(config, plot_index)
        cell_dir.mkdir(parents=True, exist_ok=True)
        output_path = plot_single_selected_cell_uncertainty_interval(
            record,
            cell_dir / "uncertainty_interval.png",
        )
        append_uncertainty_metadata(cell_dir, record)
        saved_paths[f"cell_{plot_index}"] = str(output_path)
    return saved_paths


def build_selected_cell_uncertainty_records(
    reconstructed_datasets: list[xr.DataArray],
    selected_fit_results: list[dict],
) -> list[dict]:
    """Compute selected-cell uncertainty records from the reconstructed ensemble."""
    records = []
    for index, fit_result in enumerate(selected_fit_results, start=1):
        cell = fit_result["cell"]
        time_index = int(cell["time_index"])
        lat_index = int(cell["lat_index"])
        lon_index = int(cell["lon_index"])
        values = np.asarray(
            [
                dataset.values[time_index, lat_index, lon_index]
                for dataset in reconstructed_datasets
            ],
            dtype=float,
        )
        finite_values = values[np.isfinite(values)]
        if finite_values.size == 0:
            continue

        records.append(
            {
                "label": (
                    f"{index}: lat {float(cell['lat_value']):.2f}, "
                    f"lon {float(cell['lon_value']):.2f}"
                ),
                "cell": cell,
                "chosen_model": fit_result.get("chosen_model"),
                "simulation_count": int(len(finite_values)),
                "mean": float(np.mean(finite_values)),
                "p05": float(np.percentile(finite_values, 5)),
                "p95": float(np.percentile(finite_values, 95)),
            }
        )
    return records


def plot_single_selected_cell_uncertainty_interval(record: dict, output_path: Path) -> Path:
    """Plot one selected cell's ensemble mean and p05-p95 interval."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    mean_value = float(record["mean"])
    p05_value = float(record["p05"])
    p95_value = float(record["p95"])

    fig, ax = plt.subplots(figsize=(5.6, 5), constrained_layout=True)
    ax.vlines(
        0.0,
        p05_value,
        p95_value,
        color=plt.get_cmap(BAR_COLORMAP)(0.24),
        linewidth=7.0,
        alpha=0.78,
        label="p05-p95 interval",
    )
    ax.scatter([0.0], [mean_value], color="#111111", s=68, zorder=3, label="Ensemble mean")
    ax.scatter([0.0], [p05_value], marker="_", s=220, color="#2b6cb0", zorder=4, label="p05")
    ax.scatter([0.0], [p95_value], marker="_", s=220, color="#c2410c", zorder=4, label="p95")

    padding = max((p95_value - p05_value) * 0.22, abs(mean_value) * 0.04, 0.01)
    ax.set_xlim(-0.55, 0.55)
    ax.set_ylim(max(0.0, p05_value - padding), p95_value + padding)
    ax.set_xticks([0.0])
    ax.set_xticklabels(["Selected cell"])
    ax.set_title(f"Uncertainty Interval\n{format_cell_location(record['cell'])}")
    ax.set_ylabel(CHLOROPHYLL_LABEL)
    ax.legend()
    ax.grid(True, axis="y", linewidth=0.4, alpha=0.35)

    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return output_path


def write_selected_cell_metadata(
    cell_dir: Path,
    fit_result: dict,
    sample_values: np.ndarray,
) -> Path:
    """Write selected-cell model metadata as plain text."""
    metadata_path = cell_dir / "metadata.txt"
    cell = fit_result["cell"]
    model_name = str(fit_result.get("chosen_model", "unknown")).lower()
    model_stats = fit_result.get("candidate_model_statistics", {}).get(model_name, {})
    parameters = model_stats.get("parameters")
    lines = [
        "Selected Cell Metadata",
        "======================",
        f"Cell key: {make_cell_key(cell)}",
        f"Time index: {cell['time_index']}",
        f"Time value: {cell['time_value']}",
        f"Latitude index: {cell['lat_index']}",
        f"Latitude: {float(cell['lat_value']):.6f} degrees_north",
        f"Longitude index: {cell['lon_index']}",
        f"Longitude: {float(cell['lon_value']):.6f} degrees_east",
        f"Selected model: {format_model_name(model_name)}",
        f"Fitting log-space sample count: {int(sample_values.size)}",
        "Fitting space: natural log of chlorophyll-a concentration",
        "Histogram/CDF plot unit: chlorophyll-a concentration, mg m^-3",
        "Q-Q plot unit: natural log of chlorophyll-a concentration",
    ]
    if fit_result.get("chosen_p_value") is not None:
        lines.append(f"Chosen-model KS p-value: {float(fit_result['chosen_p_value']):.6g}")
    if model_stats.get("ks_statistic") is not None:
        lines.append(f"Chosen-model KS statistic: {float(model_stats['ks_statistic']):.6g}")
    if parameters:
        lines.append("Fitted parameters in log-space:")
        for name, value in parameters.items():
            lines.append(f"  {name}: {float(value):.10g}")

    metadata_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return metadata_path


def append_uncertainty_metadata(cell_dir: Path, record: dict) -> Path:
    """Append selected-cell uncertainty metadata as plain text."""
    metadata_path = cell_dir / "metadata.txt"
    lines = [
        "",
        "Monte Carlo Uncertainty",
        "=======================",
        f"Simulation count with finite values: {record['simulation_count']}",
        f"Ensemble mean: {record['mean']:.10g} mg m^-3",
        f"5th percentile: {record['p05']:.10g} mg m^-3",
        f"95th percentile: {record['p95']:.10g} mg m^-3",
        f"Uncertainty interval: [{record['p05']:.10g}, {record['p95']:.10g}] mg m^-3",
    ]
    with metadata_path.open("a", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")
    return metadata_path


def extract_plot_sample_values(data_array: xr.DataArray, fit_result: dict) -> np.ndarray:
    """Use the same log-space support values as distribution fitting."""
    sample_values = extract_cell_time_series_samples(data_array, fit_result["cell"])
    sample_values = np.asarray(sample_values, dtype=float)
    return sample_values[np.isfinite(sample_values)]


def plot_selected_cell_histogram_pdf(
    log_sample_values: np.ndarray,
    fit_result: dict,
    output_path: Path,
) -> Path:
    """Plot sample histogram with only the selected best-fit PDF overlaid."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    model_name = str(fit_result.get("chosen_model", "unknown")).lower()
    chl_sample_values = np.exp(log_sample_values)
    x_values = build_pdf_x_values(chl_sample_values)
    pdf_values = evaluate_selected_model_pdf(
        x_values,
        log_sample_values,
        fit_result,
    )

    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    ax.hist(
        chl_sample_values,
        bins=choose_histogram_bins(chl_sample_values),
        density=True,
        alpha=0.68,
        color=plt.get_cmap(BAR_COLORMAP)(0.22),
        edgecolor="white",
        linewidth=0.7,
        label="Observed samples",
    )
    ax.plot(
        x_values,
        pdf_values,
        color="#111111",
        linewidth=2.2,
        label=f"{format_model_name(model_name)} PDF",
    )
    ax.set_title(f"Histogram + Best-Fit PDF\n{format_cell_location(fit_result['cell'])} | {format_model_name(model_name)}")
    ax.set_xlabel(CHLOROPHYLL_LABEL)
    ax.set_ylabel("Density")
    ax.legend()
    ax.grid(True, axis="y", linewidth=0.4, alpha=0.35)

    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return output_path


def plot_selected_cell_qq(
    log_sample_values: np.ndarray,
    fit_result: dict,
    output_path: Path,
) -> Path | None:
    """Plot Q-Q diagnostics in log-space for supported parametric models."""
    model_name = str(fit_result.get("chosen_model", "")).lower()
    probplot_dist, sparams = get_probplot_distribution(model_name, fit_result)
    if probplot_dist is None:
        return None

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 6), constrained_layout=True)
    stats.probplot(
        log_sample_values,
        dist=probplot_dist,
        sparams=sparams,
        plot=ax,
    )
    ax.get_lines()[0].set_markerfacecolor(plt.get_cmap(BAR_COLORMAP)(0.28))
    ax.get_lines()[0].set_markeredgecolor("#333333")
    ax.get_lines()[1].set_color("#111111")
    ax.get_lines()[1].set_linewidth(1.8)
    ax.set_title(f"Q-Q Plot (Log-Space)\n{format_cell_location(fit_result['cell'])} | {format_model_name(model_name)}")
    ax.set_xlabel("Theoretical log-space quantiles")
    ax.set_ylabel("Ordered log(chlorophyll-a)")
    ax.grid(True, linewidth=0.4, alpha=0.35)

    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return output_path


def plot_selected_cell_cdf_comparison(
    log_sample_values: np.ndarray,
    fit_result: dict,
    output_path: Path,
) -> Path:
    """Plot empirical CDF against the selected theoretical CDF."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    model_name = str(fit_result.get("chosen_model", "unknown")).lower()
    sorted_values = np.sort(np.exp(log_sample_values))
    empirical_cdf = np.arange(1, sorted_values.size + 1, dtype=float) / sorted_values.size
    theoretical_cdf = evaluate_selected_model_cdf(
        sorted_values,
        log_sample_values,
        fit_result,
    )

    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    ax.step(
        sorted_values,
        empirical_cdf,
        where="post",
        linewidth=2.0,
        color=plt.get_cmap(BAR_COLORMAP)(0.22),
        label="Empirical CDF",
    )
    ax.plot(
        sorted_values,
        theoretical_cdf,
        linewidth=2.2,
        color="#111111",
        label=f"{format_model_name(model_name)} CDF",
    )
    ax.set_title(f"CDF Comparison\n{format_cell_location(fit_result['cell'])} | {format_model_name(model_name)}")
    ax.set_xlabel(CHLOROPHYLL_LABEL)
    ax.set_ylabel("Cumulative probability")
    ax.set_ylim(-0.03, 1.03)
    ax.legend()
    ax.grid(True, linewidth=0.4, alpha=0.35)

    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return output_path


def build_pdf_x_values(sample_values: np.ndarray) -> np.ndarray:
    finite_values = np.asarray(sample_values, dtype=float)
    finite_values = finite_values[np.isfinite(finite_values)]
    minimum = float(np.min(finite_values))
    maximum = float(np.max(finite_values))
    if np.isclose(minimum, maximum):
        padding = max(abs(minimum) * 0.1, 0.01)
    else:
        padding = (maximum - minimum) * 0.1
    lower = max(1e-12, minimum - padding)
    upper = maximum + padding
    return np.linspace(lower, upper, 250)


def evaluate_selected_model_pdf(
    x_values: np.ndarray,
    log_sample_values: np.ndarray,
    fit_result: dict,
) -> np.ndarray:
    x_values = np.asarray(x_values, dtype=float)
    positive_x = np.clip(x_values, 1e-12, None)
    log_x_values = np.log(positive_x)
    model_name = str(fit_result.get("chosen_model", "")).lower()
    if model_name == "normal":
        params = get_model_parameters(fit_result, "normal")
        return stats.norm.pdf(
            log_x_values,
            loc=params["loc"],
            scale=params["scale"],
        ) / positive_x
    if model_name == "lognormal":
        params = get_model_parameters(fit_result, "lognormal")
        return stats.lognorm.pdf(
            positive_x,
            s=params["shape"],
            loc=params["loc"],
            scale=params["scale"],
        )
    if model_name == "gamma":
        params = get_model_parameters(fit_result, "gamma")
        return stats.gamma.pdf(
            log_x_values,
            a=params["shape"],
            loc=params["loc"],
            scale=params["scale"],
        ) / positive_x
    if model_name == "kde":
        kde = build_selected_kde(log_sample_values, fit_result)
        return kde.evaluate(log_x_values) / positive_x
    return np.full_like(x_values, np.nan, dtype=float)


def evaluate_selected_model_cdf(
    x_values: np.ndarray,
    log_sample_values: np.ndarray,
    fit_result: dict,
) -> np.ndarray:
    x_values = np.asarray(x_values, dtype=float)
    positive_x = np.clip(x_values, 1e-12, None)
    log_x_values = np.log(positive_x)
    model_name = str(fit_result.get("chosen_model", "")).lower()
    if model_name == "normal":
        params = get_model_parameters(fit_result, "normal")
        return stats.norm.cdf(
            log_x_values,
            loc=params["loc"],
            scale=params["scale"],
        )
    if model_name == "lognormal":
        params = get_model_parameters(fit_result, "lognormal")
        return stats.lognorm.cdf(
            positive_x,
            s=params["shape"],
            loc=params["loc"],
            scale=params["scale"],
        )
    if model_name == "gamma":
        params = get_model_parameters(fit_result, "gamma")
        return stats.gamma.cdf(
            log_x_values,
            a=params["shape"],
            loc=params["loc"],
            scale=params["scale"],
        )
    if model_name == "kde":
        kde = build_selected_kde(log_sample_values, fit_result)
        return np.asarray([kde.integrate_box_1d(-np.inf, value) for value in log_x_values])
    return np.full_like(x_values, np.nan, dtype=float)


def get_probplot_distribution(model_name: str, fit_result: dict) -> tuple[str | None, tuple]:
    if model_name == "normal":
        params = get_model_parameters(fit_result, "normal")
        return "norm", (params["loc"], params["scale"])
    if model_name == "lognormal":
        params = get_model_parameters(fit_result, "lognormal")
        return "lognorm", (params["shape"], params["loc"], params["scale"])
    if model_name == "gamma":
        params = get_model_parameters(fit_result, "gamma")
        return "gamma", (params["shape"], params["loc"], params["scale"])
    return None, ()


def get_model_parameters(fit_result: dict, model_name: str) -> dict:
    return fit_result["candidate_model_statistics"][model_name]["parameters"]


def build_selected_kde(sample_values: np.ndarray, fit_result: dict):
    kde_status = fit_result.get("candidate_model_statistics", {}).get("kde", {})
    bandwidth = kde_status.get("bandwidth")
    if bandwidth is None:
        return stats.gaussian_kde(sample_values)
    return stats.gaussian_kde(
        sample_values,
        bw_method=make_gaussian_kde_bw_method(sample_values, float(bandwidth)),
    )


def choose_histogram_bins(sample_values: np.ndarray) -> str | int:
    if sample_values.size < 10:
        return max(3, int(sample_values.size))
    return "auto"


def build_selected_cell_plot_dir(config, plot_index: int) -> Path:
    return Path(config.sampled_cells_dir) / "plots" / f"cell_{plot_index}"


def format_cell_location(cell: dict) -> str:
    return (
        f"lat {float(cell['lat_value']):.3f}, "
        f"lon {float(cell['lon_value']):.3f}, "
        f"time {cell['time_value']}"
    )


def format_model_name(model_name: str) -> str:
    labels = {
        "normal": "Normal",
        "lognormal": "Lognormal",
        "gamma": "Gamma",
        "kde": "KDE",
    }
    return labels.get(model_name.lower(), model_name.title())


def plot_missing_percentage_map(
    data_array: xr.DataArray,
    title: str,
    output_path: Path,
) -> Path:
    missing_percent = data_array.isnull().mean(dim="time", skipna=False) * 100.0
    vmax = 25.0
    return plot_spatial_field(
        missing_percent,
        title,
        output_path,
        colorbar_label="Missing data (%)",
        cmap=MAP_COLORMAP,
        vmin=0.0,
        vmax=vmax,
        extend="max",
        colorbar_ticks=np.arange(0.0, vmax + 0.1, 5.0),
        over_color="black",
        metadata_text=build_map_metadata(
            data_array,
            "Each pixel is percent missing across all time steps.\nValues above 25% are shown in black.",
        ),
    )


def plot_chlorophyll_mean_map(
    data_array: xr.DataArray,
    title: str,
    output_path: Path,
    vmax: float | None = None,
) -> Path:
    mean_chlorophyll = data_array.mean(dim="time", skipna=True)
    if vmax is None:
        vmax = rounded_colorbar_max(mean_chlorophyll, fallback=1.0, step=0.05)
    return plot_spatial_field(
        mean_chlorophyll,
        title,
        output_path,
        colorbar_label=CHLOROPHYLL_LABEL,
        cmap=MAP_COLORMAP,
        vmin=0.0,
        vmax=vmax,
        extend="max",
        metadata_text=build_map_metadata(
            data_array,
            "Map shows time-mean chlorophyll concentration.",
        ),
    )


def plot_spatial_field(
    field: xr.DataArray,
    title: str,
    output_path: Path,
    colorbar_label: str,
    cmap: str,
    vmin: float | None = None,
    vmax: float | None = None,
    extend: str = "neither",
    colorbar_ticks: np.ndarray | None = None,
    over_color: str | None = None,
    metadata_text: str | None = None,
    use_cartopy: bool | None = None,
) -> Path:
    """Plot one lat-lon field using available xarray coordinates."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    values = np.asarray(field.values, dtype=float)
    lon = np.asarray(field["lon"].values, dtype=float)
    lat = np.asarray(field["lat"].values, dtype=float)

    cmap_object = plt.get_cmap(cmap).copy()
    cmap_object.set_bad("white")
    if over_color is not None:
        cmap_object.set_over(over_color)

    if use_cartopy is None:
        use_cartopy = ccrs is not None

    projection_kwargs = {}
    plot_kwargs = {}
    if use_cartopy and ccrs is not None:
        projection_kwargs["projection"] = ccrs.PlateCarree()
        plot_kwargs["transform"] = ccrs.PlateCarree()

    fig, ax = plt.subplots(
        figsize=(10, 6),
        constrained_layout=True,
        subplot_kw=projection_kwargs or None,
    )
    mesh = ax.pcolormesh(
        lon,
        lat,
        np.ma.masked_invalid(values),
        shading="auto",
        cmap=cmap_object,
        vmin=vmin,
        vmax=vmax,
        **plot_kwargs,
    )
    colorbar = fig.colorbar(
        mesh,
        ax=ax,
        orientation="horizontal",
        pad=0.08,
        extend=extend,
        ticks=colorbar_ticks,
    )
    colorbar.set_label(colorbar_label)

    add_map_context(ax)
    ax.set_title(title)
    ax.set_xlabel("Longitude ($^\\circ$E)")
    ax.set_ylabel("Latitude ($^\\circ$N)")
    set_map_extent(ax, lon, lat)
    format_map_ticks(ax)
    if metadata_text:
        add_metadata_box(ax, metadata_text)

    try:
        fig.savefig(output_path, dpi=160)
    except Exception:
        plt.close(fig)
        if use_cartopy:
            return plot_spatial_field(
                field,
                title,
                output_path,
                colorbar_label,
                cmap,
                vmin=vmin,
                vmax=vmax,
                extend=extend,
                colorbar_ticks=colorbar_ticks,
                over_color=over_color,
                metadata_text=metadata_text,
                use_cartopy=False,
            )
        raise
    plt.close(fig)
    return output_path


def plot_missing_cells_by_stage(stage_summaries: list[dict], output_path: Path) -> Path:
    """Line plot of missing-cell percentage across major pipeline stages."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    labels = [format_stage_label(item["label"]) for item in stage_summaries]
    values = [float(item["nan_percent"]) for item in stage_summaries]

    fig, ax = plt.subplots(figsize=(9, 5), constrained_layout=True)
    x_positions = np.arange(len(labels))
    ax.plot(x_positions, values, marker="o", linewidth=2.4, color=plt.get_cmap(BAR_COLORMAP)(0.18))

    for x_position, value in zip(x_positions, values):
        ax.annotate(
            f"{value:.2f}%",
            (x_position, value),
            textcoords="offset points",
            xytext=(0, 8),
            ha="center",
            fontsize=8,
        )

    ax.set_title("Missing Data Through Pipeline")
    ax.set_xlabel("Pipeline stage")
    ax.set_ylabel("Missing data (%)")
    ax.set_xticks(x_positions)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=100.0))
    ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=6))
    ax.tick_params(axis="both", which="major", length=5, width=0.8)
    ax.grid(True, axis="y", linewidth=0.4, alpha=0.4)
    add_metadata_box(
        ax,
        "Percentages are normalized by each stage's own total cell count.",
        location="upper right",
    )

    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return output_path


def plot_interpolation_contribution(summary: dict, output_path: Path) -> Path:
    """Bar chart showing how many cells each interpolation pass filled."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pass_summaries = summary.get("pass_summaries", [])
    labels = [format_axis_label(item.get("axis", "")) for item in pass_summaries]
    values = [int(item.get("filled_cells", 0)) for item in pass_summaries]

    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    colors = color_sequence(len(labels))
    bars = ax.bar(labels, values, color=colors)
    ax.bar_label(bars, labels=[f"{value:,}" for value in values], padding=3, fontsize=9)

    before = int(summary.get("nan_count_before", 0))
    after = int(summary.get("nan_count_after", 0))
    filled = int(summary.get("filled_cells", 0))
    status = "All missing cells filled" if after == 0 else f"{after:,} cells still missing"
    ax.text(
        0.01,
        0.98,
        f"Before: {before:,}\nFilled: {filled:,}\nAfter: {after:,}\n{status}",
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=9,
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "edgecolor": "#cccccc"},
    )

    ax.set_title("Interpolation Contribution By Pass")
    ax.set_xlabel("Interpolation pass")
    ax.set_ylabel("Cells filled")
    ax.yaxis.set_major_formatter(ticker.StrMethodFormatter("{x:,.0f}"))
    ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=6, integer=True))
    ax.tick_params(axis="both", which="major", length=5, width=0.8)
    ax.grid(True, axis="y", linewidth=0.4, alpha=0.4)

    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return output_path


def plot_probability_model_counts(summary: dict, output_path: Path) -> Path:
    """Bar chart of chosen probability models for remaining missing cells."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    counts = summary.get("model_counts", {})
    labels = ["normal", "gamma", "kde", "unresolved"]
    values = [int(counts.get(label, 0)) for label in labels]
    display_labels = ["Normal", "Gamma", "KDE", "Unresolved"]

    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    bars = ax.bar(display_labels, values, color=color_sequence(len(display_labels)))
    ax.bar_label(bars, labels=[f"{value:,}" for value in values], padding=3, fontsize=9)

    ax.set_title("Probability Model Counts")
    ax.set_xlabel("Chosen model")
    ax.set_ylabel("Number of cells")
    ax.yaxis.set_major_formatter(ticker.StrMethodFormatter("{x:,.0f}"))
    ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=6, integer=True))
    ax.tick_params(axis="both", which="major", length=5, width=0.8)
    ax.grid(True, axis="y", linewidth=0.4, alpha=0.4)
    total_cells = int(summary.get("total_remaining_missing_cells", sum(values)))
    modeled_cells = int(summary.get("successfully_modeled_cells", sum(values[:-1])))
    add_metadata_box(
        ax,
        f"Remaining missing cells fitted after interpolation.\nModeled: {modeled_cells:,} / {total_cells:,}",
        location="upper right",
    )

    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return output_path


def format_stage_label(label: str) -> str:
    label_map = {
        "raw load": "Raw data",
        "spatial regridding": "After regridding",
        "ordered interpolation": "After interpolation",
        "final reconstruction": "After Monte Carlo",
    }
    return label_map.get(label, label.replace("_", " ").title())


def format_axis_label(axis: str) -> str:
    axis_map = {
        "lon": "Longitude",
        "lat": "Latitude",
        "time": "Time",
    }
    return axis_map.get(axis, str(axis).title())


def color_sequence(count: int) -> list:
    cmap = plt.get_cmap(BAR_COLORMAP)
    if count <= 1:
        return [cmap(0.2)]
    return [cmap(position) for position in np.linspace(0.12, 0.88, count)]


def add_map_context(ax) -> None:
    if ccrs is None or cfeature is None or not hasattr(ax, "add_feature"):
        return
    ax.add_feature(cfeature.LAND, facecolor="#d9d9d9", edgecolor="black", linewidth=0.4, zorder=3)
    ax.coastlines(linewidth=0.5, color="black", zorder=4)


def set_map_extent(ax, lon: np.ndarray, lat: np.ndarray) -> None:
    extent = [
        float(np.nanmin(lon)),
        float(np.nanmax(lon)),
        float(np.nanmin(lat)),
        float(np.nanmax(lat)),
    ]
    if ccrs is not None and hasattr(ax, "set_extent"):
        ax.set_extent(extent, crs=ccrs.PlateCarree())
    else:
        ax.set_xlim(extent[0], extent[1])
        ax.set_ylim(extent[2], extent[3])


def format_map_ticks(ax) -> None:
    x_min, x_max = ax.get_xlim()
    y_min, y_max = ax.get_ylim()
    x_ticks = build_degree_ticks(x_min, x_max, spacing=10.0)
    y_ticks = build_degree_ticks(y_min, y_max, spacing=10.0)

    if ccrs is not None and hasattr(ax, "set_xticks"):
        try:
            ax.set_xticks(x_ticks, crs=ccrs.PlateCarree())
            ax.set_yticks(y_ticks, crs=ccrs.PlateCarree())
        except Exception:
            ax.set_xticks(x_ticks)
            ax.set_yticks(y_ticks)
    else:
        ax.set_xticks(x_ticks)
        ax.set_yticks(y_ticks)

    if ccrs is not None and hasattr(ax, "gridlines"):
        ax.gridlines(draw_labels=False, linewidth=0.35, color="white", alpha=0.65)
    else:
        ax.grid(True, linewidth=0.35, color="white", alpha=0.65)

    ax.xaxis.set_major_formatter(ticker.FuncFormatter(format_longitude_tick))
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(format_latitude_tick))
    ax.tick_params(axis="both", which="major", length=5, width=0.8, labelsize=9)


def format_longitude_tick(value: float, _position: int) -> str:
    suffix = "E" if value >= 0 else "W"
    return f"{abs(value):.0f}$^\\circ${suffix}"


def format_latitude_tick(value: float, _position: int) -> str:
    if np.isclose(value, 0.0):
        return "0$^\\circ$"
    suffix = "N" if value > 0 else "S"
    return f"{abs(value):.0f}$^\\circ${suffix}"


def build_degree_ticks(min_value: float, max_value: float, spacing: float) -> np.ndarray:
    lower = float(np.floor(min_value / spacing) * spacing)
    upper = float(np.ceil(max_value / spacing) * spacing)
    ticks = np.arange(lower, upper + spacing * 0.5, spacing)
    return ticks[(ticks >= min_value - 1e-9) & (ticks <= max_value + 1e-9)]


def add_metadata_box(ax, text: str, location: str = "lower left") -> None:
    anchors = {
        "lower left": (0.01, 0.02, "left", "bottom"),
        "upper left": (0.01, 0.98, "left", "top"),
        "upper right": (0.99, 0.98, "right", "top"),
    }
    x_pos, y_pos, horizontal_alignment, vertical_alignment = anchors.get(
        location,
        anchors["lower left"],
    )
    ax.text(
        x_pos,
        y_pos,
        text,
        transform=ax.transAxes,
        ha=horizontal_alignment,
        va=vertical_alignment,
        fontsize=8,
        bbox={
            "boxstyle": "round,pad=0.32",
            "facecolor": "white",
            "edgecolor": "#cccccc",
            "alpha": 0.88,
        },
        zorder=10,
    )


def build_map_metadata(data_array: xr.DataArray, description: str) -> str:
    pieces = [description]
    if "time" in data_array.dims and data_array.sizes.get("time", 0) > 0:
        time_values = np.asarray(data_array["time"].values)
        pieces.append(
            f"Period: {format_date_value(time_values.min())} to {format_date_value(time_values.max())}"
        )
        pieces.append(f"Time steps: {int(data_array.sizes['time'])}")
    pieces.append(
        f"Grid: {int(data_array.sizes['lat'])} lat x {int(data_array.sizes['lon'])} lon"
    )
    return "\n".join(pieces)


def format_date_value(value) -> str:
    return str(np.datetime_as_string(value, unit="D"))


def rounded_colorbar_max(
    field: xr.DataArray,
    fallback: float,
    step: float,
    upper_limit: float | None = None,
) -> float:
    values = np.asarray(field.values, dtype=float)
    finite_values = values[np.isfinite(values)]
    if finite_values.size == 0:
        return fallback

    high_value = float(np.nanpercentile(finite_values, 98.0))
    if not np.isfinite(high_value) or high_value <= 0:
        return fallback

    rounded = float(np.ceil(high_value / step) * step)
    if upper_limit is not None:
        rounded = min(rounded, upper_limit)
    return max(rounded, step)


def compute_shared_chlorophyll_vmax(
    raw_data: xr.DataArray,
    final_mean_reconstruction: xr.DataArray,
) -> float:
    """Use one chlorophyll color scale for raw and reconstructed map comparison."""
    raw_mean = raw_data.mean(dim="time", skipna=True)
    reconstructed_mean = final_mean_reconstruction.mean(dim="time", skipna=True)
    combined_values = np.concatenate(
        [
            np.asarray(raw_mean.values, dtype=float).ravel(),
            np.asarray(reconstructed_mean.values, dtype=float).ravel(),
        ]
    )
    finite_values = combined_values[np.isfinite(combined_values)]
    if finite_values.size == 0:
        return 1.0

    high_value = float(np.nanpercentile(finite_values, 98.0))
    if not np.isfinite(high_value) or high_value <= 0:
        return 1.0
    return max(float(np.ceil(high_value / 0.05) * 0.05), 0.05)

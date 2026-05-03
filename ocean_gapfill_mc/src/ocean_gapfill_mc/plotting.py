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

from .distribution_fit import extract_cell_time_series_samples, make_cell_key

try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
except ImportError:  # pragma: no cover - cartopy is optional
    ccrs = None
    cfeature = None


MAP_COLORMAP = "jet"
BAR_COLORMAP = "turbo"


def generate_pipeline_plots(
    raw_data: xr.DataArray,
    regridded_data: xr.DataArray,
    interpolated_data: xr.DataArray,
    reconstructed_datasets: list[xr.DataArray],
    nan_stage_summaries: list[dict],
    interpolation_summary: dict,
    fit_summary: dict,
    selected_fit_results: list[dict] | None,
    selected_uncertainty_summary: list[dict] | None,
    config,
) -> dict[str, str]:
    """Create all standard pipeline plots and return their output paths."""
    output_dir = Path(config.plots_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    final_mean_reconstruction = build_reconstruction_mean(reconstructed_datasets)
    shared_chlorophyll_vmax = compute_shared_chlorophyll_vmax(
        raw_data,
        final_mean_reconstruction,
        config,
    )
    saved_paths = {
        "missing_percentage_raw": str(
            plot_missing_percentage_map(
                raw_data,
                "Raw Data Missing Percentage",
                output_dir / "missing_percentage_raw_data.png",
                config,
            )
        ),
        "missing_percentage_after_interpolation": str(
            plot_missing_percentage_map(
                interpolated_data,
                "After Interpolation Missing Percentage",
                output_dir / "missing_percentage_after_interpolation.png",
                config,
            )
        ),
        "missing_percentage_final_mean_reconstruction": str(
            plot_missing_percentage_map(
                final_mean_reconstruction,
                "Final Mean Reconstruction Missing Percentage",
                output_dir / "missing_percentage_final_mean_reconstruction.png",
                config,
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
                config,
                vmax=shared_chlorophyll_vmax,
            )
        ),
        "final_reconstructed_mean_chlorophyll": str(
            plot_chlorophyll_mean_map(
                final_mean_reconstruction,
                "Reconstructed Chlorophyll",
                output_dir / "final_reconstructed_mean_chlorophyll.png",
                config,
                vmax=shared_chlorophyll_vmax,
            )
        ),
    }
    saved_paths.update(
        generate_selected_cell_fit_plots(
            interpolated_data,
            selected_fit_results or [],
            selected_uncertainty_summary or [],
            Path(config.sampled_cells_dir) / "plots",
        )
    )

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
    interpolated_data: xr.DataArray,
    selected_fit_results: list[dict],
    selected_uncertainty_summary: list[dict],
    output_dir: Path,
) -> dict[str, str]:
    """Generate distribution diagnostic plots for selected debug cells."""
    saved_paths: dict[str, str] = {}
    output_dir.mkdir(parents=True, exist_ok=True)
    uncertainty_by_cell = {
        make_cell_key(item): item
        for item in selected_uncertainty_summary
    }

    for position, fit_result in enumerate(selected_fit_results, start=1):
        chosen_model = str(fit_result.get("chosen_model") or "").lower()
        if chosen_model not in {"normal", "lognormal", "gamma", "kde"}:
            continue

        sample_values = extract_cell_time_series_samples(interpolated_data, fit_result["cell"])
        sample_values = np.asarray(sample_values, dtype=float)
        sample_values = sample_values[np.isfinite(sample_values)]
        if sample_values.size == 0:
            continue

        cell_dir = output_dir / f"cell_{position}"
        cell_dir.mkdir(parents=True, exist_ok=True)
        cell_label = format_cell_label(fit_result["cell"])
        uncertainty = uncertainty_by_cell.get(make_cell_key(fit_result["cell"]))

        histogram_path = plot_selected_cell_histogram_pdf(
            sample_values,
            fit_result,
            cell_dir / "histogram_best_fit_pdf.png",
            cell_label,
        )
        qq_path = plot_selected_cell_qq(
            sample_values,
            fit_result,
            cell_dir / "qq_plot.png",
            cell_label,
        )
        cdf_path = plot_selected_cell_cdf(
            sample_values,
            fit_result,
            cell_dir / "cdf_comparison.png",
            cell_label,
        )
        uncertainty_path = None
        if uncertainty is not None:
            uncertainty_path = plot_selected_cell_uncertainty(
                uncertainty,
                cell_dir / "uncertainty_interval.png",
            )
        metadata_path = write_selected_cell_metadata(
            fit_result,
            sample_values,
            uncertainty,
            cell_dir / "metadata.txt",
        )

        saved_paths[f"selected_cell_{position:02d}_histogram_pdf"] = str(histogram_path)
        if qq_path is not None:
            saved_paths[f"selected_cell_{position:02d}_qq"] = str(qq_path)
        saved_paths[f"selected_cell_{position:02d}_cdf"] = str(cdf_path)
        if uncertainty_path is not None:
            saved_paths[f"selected_cell_{position:02d}_uncertainty_interval"] = str(uncertainty_path)
        saved_paths[f"selected_cell_{position:02d}_metadata"] = str(metadata_path)

    return saved_paths


def plot_selected_cell_histogram_pdf(
    sample_values: np.ndarray,
    fit_result: dict,
    output_path: Path,
    cell_label: str,
) -> Path:
    """Plot sample histogram with only the selected best-fit PDF overlaid."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    chosen_model = str(fit_result.get("chosen_model")).lower()
    x_values = build_distribution_x_values(sample_values)
    pdf_values = evaluate_selected_model_pdf(x_values, sample_values, fit_result)

    fig, ax = plt.subplots(figsize=(7.5, 5), constrained_layout=True)
    ax.hist(sample_values, bins="auto", density=True, alpha=0.62, color="#4c78a8", edgecolor="white")
    ax.plot(x_values, pdf_values, color="#d62728", linewidth=2.2, label=f"{format_model_name(chosen_model)} PDF")
    ax.set_title(f"Histogram + Best-Fit PDF\n{cell_label} | {format_model_name(chosen_model)}")
    ax.set_xlabel("Chlorophyll-a concentration (mg m$^{-3}$)")
    ax.set_ylabel("Probability density ((mg m$^{-3}$)$^{-1}$)")
    ax.legend()
    ax.grid(True, axis="y", linewidth=0.4, alpha=0.35)

    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return output_path


def plot_selected_cell_qq(
    sample_values: np.ndarray,
    fit_result: dict,
    output_path: Path,
    cell_label: str,
) -> Path | None:
    """Plot selected-cell Q-Q diagnostics using scipy.stats.probplot."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    chosen_model = str(fit_result.get("chosen_model")).lower()
    distribution = build_probplot_distribution(sample_values, fit_result)
    if distribution is None:
        return None

    fig, ax = plt.subplots(figsize=(6, 5.5), constrained_layout=True)
    stats.probplot(sample_values, dist=distribution, plot=ax)
    ax.get_lines()[0].set_markerfacecolor("#4c78a8")
    ax.get_lines()[0].set_markeredgecolor("#4c78a8")
    ax.get_lines()[1].set_color("#d62728")
    ax.set_title(f"Q-Q Plot\n{cell_label} | {format_model_name(chosen_model)}")
    ax.set_xlabel("Theoretical quantiles")
    ax.set_ylabel("Ordered sample chlorophyll-a concentration (mg m$^{-3}$)")
    ax.grid(True, linewidth=0.4, alpha=0.35)

    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return output_path


def plot_selected_cell_cdf(
    sample_values: np.ndarray,
    fit_result: dict,
    output_path: Path,
    cell_label: str,
) -> Path:
    """Plot empirical CDF against the selected theoretical CDF."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    chosen_model = str(fit_result.get("chosen_model")).lower()
    sorted_values = np.sort(sample_values)
    empirical_cdf = np.arange(1, sorted_values.size + 1, dtype=float) / sorted_values.size
    theoretical_cdf = evaluate_selected_model_cdf(sorted_values, sample_values, fit_result)

    fig, ax = plt.subplots(figsize=(7.5, 5), constrained_layout=True)
    ax.step(sorted_values, empirical_cdf, where="post", linewidth=2.0, color="#4c78a8", label="Empirical CDF")
    ax.plot(sorted_values, theoretical_cdf, linewidth=2.0, color="#d62728", label=f"{format_model_name(chosen_model)} CDF")
    ax.set_title(f"CDF Comparison\n{cell_label} | {format_model_name(chosen_model)}")
    ax.set_xlabel("Chlorophyll-a concentration (mg m$^{-3}$)")
    ax.set_ylabel("Cumulative probability")
    ax.set_ylim(-0.03, 1.03)
    ax.legend()
    ax.grid(True, linewidth=0.4, alpha=0.35)

    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return output_path


def plot_selected_cell_uncertainty(
    selected_uncertainty: dict,
    output_path: Path,
) -> Path | None:
    """Plot mean value with p05-p95 uncertainty interval for one selected cell."""
    if not selected_uncertainty:
        return None

    output_path.parent.mkdir(parents=True, exist_ok=True)
    mean_value = float(selected_uncertainty["mean"])
    lower = float(selected_uncertainty["lower_percentile"])
    upper = float(selected_uncertainty["upper_percentile"])
    lower_percentile = selected_uncertainty.get("lower_percentile_value", 5)
    upper_percentile = selected_uncertainty.get("upper_percentile_value", 95)
    lower_label = format_percentile_label(lower_percentile)
    upper_label = format_percentile_label(upper_percentile)
    yerr = np.asarray([[mean_value - lower], [upper - mean_value]], dtype=float)
    label = format_cell_label(selected_uncertainty)

    fig, ax = plt.subplots(figsize=(6.8, 5.2), constrained_layout=True)
    ax.errorbar(
        [0],
        [mean_value],
        yerr=yerr,
        fmt="o",
        markersize=8,
        capsize=6,
        linewidth=1.8,
        color="#1f77b4",
        ecolor="#d62728",
        label="Mean with percentile interval",
    )
    ax.scatter([0], [lower], marker="_", s=220, color="#d62728", label=lower_label)
    ax.scatter([0], [upper], marker="_", s=220, color="#d62728", label=upper_label)
    ax.set_title(f"Uncertainty Interval ({lower_label}-{upper_label})\n{label}")
    ax.set_xlabel("Selected cell")
    ax.set_ylabel("Chlorophyll-a concentration (mg m$^{-3}$)")
    ax.set_xticks([0])
    ax.set_xticklabels(["Monte Carlo estimate"])
    ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=6))
    ax.grid(True, axis="y", linewidth=0.4, alpha=0.35)
    ax.legend()

    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return output_path


def write_selected_cell_metadata(
    fit_result: dict,
    sample_values: np.ndarray,
    uncertainty: dict | None,
    output_path: Path,
) -> Path:
    """Write readable metadata beside the selected-cell plots."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cell = fit_result["cell"]
    chosen_model = str(fit_result.get("chosen_model") or "unresolved").lower()
    lines = [
        "Selected Cell Plot Metadata",
        "===========================",
        f"Cell: {format_cell_label(cell)}",
        f"Indices: time={int(cell['time_index'])}, lat={int(cell['lat_index'])}, lon={int(cell['lon_index'])}",
        f"Chosen distribution: {format_model_name(chosen_model)}",
        "Units: chlorophyll-a concentration in mg m^-3",
        f"Finite fitting sample count: {int(sample_values.size)}",
        f"Finite fitting sample mean: {float(np.mean(sample_values)):.6g} mg m^-3",
        f"Finite fitting sample standard deviation: {float(np.std(sample_values, ddof=0)):.6g} mg m^-3",
        f"Finite fitting sample minimum: {float(np.min(sample_values)):.6g} mg m^-3",
        f"Finite fitting sample maximum: {float(np.max(sample_values)):.6g} mg m^-3",
        "",
        "Fitted model parameters:",
    ]

    model_stats = fit_result.get("candidate_model_statistics", {}).get(chosen_model, {})
    params = model_stats.get("parameters")
    if params:
        for name, value in params.items():
            lines.append(f"  {name}: {float(value):.6g}")
    elif chosen_model == "kde":
        bandwidth = model_stats.get("bandwidth")
        lines.append(f"  bandwidth: {float(bandwidth):.6g}" if bandwidth is not None else "  bandwidth: not available")
    else:
        lines.append("  not available")

    if uncertainty is not None:
        lower_percentile = float(uncertainty.get("lower_percentile_value", 5))
        upper_percentile = float(uncertainty.get("upper_percentile_value", 95))
        lower_label = format_percentile_label(lower_percentile)
        upper_label = format_percentile_label(upper_percentile)
        lines.extend(
            [
                "",
                "Monte Carlo uncertainty:",
                f"  mean: {float(uncertainty['mean']):.6g} mg m^-3",
                f"  {lower_label}: {float(uncertainty['lower_percentile']):.6g} mg m^-3",
                f"  {upper_label}: {float(uncertainty['upper_percentile']):.6g} mg m^-3",
                f"  standard deviation: {float(uncertainty['std']):.6g} mg m^-3",
                f"  provenance status: {uncertainty.get('provenance_status', 'unknown')}",
            ]
        )

    lines.extend(
        [
            "",
        "Generated plots:",
        "  histogram_best_fit_pdf.png: probability-density histogram with the selected best-fit PDF only",
        "    Note: density is not probability or percent. The area under the histogram is 1, so the y-axis can be larger for narrow value ranges and smaller for wide value ranges.",
        "  qq_plot.png: Q-Q plot using scipy.stats.probplot and the selected distribution",
            "  cdf_comparison.png: empirical CDF compared with the selected theoretical CDF",
            "  uncertainty_interval.png: mean with percentile uncertainty interval",
        ]
    )

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output_path


def build_distribution_x_values(sample_values: np.ndarray) -> np.ndarray:
    lower = float(np.nanmin(sample_values))
    upper = float(np.nanmax(sample_values))
    if np.isclose(lower, upper):
        padding = max(abs(lower) * 0.1, 1.0)
    else:
        padding = 0.08 * (upper - lower)
    return np.linspace(lower - padding, upper + padding, 300)


def evaluate_selected_model_pdf(
    x_values: np.ndarray,
    sample_values: np.ndarray,
    fit_result: dict,
) -> np.ndarray:
    chosen_model = str(fit_result.get("chosen_model")).lower()
    distribution = build_fitted_distribution(sample_values, fit_result)
    if chosen_model == "kde":
        return distribution.evaluate(x_values)
    return distribution.pdf(x_values)


def evaluate_selected_model_cdf(
    x_values: np.ndarray,
    sample_values: np.ndarray,
    fit_result: dict,
) -> np.ndarray:
    chosen_model = str(fit_result.get("chosen_model")).lower()
    distribution = build_fitted_distribution(sample_values, fit_result)
    if chosen_model == "kde":
        return np.asarray([distribution.integrate_box_1d(-np.inf, float(value)) for value in x_values])
    return distribution.cdf(x_values)


def build_probplot_distribution(sample_values: np.ndarray, fit_result: dict):
    chosen_model = str(fit_result.get("chosen_model")).lower()
    if chosen_model == "normal":
        params = get_selected_model_parameters(fit_result, "normal")
        return stats.norm(loc=params["loc"], scale=params["scale"])
    if chosen_model == "lognormal":
        params = get_selected_model_parameters(fit_result, "lognormal")
        return stats.lognorm(s=params["shape"], loc=params["loc"], scale=params["scale"])
    if chosen_model == "gamma":
        params = get_selected_model_parameters(fit_result, "gamma")
        return stats.gamma(a=params["shape"], loc=params["loc"], scale=params["scale"])
    if chosen_model == "kde":
        return EmpiricalPpfDistribution(build_kde_reference_quantiles(sample_values, fit_result))
    return None


def build_fitted_distribution(sample_values: np.ndarray, fit_result: dict):
    chosen_model = str(fit_result.get("chosen_model")).lower()
    if chosen_model == "normal":
        params = get_selected_model_parameters(fit_result, "normal")
        return stats.norm(loc=params["loc"], scale=params["scale"])
    if chosen_model == "lognormal":
        params = get_selected_model_parameters(fit_result, "lognormal")
        return stats.lognorm(s=params["shape"], loc=params["loc"], scale=params["scale"])
    if chosen_model == "gamma":
        params = get_selected_model_parameters(fit_result, "gamma")
        return stats.gamma(a=params["shape"], loc=params["loc"], scale=params["scale"])
    if chosen_model == "kde":
        return build_selected_cell_kde(sample_values, fit_result)
    raise ValueError(f"Unsupported selected model for plotting: {chosen_model}")


def get_selected_model_parameters(fit_result: dict, model_name: str) -> dict:
    model_stats = fit_result.get("candidate_model_statistics", {}).get(model_name, {})
    params = model_stats.get("parameters")
    if not params:
        raise ValueError(f"Selected {model_name} model is missing fitted parameters.")
    return params


def build_selected_cell_kde(sample_values: np.ndarray, fit_result: dict) -> stats.gaussian_kde:
    kde_status = fit_result.get("candidate_model_statistics", {}).get("kde", {})
    bandwidth = kde_status.get("bandwidth")
    if bandwidth is None:
        return stats.gaussian_kde(sample_values)

    sample_std = float(np.std(sample_values, ddof=1))
    if not np.isfinite(sample_std) or sample_std <= 0.0:
        return stats.gaussian_kde(sample_values)
    return stats.gaussian_kde(sample_values, bw_method=float(bandwidth) / sample_std)


def build_kde_reference_quantiles(sample_values: np.ndarray, fit_result: dict) -> np.ndarray:
    kde = build_selected_cell_kde(sample_values, fit_result)
    seed = int(
        int(fit_result["cell"]["time_index"]) * 1_000_003
        + int(fit_result["cell"]["lat_index"]) * 10_007
        + int(fit_result["cell"]["lon_index"])
    )
    rng = np.random.default_rng(seed)
    reference_sample_count = max(10_000, int(sample_values.size) * 500)
    sampled = kde.resample(reference_sample_count, seed=rng)
    return np.sort(np.asarray(sampled, dtype=float).reshape(-1))


class EmpiricalPpfDistribution:
    """Small ppf adapter so scipy.stats.probplot can draw KDE Q-Q plots."""

    def __init__(self, reference_quantiles: np.ndarray):
        self.reference_quantiles = np.asarray(reference_quantiles, dtype=float)
        self.reference_probabilities = np.linspace(
            1.0 / (self.reference_quantiles.size + 1),
            self.reference_quantiles.size / (self.reference_quantiles.size + 1),
            self.reference_quantiles.size,
        )

    def ppf(self, probabilities):
        clipped = np.clip(np.asarray(probabilities, dtype=float), 0.0, 1.0)
        return np.interp(
            clipped,
            self.reference_probabilities,
            self.reference_quantiles,
            left=float(self.reference_quantiles[0]),
            right=float(self.reference_quantiles[-1]),
        )


def format_cell_label(cell: dict) -> str:
    return (
        f"time={cell['time_value']}, "
        f"lat={float(cell['lat_value']):.3f}, lon={float(cell['lon_value']):.3f}"
    )


def format_model_name(model_name: str) -> str:
    names = {
        "normal": "Normal",
        "lognormal": "Lognormal",
        "gamma": "Gamma",
        "kde": "KDE",
    }
    return names.get(model_name, str(model_name).title())


def format_percentile_label(percentile: float) -> str:
    value = float(percentile)
    if value.is_integer():
        return f"p{int(value):02d}"
    return f"p{value:g}"


def plot_missing_percentage_map(
    data_array: xr.DataArray,
    title: str,
    output_path: Path,
    config,
) -> Path:
    missing_percent = data_array.isnull().mean(dim="time", skipna=False) * 100.0
    vmax = float(config.missing_percentage_plot_vmax)
    tick_step = float(config.missing_percentage_plot_tick_step)
    return plot_spatial_field(
        missing_percent,
        title,
        output_path,
        colorbar_label="Missing data (%)",
        cmap=MAP_COLORMAP,
        vmin=0.0,
        vmax=vmax,
        extend="max",
        colorbar_ticks=np.arange(0.0, vmax + tick_step * 0.5, tick_step),
        over_color="black",
        metadata_text=build_map_metadata(
            data_array,
            f"Each pixel is percent missing across all time steps.\nValues above {vmax:g}% are shown in black.",
        ),
        map_tick_spacing=float(config.map_tick_spacing),
    )


def plot_chlorophyll_mean_map(
    data_array: xr.DataArray,
    title: str,
    output_path: Path,
    config,
    vmax: float | None = None,
) -> Path:
    mean_chlorophyll = data_array.mean(dim="time", skipna=True)
    if vmax is None:
        vmax = rounded_colorbar_max(
            mean_chlorophyll,
            fallback=1.0,
            step=float(config.chlorophyll_colorbar_step),
            percentile=float(config.chlorophyll_colorbar_percentile),
        )
    return plot_spatial_field(
        mean_chlorophyll,
        title,
        output_path,
        colorbar_label="Chlorophyll concentration (mg m$^{-3}$)",
        cmap=MAP_COLORMAP,
        vmin=0.0,
        vmax=vmax,
        extend="max",
        metadata_text=build_map_metadata(
            data_array,
            "Map shows time-mean chlorophyll concentration.",
        ),
        map_tick_spacing=float(config.map_tick_spacing),
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
    map_tick_spacing: float = 10.0,
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
    format_map_ticks(ax, map_tick_spacing)
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
                map_tick_spacing=map_tick_spacing,
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
    labels = ["normal", "lognormal", "gamma", "kde", "unresolved"]
    values = [int(counts.get(label, 0)) for label in labels]
    display_labels = ["Normal", "Lognormal", "Gamma", "KDE", "Unresolved"]

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


def format_map_ticks(ax, spacing: float) -> None:
    x_min, x_max = ax.get_xlim()
    y_min, y_max = ax.get_ylim()
    x_ticks = build_degree_ticks(x_min, x_max, spacing=spacing)
    y_ticks = build_degree_ticks(y_min, y_max, spacing=spacing)

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
    percentile: float,
    upper_limit: float | None = None,
) -> float:
    values = np.asarray(field.values, dtype=float)
    finite_values = values[np.isfinite(values)]
    if finite_values.size == 0:
        return fallback

    high_value = float(np.nanpercentile(finite_values, percentile))
    if not np.isfinite(high_value) or high_value <= 0:
        return fallback

    rounded = float(np.ceil(high_value / step) * step)
    if upper_limit is not None:
        rounded = min(rounded, upper_limit)
    return max(rounded, step)


def compute_shared_chlorophyll_vmax(
    raw_data: xr.DataArray,
    final_mean_reconstruction: xr.DataArray,
    config,
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

    high_value = float(np.nanpercentile(finite_values, float(config.chlorophyll_colorbar_percentile)))
    if not np.isfinite(high_value) or high_value <= 0:
        return 1.0
    step = float(config.chlorophyll_colorbar_step)
    return max(float(np.ceil(high_value / step) * step), step)

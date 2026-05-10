"""Simple JSON configuration loading and validation utilities."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class AppConfig:
    """Application configuration for the ocean gap-filling pipeline."""

    input_directory: Path
    variable_name: str
    latitude_dim: str
    longitude_dim: str
    time_dim: str
    output_directory: str
    raw_file_pattern: str | list[str] | tuple[str, ...] | None = None
    study_area_bounds: dict[str, float] | None = None
    comparison_year: int | None = None
    filled_data_directory: Path | None = None
    filled_data_pattern: str | list[str] | tuple[str, ...] = "*.nc*"
    filled_variable_name: str | None = None
    reconstructed_mean_data_file: Path | None = None
    enable_study_area_crop: bool = False
    enable_8day_compositing: bool = True
    composite_window_size: int = 8
    composite_min_valid_fraction: float = 0.6
    preprocessed_data_file: Path | None = None
    enable_regridding: bool = True
    target_grid_resolution: float = 1.0
    random_seed: int = 42
    sampled_cell_count: int = 5
    monte_carlo_simulations: int = 10
    ks_pvalue_threshold: float = 0.05
    min_parametric_sample_size: int = 5
    min_positive_variance: float = 1e-8
    min_kde_sample_size: int = 5
    min_kde_unique_values: int = 2
    kde_cv_folds: int = 5
    kde_bandwidth_min: float = 0.01
    kde_bandwidth_max: float = 10.0
    kde_bandwidth_grid_size: int = 50
    positive_distribution_min_value: float = 0.0
    monte_carlo_sample_min_value: float = 0.0
    monte_carlo_summary_percentiles: list[float] | tuple[float, ...] = (5, 25, 50, 75, 95)
    monte_carlo_preview_sample_count: int = 20
    uncertainty_lower_percentile: float = 5.0
    uncertainty_upper_percentile: float = 95.0
    missing_percentage_plot_vmax: float = 25.0
    missing_percentage_plot_tick_step: float = 5.0
    chlorophyll_colorbar_percentile: float = 98.0
    chlorophyll_colorbar_step: float = 0.05
    map_tick_spacing: float = 10.0
    bloom_smoothing_window: int = 3
    bloom_threshold_method: str = "median_multiplier"
    bloom_threshold_multiplier: float = 1.05
    bloom_threshold_percentile: float = 60.0
    bloom_detection_year: int | None = None
    save_reconstructed_datasets: bool = True
    config_path: str | None = None
    config_directory: str | None = None

    # This is a property, means we can call it like config.logs_dir and it will return the path to logs directory. It is derived from output_directory, so if output_directory is /path/to/output, then logs_dir will be /path/to/output/logs. This way we can keep all outputs organized under the main output directory. 
    @property
    def logs_dir(self) -> str:                                  
        return str(Path(self.output_directory) / "logs")

    @property
    def sampled_cells_dir(self) -> str:
        return str(Path(self.output_directory) / "sampled_cells")

    @property
    def reconstructed_dir(self) -> str:
        return str(Path(self.output_directory) / "reconstructed")

    @property
    def annual_cycle_dir(self) -> str:
        return str(Path(self.output_directory) / "annual_cycle")

    @property
    def pre_processing_dir(self) -> str:
        return str(Path(self.output_directory) / "pre_processing")

    @property
    def filled_data_comparison_dir(self) -> str:
        return str(Path(self.output_directory) / "filled_data_comparison")

    @property
    def bloom_dir(self) -> str:
        return str(Path(self.output_directory) / "bloom")

    @property
    def missing_percentage_dir(self) -> str:
        return str(Path(self.output_directory) / "missing_percentage")

    @property
    def pipeline_diagnostics_dir(self) -> str:
        return str(Path(self.output_directory) / "pipeline_diagnostics")

    @property
    def chlorophyll_maps_dir(self) -> str:
        return str(Path(self.output_directory) / "chlorophyll_maps")

    @property
    def datasets_dir(self) -> str:
        return str(Path(self.output_directory) / "datasets")

    def output_directories(self) -> list[str]:
        return [
            self.output_directory,
            self.logs_dir,
            self.sampled_cells_dir,
            self.reconstructed_dir,
            self.annual_cycle_dir,
            self.pre_processing_dir,
            self.filled_data_comparison_dir,
            self.bloom_dir,
            self.missing_percentage_dir,
            self.pipeline_diagnostics_dir,
            self.chlorophyll_maps_dir,
            self.datasets_dir,
        ]


def load_config(config_path: Path) -> AppConfig:
    """Load a JSON configuration file and return a validated config object."""
    resolved_config_path = Path(config_path).expanduser().resolve()
    with resolved_config_path.open("r", encoding="utf-8") as handle:
        raw_config = json.load(handle)
    return validate_config(raw_config, resolved_config_path)

# This validates all the rfequired fields are present in the config file and also checks that the values are in expected range. If we forgets something -> returns error.
def validate_config(raw_config: dict, config_path: Path | None = None) -> AppConfig:
    """Validate required fields and basic value ranges."""
    required_fields = [
        "input_directory",
        "variable_name",
        "latitude_dim",
        "longitude_dim",
        "time_dim",
        "output_directory",
    ]

    missing_fields = [field for field in required_fields if field not in raw_config]
    if missing_fields:
        joined = ", ".join(missing_fields)
        source = f" in {config_path}" if config_path else ""
        raise ValueError(f"Missing required config fields{source}: {joined}")

    normalized_config = dict(raw_config)
    config_base_dir = (
        Path(config_path).expanduser().resolve().parent
        if config_path is not None
        else Path.cwd()
    )

    # Normalize config-defined paths once during loading so the rest of the application can use config values without depending where we are running from.
    normalized_config["input_directory"] = _resolve_config_path(
        normalized_config["input_directory"],
        config_base_dir,
    )
    normalized_config["output_directory"] = str(
        _resolve_config_path(normalized_config["output_directory"], config_base_dir)
    )
    if normalized_config.get("preprocessed_data_file"):
        normalized_config["preprocessed_data_file"] = _resolve_config_path(
            normalized_config["preprocessed_data_file"],
            config_base_dir,
        )
    else:
        normalized_config["preprocessed_data_file"] = None
    if normalized_config.get("filled_data_directory"):
        normalized_config["filled_data_directory"] = _resolve_config_path(
            normalized_config["filled_data_directory"],
            config_base_dir,
        )
    else:
        normalized_config["filled_data_directory"] = (
            config_base_dir / "../data/filled_data"
        ).resolve()
    if normalized_config.get("reconstructed_mean_data_file"):
        normalized_config["reconstructed_mean_data_file"] = _resolve_config_path(
            normalized_config["reconstructed_mean_data_file"],
            config_base_dir,
        )
    else:
        normalized_config["reconstructed_mean_data_file"] = None

    normalized_config["config_path"] = (
        str(Path(config_path).expanduser().resolve()) if config_path is not None else None
    )
    normalized_config["config_directory"] = str(config_base_dir)

    # Convert dict → structured object
    config = AppConfig(**normalized_config)                 
    if config.preprocessed_data_file is None:
        _validate_existing_directory(config.input_directory, "input_directory")
    _validate_non_empty_string(config.variable_name, "variable_name")
    _validate_non_empty_string(config.latitude_dim, "latitude_dim")
    _validate_non_empty_string(config.longitude_dim, "longitude_dim")
    _validate_non_empty_string(config.time_dim, "time_dim")
    _validate_non_empty_string(config.output_directory, "output_directory")
    _validate_study_area_bounds(config.study_area_bounds)
    _validate_bool(config.enable_study_area_crop, "enable_study_area_crop")
    _validate_bool(config.enable_8day_compositing, "enable_8day_compositing")
    _validate_bool(config.enable_regridding, "enable_regridding")
    if config.composite_window_size <= 0:
        raise ValueError("composite_window_size must be a positive integer.")
    if not 0.0 < config.composite_min_valid_fraction <= 1.0:
        raise ValueError("composite_min_valid_fraction must be greater than 0 and at most 1.")

    # Value validation for numeric fields, helps avoid mistakes like "monte_carlo_simulations": -5.
    if config.target_grid_resolution <= 0:
        raise ValueError("target_grid_resolution must be positive.")
    if config.sampled_cell_count <= 0:
        raise ValueError("sampled_cell_count must be a positive integer.")
    if config.monte_carlo_simulations <= 0:
        raise ValueError("monte_carlo_simulations must be a positive integer.")
    if not 0.0 <= config.ks_pvalue_threshold <= 1.0:
        raise ValueError("ks_pvalue_threshold must be between 0 and 1.")
    if config.min_parametric_sample_size <= 0:
        raise ValueError("min_parametric_sample_size must be a positive integer.")
    if config.min_positive_variance <= 0:
        raise ValueError("min_positive_variance must be positive.")
    if config.min_kde_sample_size <= 0:
        raise ValueError("min_kde_sample_size must be a positive integer.")
    if config.min_kde_unique_values <= 0:
        raise ValueError("min_kde_unique_values must be a positive integer.")
    if config.kde_cv_folds <= 1:
        raise ValueError("kde_cv_folds must be greater than 1.")
    if config.min_kde_sample_size < config.kde_cv_folds:
        raise ValueError("min_kde_sample_size must be greater than or equal to kde_cv_folds.")
    if config.kde_bandwidth_min <= 0 or config.kde_bandwidth_max <= config.kde_bandwidth_min:
        raise ValueError("KDE bandwidth bounds must satisfy 0 < min < max.")
    if config.kde_bandwidth_grid_size <= 1:
        raise ValueError("kde_bandwidth_grid_size must be greater than 1.")
    if config.positive_distribution_min_value < 0:
        raise ValueError("positive_distribution_min_value must be non-negative.")
    if config.monte_carlo_sample_min_value < 0:
        raise ValueError("monte_carlo_sample_min_value must be non-negative.")
    _validate_percentile_sequence(
        config.monte_carlo_summary_percentiles,
        "monte_carlo_summary_percentiles",
    )
    if config.monte_carlo_preview_sample_count <= 0:
        raise ValueError("monte_carlo_preview_sample_count must be a positive integer.")
    _validate_percentile(config.uncertainty_lower_percentile, "uncertainty_lower_percentile")
    _validate_percentile(config.uncertainty_upper_percentile, "uncertainty_upper_percentile")
    if config.uncertainty_lower_percentile >= config.uncertainty_upper_percentile:
        raise ValueError("uncertainty_lower_percentile must be less than uncertainty_upper_percentile.")
    if config.missing_percentage_plot_vmax <= 0:
        raise ValueError("missing_percentage_plot_vmax must be positive.")
    if config.missing_percentage_plot_tick_step <= 0:
        raise ValueError("missing_percentage_plot_tick_step must be positive.")
    _validate_percentile(config.chlorophyll_colorbar_percentile, "chlorophyll_colorbar_percentile")
    if config.chlorophyll_colorbar_step <= 0:
        raise ValueError("chlorophyll_colorbar_step must be positive.")
    if config.map_tick_spacing <= 0:
        raise ValueError("map_tick_spacing must be positive.")
    if config.bloom_smoothing_window <= 0:
        raise ValueError("bloom_smoothing_window must be a positive integer.")
    if config.bloom_threshold_method not in {"median_multiplier", "percentile"}:
        raise ValueError("bloom_threshold_method must be 'median_multiplier' or 'percentile'.")
    if config.bloom_threshold_multiplier <= 0:
        raise ValueError("bloom_threshold_multiplier must be positive.")
    _validate_percentile(config.bloom_threshold_percentile, "bloom_threshold_percentile")

    return config


def _validate_non_empty_string(value: str, field_name: str) -> None:
    """Ensure a config value is a non-empty string."""
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string.")


def _validate_bool(value: bool, field_name: str) -> None:
    """Ensure a config value is a boolean."""
    if not isinstance(value, bool):
        raise ValueError(f"{field_name} must be true or false.")


def _validate_existing_directory(value: Path, field_name: str) -> None:
    """Ensure a config path points to an existing directory."""
    if not isinstance(value, Path):
        raise ValueError(f"{field_name} must be a pathlib.Path.")
    if not value.exists():
        raise FileNotFoundError(f"{field_name} does not exist: {value}")
    if not value.is_dir():
        raise NotADirectoryError(f"{field_name} is not a directory: {value}")


def _validate_study_area_bounds(bounds: dict[str, float] | None) -> None:
    if bounds is None:
        return

    required_keys = {
        "latitude_min",
        "latitude_max",
        "longitude_min",
        "longitude_max",
    }
    missing_keys = sorted(required_keys.difference(bounds))
    if missing_keys:
        joined = ", ".join(missing_keys)
        raise ValueError(f"study_area_bounds is missing required keys: {joined}")

    latitude_min = _coerce_float(bounds["latitude_min"], "study_area_bounds.latitude_min")
    latitude_max = _coerce_float(bounds["latitude_max"], "study_area_bounds.latitude_max")
    longitude_min = _coerce_float(bounds["longitude_min"], "study_area_bounds.longitude_min")
    longitude_max = _coerce_float(bounds["longitude_max"], "study_area_bounds.longitude_max")

    if not -90.0 <= latitude_min < latitude_max <= 90.0:
        raise ValueError("study_area_bounds latitude values must satisfy -90 <= min < max <= 90.")
    if not -180.0 <= longitude_min < longitude_max <= 360.0:
        raise ValueError(
            "study_area_bounds longitude values must satisfy -180 <= min < max <= 360."
        )

    bounds["latitude_min"] = latitude_min
    bounds["latitude_max"] = latitude_max
    bounds["longitude_min"] = longitude_min
    bounds["longitude_max"] = longitude_max


def _coerce_float(value, field_name: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be numeric.") from exc


def _validate_percentile(value: float, field_name: str) -> None:
    value = _coerce_float(value, field_name)
    if not 0.0 <= value <= 100.0:
        raise ValueError(f"{field_name} must be between 0 and 100.")


def _validate_percentile_sequence(values, field_name: str) -> None:
    if not values:
        raise ValueError(f"{field_name} must contain at least one percentile.")
    for value in values:
        _validate_percentile(value, field_name)


def _resolve_config_path(path_value: str, config_base_dir: Path) -> Path:
    """Resolve one config path relative to the config file directory.

    Absolute paths are preserved. Relative paths are interpreted relative to
    the config file location, not the current working directory.
    """
    candidate = Path(path_value).expanduser()
    if candidate.is_absolute():
        return candidate.resolve()
    return (config_base_dir / candidate).resolve()

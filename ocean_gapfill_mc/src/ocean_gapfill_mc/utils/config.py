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
    study_area_bounds: dict[str, float] | None = None
    preprocessed_data_file: Path | None = None
    composite_window_size: int = 8
    composite_min_valid_fraction: float = 0.6
    target_grid_resolution: float = 1.0
    random_seed: int = 42
    sampled_cell_count: int = 5
    monte_carlo_simulations: int = 10
    ks_pvalue_threshold: float = 0.05
    save_reconstructed_datasets: bool = True
    validation_mask_percent: float = 0.20
    validation_num_pairs: int = 5
    validation_random_seed: int = 42
    validation_season_timesteps: dict[str, list[int]] | None = None
    config_path: str | None = None
    config_directory: str | None = None

    # This is a property, means we can call it like config.logs_dir and it will return the path to logs directory. It is derived from output_directory, so if output_directory is /path/to/output, then logs_dir will be /path/to/output/logs. This way we can keep all outputs organized under the main output directory. 
    @property
    def logs_dir(self) -> str:                                  
        return str(Path(self.output_directory) / "logs")

    @property
    def summaries_dir(self) -> str:
        return str(Path(self.output_directory) / "summaries")

    @property
    def sampled_cells_dir(self) -> str:
        return str(Path(self.output_directory) / "sampled_cells")

    @property
    def reconstructed_dir(self) -> str:
        return str(Path(self.output_directory) / "reconstructed")

    @property
    def plots_dir(self) -> str:
        return str(Path(self.output_directory) / "plots")

    @property
    def datasets_dir(self) -> str:
        return str(Path(self.output_directory) / "datasets")

    @property
    def validation_dir(self) -> str:
        return str(Path(self.output_directory) / "validation")

    def output_directories(self) -> list[str]:
        return [
            self.output_directory,
            self.logs_dir,
            self.summaries_dir,
            self.sampled_cells_dir,
            self.reconstructed_dir,
            self.plots_dir,
            self.datasets_dir,
            self.validation_dir,
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

    normalized_config["config_path"] = (
        str(Path(config_path).expanduser().resolve()) if config_path is not None else None
    )
    normalized_config["config_directory"] = str(config_base_dir)
    normalize_validation_aliases(normalized_config)

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

    # Value validation for numeric fields, helps avoid mistakes like "monte_carlo_simulations": -5, "monte_carlo_simulations": -5
    if config.composite_window_size <= 0:
        raise ValueError("composite_window_size must be a positive integer.")
    if not 0.0 < config.composite_min_valid_fraction <= 1.0:
        raise ValueError("composite_min_valid_fraction must be between 0 and 1.")
    if config.target_grid_resolution <= 0:
        raise ValueError("target_grid_resolution must be positive.")
    if config.sampled_cell_count <= 0:
        raise ValueError("sampled_cell_count must be a positive integer.")
    if config.monte_carlo_simulations <= 0:
        raise ValueError("monte_carlo_simulations must be a positive integer.")
    if not 0.0 <= config.ks_pvalue_threshold <= 1.0:
        raise ValueError("ks_pvalue_threshold must be between 0 and 1.")
    if not 0.0 < config.validation_mask_percent <= 1.0:
        raise ValueError("validation_mask_percent must be between 0 and 1.")
    if config.validation_num_pairs <= 0:
        raise ValueError("validation_num_pairs must be a positive integer.")
    if config.validation_season_timesteps is not None:
        _validate_validation_seasons(config.validation_season_timesteps)

    return config


def normalize_validation_aliases(config: dict) -> None:
    """Allow either JSON-style or pseudocode-style validation config names."""
    aliases = {
        "MASK_PERCENT": "validation_mask_percent",
        "NUM_PAIRS": "validation_num_pairs",
        "RANDOM_SEED": "validation_random_seed",
        "SEASON_TIMESTEPS": "validation_season_timesteps",
    }
    for source, target in aliases.items():
        if source in config and target not in config:
            config[target] = config[source]
        config.pop(source, None)


def _validate_non_empty_string(value: str, field_name: str) -> None:
    """Ensure a config value is a non-empty string."""
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string.")


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


def _validate_validation_seasons(seasons: dict[str, list[int]]) -> None:
    if not isinstance(seasons, dict):
        raise ValueError("validation_season_timesteps must be a dictionary.")
    for season_name, timesteps in seasons.items():
        if not isinstance(season_name, str) or not season_name.strip():
            raise ValueError("validation season names must be non-empty strings.")
        if not isinstance(timesteps, list):
            raise ValueError(f"validation_season_timesteps.{season_name} must be a list.")
        if len(timesteps) < 2:
            raise ValueError(
                f"validation_season_timesteps.{season_name} must contain at least two timesteps."
            )
        for timestep in timesteps:
            if not isinstance(timestep, int) or timestep < 0:
                raise ValueError(
                    f"validation_season_timesteps.{season_name} contains an invalid timestep: {timestep}"
                )


def _coerce_float(value, field_name: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be numeric.") from exc


def _resolve_config_path(path_value: str, config_base_dir: Path) -> Path:
    """Resolve one config path relative to the config file directory.

    Absolute paths are preserved. Relative paths are interpreted relative to
    the config file location, not the current working directory.
    """
    candidate = Path(path_value).expanduser()
    if candidate.is_absolute():
        return candidate.resolve()
    return (config_base_dir / candidate).resolve()

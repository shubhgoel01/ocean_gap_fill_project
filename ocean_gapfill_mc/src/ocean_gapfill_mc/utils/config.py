"""Simple JSON configuration loading and validation utilities."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class AppConfig:
    """Application configuration for the ocean gap-filling pipeline."""

    input_file: str
    variable_name: str
    latitude_dim: str
    longitude_dim: str
    time_dim: str
    output_directory: str
    composite_window_size: int = 8
    composite_min_valid_fraction: float = 0.6
    target_grid_resolution: float = 1.0
    random_seed: int = 42
    sampled_cell_count: int = 5
    monte_carlo_simulations: int = 10
    ks_pvalue_threshold: float = 0.05
    save_plots: bool = True
    save_reconstructed_datasets: bool = True
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
    def plots_dir(self) -> str:
        return str(Path(self.output_directory) / "plots")

    @property
    def reconstructed_dir(self) -> str:
        return str(Path(self.output_directory) / "reconstructed")

    def output_directories(self) -> list[str]:
        return [
            self.output_directory,
            self.logs_dir,
            self.summaries_dir,
            self.sampled_cells_dir,
            self.plots_dir,
            self.reconstructed_dir,
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
        "input_file",
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
    for field_name in ("input_file", "output_directory"):
        normalized_config[field_name] = str(
            _resolve_config_path(normalized_config[field_name], config_base_dir)
        )

    normalized_config["config_path"] = (
        str(Path(config_path).expanduser().resolve()) if config_path is not None else None
    )
    normalized_config["config_directory"] = str(config_base_dir)

    # Convert dict → structured object
    config = AppConfig(**normalized_config)                 
    _validate_non_empty_string(config.input_file, "input_file")
    _validate_non_empty_string(config.variable_name, "variable_name")
    _validate_non_empty_string(config.latitude_dim, "latitude_dim")
    _validate_non_empty_string(config.longitude_dim, "longitude_dim")
    _validate_non_empty_string(config.time_dim, "time_dim")
    _validate_non_empty_string(config.output_directory, "output_directory")

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

    return config


def _validate_non_empty_string(value: str, field_name: str) -> None:
    """Ensure a config value is a non-empty string."""
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string.")


def _resolve_config_path(path_value: str, config_base_dir: Path) -> Path:
    """Resolve one config path relative to the config file directory.

    Absolute paths are preserved. Relative paths are interpreted relative to
    the config file location, not the current working directory.
    """
    candidate = Path(path_value).expanduser()
    if candidate.is_absolute():
        return candidate.resolve()
    return (config_base_dir / candidate).resolve()

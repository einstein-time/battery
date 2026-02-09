"""YAML/JSON configuration loader with validation and CLI override support.

Provides hierarchical configuration management for training, data generation,
and evaluation with type checking and default values.
"""

import json
import logging
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Optional, Union

import yaml

logger = logging.getLogger(__name__)

# Default configuration values
DEFAULTS: Dict[str, Any] = {
    "model": {
        "type": "pc_unet",
        "in_channels": 9,
        "out_channels": 1,
        "base_features": 32,
        "num_levels": 4,
        "dropout_rate": 0.1,
    },
    "training": {
        "epochs": 200,
        "batch_size": 16,
        "learning_rate": 1e-3,
        "weight_decay": 1e-5,
        "seed": 42,
        "phases": {
            "pretrain_epochs": 50,
            "physics_epochs": 100,
            "finetune_epochs": 50,
        },
        "scheduler": {
            "type": "cosine",
            "T_max": 200,
            "eta_min": 1e-6,
        },
        "grad_clip": 1.0,
        "early_stopping_patience": 20,
    },
    "data": {
        "data_path": "data/raw/thermal_dataset.h5",
        "grid_size": 128,
        "dt": 0.001,
        "total_time": 1.0,
        "n_trajectories": 2000,
        "train_ratio": 0.7,
        "val_ratio": 0.15,
        "test_ratio": 0.15,
    },
    "physics": {
        "dx": 0.001,
        "dy": 0.001,
        "rho": 2500.0,
        "cp": 700.0,
        "k_range": [0.5, 5.0],
        "q_range": [1e5, 5e6],
        "h_range": [10.0, 500.0],
        "T_amb": 298.15,
    },
    "loss": {
        "lambda_data": 1.0,
        "lambda_pde": 0.1,
        "lambda_bc": 0.1,
        "lambda_consistency": 0.05,
        "use_gradnorm": True,
        "gradnorm_alpha": 1.5,
    },
    "output": {
        "output_dir": "outputs",
        "checkpoint_dir": "checkpoints",
        "log_dir": "logs",
        "save_every": 10,
    },
}

# Required keys for validation
REQUIRED_KEYS = ["model", "training", "data"]


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Deep merge override dict into base dict.

    Args:
        base: Base configuration dictionary.
        override: Override dictionary (takes precedence).

    Returns:
        Merged dictionary.
    """
    result = deepcopy(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result


def load_config(
    path: Union[str, Path],
    overrides: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Load configuration from YAML or JSON file with defaults and overrides.

    Args:
        path: Path to configuration file (.yaml, .yml, or .json).
        overrides: Optional dictionary of overrides applied on top.

    Returns:
        Complete configuration dictionary.

    Raises:
        FileNotFoundError: If configuration file does not exist.
        ValueError: If file format is unsupported.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    suffix = path.suffix.lower()
    with open(path, "r", encoding="utf-8") as f:
        if suffix in (".yaml", ".yml"):
            file_config = yaml.safe_load(f) or {}
        elif suffix == ".json":
            file_config = json.load(f)
        else:
            raise ValueError(f"Unsupported config format: {suffix}")

    config = _deep_merge(DEFAULTS, file_config)

    if overrides:
        config = _deep_merge(config, overrides)

    validate_config(config)
    logger.info("Loaded config from %s", path)
    return config


def validate_config(config: Dict[str, Any]) -> None:
    """Validate configuration dictionary for required keys and value ranges.

    Args:
        config: Configuration dictionary to validate.

    Raises:
        ValueError: If configuration is invalid.
    """
    for key in REQUIRED_KEYS:
        if key not in config:
            raise ValueError(f"Missing required config section: '{key}'")

    training = config.get("training", {})
    if training.get("learning_rate", 0) <= 0:
        raise ValueError("learning_rate must be positive")
    if training.get("batch_size", 0) <= 0:
        raise ValueError("batch_size must be positive")
    if training.get("epochs", 0) <= 0:
        raise ValueError("epochs must be positive")

    data = config.get("data", {})
    if data.get("grid_size", 0) <= 0:
        raise ValueError("grid_size must be positive")

    ratios = (
        data.get("train_ratio", 0.7)
        + data.get("val_ratio", 0.15)
        + data.get("test_ratio", 0.15)
    )
    if abs(ratios - 1.0) > 1e-6:
        raise ValueError(f"Data split ratios must sum to 1.0, got {ratios}")

    logger.debug("Configuration validated successfully")


def config_to_flat(config: Dict[str, Any], prefix: str = "") -> Dict[str, Any]:
    """Flatten nested config dict to dot-separated keys.

    Args:
        config: Nested configuration dictionary.
        prefix: Key prefix for recursion.

    Returns:
        Flat dictionary with dot-separated keys.
    """
    flat: Dict[str, Any] = {}
    for key, value in config.items():
        full_key = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            flat.update(config_to_flat(value, full_key))
        else:
            flat[full_key] = value
    return flat

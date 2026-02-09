"""Training logger, experiment tracker, and progress bar utilities.

Provides structured logging for training experiments with CSV export,
metric tracking, and console progress bars via tqdm.
"""

import csv
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from tqdm import tqdm

logger = logging.getLogger(__name__)


def setup_logging(
    log_dir: Optional[Union[str, Path]] = None,
    level: int = logging.INFO,
    name: str = "battery_surrogate",
) -> logging.Logger:
    """Set up logging with console and optional file output.

    Args:
        log_dir: Directory for log files. If None, console only.
        level: Logging level.
        name: Logger name.

    Returns:
        Configured logger instance.
    """
    root_logger = logging.getLogger(name)
    root_logger.setLevel(level)
    root_logger.handlers.clear()

    formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    if log_dir is not None:
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(
            log_dir / "training.log", encoding="utf-8"
        )
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)

    return root_logger


class TrainingLogger:
    """Logs training metrics to CSV and console.

    Attributes:
        log_dir: Directory for log files.
        metrics_history: List of metric dictionaries per epoch.
    """

    def __init__(self, log_dir: Union[str, Path]) -> None:
        """Initialize TrainingLogger.

        Args:
            log_dir: Directory to store log files.
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_history: List[Dict[str, Any]] = []
        self._csv_path = self.log_dir / "metrics.csv"
        self._csv_initialized = False

    def log_epoch(self, epoch: int, metrics: Dict[str, float]) -> None:
        """Log metrics for one epoch.

        Args:
            epoch: Current epoch number.
            metrics: Dictionary of metric name to value.
        """
        row = {"epoch": epoch, "timestamp": time.time(), **metrics}
        self.metrics_history.append(row)

        if not self._csv_initialized:
            with open(self._csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=list(row.keys()))
                writer.writeheader()
                writer.writerow(row)
            self._csv_initialized = True
        else:
            with open(self._csv_path, "a", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=list(row.keys()))
                writer.writerow(row)

        parts = [f"Epoch {epoch:4d}"]
        for key, val in metrics.items():
            if isinstance(val, float):
                parts.append(f"{key}={val:.6f}")
            else:
                parts.append(f"{key}={val}")
        logger.info(" | ".join(parts))

    def get_best_epoch(self, metric: str = "val_loss", mode: str = "min") -> int:
        """Get epoch with best value for given metric.

        Args:
            metric: Metric name to compare.
            mode: 'min' for lower is better, 'max' for higher is better.

        Returns:
            Epoch number with best metric value.
        """
        if not self.metrics_history:
            return 0
        if mode == "min":
            best = min(self.metrics_history, key=lambda x: x.get(metric, float("inf")))
        else:
            best = max(self.metrics_history, key=lambda x: x.get(metric, float("-inf")))
        return best["epoch"]


class ExperimentTracker:
    """Tracks experiment metadata, configs, and results.

    Attributes:
        experiment_dir: Directory for this experiment.
    """

    def __init__(
        self,
        experiment_dir: Union[str, Path],
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Initialize ExperimentTracker.

        Args:
            experiment_dir: Directory for experiment outputs.
            config: Optional configuration dictionary to save.
        """
        self.experiment_dir = Path(experiment_dir)
        self.experiment_dir.mkdir(parents=True, exist_ok=True)
        self.training_logger = TrainingLogger(self.experiment_dir / "logs")

        if config is not None:
            self.save_config(config)

    def save_config(self, config: Dict[str, Any]) -> None:
        """Save configuration to JSON file.

        Args:
            config: Configuration dictionary.
        """
        config_path = self.experiment_dir / "config.json"
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2, default=str)
        logger.info("Saved config to %s", config_path)

    def save_results(self, results: Dict[str, Any]) -> None:
        """Save evaluation results to JSON file.

        Args:
            results: Results dictionary.
        """
        results_path = self.experiment_dir / "results.json"
        with open(results_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, default=str)
        logger.info("Saved results to %s", results_path)

    def log_epoch(self, epoch: int, metrics: Dict[str, float]) -> None:
        """Delegate epoch logging to TrainingLogger.

        Args:
            epoch: Current epoch number.
            metrics: Metric dictionary.
        """
        self.training_logger.log_epoch(epoch, metrics)


def create_progress_bar(
    iterable: Any,
    total: Optional[int] = None,
    desc: str = "",
    leave: bool = True,
) -> tqdm:
    """Create a tqdm progress bar.

    Args:
        iterable: Iterable to wrap.
        total: Total number of items.
        desc: Description string.
        leave: Whether to leave the progress bar after completion.

    Returns:
        tqdm progress bar wrapping the iterable.
    """
    return tqdm(
        iterable,
        total=total,
        desc=desc,
        leave=leave,
        dynamic_ncols=True,
    )

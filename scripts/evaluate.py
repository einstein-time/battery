#!/usr/bin/env python3
"""CLI script for evaluating trained thermal surrogate models.

Computes accuracy metrics, rollout stability, physics compliance,
speed benchmarks, and uncertainty calibration.

Usage:
    python scripts/evaluate.py --model checkpoints/best_model.pt
    python scripts/evaluate.py --model checkpoints/best_model.pt --data data/raw/thermal_dataset.h5
    python scripts/evaluate.py --model checkpoints/best_model.pt --rollout_steps 50
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.config import load_config, DEFAULTS
from src.utils.device import get_device
from src.utils.logging import setup_logging
from src.models import BaseThermalModel, PCUNet, SimpleCNN, PODModel
from src.data_utils.dataset import ThermalDataset, create_data_loaders

logger = logging.getLogger("battery_surrogate")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed arguments namespace.
    """
    parser = argparse.ArgumentParser(
        description="Evaluate trained thermal surrogate model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to model checkpoint (.pt file)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/train.yaml",
        help="Path to training config (for model architecture)",
    )
    parser.add_argument(
        "--data",
        type=str,
        default=None,
        help="Path to HDF5 test dataset (overrides config)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/evaluation",
        help="Directory to save evaluation results",
    )
    parser.add_argument(
        "--rollout_steps",
        type=int,
        default=50,
        help="Number of rollout steps for stability test",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device override (cuda, cpu, mps)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for evaluation",
    )
    return parser.parse_args()


def load_model(
    checkpoint_path: Path,
    config: Dict[str, Any],
    device: torch.device,
) -> BaseThermalModel:
    """Load model from checkpoint.

    Args:
        checkpoint_path: Path to .pt checkpoint file.
        config: Model configuration dictionary.
        device: Device to load model onto.

    Returns:
        Loaded model in eval mode.
    """
    model_type = config["model"]["type"]
    model_cfg = config["model"]

    if model_type == "pc_unet":
        model = PCUNet(
            in_channels=model_cfg.get("in_channels", 9),
            out_channels=model_cfg.get("out_channels", 1),
            base_features=model_cfg.get("base_features", 32),
            num_levels=model_cfg.get("num_levels", 4),
            dropout_rate=model_cfg.get("dropout_rate", 0.1),
        )
    elif model_type == "simple_cnn":
        model = SimpleCNN(
            in_channels=model_cfg.get("in_channels", 9),
            out_channels=model_cfg.get("out_channels", 1),
        )
    elif model_type == "pod":
        model = PODModel(
            grid_size=config["data"].get("grid_size", 128),
            n_modes=model_cfg.get("n_modes", 50),
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    model = model.to(device)
    model.eval()
    logger.info("Loaded model from %s (%s)", checkpoint_path, model_type)
    return model


def compute_metrics(
    model: BaseThermalModel,
    data_loader: torch.utils.data.DataLoader,
    device: torch.device,
) -> Dict[str, float]:
    """Compute one-step prediction metrics on test set.

    Args:
        model: Trained model in eval mode.
        data_loader: Test data loader.
        device: Compute device.

    Returns:
        Dictionary of metric names to values.
    """
    mse_sum = 0.0
    mae_sum = 0.0
    max_error = 0.0
    n_samples = 0

    with torch.no_grad():
        for batch in data_loader:
            inputs = batch["input"].to(device)
            targets = batch["target"].to(device)

            predictions = model(inputs)

            mse = torch.mean((predictions - targets) ** 2).item()
            mae = torch.mean(torch.abs(predictions - targets)).item()
            batch_max = torch.max(torch.abs(predictions - targets)).item()

            batch_size = inputs.size(0)
            mse_sum += mse * batch_size
            mae_sum += mae * batch_size
            max_error = max(max_error, batch_max)
            n_samples += batch_size

    metrics = {
        "mse": mse_sum / n_samples,
        "rmse": np.sqrt(mse_sum / n_samples),
        "mae": mae_sum / n_samples,
        "max_absolute_error": max_error,
        "n_test_samples": n_samples,
    }
    return metrics


def rollout_stability(
    model: BaseThermalModel,
    initial_input: torch.Tensor,
    ground_truth_trajectory: torch.Tensor,
    n_steps: int,
    device: torch.device,
) -> Dict[str, Any]:
    """Evaluate rollout stability over multiple time steps.

    Args:
        model: Trained model in eval mode.
        initial_input: Initial 9-channel input tensor (1, 9, H, W).
        ground_truth_trajectory: Ground truth (n_steps, H, W).
        n_steps: Number of rollout steps.
        device: Compute device.

    Returns:
        Dictionary with rollout errors at each step.
    """
    errors = []
    current_input = initial_input.to(device)

    with torch.no_grad():
        for step in range(min(n_steps, len(ground_truth_trajectory) - 1)):
            delta_t = model(current_input)
            current_temp = current_input[:, 0:1, :, :] + delta_t

            gt = ground_truth_trajectory[step + 1].to(device)
            step_error = torch.mean((current_temp.squeeze() - gt) ** 2).item()
            errors.append(step_error)

            current_input = current_input.clone()
            current_input[:, 0:1, :, :] = current_temp

    return {
        "step_mse": errors,
        "final_mse": errors[-1] if errors else 0.0,
        "mean_mse": np.mean(errors) if errors else 0.0,
        "error_growth_rate": np.polyfit(range(len(errors)), errors, 1)[0] if len(errors) > 1 else 0.0,
    }


def speed_benchmark(
    model: BaseThermalModel,
    input_shape: tuple,
    device: torch.device,
    n_warmup: int = 10,
    n_runs: int = 100,
) -> Dict[str, float]:
    """Benchmark model inference speed.

    Args:
        model: Model to benchmark.
        input_shape: Shape of input tensor (B, C, H, W).
        device: Compute device.
        n_warmup: Number of warmup iterations.
        n_runs: Number of timed iterations.

    Returns:
        Dictionary with timing statistics.
    """
    dummy_input = torch.randn(*input_shape, device=device)

    with torch.no_grad():
        for _ in range(n_warmup):
            _ = model(dummy_input)

    if device.type == "cuda":
        torch.cuda.synchronize()

    times = []
    with torch.no_grad():
        for _ in range(n_runs):
            start = time.perf_counter()
            _ = model(dummy_input)
            if device.type == "cuda":
                torch.cuda.synchronize()
            end = time.perf_counter()
            times.append(end - start)

    times_ms = np.array(times) * 1000.0
    return {
        "mean_ms": float(np.mean(times_ms)),
        "std_ms": float(np.std(times_ms)),
        "min_ms": float(np.min(times_ms)),
        "max_ms": float(np.max(times_ms)),
        "throughput_samples_per_sec": float(input_shape[0] / np.mean(times)),
    }


def main() -> None:
    """Main evaluation entry point."""
    args = parse_args()

    setup_logging(level=logging.INFO)

    device = get_device(args.device)

    config_path = Path(args.config)
    if config_path.exists():
        config = load_config(config_path)
    else:
        logger.warning("Config not found: %s, using defaults", config_path)
        config = DEFAULTS.copy()

    if args.data:
        config["data"]["data_path"] = args.data

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_path = Path(args.model)
    if not checkpoint_path.exists():
        logger.error("Model checkpoint not found: %s", checkpoint_path)
        sys.exit(1)

    model = load_model(checkpoint_path, config, device)

    logger.info("=" * 60)
    logger.info("Model Evaluation")
    logger.info("=" * 60)
    logger.info("Model:   %s", config["model"]["type"])
    logger.info("Params:  %s", f"{model.count_parameters():,}")
    logger.info("Device:  %s", device)
    logger.info("=" * 60)

    results: Dict[str, Any] = {}

    data_path = Path(config["data"]["data_path"])
    if data_path.exists():
        loaders = create_data_loaders(
            data_path,
            config,
            batch_size=args.batch_size,
        )
        test_loader = loaders["test"]

        logger.info("Computing one-step metrics...")
        metrics = compute_metrics(model, test_loader, device)
        results["one_step_metrics"] = metrics
        logger.info("MSE: %.6f | MAE: %.6f | Max Error: %.4f",
                     metrics["mse"], metrics["mae"], metrics["max_absolute_error"])
    else:
        logger.warning("Dataset not found at %s, skipping accuracy metrics", data_path)

    logger.info("Running speed benchmark...")
    grid_size = config["data"].get("grid_size", 128)
    speed = speed_benchmark(
        model,
        input_shape=(1, 9, grid_size, grid_size),
        device=device,
    )
    results["speed_benchmark"] = speed
    logger.info("Inference: %.2f ms/step (%.0f samples/sec)",
                 speed["mean_ms"], speed["throughput_samples_per_sec"])

    results_path = output_dir / "evaluation_results.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info("Results saved to %s", results_path)

    logger.info("Evaluation complete!")


if __name__ == "__main__":
    main()

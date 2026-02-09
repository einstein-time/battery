#!/usr/bin/env python3
"""CLI training script for the battery thermal surrogate model.

Supports three model architectures (PC-UNet, SimpleCNN, POD), a three-phase
physics-informed training strategy, checkpoint resume, GPU OOM recovery,
and comprehensive experiment tracking.

Usage:
    python scripts/train.py --config configs/train.yaml
    python scripts/train.py --config configs/train.yaml --model_type simple_cnn
    python scripts/train.py --config configs/train.yaml --debug
    python scripts/train.py --config configs/train.yaml --resume outputs/exp/checkpoints/best_model.pt
"""

from __future__ import annotations

import argparse
import datetime
import logging
import math
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
import yaml

# ---------------------------------------------------------------------------
# Project root on sys.path so that ``src.*`` imports work when running the
# script directly (e.g. ``python scripts/train.py ...``).
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.device import get_device
from src.utils.config import load_config
from src.utils.logging import setup_logging, ExperimentTracker
from src.models import PCUNet, SimpleCNN, PODModel
from src.training.trainer import Trainer
from src.training.losses import PhysicsInformedLoss
from src.training.gradnorm import GradNorm
from src.data_utils.dataset import ThermalDataset, create_data_loaders

logger = logging.getLogger("battery_surrogate")


# ======================================================================
# Utility helpers
# ======================================================================

def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility across all libraries.

    Configures Python's built-in random module, NumPy, and PyTorch (CPU
    and CUDA) to use the given seed.  Also sets CuDNN to deterministic
    mode, which may reduce performance slightly but guarantees bitwise-
    reproducible results on CUDA devices.

    Args:
        seed: Integer seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def create_model(config: Dict[str, Any], device: torch.device) -> torch.nn.Module:
    """Instantiate the model architecture specified in *config*.

    Factory function that reads ``config["model"]["type"]`` and forwards
    the relevant hyper-parameters to the chosen model constructor.

    Args:
        config: Full project configuration dictionary.
        device: Target device -- the model is moved to this device before
            being returned.

    Returns:
        A model instance (subclass of :class:`BaseThermalModel`) on *device*.

    Raises:
        ValueError: If the requested model type is not recognised.
    """
    model_cfg = config.get("model", {})
    model_type: str = model_cfg.get("type", "pc_unet")
    data_cfg = config.get("data", {})

    in_channels: int = model_cfg.get("in_channels", 9)
    out_channels: int = model_cfg.get("out_channels", 1)

    if model_type == "pc_unet":
        model = PCUNet(
            in_channels=in_channels,
            out_channels=out_channels,
            base_features=model_cfg.get("base_features", 32),
            num_levels=model_cfg.get("num_levels", 4),
            dropout_rate=model_cfg.get("dropout_rate", 0.1),
        )
    elif model_type == "simple_cnn":
        model = SimpleCNN(
            in_channels=in_channels,
            out_channels=out_channels,
        )
    elif model_type == "pod":
        grid_size: int = data_cfg.get("grid_size", 128)
        model = PODModel(
            n_modes=model_cfg.get("n_modes", 50),
            field_height=grid_size,
            field_width=grid_size,
            in_channels=in_channels,
            out_channels=out_channels,
        )
    else:
        raise ValueError(
            f"Unknown model type '{model_type}'. "
            "Supported: pc_unet, simple_cnn, pod"
        )

    model = model.to(device)
    return model


def apply_cli_overrides(config: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    """Apply non-None CLI argument overrides to the configuration dict.

    Only values that the user explicitly set on the command line (i.e. that
    are not ``None``) are written into *config*.  This allows the YAML file
    to serve as the single source of truth while still permitting quick
    one-off experiments from the command line.

    Args:
        config: Configuration dictionary (modified in place).
        args: Parsed CLI arguments.

    Returns:
        The (mutated) configuration dictionary for convenience.
    """
    if args.data_path is not None:
        config.setdefault("data", {})["data_path"] = args.data_path

    if args.model_type is not None:
        config.setdefault("model", {})["type"] = args.model_type

    if args.epochs is not None:
        config.setdefault("training", {})["epochs"] = args.epochs

    if args.batch_size is not None:
        config.setdefault("training", {})["batch_size"] = args.batch_size

    if args.learning_rate is not None:
        config.setdefault("training", {})["learning_rate"] = args.learning_rate

    if args.seed is not None:
        config.setdefault("training", {})["seed"] = args.seed

    # Debug mode: tiny dataset and short run for quick smoke tests.
    if args.debug:
        config.setdefault("data", {})["n_trajectories"] = 100
        config.setdefault("training", {})["epochs"] = 5
        config["training"].setdefault("phases", {})
        config["training"]["phases"]["pretrain_epochs"] = 2
        config["training"]["phases"]["physics_epochs"] = 2
        config["training"]["phases"]["finetune_epochs"] = 1
        config.setdefault("training", {})["batch_size"] = min(
            config.get("training", {}).get("batch_size", 16), 8
        )
        logger.info("Debug mode enabled: 100 samples, 5 epochs, batch_size <= 8")

    return config


def _format_param_count(n: int) -> str:
    """Return a human-readable string for a parameter count.

    Args:
        n: Number of parameters.

    Returns:
        Formatted string, e.g. ``"1.24 M"`` or ``"543 K"``.
    """
    if n >= 1_000_000:
        return f"{n / 1_000_000:.2f} M"
    if n >= 1_000:
        return f"{n / 1_000:.1f} K"
    return str(n)


def print_banner(config: Dict[str, Any], device: torch.device, experiment_name: str) -> None:
    """Print an informative startup banner summarising the run configuration.

    Args:
        config: Full project configuration dictionary.
        device: Selected compute device.
        experiment_name: Human-readable experiment identifier.
    """
    model_cfg = config.get("model", {})
    training_cfg = config.get("training", {})
    data_cfg = config.get("data", {})
    loss_cfg = config.get("loss", {})
    phases = training_cfg.get("phases", {})

    banner = "\n".join([
        "",
        "=" * 70,
        "  Battery Thermal Surrogate -- Training",
        "=" * 70,
        f"  Experiment : {experiment_name}",
        f"  Timestamp  : {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"  Device     : {device}",
        "-" * 70,
        "  Model",
        f"    type           : {model_cfg.get('type', 'pc_unet')}",
        f"    in_channels    : {model_cfg.get('in_channels', 9)}",
        f"    out_channels   : {model_cfg.get('out_channels', 1)}",
        f"    base_features  : {model_cfg.get('base_features', '-')}",
        f"    num_levels     : {model_cfg.get('num_levels', '-')}",
        f"    dropout_rate   : {model_cfg.get('dropout_rate', '-')}",
        "-" * 70,
        "  Training",
        f"    total epochs   : {training_cfg.get('epochs', 200)}",
        f"      phase 1      : {phases.get('pretrain_epochs', 50)}  (data only)",
        f"      phase 2      : {phases.get('physics_epochs', 100)}  (physics-informed)",
        f"      phase 3      : {phases.get('finetune_epochs', 50)}  (consistency)",
        f"    batch_size     : {training_cfg.get('batch_size', 16)}",
        f"    learning_rate  : {training_cfg.get('learning_rate', 1e-3):.2e}",
        f"    weight_decay   : {training_cfg.get('weight_decay', 1e-5):.2e}",
        f"    grad_clip      : {training_cfg.get('grad_clip', 1.0)}",
        f"    early_stopping : {training_cfg.get('early_stopping_patience', 20)} epochs",
        f"    seed           : {training_cfg.get('seed', 42)}",
        "-" * 70,
        "  Data",
        f"    data_path      : {data_cfg.get('data_path', '-')}",
        f"    grid_size      : {data_cfg.get('grid_size', 128)}",
        f"    n_trajectories : {data_cfg.get('n_trajectories', 2000)}",
        f"    train/val/test : {data_cfg.get('train_ratio', 0.7):.0%} / "
        f"{data_cfg.get('val_ratio', 0.15):.0%} / "
        f"{data_cfg.get('test_ratio', 0.15):.0%}",
        "-" * 70,
        "  Loss weights",
        f"    lambda_data    : {loss_cfg.get('lambda_data', 1.0)}",
        f"    lambda_pde     : {loss_cfg.get('lambda_pde', 0.1)}",
        f"    lambda_bc      : {loss_cfg.get('lambda_bc', 0.1)}",
        f"    lambda_consist : {loss_cfg.get('lambda_consistency', 0.05)}",
        f"    use_gradnorm   : {loss_cfg.get('use_gradnorm', True)}",
        "=" * 70,
        "",
    ])
    logger.info(banner)


# ======================================================================
# CLI argument parser
# ======================================================================

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the training script.

    Returns:
        Parsed arguments namespace.
    """
    parser = argparse.ArgumentParser(
        description="Train a battery thermal surrogate model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Required
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the YAML configuration file",
    )

    # Data overrides
    parser.add_argument(
        "--data_path",
        type=str,
        default=None,
        help="Override the HDF5 data path from config",
    )

    # Model overrides
    parser.add_argument(
        "--model_type",
        type=str,
        default=None,
        choices=["pc_unet", "simple_cnn", "pod"],
        help="Override the model architecture from config",
    )

    # Output
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs",
        help="Root directory for all experiment outputs",
    )

    # Resume training
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to a checkpoint file to resume training from",
    )

    # Training hyper-parameter overrides
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Override the total number of training epochs",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=None,
        help="Override the training batch size",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=None,
        help="Override the initial learning rate",
    )

    # Device
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        choices=["cuda", "cpu", "mps"],
        help="Compute device (default: auto-detect)",
    )

    # Debug mode
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Debug mode: small dataset (100 samples) and 5 epochs",
    )

    # Reproducibility
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )

    # Experiment tracking
    parser.add_argument(
        "--experiment_name",
        type=str,
        default=None,
        help="Human-readable name for this experiment run "
             "(default: auto-generated from model type and timestamp)",
    )

    return parser.parse_args()


# ======================================================================
# Main entry point
# ======================================================================

def main() -> None:
    """Main training entry point.

    Orchestrates the full training pipeline:
    1. Parse CLI arguments and load configuration.
    2. Set up logging, random seeds, and compute device.
    3. Create output directories and experiment tracker.
    4. Load data, build model, and instantiate the Trainer.
    5. Run training with error handling (OOM, interrupt, NaN).
    6. Save the final model and print a summary.
    """
    # ------------------------------------------------------------------
    # 1. Parse arguments and load configuration
    # ------------------------------------------------------------------
    args = parse_args()

    config_path = Path(args.config)
    config = load_config(config_path)
    config = apply_cli_overrides(config, args)

    # ------------------------------------------------------------------
    # 2. Set up logging, seeds, and device
    # ------------------------------------------------------------------
    seed: int = config.get("training", {}).get("seed", 42)
    set_seed(seed)

    device = get_device(preferred=args.device)

    # Derive experiment name
    model_type: str = config.get("model", {}).get("type", "pc_unet")
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_name: str = args.experiment_name or f"{model_type}_{timestamp}"

    # ------------------------------------------------------------------
    # 3. Create output directories
    # ------------------------------------------------------------------
    output_root = Path(args.output_dir).resolve()
    experiment_dir = output_root / experiment_name
    checkpoint_dir = experiment_dir / "checkpoints"
    log_dir = experiment_dir / "logs"
    figures_dir = experiment_dir / "figures"

    for d in (checkpoint_dir, log_dir, figures_dir):
        d.mkdir(parents=True, exist_ok=True)

    # Point the config's output section at the resolved directories so
    # that the Trainer writes checkpoints to the correct location.
    config.setdefault("output", {})["checkpoint_dir"] = str(checkpoint_dir)
    config["output"]["log_dir"] = str(log_dir)
    config["output"]["output_dir"] = str(output_root)

    # Set up logging (console + file)
    log_level = logging.DEBUG if args.debug else logging.INFO
    setup_logging(log_dir=log_dir, level=log_level)

    # Print startup banner
    print_banner(config, device, experiment_name)

    # ------------------------------------------------------------------
    # 4. Create ExperimentTracker and persist config
    # ------------------------------------------------------------------
    tracker = ExperimentTracker(experiment_dir=experiment_dir, config=config)
    logger.info("Experiment directory: %s", experiment_dir)

    # Also save the raw YAML for easy reproduction
    config_save_path = experiment_dir / "config.yaml"
    with open(config_save_path, "w", encoding="utf-8") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    logger.info("Saved config YAML to %s", config_save_path)

    # ------------------------------------------------------------------
    # 5. Load data
    # ------------------------------------------------------------------
    data_path_str: str = config.get("data", {}).get(
        "data_path", "data/raw/thermal_dataset.h5"
    )
    data_path = Path(data_path_str)
    if not data_path.is_absolute():
        data_path = PROJECT_ROOT / data_path

    try:
        logger.info("Loading data from %s ...", data_path)
        loaders = create_data_loaders(
            h5_path=data_path,
            config=config,
            device=device,
        )
        train_loader = loaders["train"]
        val_loader = loaders["val"]
        logger.info(
            "Data loaded: %d train batches, %d val batches",
            len(train_loader),
            len(val_loader),
        )
    except FileNotFoundError as exc:
        logger.error(
            "Data file not found: %s. "
            "Please generate data first with: python scripts/generate_data.py --config configs/generate.yaml",
            data_path,
        )
        logger.error("Original error: %s", exc)
        sys.exit(1)
    except Exception as exc:
        logger.error("Failed to load data: %s", exc, exc_info=True)
        sys.exit(1)

    # ------------------------------------------------------------------
    # 6. Create model
    # ------------------------------------------------------------------
    model = create_model(config, device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(
        "Model: %s | Parameters: %s (%s)",
        model_type,
        f"{n_params:,}",
        _format_param_count(n_params),
    )

    # ------------------------------------------------------------------
    # 7. Create Trainer
    # ------------------------------------------------------------------
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        device=device,
        training_logger=tracker.training_logger,
    )

    # ------------------------------------------------------------------
    # 8. Optionally resume from checkpoint
    # ------------------------------------------------------------------
    if args.resume is not None:
        resume_path = Path(args.resume)
        if not resume_path.is_absolute():
            resume_path = PROJECT_ROOT / resume_path
        logger.info("Resuming from checkpoint: %s", resume_path)
        try:
            resumed_epoch = trainer.load_checkpoint(resume_path)
            logger.info("Resumed training; will continue from epoch %d", resumed_epoch + 1)
        except FileNotFoundError:
            logger.error("Checkpoint file not found: %s", resume_path)
            sys.exit(1)
        except Exception as exc:
            logger.error("Failed to load checkpoint: %s", exc, exc_info=True)
            sys.exit(1)

    # ------------------------------------------------------------------
    # 9. Train with robust error handling
    # ------------------------------------------------------------------
    training_start = time.time()
    summary: Optional[Dict[str, Any]] = None

    try:
        summary = trainer.train()

    except KeyboardInterrupt:
        logger.info("")
        logger.info("=" * 70)
        logger.info("Training interrupted by user (KeyboardInterrupt).")
        logger.info("Saving interrupted checkpoint ...")
        interrupted_path = checkpoint_dir / "interrupted.pt"
        trainer.save_checkpoint(interrupted_path, epoch=0, metrics={"interrupted": True})
        logger.info("Interrupted checkpoint saved to %s", interrupted_path)
        logger.info("You can resume later with: --resume %s", interrupted_path)
        logger.info("=" * 70)
        sys.exit(0)

    except RuntimeError as exc:
        if "out of memory" in str(exc).lower():
            logger.warning("=" * 70)
            logger.warning("GPU out-of-memory error detected!")
            logger.warning("=" * 70)

            # Attempt to recover: free memory and retry with halved batch size
            if device.type == "cuda":
                torch.cuda.empty_cache()

            current_bs: int = config.get("training", {}).get("batch_size", 16)
            new_bs = max(1, current_bs // 2)

            if new_bs == current_bs:
                logger.error("Batch size is already 1; cannot reduce further.")
                logger.error("Original error: %s", exc)
                sys.exit(1)

            logger.warning(
                "Retrying with halved batch size: %d -> %d", current_bs, new_bs
            )
            config["training"]["batch_size"] = new_bs

            # Rebuild data loaders with the smaller batch size
            try:
                loaders = create_data_loaders(
                    h5_path=data_path,
                    config=config,
                    device=device,
                )
                train_loader = loaders["train"]
                val_loader = loaders["val"]

                # Rebuild the trainer with new loaders
                model = create_model(config, device)
                trainer = Trainer(
                    model=model,
                    train_loader=train_loader,
                    val_loader=val_loader,
                    config=config,
                    device=device,
                    training_logger=tracker.training_logger,
                )

                logger.info("Retry training with batch_size=%d ...", new_bs)
                summary = trainer.train()

            except RuntimeError as retry_exc:
                logger.error(
                    "Training failed again after halving batch size: %s",
                    retry_exc,
                )
                debug_ckpt = checkpoint_dir / "debug_oom.pt"
                trainer.save_checkpoint(debug_ckpt, epoch=0, metrics={"oom": True})
                logger.error("Debug checkpoint saved to %s", debug_ckpt)
                raise
        else:
            # Non-OOM RuntimeError -- save debug checkpoint and re-raise
            logger.error("RuntimeError during training: %s", exc, exc_info=True)
            debug_ckpt = checkpoint_dir / "debug_error.pt"
            try:
                trainer.save_checkpoint(debug_ckpt, epoch=0, metrics={"error": str(exc)})
                logger.error("Debug checkpoint saved to %s", debug_ckpt)
            except Exception:
                logger.error("Failed to save debug checkpoint.")
            raise

    except Exception as exc:
        logger.error("Unexpected error during training: %s", exc, exc_info=True)
        debug_ckpt = checkpoint_dir / "debug_error.pt"
        try:
            trainer.save_checkpoint(debug_ckpt, epoch=0, metrics={"error": str(exc)})
            logger.error("Debug checkpoint saved to %s", debug_ckpt)
        except Exception:
            logger.error("Failed to save debug checkpoint.")
        raise

    # ------------------------------------------------------------------
    # 10. Post-training: save final model and print summary
    # ------------------------------------------------------------------
    training_elapsed = time.time() - training_start

    # Save a standalone model file (state_dict only, for easy inference loading)
    final_model_path = experiment_dir / "final_model.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model_config": config.get("model", {}),
            "model_type": model_type,
        },
        final_model_path,
    )
    logger.info("Final model saved to %s", final_model_path)

    # Save results to the tracker
    if summary is not None:
        summary["training_time_seconds"] = training_elapsed
        summary["experiment_name"] = experiment_name
        summary["model_type"] = model_type
        summary["n_parameters"] = n_params
        tracker.save_results(summary)

    # Print final summary
    logger.info("")
    logger.info("=" * 70)
    logger.info("  Training Complete")
    logger.info("=" * 70)
    logger.info("  Experiment   : %s", experiment_name)
    logger.info("  Model        : %s (%s params)", model_type, _format_param_count(n_params))
    logger.info("  Device       : %s", device)
    if summary is not None:
        logger.info("  Best val loss: %.6f", summary.get("best_val_loss", float("nan")))
        logger.info("  Total epochs : %d", summary.get("total_epochs_run", 0))
        logger.info("  Final phase  : %d", summary.get("final_phase", 0))
    hours, remainder = divmod(training_elapsed, 3600)
    minutes, seconds = divmod(remainder, 60)
    logger.info(
        "  Wall time    : %dh %dm %.1fs (%.1f s total)",
        int(hours), int(minutes), seconds, training_elapsed,
    )
    logger.info("  Outputs      : %s", experiment_dir)
    logger.info("=" * 70)


if __name__ == "__main__":
    main()

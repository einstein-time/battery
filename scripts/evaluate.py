#!/usr/bin/env python
"""Evaluate trained thermal surrogate model."""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_utils.dataset import ThermalDataset, create_data_splits
from src.evaluation.visualization import (
    plot_temperature_comparison,
    plot_training_curves,
)
from src.models import PCUNet, SimpleCNN
from src.physics.validation import compute_error_metrics
from src.utils.device import get_device
from src.utils.logging import setup_logging


def main():
    parser = argparse.ArgumentParser(description="Evaluate thermal surrogate model")
    parser.add_argument(
        "--model",
        type=Path,
        required=True,
        help="Path to trained model checkpoint",
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("data/raw/thermal_dataset.h5"),
        help="Path to dataset",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("outputs/evaluation"),
        help="Output directory for results",
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=10,
        help="Number of samples to visualize",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose logging",
    )

    args = parser.parse_args()

    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    setup_logging(level=log_level, verbose=args.verbose)

    logger = logging.getLogger(__name__)
    logger.info("=" * 60)
    logger.info("THERMAL SURROGATE EVALUATION")
    logger.info("=" * 60)

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Setup device
    device = get_device(verbose=True)

    # Load checkpoint
    logger.info(f"Loading model from {args.model}")
    checkpoint = torch.load(args.model, map_location=device)

    # Infer model architecture from checkpoint
    # (In a real project, you'd save this in the checkpoint)
    # For now, assume PC-UNet
    model = PCUNet(
        in_channels=9,
        out_channels=1,
        base_features=32,
        num_levels=4,
        dropout_rate=0.1,
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    logger.info(f"Model has {model.count_parameters():,} parameters")
    logger.info(f"Best validation loss: {checkpoint.get('best_val_loss', 'N/A')}")

    # Plot training curves if available
    if "history" in checkpoint:
        logger.info("Plotting training curves")
        plot_training_curves(
            checkpoint["history"],
            save_path=args.output_dir / "training_curves.png",
        )

    # Load test dataset
    logger.info(f"Loading dataset from {args.data}")

    if not args.data.exists():
        logger.error(f"Dataset not found: {args.data}")
        sys.exit(1)

    _, _, test_indices = create_data_splits(
        args.data,
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15,
        seed=42,
    )

    test_dataset = ThermalDataset(args.data, trajectory_indices=test_indices)
    test_loader = DataLoader(
        test_dataset,
        batch_size=16,
        shuffle=False,
        num_workers=0,
    )

    # Evaluate on test set
    logger.info("Evaluating on test set...")

    all_errors = []
    all_l2_errors = []

    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            inputs = batch["input"].to(device)
            targets = batch["target"].to(device)
            physics = batch["physics"].to(device)

            # Predict
            preds = model(inputs, physics)

            # Compute errors
            preds_np = preds.cpu().numpy()
            targets_np = targets.cpu().numpy()

            for j in range(preds.size(0)):
                metrics = compute_error_metrics(preds_np[j, 0], targets_np[j, 0])
                all_errors.append(metrics)
                all_l2_errors.append(metrics["l2_rel"])

            # Visualize first few samples
            if i == 0 and args.n_samples > 0:
                n_vis = min(args.n_samples, preds.size(0))
                for j in range(n_vis):
                    plot_temperature_comparison(
                        T_true=targets_np[j, 0],
                        T_pred=preds_np[j, 0],
                        save_path=args.output_dir / f"comparison_{j:03d}.png",
                        title=f"Sample {j}",
                    )

    # Aggregate statistics
    mean_l2 = np.mean(all_l2_errors)
    std_l2 = np.std(all_l2_errors)
    max_l2 = np.max(all_l2_errors)

    mean_mae = np.mean([e["mae"] for e in all_errors])
    mean_rmse = np.mean([e["rmse"] for e in all_errors])
    mean_max_error = np.mean([e["max_abs"] for e in all_errors])

    logger.info("=" * 60)
    logger.info("TEST SET RESULTS")
    logger.info("=" * 60)
    logger.info(f"Relative L2 Error:  {mean_l2:.4f} ± {std_l2:.4f} (max: {max_l2:.4f})")
    logger.info(f"MAE:                {mean_mae:.4f} K")
    logger.info(f"RMSE:               {mean_rmse:.4f} K")
    logger.info(f"Mean Max Error:     {mean_max_error:.4f} K")
    logger.info("=" * 60)

    # Save results to file
    results_file = args.output_dir / "results.txt"
    with open(results_file, "w") as f:
        f.write("THERMAL SURROGATE EVALUATION RESULTS\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Model: {args.model}\n")
        f.write(f"Dataset: {args.data}\n")
        f.write(f"Test samples: {len(all_errors)}\n\n")
        f.write(f"Relative L2 Error:  {mean_l2:.4f} ± {std_l2:.4f} (max: {max_l2:.4f})\n")
        f.write(f"MAE:                {mean_mae:.4f} K\n")
        f.write(f"RMSE:               {mean_rmse:.4f} K\n")
        f.write(f"Mean Max Error:     {mean_max_error:.4f} K\n")

    logger.info(f"Results saved to {results_file}")
    logger.info(f"Visualizations saved to {args.output_dir}")


if __name__ == "__main__":
    main()

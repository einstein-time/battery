#!/usr/bin/env python
"""Train physics-informed thermal surrogate model."""

import argparse
import logging
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_utils.dataset import ThermalDataset, create_data_splits
from src.models import PCUNet, SimpleCNN
from src.training.losses import PhysicsInformedLoss
from src.training.trainer import Trainer
from src.utils.config import load_config
from src.utils.device import setup_device
from src.utils.logging import setup_logging


def main():
    parser = argparse.ArgumentParser(description="Train thermal surrogate model")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/train.yaml"),
        help="Path to configuration file",
    )
    parser.add_argument(
        "--resume",
        type=Path,
        default=None,
        help="Resume training from checkpoint",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Debug mode (small dataset, few epochs)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Override number of epochs from config",
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
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    setup_logging(
        log_file=log_dir / "train.log",
        level=log_level,
        verbose=args.verbose,
    )

    logger = logging.getLogger(__name__)
    logger.info("=" * 60)
    logger.info("THERMAL SURROGATE TRAINING")
    logger.info("=" * 60)

    # Load configuration
    config = load_config(args.config)

    # Debug mode overrides
    if args.debug:
        logger.info("DEBUG MODE: Using small dataset and few epochs")
        config["training"]["epochs"] = args.epochs if args.epochs else 5
        config["training"]["batch_size"] = 4
        config["data"]["num_workers"] = 0

    if args.epochs:
        config["training"]["epochs"] = args.epochs

    # Setup device
    device = setup_device(seed=config["training"]["seed"])

    # Create datasets
    data_path = Path(config["data"]["data_path"])
    if not data_path.exists():
        logger.error(f"Dataset not found: {data_path}")
        logger.error("Please run 'python scripts/generate_data.py' first")
        sys.exit(1)

    logger.info(f"Loading dataset from {data_path}")

    train_indices, val_indices, test_indices = create_data_splits(
        data_path,
        train_ratio=config["data"]["train_ratio"],
        val_ratio=config["data"]["val_ratio"],
        test_ratio=config["data"]["test_ratio"],
        seed=config["training"]["seed"],
    )

    train_dataset = ThermalDataset(data_path, trajectory_indices=train_indices)
    val_dataset = ThermalDataset(data_path, trajectory_indices=val_indices)

    # Create data loaders
    num_workers = 0 if sys.platform == "win32" else config["data"]["num_workers"]
    train_loader = DataLoader(
        train_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=True,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
    )

    # Create model
    model_type = config["model"]["type"]
    logger.info(f"Creating model: {model_type}")

    if model_type == "pc_unet":
        model = PCUNet(
            in_channels=config["model"]["in_channels"],
            out_channels=config["model"]["out_channels"],
            base_features=config["model"]["base_features"],
            num_levels=config["model"]["num_levels"],
            dropout_rate=config["model"]["dropout_rate"],
        )
    elif model_type == "simple_cnn":
        model = SimpleCNN(
            in_channels=config["model"]["in_channels"],
            out_channels=config["model"]["out_channels"],
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    model = model.to(device)
    logger.info(f"Model has {model.count_parameters():,} trainable parameters")

    # Create loss function
    loss_fn = PhysicsInformedLoss(
        lambda_data=config["loss"]["lambda_data"],
        lambda_pde=config["loss"]["lambda_pde"],
        lambda_bc=config["loss"]["lambda_bc"],
        lambda_consistency=config["loss"]["lambda_consistency"],
        dx=config["physics"]["dx"],
        dy=config["physics"]["dy"],
        dt=config["data"]["dt"],
        rho=config["physics"]["rho"],
        cp=config["physics"]["cp"],
        T_amb=config["physics"]["T_amb"],
    )

    # Create optimizer and scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["training"]["learning_rate"],
        weight_decay=config["training"]["weight_decay"],
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config["training"]["epochs"],
        eta_min=config["training"]["scheduler"]["eta_min"],
    )

    # Create trainer
    use_mixed_precision = config["training"]["mixed_precision"] and device.type == "cuda"

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        loss_fn=loss_fn,
        device=device,
        scheduler=scheduler,
        grad_clip=config["training"]["grad_clip"],
        mixed_precision=use_mixed_precision,
    )

    # Resume from checkpoint if requested
    if args.resume:
        logger.info(f"Resuming from checkpoint: {args.resume}")
        trainer.load_checkpoint(args.resume)

    # Create checkpoint directory
    checkpoint_dir = Path(config["output"]["checkpoint_dir"])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Train
    phase_epochs = (
        config["training"]["phases"]["pretrain_epochs"],
        config["training"]["phases"]["physics_epochs"],
        config["training"]["phases"]["finetune_epochs"],
    )

    try:
        trainer.train(
            train_loader=train_loader,
            val_loader=val_loader,
            num_epochs=config["training"]["epochs"],
            phase_epochs=phase_epochs,
            checkpoint_dir=checkpoint_dir,
            early_stopping_patience=config["training"]["early_stopping_patience"],
        )

        logger.info("Training completed successfully!")

        # Save final model
        final_path = checkpoint_dir / "final_model.pt"
        trainer.save_checkpoint(final_path, config["training"]["epochs"], trainer.best_val_loss)
        logger.info(f"Saved final model to {final_path}")

    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Training failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()

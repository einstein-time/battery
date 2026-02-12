"""Trainer class for physics-informed thermal surrogate models."""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.training.losses import PhysicsInformedLoss

logger = logging.getLogger(__name__)


class Trainer:
    """Trainer for physics-informed thermal surrogate models.

    Implements 3-phase training:
      - Phase 1: Data-only pretraining
      - Phase 2: Physics-informed training (data + PDE + BC losses)
      - Phase 3: Consistency fine-tuning (all losses)

    Args:
        model: Neural network model.
        optimizer: PyTorch optimizer.
        loss_fn: Physics-informed loss function.
        device: Device to train on.
        scheduler: Optional learning rate scheduler.
        grad_clip: Gradient clipping value (default 1.0).
        mixed_precision: Use automatic mixed precision (AMP) on CUDA.
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        loss_fn: PhysicsInformedLoss,
        device: torch.device,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        grad_clip: float = 1.0,
        mixed_precision: bool = False,
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.device = device
        self.scheduler = scheduler
        self.grad_clip = grad_clip
        self.mixed_precision = mixed_precision and device.type == "cuda"

        # GradScaler for mixed precision
        self.scaler: Optional[torch.cuda.amp.GradScaler] = None
        if self.mixed_precision:
            self.scaler = torch.cuda.amp.GradScaler()

        # Training history
        self.history: Dict[str, list] = {
            "train_loss": [],
            "val_loss": [],
            "epoch_time": [],
        }

        # Best model tracking
        self.best_val_loss = float("inf")
        self.epochs_without_improvement = 0

    def train_epoch(
        self,
        train_loader: DataLoader,
        epoch: int,
        phase: int,
    ) -> Dict[str, float]:
        """Train for one epoch.

        Args:
            train_loader: DataLoader for training data.
            epoch: Current epoch number.
            phase: Training phase (1, 2, or 3).

        Returns:
            Dictionary of average metrics for this epoch.
        """
        self.model.train()
        epoch_metrics = {"loss": 0.0, "data": 0.0, "pde": 0.0, "bc": 0.0}
        n_batches = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch} [Phase {phase}]")

        for batch in pbar:
            # Move batch to device
            inputs = batch["input"].to(self.device)       # (B, 9, H, W)
            targets = batch["target"].to(self.device)     # (B, 1, H, W)
            T_input = inputs[:, 0:1, :, :]                # Temperature channel
            k = inputs[:, 4:5, :, :]                      # Conductivity channel
            q = inputs[:, 5:6, :, :]                      # Heat generation channel
            h = inputs[:, 6:7, :, :]                      # Convection channel

            # Extract physics parameters if available
            physics = None
            if "physics" in batch:
                physics = batch["physics"].to(self.device)

            # Forward pass
            if self.mixed_precision:
                with torch.cuda.amp.autocast():
                    pred = self.model(inputs, physics)
                    loss, breakdown = self.loss_fn(pred, targets, T_input, k, q, h, phase)
            else:
                pred = self.model(inputs, physics)
                loss, breakdown = self.loss_fn(pred, targets, T_input, k, q, h, phase)

            # Backward pass
            self.optimizer.zero_grad()

            if self.mixed_precision:
                self.scaler.scale(loss).backward()
                if self.grad_clip > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                if self.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                self.optimizer.step()

            # Accumulate metrics
            epoch_metrics["loss"] += loss.item()
            for key, value in breakdown.items():
                if key not in epoch_metrics:
                    epoch_metrics[key] = 0.0
                epoch_metrics[key] += value

            n_batches += 1

            # Update progress bar
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        # Average metrics
        for key in epoch_metrics:
            epoch_metrics[key] /= n_batches

        return epoch_metrics

    def validate(
        self,
        val_loader: DataLoader,
        phase: int,
    ) -> Dict[str, float]:
        """Validate the model.

        Args:
            val_loader: DataLoader for validation data.
            phase: Training phase.

        Returns:
            Dictionary of validation metrics.
        """
        self.model.eval()
        val_metrics = {"loss": 0.0, "data": 0.0}
        n_batches = 0

        with torch.no_grad():
            for batch in val_loader:
                inputs = batch["input"].to(self.device)
                targets = batch["target"].to(self.device)
                T_input = inputs[:, 0:1, :, :]
                k = inputs[:, 4:5, :, :]
                q = inputs[:, 5:6, :, :]
                h = inputs[:, 6:7, :, :]

                physics = None
                if "physics" in batch:
                    physics = batch["physics"].to(self.device)

                pred = self.model(inputs, physics)
                loss, breakdown = self.loss_fn(pred, targets, T_input, k, q, h, phase)

                val_metrics["loss"] += loss.item()
                for key, value in breakdown.items():
                    if key not in val_metrics:
                        val_metrics[key] = 0.0
                    val_metrics[key] += value

                n_batches += 1

        # Average metrics
        for key in val_metrics:
            val_metrics[key] /= n_batches

        return val_metrics

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        num_epochs: int,
        phase_epochs: Tuple[int, int, int] = (50, 100, 50),
        checkpoint_dir: Optional[Path] = None,
        early_stopping_patience: int = 20,
    ) -> None:
        """Run full training loop with 3-phase curriculum.

        Args:
            train_loader: Training data loader.
            val_loader: Validation data loader.
            num_epochs: Total number of epochs.
            phase_epochs: Tuple of (pretrain, physics, finetune) epoch counts.
            checkpoint_dir: Directory to save checkpoints.
            early_stopping_patience: Patience for early stopping.
        """
        pretrain_epochs, physics_epochs, finetune_epochs = phase_epochs

        if checkpoint_dir:
            checkpoint_dir = Path(checkpoint_dir)
            checkpoint_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Starting training for {num_epochs} epochs")
        logger.info(f"Phase epochs: pretrain={pretrain_epochs}, physics={physics_epochs}, finetune={finetune_epochs}")

        for epoch in range(1, num_epochs + 1):
            epoch_start = time.time()

            # Determine training phase
            if epoch <= pretrain_epochs:
                phase = 1
            elif epoch <= pretrain_epochs + physics_epochs:
                phase = 2
            else:
                phase = 3

            # Train and validate
            train_metrics = self.train_epoch(train_loader, epoch, phase)
            val_metrics = self.validate(val_loader, phase)

            epoch_time = time.time() - epoch_start

            # Log metrics
            logger.info(
                f"Epoch {epoch}/{num_epochs} | Phase {phase} | "
                f"Train Loss: {train_metrics['loss']:.4f} | "
                f"Val Loss: {val_metrics['loss']:.4f} | "
                f"Time: {epoch_time:.1f}s"
            )

            # Update history
            self.history["train_loss"].append(train_metrics["loss"])
            self.history["val_loss"].append(val_metrics["loss"])
            self.history["epoch_time"].append(epoch_time)

            # Learning rate scheduling
            if self.scheduler:
                self.scheduler.step()

            # Save best model
            if val_metrics["loss"] < self.best_val_loss:
                self.best_val_loss = val_metrics["loss"]
                self.epochs_without_improvement = 0

                if checkpoint_dir:
                    best_path = checkpoint_dir / "best_model.pt"
                    self.save_checkpoint(best_path, epoch, val_metrics["loss"])
                    logger.info(f"Saved best model to {best_path}")
            else:
                self.epochs_without_improvement += 1

            # Periodic checkpoint
            if checkpoint_dir and epoch % 10 == 0:
                ckpt_path = checkpoint_dir / f"checkpoint_epoch_{epoch}.pt"
                self.save_checkpoint(ckpt_path, epoch, val_metrics["loss"])

            # Early stopping
            if self.epochs_without_improvement >= early_stopping_patience:
                logger.info(f"Early stopping triggered after {epoch} epochs")
                break

        logger.info("Training completed!")

    def save_checkpoint(
        self,
        path: Path,
        epoch: int,
        val_loss: float,
    ) -> None:
        """Save model checkpoint.

        Args:
            path: Path to save checkpoint.
            epoch: Current epoch number.
            val_loss: Validation loss.
        """
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "val_loss": val_loss,
            "best_val_loss": self.best_val_loss,
            "history": self.history,
        }

        if self.scheduler:
            checkpoint["scheduler_state_dict"] = self.scheduler.state_dict()

        torch.save(checkpoint, path)

    def load_checkpoint(self, path: Path) -> None:
        """Load model checkpoint.

        Args:
            path: Path to checkpoint file.
        """
        checkpoint = torch.load(path, map_location=self.device)

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        if self.scheduler and "scheduler_state_dict" in checkpoint:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        self.best_val_loss = checkpoint.get("best_val_loss", float("inf"))
        self.history = checkpoint.get("history", {})

        logger.info(f"Loaded checkpoint from {path}")

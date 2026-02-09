"""Three-phase training loop for the 2D battery thermal surrogate model.

Orchestrates a staged training strategy:

* **Phase 1 -- Data pre-training** (``pretrain_epochs``):  Pure data-driven
  learning with MSE loss and a warm learning rate.
* **Phase 2 -- Physics-informed** (``physics_epochs``):  Adds PDE residual
  and boundary-condition losses alongside data MSE.
* **Phase 3 -- Consistency fine-tuning** (``finetune_epochs``):  Adds
  multi-step temporal consistency loss for stable autoregressive rollout.

Supports mixed-precision training on CUDA, gradient clipping, early
stopping, GradNorm adaptive loss weighting, and checkpoint save / resume.

Typical usage::

    trainer = Trainer(model, train_loader, val_loader, config, device, logger)
    trainer.train()
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from src.models.base_model import BaseThermalModel
from src.training.gradnorm import GradNorm
from src.training.losses import PhysicsInformedLoss
from src.utils.logging import TrainingLogger

logger = logging.getLogger(__name__)


class Trainer:
    """Three-phase trainer for physics-informed thermal surrogate models.

    Attributes:
        model: The neural-network model to train.
        train_loader: DataLoader yielding training batches.
        val_loader: DataLoader yielding validation batches.
        config: Full configuration dictionary (see ``src.utils.config``).
        device: Torch device (CPU / CUDA / MPS).
        training_logger: Logger for per-epoch metric recording.
        optimizer: AdamW optimiser instance.
        scheduler: Cosine-annealing LR scheduler.
        loss_fn: :class:`PhysicsInformedLoss` instance.
        grad_norm: Optional :class:`GradNorm` instance for adaptive
            loss balancing (enabled when ``config["loss"]["use_gradnorm"]``
            is ``True``).
        best_val_loss: Best validation loss seen so far (for early stopping
            and best-model saving).
        use_amp: Whether mixed-precision training is active.
    """

    def __init__(
        self,
        model: BaseThermalModel,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: Dict[str, Any],
        device: torch.device,
        training_logger: TrainingLogger,
    ) -> None:
        """Initialise the Trainer.

        Args:
            model: A :class:`BaseThermalModel` subclass already moved to
                ``device``.
            train_loader: Training data loader.  Each batch is a dict with
                keys ``'input'`` (B, 9, H, W), ``'target'`` (B, 1, H, W),
                and optionally ``'k_field'``, ``'q_field'``, ``'h_field'``
                (each B, 1, H, W).
            val_loader: Validation data loader (same batch format).
            config: Hierarchical configuration dictionary.
            device: Compute device.
            training_logger: :class:`TrainingLogger` for metric logging.
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device
        self.training_logger = training_logger

        # ---- Extract sub-configs ------------------------------------
        self._train_cfg = config.get("training", {})
        self._loss_cfg = config.get("loss", {})
        self._physics_cfg = config.get("physics", {})
        self._output_cfg = config.get("output", {})
        self._phase_cfg = self._train_cfg.get("phases", {})

        # ---- Phase boundaries (cumulative epoch counts) -------------
        self.pretrain_epochs: int = self._phase_cfg.get("pretrain_epochs", 50)
        self.physics_epochs: int = self._phase_cfg.get("physics_epochs", 100)
        self.finetune_epochs: int = self._phase_cfg.get("finetune_epochs", 50)
        self.total_epochs: int = (
            self.pretrain_epochs + self.physics_epochs + self.finetune_epochs
        )

        # ---- Loss function ------------------------------------------
        self.loss_fn = PhysicsInformedLoss(
            lambda_data=self._loss_cfg.get("lambda_data", 1.0),
            lambda_pde=self._loss_cfg.get("lambda_pde", 0.1),
            lambda_bc=self._loss_cfg.get("lambda_bc", 0.1),
            lambda_consistency=self._loss_cfg.get("lambda_consistency", 0.05),
            dx=self._physics_cfg.get("dx", 0.001),
            dy=self._physics_cfg.get("dy", 0.001),
            dt=config.get("data", {}).get("dt", 0.001),
            rho=self._physics_cfg.get("rho", 2500.0),
            cp=self._physics_cfg.get("cp", 700.0),
            T_amb=self._physics_cfg.get("T_amb", 298.15),
        )

        # ---- GradNorm (optional) ------------------------------------
        self.use_gradnorm: bool = self._loss_cfg.get("use_gradnorm", True)
        if self.use_gradnorm:
            self.grad_norm = GradNorm(
                n_tasks=4,
                alpha=self._loss_cfg.get("gradnorm_alpha", 1.5),
                lr=0.025,
            ).to(device)
        else:
            self.grad_norm: Optional[GradNorm] = None

        # ---- Optimiser and scheduler --------------------------------
        self.optimizer = self.setup_optimizer()
        self.scheduler = self.setup_scheduler()

        # ---- Training state -----------------------------------------
        self.grad_clip: float = self._train_cfg.get("grad_clip", 1.0)
        self.patience: int = self._train_cfg.get("early_stopping_patience", 20)
        self.save_every: int = self._output_cfg.get("save_every", 10)
        self.checkpoint_dir = Path(
            self._output_cfg.get("checkpoint_dir", "checkpoints")
        )
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.best_val_loss: float = float("inf")
        self._epochs_without_improvement: int = 0
        self._start_epoch: int = 0

        # ---- Mixed precision ----------------------------------------
        self.use_amp: bool = device.type == "cuda"
        self._scaler: Optional[GradScaler] = (
            GradScaler() if self.use_amp else None
        )

    # ------------------------------------------------------------------
    # Optimiser / Scheduler factories
    # ------------------------------------------------------------------

    def setup_optimizer(self) -> AdamW:
        """Create an AdamW optimiser for the model parameters.

        The learning rate is set to the *phase-1* value; it will be
        adjusted at phase boundaries by :meth:`_adjust_lr_for_phase`.

        Returns:
            Configured :class:`AdamW` optimiser.
        """
        lr = self._train_cfg.get("learning_rate", 1e-3)
        weight_decay = self._train_cfg.get("weight_decay", 1e-5)
        optimizer = AdamW(
            self.model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
        )
        logger.info("AdamW optimiser created (lr=%.2e, wd=%.2e)", lr, weight_decay)
        return optimizer

    def setup_scheduler(self) -> CosineAnnealingLR:
        """Create a cosine-annealing learning-rate scheduler.

        Returns:
            Configured :class:`CosineAnnealingLR` scheduler.
        """
        sched_cfg = self._train_cfg.get("scheduler", {})
        T_max = sched_cfg.get("T_max", self.total_epochs)
        eta_min = sched_cfg.get("eta_min", 1e-6)
        scheduler = CosineAnnealingLR(
            self.optimizer, T_max=T_max, eta_min=eta_min
        )
        logger.info(
            "CosineAnnealingLR scheduler created (T_max=%d, eta_min=%.2e)",
            T_max,
            eta_min,
        )
        return scheduler

    # ------------------------------------------------------------------
    # Phase management
    # ------------------------------------------------------------------

    def get_current_phase(self, epoch: int) -> int:
        """Determine the training phase for a given epoch number.

        Args:
            epoch: Zero-based epoch index.

        Returns:
            Phase number (1, 2, or 3).
        """
        if epoch < self.pretrain_epochs:
            return 1
        elif epoch < self.pretrain_epochs + self.physics_epochs:
            return 2
        else:
            return 3

    def _get_phase_lr(self, phase: int) -> float:
        """Return the base learning rate for a given phase.

        Args:
            phase: Training phase (1, 2, or 3).

        Returns:
            Learning rate for the phase.
        """
        phase_lrs = {1: 1e-3, 2: 5e-4, 3: 1e-4}
        return phase_lrs.get(phase, 1e-4)

    def _adjust_lr_for_phase(self, phase: int) -> None:
        """Set the optimiser learning rate to match the current phase.

        Also resets the GradNorm state when transitioning between phases
        (since the loss landscape changes significantly).

        Args:
            phase: Current training phase.
        """
        new_lr = self._get_phase_lr(phase)
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = new_lr
        logger.info("Phase %d: learning rate set to %.2e", phase, new_lr)

        # Reset GradNorm baselines at phase boundaries
        if self.grad_norm is not None:
            self.grad_norm.reset()

    # ------------------------------------------------------------------
    # Main training loop
    # ------------------------------------------------------------------

    def train(self) -> Dict[str, Any]:
        """Run the full three-phase training loop.

        Iterates over all epochs across phases 1-3, validates after each
        epoch, saves periodic and best checkpoints, and honours early
        stopping.

        Returns:
            Dictionary of final training summary metrics including
            ``best_val_loss``, ``total_epochs_run``, and ``final_phase``.
        """
        logger.info(
            "Starting training: %d total epochs "
            "(Phase 1: %d, Phase 2: %d, Phase 3: %d)",
            self.total_epochs,
            self.pretrain_epochs,
            self.physics_epochs,
            self.finetune_epochs,
        )
        logger.info("Model parameters: %s", f"{self.model.count_parameters():,}")
        logger.info("Device: %s | AMP: %s", self.device, self.use_amp)

        prev_phase: int = 0
        final_epoch: int = self._start_epoch

        try:
            for epoch in range(self._start_epoch, self.total_epochs):
                epoch_start = time.time()
                phase = self.get_current_phase(epoch)

                # Handle phase transitions
                if phase != prev_phase:
                    logger.info(
                        "=== Entering Phase %d at epoch %d ===", phase, epoch
                    )
                    self._adjust_lr_for_phase(phase)
                    prev_phase = phase

                # Train and validate
                train_metrics = self.train_epoch(epoch, phase)
                val_loss = self.validate(epoch)

                # Scheduler step
                self.scheduler.step()

                # Compile epoch metrics
                epoch_time = time.time() - epoch_start
                metrics: Dict[str, float] = {
                    "phase": float(phase),
                    "lr": self.optimizer.param_groups[0]["lr"],
                    "epoch_time": epoch_time,
                    "val_loss": val_loss,
                    **{f"train_{k}": v for k, v in train_metrics.items()},
                }
                self.training_logger.log_epoch(epoch, metrics)

                # --- Best-model tracking and early stopping --------------
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self._epochs_without_improvement = 0
                    self.save_checkpoint(
                        self.checkpoint_dir / "best_model.pt",
                        epoch,
                        metrics,
                    )
                    logger.info(
                        "New best val_loss=%.6f at epoch %d", val_loss, epoch
                    )
                else:
                    self._epochs_without_improvement += 1

                # Periodic checkpoint
                if (epoch + 1) % self.save_every == 0:
                    self.save_checkpoint(
                        self.checkpoint_dir / f"checkpoint_epoch_{epoch:04d}.pt",
                        epoch,
                        metrics,
                    )

                # Early stopping (per-phase patience)
                if self._epochs_without_improvement >= self.patience:
                    logger.info(
                        "Early stopping triggered at epoch %d "
                        "(no improvement for %d epochs)",
                        epoch,
                        self.patience,
                    )
                    # If not in the final phase, advance to the next phase
                    if phase < 3:
                        remaining_in_phase = self._epochs_remaining_in_phase(
                            epoch, phase
                        )
                        logger.info(
                            "Skipping remaining %d epochs in phase %d",
                            remaining_in_phase,
                            phase,
                        )
                        # Fast-forward epoch counter to next phase boundary
                        # This is handled by resetting patience and continuing
                        self._epochs_without_improvement = 0
                        self._start_epoch = self._phase_end_epoch(phase)
                        # Jump to next phase
                        final_epoch = epoch
                        continue
                    else:
                        final_epoch = epoch
                        break

                final_epoch = epoch

        except KeyboardInterrupt:
            logger.info(
                "Training interrupted by user at epoch %d. "
                "Saving emergency checkpoint.",
                final_epoch,
            )
            self.save_checkpoint(
                self.checkpoint_dir / "interrupted.pt",
                final_epoch,
                {"interrupted": True},
            )

        # Final checkpoint
        self.save_checkpoint(
            self.checkpoint_dir / "final_model.pt",
            final_epoch,
            {"best_val_loss": self.best_val_loss},
        )

        summary = {
            "best_val_loss": self.best_val_loss,
            "total_epochs_run": final_epoch + 1,
            "final_phase": self.get_current_phase(final_epoch),
        }
        logger.info("Training complete. Summary: %s", summary)
        return summary

    # ------------------------------------------------------------------
    # Single epoch
    # ------------------------------------------------------------------

    def train_epoch(self, epoch: int, phase: int) -> Dict[str, float]:
        """Execute one training epoch.

        Args:
            epoch: Current epoch index (zero-based).
            phase: Current training phase (1, 2, or 3).

        Returns:
            Dictionary of mean training loss values for the epoch.
        """
        self.model.train()
        running_losses: Dict[str, float] = {}
        n_batches: int = 0

        for batch in self.train_loader:
            batch_losses = self._train_step(batch, phase)

            for key, val in batch_losses.items():
                running_losses[key] = running_losses.get(key, 0.0) + val
            n_batches += 1

        # Average over batches
        avg_losses = {k: v / max(n_batches, 1) for k, v in running_losses.items()}
        return avg_losses

    def _train_step(
        self,
        batch: Dict[str, torch.Tensor],
        phase: int,
    ) -> Dict[str, float]:
        """Execute a single training step on one batch.

        Args:
            batch: Batch dictionary with at least ``'input'`` and
                ``'target'`` keys.
            phase: Current training phase.

        Returns:
            Dictionary of loss values for this batch.
        """
        # Move tensors to device
        x = batch["input"].to(self.device)           # (B, 9, H, W)
        target = batch["target"].to(self.device)      # (B, 1, H, W)

        # Extract sub-fields
        T_input = x[:, 0:1, :, :]                     # (B, 1, H, W)
        params = x[:, 1:, :, :]                        # (B, 8, H, W)

        # Optional physics fields from the batch
        k_field = (
            batch["k_field"].to(self.device)
            if "k_field" in batch else x[:, 4:5, :, :]
        )
        q_field = (
            batch["q_field"].to(self.device)
            if "q_field" in batch else x[:, 5:6, :, :]
        )
        h_field = (
            batch["h_field"].to(self.device)
            if "h_field" in batch else x[:, 6:7, :, :]
        )

        # Cache params on the model for consistency loss
        self.model._cached_params = params  # type: ignore[attr-defined]

        self.optimizer.zero_grad()

        if self.use_amp:
            with autocast(device_type="cuda"):
                pred = self.model(x)
                total_loss, loss_dict = self.loss_fn(
                    pred=pred,
                    target=target,
                    T_input=T_input,
                    k_field=k_field,
                    q_field=q_field,
                    h_field=h_field,
                    model=self.model,
                    phase=phase,
                )

            self._scaler.scale(total_loss).backward()  # type: ignore[union-attr]

            # Unscale before clipping
            self._scaler.unscale_(self.optimizer)
            nn.utils.clip_grad_norm_(
                self.model.parameters(), max_norm=self.grad_clip
            )

            self._scaler.step(self.optimizer)
            self._scaler.update()  # type: ignore[union-attr]
        else:
            pred = self.model(x)
            total_loss, loss_dict = self.loss_fn(
                pred=pred,
                target=target,
                T_input=T_input,
                k_field=k_field,
                q_field=q_field,
                h_field=h_field,
                model=self.model,
                phase=phase,
            )

            total_loss.backward()
            nn.utils.clip_grad_norm_(
                self.model.parameters(), max_norm=self.grad_clip
            )
            self.optimizer.step()

        # GradNorm update (phase 2+ with multiple active losses)
        if (
            self.grad_norm is not None
            and phase >= 2
            and len(loss_dict) > 2  # more than just data_loss + total_loss
        ):
            # Collect individual unweighted losses that are active
            active_losses: List[torch.Tensor] = []
            if "data_loss" in loss_dict:
                active_losses.append(
                    self.loss_fn.data_loss(pred.detach(), target)
                )
            if "pde_loss" in loss_dict:
                active_losses.append(
                    self.loss_fn.pde_residual_loss(
                        pred.detach(), T_input, k_field, q_field
                    )
                )
            if "bc_loss" in loss_dict:
                active_losses.append(
                    self.loss_fn.boundary_loss(
                        pred.detach(), T_input, h_field
                    )
                )
            if "consistency_loss" in loss_dict:
                active_losses.append(
                    torch.tensor(
                        loss_dict["consistency_loss"],
                        device=self.device,
                    )
                )

            # Pad to n_tasks if fewer losses are active
            while len(active_losses) < self.grad_norm.n_tasks:
                active_losses.append(
                    torch.tensor(0.0, device=self.device)
                )

            # Find shared layer (first conv / linear in the model)
            shared_layer = self._find_shared_layer()
            if shared_layer is not None:
                try:
                    self.grad_norm.update(active_losses, shared_layer)
                except Exception as e:
                    logger.debug("GradNorm update skipped: %s", e)

        # Clean up cached params
        self.model._cached_params = None  # type: ignore[attr-defined]

        return loss_dict

    def _find_shared_layer(self) -> Optional[nn.Module]:
        """Locate a shared layer in the model for GradNorm gradient norms.

        Searches for the first ``Conv2d`` or ``Linear`` layer, which is
        typically part of the shared encoder.

        Returns:
            An ``nn.Module`` suitable as the ``shared_layer`` argument to
            :meth:`GradNorm.update`, or ``None`` if no candidate is found.
        """
        for module in self.model.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                return module
        return None

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    @torch.no_grad()
    def validate(self, epoch: int) -> float:
        """Run a validation pass and return the mean validation loss.

        Only the data-fidelity (MSE) loss is used for validation to
        provide a stable, phase-independent metric for early stopping
        and model selection.

        Args:
            epoch: Current epoch index (for logging purposes).

        Returns:
            Mean validation MSE loss (float).
        """
        self.model.eval()
        total_loss: float = 0.0
        n_batches: int = 0

        for batch in self.val_loader:
            x = batch["input"].to(self.device)
            target = batch["target"].to(self.device)

            if self.use_amp:
                with autocast(device_type="cuda"):
                    pred = self.model(x)
                    loss = self.loss_fn.data_loss(pred, target)
            else:
                pred = self.model(x)
                loss = self.loss_fn.data_loss(pred, target)

            total_loss += loss.item()
            n_batches += 1

        avg_loss = total_loss / max(n_batches, 1)
        return avg_loss

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------

    def save_checkpoint(
        self,
        path: Union[str, Path],
        epoch: int,
        metrics: Dict[str, Any],
    ) -> None:
        """Save a training checkpoint to disk.

        The checkpoint contains everything needed to resume training:
        model weights, optimiser state, scheduler state, GradNorm state
        (if applicable), configuration, epoch number, and metrics.

        Args:
            path: File path for the checkpoint (``.pt``).
            epoch: Current epoch number.
            metrics: Dictionary of metrics to store alongside the
                checkpoint.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        checkpoint: Dict[str, Any] = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "config": self.config,
            "best_val_loss": self.best_val_loss,
            "metrics": metrics,
        }

        if self.grad_norm is not None:
            checkpoint["gradnorm_state_dict"] = self.grad_norm.state_dict()

        if self._scaler is not None:
            checkpoint["scaler_state_dict"] = self._scaler.state_dict()

        torch.save(checkpoint, path)
        logger.info("Checkpoint saved to %s (epoch %d)", path, epoch)

    def load_checkpoint(self, path: Union[str, Path]) -> int:
        """Load a training checkpoint and restore all state.

        Args:
            path: Path to the checkpoint file.

        Returns:
            The epoch number stored in the checkpoint (training resumes
            from ``epoch + 1``).

        Raises:
            FileNotFoundError: If the checkpoint file does not exist.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {path}")

        checkpoint = torch.load(path, map_location=self.device, weights_only=False)

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        self.best_val_loss = checkpoint.get("best_val_loss", float("inf"))

        if (
            self.grad_norm is not None
            and "gradnorm_state_dict" in checkpoint
        ):
            self.grad_norm.load_state_dict(checkpoint["gradnorm_state_dict"])

        if self._scaler is not None and "scaler_state_dict" in checkpoint:
            self._scaler.load_state_dict(checkpoint["scaler_state_dict"])

        epoch = checkpoint["epoch"]
        self._start_epoch = epoch + 1
        logger.info(
            "Resumed from checkpoint %s (epoch %d, best_val=%.6f)",
            path,
            epoch,
            self.best_val_loss,
        )
        return epoch

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _epochs_remaining_in_phase(self, current_epoch: int, phase: int) -> int:
        """Count how many epochs remain in the given phase.

        Args:
            current_epoch: Current epoch index.
            phase: Current phase number.

        Returns:
            Number of remaining epochs in the phase.
        """
        end = self._phase_end_epoch(phase)
        return max(0, end - current_epoch - 1)

    def _phase_end_epoch(self, phase: int) -> int:
        """Return the first epoch index that belongs to the *next* phase.

        Args:
            phase: Phase number (1, 2, or 3).

        Returns:
            Epoch index marking the start of the subsequent phase.
        """
        if phase == 1:
            return self.pretrain_epochs
        elif phase == 2:
            return self.pretrain_epochs + self.physics_epochs
        else:
            return self.total_epochs

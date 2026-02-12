"""Visualization utilities for thermal simulation results."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

logger = logging.getLogger(__name__)

# Set plotting style
sns.set_style("whitegrid")
plt.rcParams["figure.dpi"] = 120


def plot_temperature_comparison(
    T_true: np.ndarray,
    T_pred: np.ndarray,
    save_path: Optional[Path] = None,
    title: str = "Temperature Comparison",
) -> None:
    """Plot side-by-side comparison of true vs predicted temperature fields.

    Args:
        T_true: Ground truth temperature field of shape ``(H, W)``.
        T_pred: Predicted temperature field of shape ``(H, W)``.
        save_path: Optional path to save the figure.
        title: Figure title.
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    vmin = min(T_true.min(), T_pred.min())
    vmax = max(T_true.max(), T_pred.max())

    # Ground truth
    im0 = axes[0].imshow(T_true, cmap="hot", origin="lower", vmin=vmin, vmax=vmax)
    axes[0].set_title("Ground Truth")
    axes[0].set_xlabel("x (grid cells)")
    axes[0].set_ylabel("y (grid cells)")
    plt.colorbar(im0, ax=axes[0], label="Temperature [K]")

    # Prediction
    im1 = axes[1].imshow(T_pred, cmap="hot", origin="lower", vmin=vmin, vmax=vmax)
    axes[1].set_title("Prediction")
    axes[1].set_xlabel("x (grid cells)")
    axes[1].set_ylabel("y (grid cells)")
    plt.colorbar(im1, ax=axes[1], label="Temperature [K]")

    # Error
    error = np.abs(T_pred - T_true)
    im2 = axes[2].imshow(error, cmap="Reds", origin="lower")
    axes[2].set_title(f"Absolute Error (max={error.max():.2f} K)")
    axes[2].set_xlabel("x (grid cells)")
    axes[2].set_ylabel("y (grid cells)")
    plt.colorbar(im2, ax=axes[2], label="Error [K]")

    fig.suptitle(title, fontsize=14, y=1.02)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"Saved figure to {save_path}")

    plt.show()


def plot_error_map(
    error: np.ndarray,
    save_path: Optional[Path] = None,
    title: str = "Error Map",
) -> None:
    """Plot spatial error distribution.

    Args:
        error: Error field of shape ``(H, W)``.
        save_path: Optional path to save the figure.
        title: Figure title.
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    im = ax.imshow(error, cmap="Reds", origin="lower")
    ax.set_title(title)
    ax.set_xlabel("x (grid cells)")
    ax.set_ylabel("y (grid cells)")
    plt.colorbar(im, ax=ax, label="Absolute Error [K]")

    # Add statistics text box
    stats_text = (
        f"Mean: {error.mean():.3f} K\n"
        f"Max:  {error.max():.3f} K\n"
        f"Std:  {error.std():.3f} K"
    )
    ax.text(
        0.02, 0.98, stats_text,
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"Saved figure to {save_path}")

    plt.show()


def plot_training_curves(
    history: Dict[str, List[float]],
    save_path: Optional[Path] = None,
) -> None:
    """Plot training and validation loss curves.

    Args:
        history: Dictionary with 'train_loss' and 'val_loss' lists.
        save_path: Optional path to save the figure.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    epochs = range(1, len(history["train_loss"]) + 1)

    ax.plot(epochs, history["train_loss"], label="Train Loss", linewidth=2)
    ax.plot(epochs, history["val_loss"], label="Val Loss", linewidth=2)

    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Loss", fontsize=12)
    ax.set_title("Training and Validation Loss", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    # Find best epoch
    best_epoch = np.argmin(history["val_loss"]) + 1
    best_val_loss = min(history["val_loss"])

    ax.axvline(best_epoch, color="red", linestyle="--", linewidth=1.5, alpha=0.7,
               label=f"Best Epoch ({best_epoch})")

    ax.text(
        0.95, 0.95,
        f"Best Val Loss: {best_val_loss:.4f}\nEpoch: {best_epoch}",
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment="top",
        horizontalalignment="right",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"Saved figure to {save_path}")

    plt.show()


def plot_rollout_comparison(
    true_trajectory: np.ndarray,
    pred_trajectory: np.ndarray,
    time_indices: Optional[List[int]] = None,
    save_path: Optional[Path] = None,
) -> None:
    """Plot multi-step rollout comparison at selected time steps.

    Args:
        true_trajectory: Ground truth trajectory of shape ``(T, H, W)``.
        pred_trajectory: Predicted trajectory of shape ``(T, H, W)``.
        time_indices: List of time indices to visualize.
        save_path: Optional path to save the figure.
    """
    if time_indices is None:
        # Evenly spaced time indices
        n_snapshots = min(5, len(true_trajectory))
        time_indices = np.linspace(0, len(true_trajectory) - 1, n_snapshots, dtype=int).tolist()

    n_times = len(time_indices)
    fig, axes = plt.subplots(3, n_times, figsize=(4 * n_times, 10))

    if n_times == 1:
        axes = axes[:, np.newaxis]

    vmin = min(true_trajectory.min(), pred_trajectory.min())
    vmax = max(true_trajectory.max(), pred_trajectory.max())

    for col, t_idx in enumerate(time_indices):
        T_true = true_trajectory[t_idx]
        T_pred = pred_trajectory[t_idx]
        error = np.abs(T_pred - T_true)

        # Ground truth
        axes[0, col].imshow(T_true, cmap="hot", origin="lower", vmin=vmin, vmax=vmax)
        axes[0, col].set_title(f"t={t_idx}")
        axes[0, col].set_xticks([])
        axes[0, col].set_yticks([])

        # Prediction
        axes[1, col].imshow(T_pred, cmap="hot", origin="lower", vmin=vmin, vmax=vmax)
        axes[1, col].set_xticks([])
        axes[1, col].set_yticks([])

        # Error
        axes[2, col].imshow(error, cmap="Reds", origin="lower")
        axes[2, col].set_xlabel(f"Max err: {error.max():.2f} K")
        axes[2, col].set_xticks([])
        axes[2, col].set_yticks([])

    axes[0, 0].set_ylabel("Ground Truth", fontsize=12)
    axes[1, 0].set_ylabel("Prediction", fontsize=12)
    axes[2, 0].set_ylabel("Absolute Error", fontsize=12)

    fig.suptitle("Multi-Step Rollout Comparison", fontsize=14, y=1.00)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"Saved figure to {save_path}")

    plt.show()

"""Data normalization and train/val/test splitting utilities.

Provides streaming computation of per-channel normalization statistics,
z-score normalization / denormalization transforms, deterministic data
splitting, and JSON serialization for reproducible preprocessing in the
2D battery thermal surrogate pipeline.

Typical usage::

    from src.data_utils.preprocess import (
        compute_normalization_stats,
        normalize_data,
        denormalize_data,
        create_splits,
        save_normalization_stats,
        load_normalization_stats,
    )

    splits = create_splits(n_samples=2000)
    dataset = ThermalDataset(h5_path, split="train", indices=splits["train"])
    stats = compute_normalization_stats(dataset)
    save_normalization_stats(stats, "data/processed/norm_stats.json")

    # In the training loop:
    sample = dataset[0]
    normed = normalize_data(sample["input"], stats)
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)


# ======================================================================
# Normalization statistics
# ======================================================================

@dataclass
class NormalizationStats:
    """Per-channel normalization statistics for thermal simulation data.

    Stores mean, standard deviation, minimum, and maximum values for each
    channel of the 9-channel input tensor and the 1-channel target tensor.
    These statistics are computed over the training set only and applied
    identically to all splits to prevent data leakage.

    Attributes:
        input_mean: Per-channel means of shape ``(C_in,)``.
        input_std: Per-channel standard deviations of shape ``(C_in,)``.
        input_min: Per-channel minima of shape ``(C_in,)``.
        input_max: Per-channel maxima of shape ``(C_in,)``.
        target_mean: Per-channel means of shape ``(C_out,)``.
        target_std: Per-channel standard deviations of shape ``(C_out,)``.
        target_min: Per-channel minima of shape ``(C_out,)``.
        target_max: Per-channel maxima of shape ``(C_out,)``.
        n_samples: Number of samples used to compute the statistics.
        channel_names: Human-readable names for each input channel.
    """

    input_mean: np.ndarray    # (C_in,)
    input_std: np.ndarray     # (C_in,)
    input_min: np.ndarray     # (C_in,)
    input_max: np.ndarray     # (C_in,)
    target_mean: np.ndarray   # (C_out,)
    target_std: np.ndarray    # (C_out,)
    target_min: np.ndarray    # (C_out,)
    target_max: np.ndarray    # (C_out,)
    n_samples: int = 0
    channel_names: List[str] = field(default_factory=lambda: [
        "T_t", "mask_cell", "mask_coolant", "mask_insulation",
        "k_field", "q_field", "h_field", "sdf_cell", "sdf_coolant",
    ])


# ======================================================================
# Statistics computation (streaming / online)
# ======================================================================

def compute_normalization_stats(
    dataset: Any,
    max_samples: Optional[int] = None,
    batch_print_interval: int = 500,
) -> NormalizationStats:
    """Compute per-channel normalization statistics from a dataset.

    Uses Welford's online algorithm to compute mean and variance in a
    single streaming pass without loading the entire dataset into memory.
    Min and max are tracked simultaneously.

    Args:
        dataset: A :class:`~src.data_utils.dataset.ThermalDataset` (or any
            object supporting ``__len__`` and ``__getitem__`` that returns
            dicts with ``'input'`` and ``'target'`` tensors).
        max_samples: Optional cap on the number of samples to process.  If
            ``None``, all samples in the dataset are used.
        batch_print_interval: Log progress every this many samples.

    Returns:
        A :class:`NormalizationStats` instance with the computed statistics.

    Note:
        This function should be called **only on the training split** to
        prevent information leakage from validation or test data.
    """
    n_total = len(dataset)
    n_samples = min(n_total, max_samples) if max_samples is not None else n_total

    logger.info(
        "Computing normalization statistics over %d / %d samples (streaming)",
        n_samples, n_total,
    )

    # Peek at the first sample to determine channel counts
    first_sample = dataset[0]
    c_in: int = first_sample["input"].shape[0]
    c_out: int = first_sample["target"].shape[0]

    # Welford accumulators for input channels
    in_count: int = 0
    in_mean = np.zeros(c_in, dtype=np.float64)
    in_m2 = np.zeros(c_in, dtype=np.float64)
    in_min = np.full(c_in, np.inf, dtype=np.float64)
    in_max = np.full(c_in, -np.inf, dtype=np.float64)

    # Welford accumulators for target channels
    tgt_count: int = 0
    tgt_mean = np.zeros(c_out, dtype=np.float64)
    tgt_m2 = np.zeros(c_out, dtype=np.float64)
    tgt_min = np.full(c_out, np.inf, dtype=np.float64)
    tgt_max = np.full(c_out, -np.inf, dtype=np.float64)

    for i in range(n_samples):
        sample = dataset[i]
        inp: np.ndarray = sample["input"].numpy()    # (C_in, H, W)
        tgt: np.ndarray = sample["target"].numpy()   # (C_out, H, W)

        # Per-channel statistics: reduce over spatial dims (H, W)
        # Each sample contributes H*W pixel observations per channel.
        n_pixels = inp.shape[1] * inp.shape[2]

        for c in range(c_in):
            channel_data = inp[c].ravel().astype(np.float64)
            for val in [channel_data.mean()]:
                # Batch Welford update: treat the spatial mean as a single observation
                # For more accuracy we track pixel-level stats
                pass

            # Pixel-level Welford update (batched)
            old_count = in_count
            batch_mean = channel_data.mean()
            batch_var = channel_data.var()
            new_count = old_count + n_pixels

            delta = batch_mean - in_mean[c]
            in_mean[c] += delta * n_pixels / new_count
            # Combine variances using parallel algorithm
            in_m2[c] += batch_var * n_pixels + delta ** 2 * old_count * n_pixels / new_count

            in_min[c] = min(in_min[c], channel_data.min())
            in_max[c] = max(in_max[c], channel_data.max())

        for c in range(c_out):
            channel_data = tgt[c].ravel().astype(np.float64)
            old_count = tgt_count
            batch_mean = channel_data.mean()
            batch_var = channel_data.var()
            new_count = old_count + n_pixels

            delta = batch_mean - tgt_mean[c]
            tgt_mean[c] += delta * n_pixels / new_count
            tgt_m2[c] += batch_var * n_pixels + delta ** 2 * old_count * n_pixels / new_count

            tgt_min[c] = min(tgt_min[c], channel_data.min())
            tgt_max[c] = max(tgt_max[c], channel_data.max())

        # Update shared count (same n_pixels for input and target)
        in_count += n_pixels
        tgt_count += n_pixels

        if (i + 1) % batch_print_interval == 0:
            logger.info("  Processed %d / %d samples", i + 1, n_samples)

    # Finalise variance -> std
    in_std = np.sqrt(in_m2 / in_count)
    tgt_std = np.sqrt(tgt_m2 / tgt_count)

    # Avoid zero std (binary masks, constant fields)
    eps = 1e-8
    in_std = np.where(in_std < eps, 1.0, in_std)
    tgt_std = np.where(tgt_std < eps, 1.0, tgt_std)

    stats = NormalizationStats(
        input_mean=in_mean.astype(np.float32),
        input_std=in_std.astype(np.float32),
        input_min=in_min.astype(np.float32),
        input_max=in_max.astype(np.float32),
        target_mean=tgt_mean.astype(np.float32),
        target_std=tgt_std.astype(np.float32),
        target_min=tgt_min.astype(np.float32),
        target_max=tgt_max.astype(np.float32),
        n_samples=n_samples,
    )

    logger.info("Normalization stats computed (n_samples=%d)", n_samples)
    for c in range(c_in):
        name = stats.channel_names[c] if c < len(stats.channel_names) else f"ch{c}"
        logger.debug(
            "  Input  ch%d (%s): mean=%.4f, std=%.4f, min=%.4f, max=%.4f",
            c, name, in_mean[c], in_std[c], in_min[c], in_max[c],
        )
    for c in range(c_out):
        logger.debug(
            "  Target ch%d: mean=%.6f, std=%.6f, min=%.6f, max=%.6f",
            c, tgt_mean[c], tgt_std[c], tgt_min[c], tgt_max[c],
        )

    return stats


# ======================================================================
# Normalize / denormalize
# ======================================================================

def normalize_data(
    data: torch.Tensor,
    stats: NormalizationStats,
    mode: str = "input",
) -> torch.Tensor:
    """Apply z-score normalization to a tensor.

    Performs channel-wise normalization: ``(x - mean) / std``.

    Args:
        data: Tensor of shape ``(C, H, W)`` or ``(B, C, H, W)``.
        stats: Normalization statistics (typically from the training set).
        mode: Which statistics to use -- ``"input"`` for the 9-channel input
            tensor or ``"target"`` for the 1-channel target tensor.

    Returns:
        Normalized tensor of the same shape and dtype.

    Raises:
        ValueError: If *mode* is not ``"input"`` or ``"target"``.
    """
    mean, std = _get_mean_std(stats, mode, data.device)
    return _broadcast_normalize(data, mean, std)


def denormalize_data(
    data: torch.Tensor,
    stats: NormalizationStats,
    mode: str = "target",
) -> torch.Tensor:
    """Reverse z-score normalization.

    Performs the inverse transform: ``x * std + mean``.

    Args:
        data: Normalized tensor of shape ``(C, H, W)`` or ``(B, C, H, W)``.
        stats: Normalization statistics used during normalization.
        mode: Which statistics to use (``"input"`` or ``"target"``).

    Returns:
        Denormalized tensor of the same shape and dtype.

    Raises:
        ValueError: If *mode* is not ``"input"`` or ``"target"``.
    """
    mean, std = _get_mean_std(stats, mode, data.device)
    return _broadcast_denormalize(data, mean, std)


def _get_mean_std(
    stats: NormalizationStats,
    mode: str,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Extract mean and std tensors for the requested mode.

    Args:
        stats: Normalization statistics dataclass.
        mode: ``"input"`` or ``"target"``.
        device: Device to place the returned tensors on.

    Returns:
        Tuple of ``(mean, std)`` tensors of shape ``(C,)``.

    Raises:
        ValueError: If *mode* is unrecognised.
    """
    if mode == "input":
        mean = torch.from_numpy(stats.input_mean).to(device=device, dtype=torch.float32)
        std = torch.from_numpy(stats.input_std).to(device=device, dtype=torch.float32)
    elif mode == "target":
        mean = torch.from_numpy(stats.target_mean).to(device=device, dtype=torch.float32)
        std = torch.from_numpy(stats.target_std).to(device=device, dtype=torch.float32)
    else:
        raise ValueError(f"mode must be 'input' or 'target', got {mode!r}")
    return mean, std


def _broadcast_normalize(
    data: torch.Tensor,
    mean: torch.Tensor,
    std: torch.Tensor,
) -> torch.Tensor:
    """Z-score normalize with proper broadcasting for (C,H,W) or (B,C,H,W).

    Args:
        data: Input tensor.
        mean: Per-channel mean of shape ``(C,)``.
        std: Per-channel std of shape ``(C,)``.

    Returns:
        Normalized tensor.
    """
    if data.ndim == 3:
        # (C, H, W) -> reshape mean/std to (C, 1, 1)
        mean = mean.view(-1, 1, 1)
        std = std.view(-1, 1, 1)
    elif data.ndim == 4:
        # (B, C, H, W) -> reshape mean/std to (1, C, 1, 1)
        mean = mean.view(1, -1, 1, 1)
        std = std.view(1, -1, 1, 1)
    else:
        raise ValueError(f"Expected 3-D or 4-D tensor, got {data.ndim}-D")
    return (data - mean) / std


def _broadcast_denormalize(
    data: torch.Tensor,
    mean: torch.Tensor,
    std: torch.Tensor,
) -> torch.Tensor:
    """Reverse z-score normalization with proper broadcasting.

    Args:
        data: Normalized tensor.
        mean: Per-channel mean of shape ``(C,)``.
        std: Per-channel std of shape ``(C,)``.

    Returns:
        Denormalized tensor.
    """
    if data.ndim == 3:
        mean = mean.view(-1, 1, 1)
        std = std.view(-1, 1, 1)
    elif data.ndim == 4:
        mean = mean.view(1, -1, 1, 1)
        std = std.view(1, -1, 1, 1)
    else:
        raise ValueError(f"Expected 3-D or 4-D tensor, got {data.ndim}-D")
    return data * std + mean


# ======================================================================
# Data splitting
# ======================================================================

def create_splits(
    n_samples: int,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> Dict[str, List[int]]:
    """Create deterministic train / validation / test index splits.

    Uses :func:`sklearn.model_selection.train_test_split` for two-stage
    stratification-compatible splitting.  The first stage separates the
    training set; the second stage splits the remainder into validation
    and test sets.

    Args:
        n_samples: Total number of samples (e.g., trajectory count).
        train_ratio: Fraction of samples for training.  Default ``0.7``.
        val_ratio: Fraction of samples for validation.  Default ``0.15``.
        test_ratio: Fraction of samples for testing.  Default ``0.15``.
        seed: Random seed for reproducibility.  Default ``42``.

    Returns:
        Dictionary with keys ``"train"``, ``"val"``, ``"test"`` mapping to
        **sorted** lists of integer indices.

    Raises:
        ValueError: If the ratios do not sum to 1.0 (within tolerance) or
            if *n_samples* is too small for the requested split.
    """
    ratio_sum = train_ratio + val_ratio + test_ratio
    if abs(ratio_sum - 1.0) > 1e-6:
        raise ValueError(
            f"Split ratios must sum to 1.0, got "
            f"{train_ratio} + {val_ratio} + {test_ratio} = {ratio_sum}"
        )

    if n_samples < 3:
        raise ValueError(
            f"Need at least 3 samples for a 3-way split, got {n_samples}"
        )

    indices = list(range(n_samples))

    # Stage 1: separate train from (val + test)
    val_test_ratio = val_ratio + test_ratio
    train_indices, val_test_indices = train_test_split(
        indices,
        test_size=val_test_ratio,
        random_state=seed,
        shuffle=True,
    )

    # Stage 2: split remainder into val and test
    # Relative ratio of test within (val + test)
    relative_test_ratio = test_ratio / val_test_ratio if val_test_ratio > 0 else 0.5

    if len(val_test_indices) < 2:
        # Edge case: too few samples to split further
        val_indices = val_test_indices
        test_indices: List[int] = []
    else:
        val_indices, test_indices = train_test_split(
            val_test_indices,
            test_size=relative_test_ratio,
            random_state=seed,
            shuffle=True,
        )

    # Sort indices for reproducible iteration order
    train_indices = sorted(train_indices)
    val_indices = sorted(val_indices)
    test_indices = sorted(test_indices)

    logger.info(
        "Data splits: train=%d (%.1f%%), val=%d (%.1f%%), test=%d (%.1f%%)",
        len(train_indices), 100.0 * len(train_indices) / n_samples,
        len(val_indices), 100.0 * len(val_indices) / n_samples,
        len(test_indices), 100.0 * len(test_indices) / n_samples,
    )

    return {
        "train": train_indices,
        "val": val_indices,
        "test": test_indices,
    }


# ======================================================================
# Stats serialisation (JSON)
# ======================================================================

def save_normalization_stats(
    stats: NormalizationStats,
    path: Union[str, Path],
) -> Path:
    """Serialize normalization statistics to a JSON file.

    NumPy arrays are converted to plain Python lists for JSON compatibility.

    Args:
        stats: The :class:`NormalizationStats` to save.
        path: Destination file path (should end in ``.json``).

    Returns:
        Resolved :class:`~pathlib.Path` to the saved file.
    """
    path = Path(path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)

    payload: Dict[str, Any] = {
        "input_mean": stats.input_mean.tolist(),
        "input_std": stats.input_std.tolist(),
        "input_min": stats.input_min.tolist(),
        "input_max": stats.input_max.tolist(),
        "target_mean": stats.target_mean.tolist(),
        "target_std": stats.target_std.tolist(),
        "target_min": stats.target_min.tolist(),
        "target_max": stats.target_max.tolist(),
        "n_samples": stats.n_samples,
        "channel_names": stats.channel_names,
    }

    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    logger.info("Normalization stats saved to %s", path)
    return path


def load_normalization_stats(path: Union[str, Path]) -> NormalizationStats:
    """Load normalization statistics from a JSON file.

    Args:
        path: Path to the JSON file written by
            :func:`save_normalization_stats`.

    Returns:
        Reconstructed :class:`NormalizationStats` instance.

    Raises:
        FileNotFoundError: If *path* does not exist.
        KeyError: If the JSON file is missing required fields.
    """
    path = Path(path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"Normalization stats file not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        payload: Dict[str, Any] = json.load(f)

    required_keys = [
        "input_mean", "input_std", "input_min", "input_max",
        "target_mean", "target_std", "target_min", "target_max",
    ]
    for key in required_keys:
        if key not in payload:
            raise KeyError(
                f"Required key '{key}' not found in {path}. "
                f"Available keys: {list(payload.keys())}"
            )

    stats = NormalizationStats(
        input_mean=np.array(payload["input_mean"], dtype=np.float32),
        input_std=np.array(payload["input_std"], dtype=np.float32),
        input_min=np.array(payload["input_min"], dtype=np.float32),
        input_max=np.array(payload["input_max"], dtype=np.float32),
        target_mean=np.array(payload["target_mean"], dtype=np.float32),
        target_std=np.array(payload["target_std"], dtype=np.float32),
        target_min=np.array(payload["target_min"], dtype=np.float32),
        target_max=np.array(payload["target_max"], dtype=np.float32),
        n_samples=int(payload.get("n_samples", 0)),
        channel_names=payload.get("channel_names", []),
    )

    logger.info(
        "Normalization stats loaded from %s (n_samples=%d)",
        path, stats.n_samples,
    )
    return stats

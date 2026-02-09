"""Comprehensive visualization module for the battery thermal surrogate project.

Provides five visualizer classes and module-level helper functions for
creating publication-quality figures from model predictions, error
diagnostics, physics validation, uncertainty quantification, and training
monitoring.

All public plotting methods accept both ``torch.Tensor`` and ``numpy.ndarray``
inputs, save to file in multiple formats when a path is given, and return the
``matplotlib.figure.Figure`` object for interactive use.

Typical usage::

    from src.evaluation.visualization import (
        TemperatureFieldVisualizer,
        ErrorAnalysisVisualizer,
        TrainingMonitor,
    )

    viz = TemperatureFieldVisualizer()
    fig = viz.plot_comparison(pred, target, input_field=T_input,
                              save_path="results/comparison.png")

    monitor = TrainingMonitor()
    fig = monitor.plot_loss_curves(train_losses, val_losses,
                                   save_path="results/loss_curves.png")
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (side-effect import)

logger = logging.getLogger(__name__)

# Try importing torch; if unavailable, torch tensors simply won't be passed.
try:
    import torch
except ImportError:  # pragma: no cover
    torch = None  # type: ignore[assignment]

# ---------------------------------------------------------------------------
# Module-level helper functions
# ---------------------------------------------------------------------------


def _to_numpy(data: Any) -> np.ndarray:
    """Convert a ``torch.Tensor`` or ``numpy.ndarray`` to a plain NumPy array.

    Handles ``detach()`` and ``.cpu()`` transparently when the input is a
    CUDA tensor.  If *data* is already a NumPy array it is returned as-is.
    If the resulting array is 4-D (batch dimension present), the first sample
    is selected automatically.

    Args:
        data: Input data -- either a ``torch.Tensor`` or ``numpy.ndarray``.

    Returns:
        A NumPy array with at most 3 dimensions (C, H, W) or 2 dimensions
        (H, W).

    Raises:
        TypeError: If *data* is neither a tensor nor an ndarray.
    """
    if torch is not None and isinstance(data, torch.Tensor):
        arr: np.ndarray = data.detach().cpu().numpy()
    elif isinstance(data, np.ndarray):
        arr = data
    else:
        raise TypeError(
            f"Expected torch.Tensor or numpy.ndarray, got {type(data).__name__}."
        )

    # Squeeze batch dimension when present (B, C, H, W) -> (C, H, W).
    if arr.ndim == 4:
        arr = arr[0]

    # Squeeze channel dimension when it is 1: (1, H, W) -> (H, W).
    if arr.ndim == 3 and arr.shape[0] == 1:
        arr = arr[0]

    return arr


def _setup_figure(
    nrows: int = 1,
    ncols: int = 1,
    figsize: Optional[Tuple[float, float]] = None,
    dpi: int = 150,
) -> Tuple[Figure, Any]:
    """Create a standard matplotlib figure with consistent styling.

    Applies the seaborn ``whitegrid`` style globally before creating the
    figure so that all downstream plots share a uniform appearance.

    Args:
        nrows: Number of subplot rows.
        ncols: Number of subplot columns.
        figsize: Figure size ``(width, height)`` in inches.  Defaults to
            ``(ncols * 5, nrows * 4)`` if not specified.
        dpi: Dots per inch for the figure (default 150).

    Returns:
        Tuple of ``(fig, axes)`` where *axes* is the array (or scalar)
        of ``Axes`` objects returned by ``plt.subplots``.
    """
    sns.set_style("whitegrid")
    if figsize is None:
        figsize = (ncols * 5, nrows * 4)
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, dpi=dpi)
    return fig, axes


def _save_figure(
    fig: Figure,
    save_path: Optional[Union[str, Path]],
    formats: Tuple[str, ...] = ("png", "pdf"),
) -> None:
    """Save a figure to disk in one or more formats.

    The parent directory is created if it does not exist.  Each requested
    format is saved by replacing the suffix of *save_path*.

    Args:
        fig: The matplotlib figure to save.
        save_path: Destination file path.  If ``None``, the function is a
            no-op.
        formats: Iterable of file-format extensions to write (default
            ``('png', 'pdf')``).
    """
    if save_path is None:
        return

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    for fmt in formats:
        out = save_path.with_suffix(f".{fmt}")
        fig.savefig(str(out), dpi=fig.dpi, bbox_inches="tight")
        logger.info("Saved figure to %s", out)


# ---------------------------------------------------------------------------
# 1. TemperatureFieldVisualizer
# ---------------------------------------------------------------------------


class TemperatureFieldVisualizer:
    """Visualizes 2-D temperature fields from the thermal surrogate model.

    Provides comparison plots between predictions and ground truth,
    time-evolution snapshots, and 3-D surface views of temperature
    distributions.

    Attributes:
        cmap: Matplotlib colormap name used for temperature fields.
        vmin: Lower bound of the temperature colour scale [K].
        vmax: Upper bound of the temperature colour scale [K].
    """

    def __init__(
        self,
        cmap: str = "hot",
        vmin: float = 290.0,
        vmax: float = 350.0,
    ) -> None:
        """Initialise the temperature field visualizer.

        Args:
            cmap: Matplotlib colormap for temperature rendering (default
                ``'hot'``).
            vmin: Minimum temperature for the colour scale (default 290 K).
            vmax: Maximum temperature for the colour scale (default 350 K).
        """
        self.cmap = cmap
        self.vmin = vmin
        self.vmax = vmax

    def plot_comparison(
        self,
        pred: Any,
        target: Any,
        input_field: Optional[Any] = None,
        save_path: Optional[Union[str, Path]] = None,
    ) -> Figure:
        """Plot a 2x2 comparison between prediction, ground truth, and error.

        The four panels are:

        1. **Input T(t)** -- the current temperature field (or blank if not
           supplied).
        2. **Predicted T(t+dt)** -- the model's predicted next step.
        3. **Ground Truth T(t+dt)** -- the reference simulation output.
        4. **Absolute Error** -- pixel-wise ``|pred - target|`` with MAE
           annotated in the title.

        Args:
            pred: Predicted temperature field.  Shape ``(B, 1, H, W)`` or
                ``(H, W)``.
            target: Ground-truth temperature field (same shape convention).
            input_field: Optional input temperature field T(t).  If ``None``,
                the first panel displays a placeholder message.
            save_path: Optional file path for saving the figure.

        Returns:
            The ``matplotlib.figure.Figure`` object.
        """
        pred_np = _to_numpy(pred)
        target_np = _to_numpy(target)
        abs_error = np.abs(pred_np - target_np)
        mae = float(np.mean(abs_error))

        fig, axes = _setup_figure(nrows=2, ncols=2, figsize=(12, 10))

        # Panel 1: Input T(t)
        ax0 = axes[0, 0]
        if input_field is not None:
            input_np = _to_numpy(input_field)
            im0 = ax0.imshow(
                input_np, cmap=self.cmap, vmin=self.vmin, vmax=self.vmax,
                aspect="auto", origin="lower",
            )
            fig.colorbar(im0, ax=ax0, label="Temperature [K]")
            ax0.set_title("Input T(t)")
        else:
            ax0.text(
                0.5, 0.5, "No input field provided",
                ha="center", va="center", transform=ax0.transAxes,
                fontsize=12, color="gray",
            )
            ax0.set_title("Input T(t) [N/A]")
        ax0.set_xlabel("x")
        ax0.set_ylabel("y")

        # Panel 2: Predicted T(t+dt)
        ax1 = axes[0, 1]
        im1 = ax1.imshow(
            pred_np, cmap=self.cmap, vmin=self.vmin, vmax=self.vmax,
            aspect="auto", origin="lower",
        )
        fig.colorbar(im1, ax=ax1, label="Temperature [K]")
        ax1.set_title("Predicted T(t+dt)")
        ax1.set_xlabel("x")
        ax1.set_ylabel("y")

        # Panel 3: Ground truth T(t+dt)
        ax2 = axes[1, 0]
        im2 = ax2.imshow(
            target_np, cmap=self.cmap, vmin=self.vmin, vmax=self.vmax,
            aspect="auto", origin="lower",
        )
        fig.colorbar(im2, ax=ax2, label="Temperature [K]")
        ax2.set_title("Ground Truth T(t+dt)")
        ax2.set_xlabel("x")
        ax2.set_ylabel("y")

        # Panel 4: Absolute error
        ax3 = axes[1, 1]
        im3 = ax3.imshow(
            abs_error, cmap="inferno", aspect="auto", origin="lower",
        )
        fig.colorbar(im3, ax=ax3, label="|Error| [K]")
        ax3.set_title(f"Absolute Error (MAE = {mae:.4f} K)")
        ax3.set_xlabel("x")
        ax3.set_ylabel("y")

        fig.tight_layout()
        _save_figure(fig, save_path)
        return fig

    def plot_time_evolution(
        self,
        trajectory: Any,
        times: Optional[Sequence[float]] = None,
        n_snapshots: int = 6,
        save_path: Optional[Union[str, Path]] = None,
    ) -> Figure:
        """Show the temperature field at multiple time steps in a single row.

        Args:
            trajectory: Temperature trajectory of shape
                ``(B, T, H, W)`` or ``(T, H, W)``.  If 4-D, the first
                batch element is used.
            times: Optional sequence of physical time values corresponding
                to each snapshot.  If ``None``, integer step indices are
                used as labels.
            n_snapshots: Number of equally spaced snapshots to display
                (default 6).
            save_path: Optional file path for saving the figure.

        Returns:
            The ``matplotlib.figure.Figure`` object.
        """
        traj_np = _to_numpy(trajectory)
        # traj_np is now (T, H, W) or (H, W). Force 3-D.
        if traj_np.ndim == 2:
            traj_np = traj_np[np.newaxis, ...]

        n_total = traj_np.shape[0]
        indices = np.linspace(0, n_total - 1, n_snapshots, dtype=int)

        fig, axes = _setup_figure(nrows=1, ncols=n_snapshots,
                                  figsize=(n_snapshots * 4, 4))
        if n_snapshots == 1:
            axes = [axes]

        for i, idx in enumerate(indices):
            ax = axes[i]
            im = ax.imshow(
                traj_np[idx], cmap=self.cmap, vmin=self.vmin, vmax=self.vmax,
                aspect="auto", origin="lower",
            )
            if times is not None and idx < len(times):
                label = f"t = {times[idx]:.4g} s"
            else:
                label = f"Step {idx}"
            ax.set_title(label, fontsize=10)
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        fig.suptitle("Temperature Field Evolution", fontsize=14, y=1.02)
        fig.tight_layout()
        _save_figure(fig, save_path)
        return fig

    def plot_3d_surface(
        self,
        field: Any,
        title: str = "Temperature Field",
        save_path: Optional[Union[str, Path]] = None,
    ) -> Figure:
        """Create a 3-D surface plot of a temperature field.

        The x- and y-axes correspond to spatial pixel indices while the
        z-axis shows the temperature value at each pixel.

        Args:
            field: 2-D temperature field of shape ``(B, 1, H, W)`` or
                ``(H, W)``.
            title: Plot title (default ``'Temperature Field'``).
            save_path: Optional file path for saving the figure.

        Returns:
            The ``matplotlib.figure.Figure`` object.
        """
        field_np = _to_numpy(field)

        fig = plt.figure(figsize=(10, 7), dpi=150)
        ax = fig.add_subplot(111, projection="3d")

        h, w = field_np.shape
        x = np.arange(w)
        y = np.arange(h)
        X, Y = np.meshgrid(x, y)

        surf = ax.plot_surface(
            X, Y, field_np,
            cmap=self.cmap, edgecolor="none", alpha=0.9,
            vmin=self.vmin, vmax=self.vmax,
        )
        fig.colorbar(surf, ax=ax, shrink=0.5, aspect=15, label="Temperature [K]")
        ax.set_xlabel("x (pixels)")
        ax.set_ylabel("y (pixels)")
        ax.set_zlabel("Temperature [K]")
        ax.set_title(title, fontsize=13)

        fig.tight_layout()
        _save_figure(fig, save_path)
        return fig


# ---------------------------------------------------------------------------
# 2. ErrorAnalysisVisualizer
# ---------------------------------------------------------------------------


class ErrorAnalysisVisualizer:
    """Visualizes spatial and temporal error characteristics of predictions.

    Provides heatmaps of spatial error, histograms of pixel-wise error
    distributions, error growth curves over rollout steps, and boxplots of
    errors grouped by material region.
    """

    def plot_error_heatmap(
        self,
        pred: Any,
        target: Any,
        save_path: Optional[Union[str, Path]] = None,
    ) -> Figure:
        """Plot a spatial heatmap of the absolute prediction error.

        Uses the ``RdYlBu_r`` colormap so that large errors appear red
        and small errors appear blue.

        Args:
            pred: Predicted temperature field.  Shape ``(B, 1, H, W)`` or
                ``(H, W)``.
            target: Ground-truth temperature field (same shape convention).
            save_path: Optional file path for saving the figure.

        Returns:
            The ``matplotlib.figure.Figure`` object.
        """
        pred_np = _to_numpy(pred)
        target_np = _to_numpy(target)
        abs_error = np.abs(pred_np - target_np)

        fig, ax = _setup_figure(nrows=1, ncols=1, figsize=(7, 5))
        im = ax.imshow(
            abs_error, cmap="RdYlBu_r", aspect="auto", origin="lower",
        )
        fig.colorbar(im, ax=ax, label="|Error| [K]")
        mae = float(np.mean(abs_error))
        max_err = float(np.max(abs_error))
        ax.set_title(f"Spatial Error Map  (MAE={mae:.4f} K, Max={max_err:.4f} K)")
        ax.set_xlabel("x")
        ax.set_ylabel("y")

        fig.tight_layout()
        _save_figure(fig, save_path)
        return fig

    def plot_error_histogram(
        self,
        pred: Any,
        target: Any,
        n_bins: int = 50,
        save_path: Optional[Union[str, Path]] = None,
    ) -> Figure:
        """Plot a histogram of pixel-wise absolute errors.

        Args:
            pred: Predicted temperature field.
            target: Ground-truth temperature field.
            n_bins: Number of histogram bins (default 50).
            save_path: Optional file path for saving the figure.

        Returns:
            The ``matplotlib.figure.Figure`` object.
        """
        pred_np = _to_numpy(pred).ravel()
        target_np = _to_numpy(target).ravel()
        errors = np.abs(pred_np - target_np)

        fig, ax = _setup_figure(nrows=1, ncols=1, figsize=(7, 5))
        ax.hist(errors, bins=n_bins, color="steelblue", edgecolor="white",
                alpha=0.85, density=True)

        mae = float(np.mean(errors))
        std = float(np.std(errors))
        ax.axvline(mae, color="red", linestyle="--", linewidth=1.5,
                   label=f"MAE = {mae:.4f} K")
        ax.axvline(mae + std, color="orange", linestyle=":", linewidth=1.2,
                   label=f"MAE + 1$\\sigma$ = {mae + std:.4f} K")

        ax.set_xlabel("Absolute Error [K]")
        ax.set_ylabel("Density")
        ax.set_title("Pixel-wise Error Distribution")
        ax.legend(frameon=True)

        fig.tight_layout()
        _save_figure(fig, save_path)
        return fig

    def plot_error_vs_time(
        self,
        errors_per_step: Any,
        save_path: Optional[Union[str, Path]] = None,
    ) -> Figure:
        """Plot rollout error growth over time steps.

        Args:
            errors_per_step: 1-D array-like of error values (e.g. MAE)
                at each rollout step.
            save_path: Optional file path for saving the figure.

        Returns:
            The ``matplotlib.figure.Figure`` object.
        """
        errors = _to_numpy(np.asarray(errors_per_step)).ravel()
        steps = np.arange(1, len(errors) + 1)

        fig, ax = _setup_figure(nrows=1, ncols=1, figsize=(8, 5))
        ax.plot(steps, errors, marker="o", markersize=4, linewidth=1.8,
                color="crimson", label="Rollout MAE")
        ax.fill_between(steps, 0, errors, alpha=0.15, color="crimson")

        ax.set_xlabel("Rollout Step")
        ax.set_ylabel("Mean Absolute Error [K]")
        ax.set_title("Error Growth Over Autoregressive Rollout")
        ax.legend(frameon=True)

        fig.tight_layout()
        _save_figure(fig, save_path)
        return fig

    def plot_error_by_region(
        self,
        pred: Any,
        target: Any,
        mask: Any,
        save_path: Optional[Union[str, Path]] = None,
    ) -> Figure:
        """Boxplot of prediction errors grouped by material region.

        The *mask* array encodes region labels as integer values (e.g. 0 for
        background, 1 for cell, 2 for coolant, 3 for insulation).  A
        separate box is drawn for each unique region label.

        Args:
            pred: Predicted temperature field.
            target: Ground-truth temperature field.
            mask: Integer region-label map of shape ``(H, W)`` or
                ``(B, 1, H, W)``.
            save_path: Optional file path for saving the figure.

        Returns:
            The ``matplotlib.figure.Figure`` object.
        """
        pred_np = _to_numpy(pred).ravel()
        target_np = _to_numpy(target).ravel()
        mask_np = _to_numpy(mask).ravel().astype(int)
        errors = np.abs(pred_np - target_np)

        region_labels: List[str] = []
        region_errors: List[np.ndarray] = []
        for region_id in sorted(np.unique(mask_np)):
            idx = mask_np == region_id
            region_errors.append(errors[idx])
            region_labels.append(f"Region {region_id}")

        fig, ax = _setup_figure(nrows=1, ncols=1, figsize=(8, 5))
        bp = ax.boxplot(region_errors, labels=region_labels, patch_artist=True,
                        showfliers=True, notch=False)

        palette = sns.color_palette("Set2", n_colors=len(region_labels))
        for patch, color in zip(bp["boxes"], palette):
            patch.set_facecolor(color)
            patch.set_alpha(0.75)

        ax.set_xlabel("Material Region")
        ax.set_ylabel("Absolute Error [K]")
        ax.set_title("Error Distribution by Material Region")

        fig.tight_layout()
        _save_figure(fig, save_path)
        return fig


# ---------------------------------------------------------------------------
# 3. PhysicsValidationVisualizer
# ---------------------------------------------------------------------------


class PhysicsValidationVisualizer:
    """Visualizes physics-based validation diagnostics.

    Includes PDE residual heatmaps, boundary-condition compliance bar
    charts, and energy conservation tracking plots.
    """

    def plot_pde_residual(
        self,
        residual_field: Any,
        save_path: Optional[Union[str, Path]] = None,
    ) -> Figure:
        """Plot a heatmap of the PDE residual field.

        The residual is computed from the heat equation:
        ``rho * cp * dT/dt - div(k grad T) - q``.  Ideally, the residual
        should be close to zero everywhere.

        Args:
            residual_field: 2-D residual field of shape ``(B, 1, H, W)``
                or ``(H, W)``.
            save_path: Optional file path for saving the figure.

        Returns:
            The ``matplotlib.figure.Figure`` object.
        """
        residual_np = _to_numpy(residual_field)

        fig, ax = _setup_figure(nrows=1, ncols=1, figsize=(7, 5))

        # Use a symmetric colormap centred at zero.
        vabs = float(np.max(np.abs(residual_np)))
        im = ax.imshow(
            residual_np, cmap="RdBu_r", vmin=-vabs, vmax=vabs,
            aspect="auto", origin="lower",
        )
        fig.colorbar(im, ax=ax, label="PDE Residual")

        rms = float(np.sqrt(np.mean(residual_np ** 2)))
        ax.set_title(f"PDE Residual Field  (RMS = {rms:.4e})")
        ax.set_xlabel("x")
        ax.set_ylabel("y")

        fig.tight_layout()
        _save_figure(fig, save_path)
        return fig

    def plot_boundary_compliance(
        self,
        boundary_errors: Dict[str, float],
        save_path: Optional[Union[str, Path]] = None,
    ) -> Figure:
        """Bar chart of boundary-condition errors per domain edge.

        Args:
            boundary_errors: Dictionary mapping edge names (e.g.
                ``'top'``, ``'bottom'``, ``'left'``, ``'right'``) to their
                mean absolute BC residual values.
            save_path: Optional file path for saving the figure.

        Returns:
            The ``matplotlib.figure.Figure`` object.
        """
        edges = list(boundary_errors.keys())
        values = [boundary_errors[e] for e in edges]

        fig, ax = _setup_figure(nrows=1, ncols=1, figsize=(7, 5))
        palette = sns.color_palette("coolwarm", n_colors=len(edges))
        bars = ax.bar(edges, values, color=palette, edgecolor="black",
                      linewidth=0.8, alpha=0.85)

        # Annotate each bar with its value.
        for bar, val in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                bar.get_height() + max(values) * 0.02,
                f"{val:.4e}",
                ha="center", va="bottom", fontsize=9,
            )

        ax.set_xlabel("Boundary Edge")
        ax.set_ylabel("Mean |BC Residual|")
        ax.set_title("Boundary Condition Compliance")

        fig.tight_layout()
        _save_figure(fig, save_path)
        return fig

    def plot_energy_conservation(
        self,
        energy_over_time: Any,
        save_path: Optional[Union[str, Path]] = None,
    ) -> Figure:
        """Line plot of total thermal energy vs time.

        Total thermal energy is typically proportional to the spatial
        integral of the temperature field.  A well-behaved simulation
        should show smooth, physically plausible energy evolution.

        Args:
            energy_over_time: 1-D array-like of total energy values at
                each time step.
            save_path: Optional file path for saving the figure.

        Returns:
            The ``matplotlib.figure.Figure`` object.
        """
        energy = _to_numpy(np.asarray(energy_over_time)).ravel()
        steps = np.arange(len(energy))

        fig, ax = _setup_figure(nrows=1, ncols=1, figsize=(8, 5))
        ax.plot(steps, energy, linewidth=2.0, color="teal", label="Total Energy")

        # Show relative drift from initial.
        if len(energy) > 1 and energy[0] != 0.0:
            drift_pct = (energy - energy[0]) / np.abs(energy[0]) * 100.0
            ax2 = ax.twinx()
            ax2.plot(steps, drift_pct, linewidth=1.2, linestyle="--",
                     color="coral", alpha=0.7, label="Relative Drift [%]")
            ax2.set_ylabel("Relative Drift [%]", color="coral")
            ax2.tick_params(axis="y", labelcolor="coral")
            ax2.legend(loc="upper left", frameon=True)

        ax.set_xlabel("Time Step")
        ax.set_ylabel("Total Energy [J]")
        ax.set_title("Energy Conservation Over Time")
        ax.legend(loc="upper right", frameon=True)

        fig.tight_layout()
        _save_figure(fig, save_path)
        return fig


# ---------------------------------------------------------------------------
# 4. UncertaintyVisualizer
# ---------------------------------------------------------------------------


class UncertaintyVisualizer:
    """Visualizes uncertainty quantification results from ensemble or
    MC-dropout inference.

    Provides reliability diagrams (calibration curves), side-by-side
    prediction-uncertainty maps, and time-series prediction intervals.
    """

    def plot_calibration_curve(
        self,
        predicted_confidence: Any,
        observed_frequency: Any,
        save_path: Optional[Union[str, Path]] = None,
    ) -> Figure:
        """Plot a reliability diagram (calibration curve).

        A perfectly calibrated model lies on the diagonal.  Points above
        the diagonal indicate under-confidence; points below indicate
        over-confidence.

        Args:
            predicted_confidence: 1-D array of nominal confidence levels
                (x-axis).
            observed_frequency: 1-D array of corresponding empirical
                coverage fractions (y-axis).
            save_path: Optional file path for saving the figure.

        Returns:
            The ``matplotlib.figure.Figure`` object.
        """
        conf = _to_numpy(np.asarray(predicted_confidence)).ravel()
        freq = _to_numpy(np.asarray(observed_frequency)).ravel()

        fig, ax = _setup_figure(nrows=1, ncols=1, figsize=(6, 6))

        # Diagonal reference (perfect calibration).
        ax.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1.2,
                label="Perfect Calibration")

        ax.plot(conf, freq, marker="s", markersize=6, linewidth=2.0,
                color="dodgerblue", label="Model Calibration")

        # Shade the gap between model and ideal.
        ax.fill_between(conf, conf, freq, alpha=0.15, color="dodgerblue")

        # Compute calibration error summary.
        cal_error = float(np.mean(np.abs(conf - freq)))
        ax.text(
            0.05, 0.92,
            f"Mean Cal. Error = {cal_error:.4f}",
            transform=ax.transAxes, fontsize=10,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                      edgecolor="gray", alpha=0.8),
        )

        ax.set_xlabel("Predicted Confidence")
        ax.set_ylabel("Observed Frequency")
        ax.set_title("Reliability Diagram")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect("equal")
        ax.legend(loc="lower right", frameon=True)

        fig.tight_layout()
        _save_figure(fig, save_path)
        return fig

    def plot_uncertainty_map(
        self,
        mean_pred: Any,
        uncertainty: Any,
        save_path: Optional[Union[str, Path]] = None,
    ) -> Figure:
        """Side-by-side plot of predicted mean and uncertainty map.

        Args:
            mean_pred: Predicted mean temperature field of shape
                ``(B, 1, H, W)`` or ``(H, W)``.
            uncertainty: Predicted standard deviation (same shape).
            save_path: Optional file path for saving the figure.

        Returns:
            The ``matplotlib.figure.Figure`` object.
        """
        mean_np = _to_numpy(mean_pred)
        unc_np = _to_numpy(uncertainty)

        fig, axes = _setup_figure(nrows=1, ncols=2, figsize=(12, 5))

        # Left: mean prediction.
        im0 = axes[0].imshow(
            mean_np, cmap="hot", aspect="auto", origin="lower",
        )
        fig.colorbar(im0, ax=axes[0], label="Temperature [K]")
        axes[0].set_title("Predicted Mean")
        axes[0].set_xlabel("x")
        axes[0].set_ylabel("y")

        # Right: uncertainty (std dev).
        im1 = axes[1].imshow(
            unc_np, cmap="viridis", aspect="auto", origin="lower",
        )
        fig.colorbar(im1, ax=axes[1], label="Std. Dev. [K]")
        axes[1].set_title(
            f"Uncertainty Map  (mean $\\sigma$ = {float(np.mean(unc_np)):.4f} K)"
        )
        axes[1].set_xlabel("x")
        axes[1].set_ylabel("y")

        fig.tight_layout()
        _save_figure(fig, save_path)
        return fig

    def plot_prediction_intervals(
        self,
        predictions: Any,
        uncertainties: Any,
        targets: Any,
        pixel_idx: Optional[Tuple[int, int]] = None,
        save_path: Optional[Union[str, Path]] = None,
    ) -> Figure:
        """Plot time-series prediction intervals vs actual values.

        For a selected spatial pixel, this shows the predicted mean over
        time with shaded +/-1 sigma and +/-2 sigma intervals, overlaid
        with the ground-truth trajectory.

        Args:
            predictions: Predicted means of shape ``(T,)`` (already
                extracted for one pixel) or ``(T, H, W)`` / ``(B, T, H, W)``.
            uncertainties: Predicted standard deviations (same shape as
                *predictions*).
            targets: Ground-truth values (same shape as *predictions*).
            pixel_idx: Optional ``(row, col)`` tuple selecting the pixel
                to plot when inputs are spatial.  If ``None`` and inputs are
                spatial, the centre pixel is used.
            save_path: Optional file path for saving the figure.

        Returns:
            The ``matplotlib.figure.Figure`` object.
        """
        pred_np = _to_numpy(predictions)
        unc_np = _to_numpy(uncertainties)
        tgt_np = _to_numpy(targets)

        # Extract pixel time-series if spatial dimensions remain.
        if pred_np.ndim == 3:
            # (T, H, W) -- extract a single pixel.
            if pixel_idx is None:
                pixel_idx = (pred_np.shape[1] // 2, pred_np.shape[2] // 2)
            r, c = pixel_idx
            pred_np = pred_np[:, r, c]
            unc_np = unc_np[:, r, c]
            tgt_np = tgt_np[:, r, c]
        elif pred_np.ndim == 2:
            # Could be (T, pixels) -- use first pixel if ambiguous.
            if pred_np.shape[1] > 1:
                col = 0 if pixel_idx is None else pixel_idx[1]
                pred_np = pred_np[:, col]
                unc_np = unc_np[:, col]
                tgt_np = tgt_np[:, col]

        pred_np = pred_np.ravel()
        unc_np = unc_np.ravel()
        tgt_np = tgt_np.ravel()
        steps = np.arange(len(pred_np))

        fig, ax = _setup_figure(nrows=1, ncols=1, figsize=(10, 5))

        # Shaded intervals: +/-2 sigma (95%) then +/-1 sigma (68%).
        ax.fill_between(
            steps,
            pred_np - 2 * unc_np,
            pred_np + 2 * unc_np,
            alpha=0.18, color="steelblue", label="$\\pm 2\\sigma$ (95%)",
        )
        ax.fill_between(
            steps,
            pred_np - unc_np,
            pred_np + unc_np,
            alpha=0.35, color="steelblue", label="$\\pm 1\\sigma$ (68%)",
        )

        ax.plot(steps, pred_np, linewidth=1.8, color="navy",
                label="Predicted Mean")
        ax.plot(steps, tgt_np, linewidth=1.5, linestyle="--", color="red",
                label="Ground Truth")

        pixel_label = ""
        if pixel_idx is not None:
            pixel_label = f" at pixel ({pixel_idx[0]}, {pixel_idx[1]})"
        ax.set_xlabel("Time Step")
        ax.set_ylabel("Temperature [K]")
        ax.set_title(f"Prediction Intervals{pixel_label}")
        ax.legend(loc="best", frameon=True)

        fig.tight_layout()
        _save_figure(fig, save_path)
        return fig


# ---------------------------------------------------------------------------
# 5. TrainingMonitor
# ---------------------------------------------------------------------------


class TrainingMonitor:
    """Visualizes training diagnostics: loss curves, learning rate
    schedules, gradient norms, and weight distributions.
    """

    @staticmethod
    def _moving_average(values: np.ndarray, window: int = 10) -> np.ndarray:
        """Compute a causal (left-looking) moving average.

        Pads the beginning with the first value so the output has the
        same length as the input.

        Args:
            values: 1-D array of values to smooth.
            window: Width of the averaging window (default 10).

        Returns:
            Smoothed 1-D array of the same length.
        """
        if len(values) < window:
            window = max(1, len(values))
        kernel = np.ones(window) / window
        padded = np.concatenate([np.full(window - 1, values[0]), values])
        return np.convolve(padded, kernel, mode="valid")

    def plot_loss_curves(
        self,
        train_losses: Dict[str, List[float]],
        val_losses: Dict[str, List[float]],
        loss_names: Optional[List[str]] = None,
        save_path: Optional[Union[str, Path]] = None,
    ) -> Figure:
        """Multi-panel loss curves with moving-average smoothing.

        Creates one subplot per loss component (e.g. ``total``, ``data``,
        ``pde``, ``bc``, ``consistency``) with both training and
        validation curves plus a smoothed moving average overlay.

        Args:
            train_losses: Dictionary mapping loss component names to lists
                of per-epoch training loss values.
            val_losses: Dictionary mapping loss component names to lists
                of per-epoch validation loss values.
            loss_names: Optional explicit ordering of loss component names
                to plot.  If ``None``, all keys present in *train_losses*
                are used (sorted alphabetically).
            save_path: Optional file path for saving the figure.

        Returns:
            The ``matplotlib.figure.Figure`` object.
        """
        if loss_names is None:
            loss_names = sorted(train_losses.keys())

        n_panels = len(loss_names)
        ncols = min(n_panels, 3)
        nrows = int(np.ceil(n_panels / ncols))

        fig, axes = _setup_figure(nrows=nrows, ncols=ncols,
                                  figsize=(ncols * 5, nrows * 4))
        if n_panels == 1:
            axes = np.array([axes])
        axes_flat = np.array(axes).flatten()

        for i, name in enumerate(loss_names):
            ax = axes_flat[i]

            if name in train_losses:
                train_arr = np.array(train_losses[name])
                epochs = np.arange(1, len(train_arr) + 1)
                ax.plot(epochs, train_arr, alpha=0.35, color="steelblue",
                        linewidth=0.8)
                smoothed_train = self._moving_average(train_arr)
                ax.plot(epochs, smoothed_train, color="steelblue",
                        linewidth=2.0, label="Train (smoothed)")

            if name in val_losses:
                val_arr = np.array(val_losses[name])
                val_epochs = np.arange(1, len(val_arr) + 1)
                ax.plot(val_epochs, val_arr, alpha=0.35, color="coral",
                        linewidth=0.8)
                smoothed_val = self._moving_average(val_arr)
                ax.plot(val_epochs, smoothed_val, color="coral",
                        linewidth=2.0, label="Val (smoothed)")

            ax.set_xlabel("Epoch")
            ax.set_ylabel("Loss")
            ax.set_title(name.replace("_", " ").title())
            ax.legend(frameon=True, fontsize=8)
            ax.set_yscale("log")

        # Hide unused axes.
        for j in range(n_panels, len(axes_flat)):
            axes_flat[j].set_visible(False)

        fig.suptitle("Training & Validation Loss Curves", fontsize=14, y=1.01)
        fig.tight_layout()
        _save_figure(fig, save_path)
        return fig

    def plot_learning_rate(
        self,
        lr_history: Any,
        save_path: Optional[Union[str, Path]] = None,
    ) -> Figure:
        """Plot the learning-rate schedule over training epochs.

        Args:
            lr_history: 1-D array-like of learning rate values at each
                epoch (or step).
            save_path: Optional file path for saving the figure.

        Returns:
            The ``matplotlib.figure.Figure`` object.
        """
        lr = _to_numpy(np.asarray(lr_history)).ravel()
        epochs = np.arange(1, len(lr) + 1)

        fig, ax = _setup_figure(nrows=1, ncols=1, figsize=(8, 4))
        ax.plot(epochs, lr, linewidth=2.0, color="darkorange")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Learning Rate")
        ax.set_title("Learning Rate Schedule")
        ax.set_yscale("log")

        fig.tight_layout()
        _save_figure(fig, save_path)
        return fig

    def plot_gradient_norms(
        self,
        grad_norms: Any,
        save_path: Optional[Union[str, Path]] = None,
    ) -> Figure:
        """Plot gradient norms over training iterations.

        Args:
            grad_norms: 1-D array-like of gradient L2-norm values logged
                at each training step.
            save_path: Optional file path for saving the figure.

        Returns:
            The ``matplotlib.figure.Figure`` object.
        """
        norms = _to_numpy(np.asarray(grad_norms)).ravel()
        steps = np.arange(1, len(norms) + 1)

        fig, ax = _setup_figure(nrows=1, ncols=1, figsize=(9, 4))
        ax.plot(steps, norms, alpha=0.4, color="mediumseagreen",
                linewidth=0.6)
        smoothed = self._moving_average(norms, window=20)
        ax.plot(steps, smoothed, color="mediumseagreen", linewidth=2.0,
                label="Smoothed (window=20)")
        ax.set_xlabel("Training Step")
        ax.set_ylabel("Gradient L2 Norm")
        ax.set_title("Gradient Norm Over Training")
        ax.legend(frameon=True)

        fig.tight_layout()
        _save_figure(fig, save_path)
        return fig

    def plot_weight_distribution(
        self,
        model: Any,
        save_path: Optional[Union[str, Path]] = None,
    ) -> Figure:
        """Histogram of all trainable weight values from a PyTorch model.

        Aggregates all ``requires_grad=True`` parameters into a single
        flat vector and plots the resulting distribution.  A vertical
        line marks zero.

        Args:
            model: A ``torch.nn.Module`` whose parameters will be
                visualized.
            save_path: Optional file path for saving the figure.

        Returns:
            The ``matplotlib.figure.Figure`` object.

        Raises:
            TypeError: If *model* does not have a ``parameters()`` method.
        """
        if not hasattr(model, "parameters"):
            raise TypeError(
                f"Expected a torch.nn.Module with parameters(), got "
                f"{type(model).__name__}."
            )

        all_weights: List[np.ndarray] = []
        layer_names: List[str] = []
        for name, param in model.named_parameters():
            if param.requires_grad:
                all_weights.append(param.detach().cpu().numpy().ravel())
                layer_names.append(name)

        if not all_weights:
            logger.warning("Model has no trainable parameters to visualize.")
            fig, ax = _setup_figure(nrows=1, ncols=1, figsize=(7, 5))
            ax.text(0.5, 0.5, "No trainable parameters found",
                    ha="center", va="center", transform=ax.transAxes)
            fig.tight_layout()
            _save_figure(fig, save_path)
            return fig

        flat_weights = np.concatenate(all_weights)

        # Determine layout: overview + per-layer histograms for first few
        # layers (up to 8 layers).
        max_layer_panels = min(8, len(all_weights))
        n_panels = 1 + max_layer_panels
        ncols = min(3, n_panels)
        nrows = int(np.ceil(n_panels / ncols))

        fig, axes = _setup_figure(nrows=nrows, ncols=ncols,
                                  figsize=(ncols * 5, nrows * 4))
        axes_flat = np.array(axes).flatten()

        # Panel 0: aggregate distribution.
        ax0 = axes_flat[0]
        ax0.hist(flat_weights, bins=100, color="slategray", edgecolor="white",
                 alpha=0.85, density=True)
        ax0.axvline(0.0, color="red", linestyle="--", linewidth=1.0)
        ax0.set_title(
            f"All Weights  (n={len(flat_weights):,}, "
            f"$\\mu$={float(np.mean(flat_weights)):.4f}, "
            f"$\\sigma$={float(np.std(flat_weights)):.4f})"
        )
        ax0.set_xlabel("Weight Value")
        ax0.set_ylabel("Density")

        # Per-layer panels.
        palette = sns.color_palette("husl", n_colors=max_layer_panels)
        for i in range(max_layer_panels):
            ax = axes_flat[1 + i]
            w = all_weights[i]
            ax.hist(w, bins=60, color=palette[i], edgecolor="white",
                    alpha=0.8, density=True)
            ax.axvline(0.0, color="red", linestyle="--", linewidth=0.8)
            short_name = layer_names[i]
            if len(short_name) > 30:
                short_name = "..." + short_name[-27:]
            ax.set_title(short_name, fontsize=9)
            ax.set_xlabel("Value")
            ax.set_ylabel("Density")

        # Hide unused axes.
        for j in range(n_panels, len(axes_flat)):
            axes_flat[j].set_visible(False)

        fig.suptitle("Weight Distribution", fontsize=14, y=1.01)
        fig.tight_layout()
        _save_figure(fig, save_path)
        return fig

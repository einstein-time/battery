"""Material properties and geometry utilities for battery thermal simulation.

This module provides functions to create battery-pack material layouts,
compute signed distance fields, and manage material-specific thermal properties.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
from scipy.ndimage import distance_transform_edt

logger = logging.getLogger(__name__)


@dataclass
class MaterialProperties:
    """Thermal properties for a single material.

    Attributes:
        k: Thermal conductivity [W/(m*K)]
        rho: Density [kg/m^3]
        cp: Specific heat capacity [J/(kg*K)]
        name: Optional material name for identification
    """
    k: float
    rho: float
    cp: float
    name: str = ""

    @property
    def alpha(self) -> float:
        """Thermal diffusivity [m^2/s]."""
        return self.k / (self.rho * self.cp)


# Predefined material library
MATERIAL_LIBRARY: Dict[str, MaterialProperties] = {
    "battery_cell": MaterialProperties(k=2.0, rho=2500.0, cp=700.0, name="Battery Cell"),
    "coolant_water": MaterialProperties(k=0.6, rho=998.0, cp=4182.0, name="Water Coolant"),
    "coolant_air": MaterialProperties(k=0.026, rho=1.225, cp=1006.0, name="Air Coolant"),
    "insulation": MaterialProperties(k=0.04, rho=30.0, cp=1400.0, name="Insulation"),
}


def create_material_mask(
    grid_size: int,
    n_cells: int = 4,
    cell_spacing: float = 0.15,
    coolant_thickness: float = 0.1,
    layout: str = "grid",
) -> np.ndarray:
    """Create a 2D material mask representing a battery pack layout.

    The mask is an integer array where each value represents a material ID:
      - 0: Battery cell
      - 1: Coolant (water or air channels)
      - 2: Insulation (outer boundary)

    Args:
        grid_size: Number of grid points in each dimension (creates square domain).
        n_cells: Number of battery cells (e.g., 4 creates a 2x2 grid).
        cell_spacing: Fraction of domain width between cell centers (0.1 = 10%).
        coolant_thickness: Fraction of domain used for coolant channels.
        layout: Geometry layout, one of ["grid", "line", "single"].

    Returns:
        Material mask of shape ``(grid_size, grid_size)`` with dtype int8.

    Raises:
        ValueError: If layout is not recognized.
    """
    mask = np.full((grid_size, grid_size), 2, dtype=np.int8)  # Default: insulation

    if layout == "single":
        # Single centered battery cell
        margin = int(grid_size * 0.2)
        mask[margin:-margin, margin:-margin] = 0  # Cell

        # Coolant channels around the cell
        coolant_width = int(grid_size * coolant_thickness)
        mask[margin - coolant_width:margin, :] = 1
        mask[-margin:-margin + coolant_width, :] = 1
        mask[:, margin - coolant_width:margin] = 1
        mask[:, -margin:-margin + coolant_width] = 1

    elif layout == "line":
        # Horizontal line of cells
        cell_width = int(grid_size * 0.15)
        coolant_gap = int(grid_size * coolant_thickness)
        y_start = grid_size // 2 - cell_width // 2
        y_end = y_start + cell_width

        x_positions = np.linspace(0.2, 0.8, n_cells)
        for x_frac in x_positions:
            x_center = int(grid_size * x_frac)
            x_start = x_center - cell_width // 2
            x_end = x_start + cell_width
            mask[y_start:y_end, x_start:x_end] = 0

            # Coolant gaps between cells
            if x_end < grid_size - coolant_gap:
                mask[y_start:y_end, x_end:x_end + coolant_gap] = 1

    elif layout == "grid":
        # Grid layout (e.g., 2x2 for n_cells=4)
        n_rows = int(np.sqrt(n_cells))
        n_cols = (n_cells + n_rows - 1) // n_rows

        cell_width = int(grid_size * (1.0 - cell_spacing * (n_cols + 1)) / n_cols)
        gap = int(grid_size * cell_spacing)

        for row in range(n_rows):
            for col in range(n_cols):
                if row * n_cols + col >= n_cells:
                    break
                y_start = gap + row * (cell_width + gap)
                y_end = y_start + cell_width
                x_start = gap + col * (cell_width + gap)
                x_end = x_start + cell_width

                # Ensure within bounds
                y_end = min(y_end, grid_size - gap)
                x_end = min(x_end, grid_size - gap)

                mask[y_start:y_end, x_start:x_end] = 0

        # Fill gaps between cells with coolant
        mask[mask == 2] = 1  # Everything not cell becomes coolant

        # Outer boundary is insulation
        boundary_width = max(1, int(grid_size * 0.05))
        mask[:boundary_width, :] = 2
        mask[-boundary_width:, :] = 2
        mask[:, :boundary_width] = 2
        mask[:, -boundary_width:] = 2

    else:
        raise ValueError(f"Unknown layout: {layout}. Choose from ['grid', 'line', 'single'].")

    logger.info(
        f"Created material mask (layout={layout}, grid_size={grid_size}, n_cells={n_cells}): "
        f"{(mask == 0).sum()} cell pixels, {(mask == 1).sum()} coolant, {(mask == 2).sum()} insulation"
    )

    return mask


def compute_signed_distance(
    mask: np.ndarray,
    material_id: int,
) -> np.ndarray:
    """Compute signed distance field (SDF) to a specific material region.

    Positive values indicate distance outside the region (in grid cells),
    negative values indicate distance inside the region.

    Args:
        mask: Material mask array of shape ``(H, W)`` with integer material IDs.
        material_id: Target material ID to compute distance to.

    Returns:
        Signed distance field of shape ``(H, W)`` with dtype float32.
    """
    binary_mask = (mask == material_id).astype(np.uint8)

    # Distance transform from outside the region
    dist_outside = distance_transform_edt(1 - binary_mask)

    # Distance transform from inside the region
    dist_inside = distance_transform_edt(binary_mask)

    # Combine: negative inside, positive outside
    sdf = dist_outside - dist_inside

    return sdf.astype(np.float32)


def create_property_field(
    mask: np.ndarray,
    values: np.ndarray,
) -> np.ndarray:
    """Create a spatially-varying property field from material mask and per-material values.

    Args:
        mask: Material mask of shape ``(H, W)`` with integer material IDs.
        values: Array of property values, one per material ID. For example,
            ``values = [k_cell, k_coolant, k_insulation]`` for thermal conductivity.

    Returns:
        Property field of shape ``(H, W)`` where each pixel has the value
        corresponding to its material ID.
    """
    return values[mask].astype(np.float32)

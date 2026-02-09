"""Material properties and signed distance fields for battery thermal simulation.

This module provides material property definitions, geometry masks, signed distance
field computation, and effective conductivity calculations for a 2D battery pack
thermal management simulation. The geometry models rectangular battery cells
separated by coolant channels and surrounded by insulation.

Typical usage::

    props = MaterialProperties(coolant="water")
    mask = create_material_mask(grid_size=128)
    sdf = compute_signed_distance(mask, material_id=0)
    k_eff = get_effective_conductivity(mask, props.conductivity_array())
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Literal, Optional, Tuple, Union

import numpy as np
from scipy import ndimage


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_MATERIAL_IDS: Dict[str, int] = {
    "battery": 0,
    "coolant": 1,
    "insulation": 2,
}
"""Canonical mapping from material name to integer label used in masks."""

_DEFAULT_GRID_SIZE: int = 128
"""Default spatial resolution (cells per side) for the simulation domain."""


# ---------------------------------------------------------------------------
# Material property data
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ThermalProps:
    """Immutable thermal properties for a single material.

    Attributes:
        k: Thermal conductivity [W/(m*K)].
        rho: Density [kg/m^3].
        cp: Specific heat capacity [J/(kg*K)].
    """

    k: float
    rho: float
    cp: float

    @property
    def alpha(self) -> float:
        """Thermal diffusivity [m^2/s]."""
        return self.k / (self.rho * self.cp)


# Pre-defined property sets ------------------------------------------------

BATTERY_CELL = ThermalProps(k=3.0, rho=2500.0, cp=700.0)
"""Default battery cell properties (mid-range k=3.0 in [1.0, 5.0])."""

COOLANT_AIR = ThermalProps(k=0.026, rho=1.225, cp=1006.0)
"""Air coolant properties at ~25 degC."""

COOLANT_WATER = ThermalProps(k=0.6, rho=998.0, cp=4182.0)
"""Water coolant properties at ~25 degC."""

INSULATION = ThermalProps(k=0.04, rho=30.0, cp=1400.0)
"""Typical foam/fibre insulation properties."""


# ---------------------------------------------------------------------------
# MaterialProperties container
# ---------------------------------------------------------------------------

@dataclass
class MaterialProperties:
    """Container for all material properties used in a battery thermal simulation.

    The class stores :class:`ThermalProps` for three materials (battery cell,
    coolant, and insulation) and provides convenience accessors that return
    arrays indexed by material id.

    Args:
        coolant: Which coolant fluid to use (``"air"`` or ``"water"``).
            Defaults to ``"water"``.
        battery_k: Override battery-cell conductivity [W/(m*K)].  Must be in
            the physically realistic range [1.0, 5.0].  Defaults to 3.0.

    Raises:
        ValueError: If *coolant* is not ``"air"`` or ``"water"``, or if
            *battery_k* is outside [1.0, 5.0].

    Example::

        props = MaterialProperties(coolant="water", battery_k=2.0)
        k_arr = props.conductivity_array()   # shape (3,)
    """

    coolant: Literal["air", "water"] = "water"
    battery_k: float = 3.0

    # Derived (set in __post_init__) ----------------------------------------
    battery: ThermalProps = field(init=False, repr=False)
    coolant_props: ThermalProps = field(init=False, repr=False)
    insulation: ThermalProps = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if self.coolant not in ("air", "water"):
            raise ValueError(
                f"coolant must be 'air' or 'water', got {self.coolant!r}"
            )
        if not 1.0 <= self.battery_k <= 5.0:
            raise ValueError(
                f"battery_k must be in [1.0, 5.0], got {self.battery_k}"
            )

        self.battery = ThermalProps(k=self.battery_k, rho=2500.0, cp=700.0)
        self.coolant_props = COOLANT_WATER if self.coolant == "water" else COOLANT_AIR
        self.insulation = INSULATION

    # Convenience array builders --------------------------------------------

    def conductivity_array(self) -> np.ndarray:
        """Return k values as ``ndarray`` of shape ``(3,)`` indexed by material id.

        Returns:
            1-D float64 array ``[k_battery, k_coolant, k_insulation]``.
        """
        return np.array(
            [self.battery.k, self.coolant_props.k, self.insulation.k],
            dtype=np.float64,
        )

    def density_array(self) -> np.ndarray:
        """Return rho values as ``ndarray`` of shape ``(3,)`` indexed by material id.

        Returns:
            1-D float64 array ``[rho_battery, rho_coolant, rho_insulation]``.
        """
        return np.array(
            [self.battery.rho, self.coolant_props.rho, self.insulation.rho],
            dtype=np.float64,
        )

    def heat_capacity_array(self) -> np.ndarray:
        """Return cp values as ``ndarray`` of shape ``(3,)`` indexed by material id.

        Returns:
            1-D float64 array ``[cp_battery, cp_coolant, cp_insulation]``.
        """
        return np.array(
            [self.battery.cp, self.coolant_props.cp, self.insulation.cp],
            dtype=np.float64,
        )

    def diffusivity_array(self) -> np.ndarray:
        """Return thermal diffusivity alpha = k/(rho*cp) for each material.

        Returns:
            1-D float64 array of shape ``(3,)``.
        """
        k = self.conductivity_array()
        rho = self.density_array()
        cp = self.heat_capacity_array()
        return k / (rho * cp)

    def props_for(self, material_id: int) -> ThermalProps:
        """Return :class:`ThermalProps` for a given material id.

        Args:
            material_id: Integer label (0=battery, 1=coolant, 2=insulation).

        Returns:
            The corresponding :class:`ThermalProps` instance.

        Raises:
            KeyError: If *material_id* is not in {0, 1, 2}.
        """
        mapping = {0: self.battery, 1: self.coolant_props, 2: self.insulation}
        if material_id not in mapping:
            raise KeyError(
                f"Unknown material_id {material_id}; expected one of {set(mapping)}"
            )
        return mapping[material_id]


# ---------------------------------------------------------------------------
# Geometry / mask creation
# ---------------------------------------------------------------------------

def create_material_mask(grid_size: int = _DEFAULT_GRID_SIZE) -> np.ndarray:
    """Create a 2-D material mask for a simplified battery-pack cross-section.

    The domain is a square of *grid_size* x *grid_size* cells.  The layout is:

    * **Insulation** (id 2): outer ring of thickness ~8 % of domain width.
    * **Battery cells** (id 0): 2x3 grid of rectangular cells centred inside
      the insulation layer.
    * **Coolant channels** (id 1): everything between cells and insulation.

    Args:
        grid_size: Number of grid points along each axis.  Must be >= 16.

    Returns:
        Integer ``ndarray`` of shape ``(grid_size, grid_size)`` with values in
        ``{0, 1, 2}``.

    Raises:
        ValueError: If *grid_size* < 16.

    Example::

        mask = create_material_mask(128)
        assert mask.shape == (128, 128)
        assert set(np.unique(mask)) == {0, 1, 2}
    """
    if grid_size < 16:
        raise ValueError(f"grid_size must be >= 16, got {grid_size}")

    mask = np.ones((grid_size, grid_size), dtype=np.int32)  # default: coolant

    # --- Insulation border --------------------------------------------------
    border = max(1, int(round(0.08 * grid_size)))
    mask[:border, :] = 2
    mask[-border:, :] = 2
    mask[:, :border] = 2
    mask[:, -border:] = 2

    # --- Battery cells (2 rows x 3 columns) --------------------------------
    inner_x_start = border + max(1, int(round(0.04 * grid_size)))
    inner_x_end = grid_size - border - max(1, int(round(0.04 * grid_size)))
    inner_y_start = border + max(1, int(round(0.04 * grid_size)))
    inner_y_end = grid_size - border - max(1, int(round(0.04 * grid_size)))

    n_rows: int = 2
    n_cols: int = 3
    gap_frac: float = 0.12  # fraction of inner span used for gaps

    inner_w = inner_x_end - inner_x_start
    inner_h = inner_y_end - inner_y_start

    gap_x = max(1, int(round(gap_frac * inner_w)))
    gap_y = max(1, int(round(gap_frac * inner_h)))

    cell_w = (inner_w - (n_cols - 1) * gap_x) // n_cols
    cell_h = (inner_h - (n_rows - 1) * gap_y) // n_rows

    for row in range(n_rows):
        for col in range(n_cols):
            y0 = inner_y_start + row * (cell_h + gap_y)
            x0 = inner_x_start + col * (cell_w + gap_x)
            y1 = y0 + cell_h
            x1 = x0 + cell_w
            # Clamp to inner region
            y1 = min(y1, inner_y_end)
            x1 = min(x1, inner_x_end)
            mask[y0:y1, x0:x1] = 0

    return mask


# ---------------------------------------------------------------------------
# Signed distance field
# ---------------------------------------------------------------------------

def compute_signed_distance(
    mask: np.ndarray,
    material_id: int,
) -> np.ndarray:
    """Compute the signed distance field for a material region.

    The SDF is **negative** inside the material region, **positive** outside,
    and **zero** on the boundary.  Distances are measured in grid-cell units
    using the Euclidean distance transform from :mod:`scipy.ndimage`.

    Args:
        mask: Integer array of shape ``(H, W)`` with material labels.
        material_id: The target material label.

    Returns:
        Float64 array of shape ``(H, W)`` with the signed distance field.

    Raises:
        ValueError: If *material_id* does not appear in *mask*.

    Example::

        mask = create_material_mask(64)
        sdf = compute_signed_distance(mask, material_id=0)
        assert sdf.shape == mask.shape
        assert sdf[mask == 0].max() <= 0.0   # inside is non-positive
    """
    region = (mask == material_id)

    if not region.any():
        raise ValueError(
            f"material_id {material_id} not found in mask "
            f"(unique values: {np.unique(mask).tolist()})"
        )

    # Distance from *outside* the region to the nearest interior point.
    dist_outside = ndimage.distance_transform_edt(~region)
    # Distance from *inside* the region to the nearest exterior point.
    dist_inside = ndimage.distance_transform_edt(region)

    # Convention: negative inside, positive outside.
    sdf = dist_outside - dist_inside

    return sdf.astype(np.float64)


# ---------------------------------------------------------------------------
# Effective conductivity at interfaces
# ---------------------------------------------------------------------------

def get_effective_conductivity(
    mask: np.ndarray,
    k_values: np.ndarray,
    method: Literal["harmonic", "arithmetic", "geometric"] = "harmonic",
) -> np.ndarray:
    """Compute a spatially-varying effective thermal conductivity field.

    At every grid cell the conductivity is determined by its material label.
    At **interface cells** (cells adjacent to a different material) the
    effective conductivity is computed from the two material conductivities
    using the chosen averaging method.  This is important for stability and
    accuracy of the finite-difference heat-equation solver near interfaces.

    For a pair of conductivities *k_a* and *k_b*:

    * **harmonic** (default): ``k_eff = 2*k_a*k_b / (k_a + k_b)``
    * **arithmetic**: ``k_eff = (k_a + k_b) / 2``
    * **geometric**: ``k_eff = sqrt(k_a * k_b)``

    Args:
        mask: Integer material mask of shape ``(H, W)``.
        k_values: 1-D array of conductivities indexed by material id.
            Must have length >= ``mask.max() + 1``.
        method: Averaging scheme for interface cells.

    Returns:
        Float64 array of shape ``(H, W)`` with the effective conductivity at
        each grid point.

    Raises:
        ValueError: If *method* is not one of the three supported strings,
            or if *k_values* is too short.

    Example::

        props = MaterialProperties()
        mask = create_material_mask(128)
        k_eff = get_effective_conductivity(
            mask, props.conductivity_array(), method="harmonic"
        )
    """
    valid_methods = ("harmonic", "arithmetic", "geometric")
    if method not in valid_methods:
        raise ValueError(f"method must be one of {valid_methods}, got {method!r}")

    n_materials = int(mask.max()) + 1
    if k_values.shape[0] < n_materials:
        raise ValueError(
            f"k_values has length {k_values.shape[0]} but mask contains "
            f"material ids up to {mask.max()}; need at least {n_materials} entries"
        )

    ny, nx = mask.shape

    # Base conductivity: simply look up per-cell material.
    k_field = k_values[mask].astype(np.float64)

    # Identify interface cells (cells whose 4-connected neighbours include a
    # different material).  For each interface cell we average its own k with
    # the *minimum-k* neighbour to be conservative (worst-case heat path).
    # Pad mask to handle boundaries.
    padded = np.pad(mask, 1, mode="edge")

    # Shifts: up, down, left, right.
    neighbours = np.stack(
        [
            padded[:-2, 1:-1],   # up
            padded[2:, 1:-1],    # down
            padded[1:-1, :-2],   # left
            padded[1:-1, 2:],    # right
        ],
        axis=0,
    )  # shape (4, ny, nx)

    # Boolean: is any neighbour a different material?
    is_interface = np.any(neighbours != mask[np.newaxis, :, :], axis=0)

    if not is_interface.any():
        return k_field

    # For interface cells, gather neighbour conductivities.
    neighbour_k = k_values[neighbours]  # (4, ny, nx)

    # We average cell k with the mean of *distinct-material* neighbour k's.
    # For simplicity and robustness, average with the mean of all 4 neighbour k
    # values at interface cells.
    k_self = k_field[is_interface]
    k_neigh_mean = neighbour_k[:, is_interface].mean(axis=0)

    if method == "harmonic":
        # Avoid division by zero: add tiny epsilon.
        eps = 1e-30
        k_eff = 2.0 * k_self * k_neigh_mean / (k_self + k_neigh_mean + eps)
    elif method == "arithmetic":
        k_eff = 0.5 * (k_self + k_neigh_mean)
    else:  # geometric
        k_eff = np.sqrt(k_self * k_neigh_mean)

    k_field[is_interface] = k_eff

    return k_field


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def material_id_from_name(name: str) -> int:
    """Look up integer id for a material name.

    Args:
        name: One of ``"battery"``, ``"coolant"``, ``"insulation"``.

    Returns:
        Integer material id.

    Raises:
        KeyError: If *name* is not recognised.
    """
    try:
        return _MATERIAL_IDS[name.lower()]
    except KeyError:
        raise KeyError(
            f"Unknown material name {name!r}; "
            f"expected one of {list(_MATERIAL_IDS)}"
        ) from None


def save_mask(mask: np.ndarray, path: Union[str, Path]) -> Path:
    """Save a material mask to a ``.npy`` file.

    Args:
        mask: Integer mask array.
        path: Destination file path (extension ``.npy`` recommended).

    Returns:
        Resolved :class:`~pathlib.Path` to the saved file.
    """
    path = Path(path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, mask)
    return path


def load_mask(path: Union[str, Path]) -> np.ndarray:
    """Load a material mask from a ``.npy`` file.

    Args:
        path: Source file path.

    Returns:
        Integer ``ndarray`` with the material mask.

    Raises:
        FileNotFoundError: If *path* does not exist.
    """
    path = Path(path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"Mask file not found: {path}")
    return np.load(path)

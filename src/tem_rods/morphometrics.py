"""
Shape metrics inspired by Aviles & Lear (ACS Nanosci. Au 2025).

Provides Feret (caliper) diameters, circularity (ASTM roundness), and
equivalent-area diameter — complementary to ellipse major/minor axes.
"""

from __future__ import annotations

import math

import numpy as np


def feret_diameters_px(coords: np.ndarray, *, n_angles: int = 180) -> tuple[float, float]:
    """
    Maximum and minimum Feret (caliper) diameters in pixels.

    For each angle, project particle coordinates onto a unit axis and take the
    span (max − min). Max over angles ≈ max Feret; min over angles ≈ min Feret.
    """
    if coords is None or len(coords) < 2:
        return 0.0, 0.0
    # regionprops coords are (row, col) = (y, x); project in Cartesian (x, y).
    pts = np.asarray(coords, dtype=float)
    xy = np.column_stack([pts[:, 1], pts[:, 0]])
    angles = np.linspace(0.0, math.pi, n_angles, endpoint=False)
    spans = np.empty(n_angles, dtype=float)
    for i, theta in enumerate(angles):
        c, s = math.cos(theta), math.sin(theta)
        proj = xy[:, 0] * c + xy[:, 1] * s
        spans[i] = float(proj.max() - proj.min())
    return float(spans.max()), float(spans.min())


def circularity_from_area_perimeter(area: float, perimeter: float) -> float:
    """
    Circularity = 4π·area / perimeter² (FIJI / ASTM F1877 roundness).

    Ideal circle → 1. Discrete pixel boundaries can exceed 1 slightly; clamp.
    """
    if perimeter <= 0 or area <= 0:
        return 0.0
    value = (4.0 * math.pi * float(area)) / (float(perimeter) ** 2)
    return float(min(max(value, 0.0), 1.0))


def equivalent_diameter_px(area_px: float) -> float:
    """Area-equivalent circular diameter in pixels: 2√(A/π)."""
    if area_px <= 0:
        return 0.0
    return float(2.0 * math.sqrt(float(area_px) / math.pi))


def region_morphometrics(region) -> dict[str, float]:
    """
    Compute Aviles/Lear-style metrics for a skimage regionprops region.

    Prefer skimage ``feret_diameter_max`` when available; always compute min
    Feret via caliper projections for consistency.
    """
    coords = region.coords
    feret_max_px, feret_min_px = feret_diameters_px(coords)
    sk_max = float(getattr(region, "feret_diameter_max", 0.0) or 0.0)
    if sk_max > 0:
        feret_max_px = max(feret_max_px, sk_max)
    if feret_min_px <= 0 and feret_max_px > 0:
        feret_min_px = float(region.minor_axis_length) or feret_max_px
    if feret_max_px < feret_min_px:
        feret_max_px, feret_min_px = feret_min_px, feret_max_px

    area = float(region.area)
    perimeter = float(region.perimeter)
    return {
        "feret_max_px": feret_max_px,
        "feret_min_px": feret_min_px,
        "circularity": circularity_from_area_perimeter(area, perimeter),
        "equiv_diameter_px": equivalent_diameter_px(area),
    }

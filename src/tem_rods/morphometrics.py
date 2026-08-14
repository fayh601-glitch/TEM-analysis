"""
Shape metrics inspired by Aviles & Lear (ACS Nanosci. Au 2025).

Provides Feret (caliper) diameters, circularity (ASTM roundness), and
equivalent-area diameter — complementary to ellipse major/minor axes.
"""

from __future__ import annotations

import math

import numpy as np


def principal_caliper_px(coords: np.ndarray) -> tuple[float, float]:
    """
    Length and width as spans along the particle's principal axes.

    This matches ImageJ: a straight line along the rod and a perpendicular
    line across it. Pixel-center projections are ~1 px short of an edge-to-edge
    line, so 1 px is added to each span.
    """
    pts = np.asarray(coords, dtype=float)
    if pts.ndim != 2 or len(pts) < 2:
        return 0.0, 0.0
    xy = np.column_stack([pts[:, 1], pts[:, 0]])
    centered = xy - xy.mean(axis=0)
    if len(xy) == 2:
        length = float(np.hypot(*(xy[1] - xy[0]))) + 1.0
        return length, 1.0
    cov = np.cov(centered, rowvar=False)
    cov = np.atleast_2d(cov)
    if cov.shape != (2, 2) or not np.all(np.isfinite(cov)):
        length = float(np.ptp(xy[:, 0])) + 1.0
        width = float(np.ptp(xy[:, 1])) + 1.0
        if width > length:
            length, width = width, length
        return length, width
    evals, evecs = np.linalg.eigh(cov)
    major = evecs[:, int(np.argmax(evals))]
    minor = np.array([-major[1], major[0]], dtype=float)
    length = float(np.ptp(centered @ major)) + 1.0
    width = float(np.ptp(centered @ minor)) + 1.0
    if width > length:
        length, width = width, length
    return length, max(width, 1e-6)


def principal_axis_xy(coords: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return unit major and minor axis vectors in (x, y) image coordinates."""
    pts = np.asarray(coords, dtype=float)
    xy = np.column_stack([pts[:, 1], pts[:, 0]])
    centered = xy - xy.mean(axis=0)
    if len(xy) < 3:
        return np.array([1.0, 0.0]), np.array([0.0, 1.0])
    cov = np.atleast_2d(np.cov(centered, rowvar=False))
    if cov.shape != (2, 2) or not np.all(np.isfinite(cov)):
        return np.array([1.0, 0.0]), np.array([0.0, 1.0])
    evals, evecs = np.linalg.eigh(cov)
    major = evecs[:, int(np.argmax(evals))]
    minor = np.array([-major[1], major[0]], dtype=float)
    return major, minor


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
    directions = np.stack([np.cos(angles), np.sin(angles)], axis=0)
    proj = xy @ directions
    spans = proj.max(axis=0) - proj.min(axis=0)
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

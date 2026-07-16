"""
Shape matching — find particles similar to a user-traced outline
==============================================================

Given a template mask (from a traced polygon), score every segmented blob by
shape similarity (Hu moments + aspect/circularity) and return the best matches.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from skimage.draw import polygon as draw_polygon
from skimage.measure import label, regionprops

from tem_rods.morphometrics import circularity_from_area_perimeter, feret_diameters_px


@dataclass(frozen=True)
class ShapeFeatures:
    aspect_ratio: float
    eccentricity: float
    circularity: float
    solidity: float
    extent: float
    area_px: float
    feret_ratio: float
    hu_log: tuple[float, ...]  # log(|φ|) for 7 Hu moments


@dataclass
class ShapeMatch:
    label_id: int
    score: float  # 0 = identical, larger = more different
    features: ShapeFeatures
    centroid_y: float
    centroid_x: float
    area_px: int


def polygon_to_mask(
    points_xy: list[tuple[float, float]] | np.ndarray,
    shape: tuple[int, int],
) -> np.ndarray:
    """
    Fill a polygon into a boolean mask.

    ``points_xy`` are (x, y) image coordinates (column, row), as returned by
    click handlers. Needs ≥ 3 points.
    """
    pts = np.asarray(points_xy, dtype=float)
    if pts.ndim != 2 or pts.shape[0] < 3 or pts.shape[1] < 2:
        raise ValueError("Need at least 3 (x, y) points to form an outline.")
    rr, cc = draw_polygon(pts[:, 1], pts[:, 0], shape=shape)
    mask = np.zeros(shape, dtype=bool)
    valid = (rr >= 0) & (rr < shape[0]) & (cc >= 0) & (cc < shape[1])
    mask[rr[valid], cc[valid]] = True
    if not mask.any():
        raise ValueError("Traced outline did not cover any pixels.")
    return mask


def _hu_log_from_region(region) -> tuple[float, ...]:
    hu = np.asarray(region.moments_hu, dtype=float)
    # Log transform (OpenCV-style) for numerical stability / scale.
    return tuple(float(np.sign(v) * np.log10(abs(v) + 1e-12)) for v in hu)


def features_from_mask(mask: np.ndarray) -> ShapeFeatures:
    """Extract rotation/scale-friendly shape descriptors from a binary mask."""
    labeled = label(mask.astype(np.uint8))
    props = regionprops(labeled)
    if not props:
        raise ValueError("Empty template mask.")
    # Use the largest connected component if the trace is fragmented.
    region = max(props, key=lambda r: r.area)
    return features_from_region(region)


def features_from_region(region) -> ShapeFeatures:
    length = float(region.major_axis_length) or 1e-6
    width = float(region.minor_axis_length) or 1e-6
    fmax, fmin = feret_diameters_px(region.coords)
    fmin = fmin or 1e-6
    circ = circularity_from_area_perimeter(float(region.area), float(region.perimeter))
    return ShapeFeatures(
        aspect_ratio=length / width,
        eccentricity=float(region.eccentricity),
        circularity=circ,
        solidity=float(region.solidity),
        extent=float(region.extent),
        area_px=float(region.area),
        feret_ratio=float(fmax / fmin),
        hu_log=_hu_log_from_region(region),
    )


def shape_distance(
    a: ShapeFeatures,
    b: ShapeFeatures,
    *,
    area_weight: float = 0.08,
    mode: str = "auto",
) -> float:
    """
    Weighted distance between two shapes (lower = more similar).

    ``mode``:
      - ``"rods"``: emphasize aspect / eccentricity (better for nanorods)
      - ``"dots"``: emphasize circularity / Hu moments
      - ``"auto"``: pick rods if template aspect ≥ 2
    """
    if mode == "auto":
        mode = "rods" if a.aspect_ratio >= 2.0 else "dots"

    def _rel(x: float, y: float) -> float:
        denom = max(abs(x), abs(y), 1e-6)
        return abs(x - y) / denom

    hu_dist = float(np.mean(np.abs(np.asarray(a.hu_log) - np.asarray(b.hu_log))))
    # Cap Hu contribution — small TEM rods are noisy in moment space.
    hu_dist = min(hu_dist, 1.5)

    if mode == "rods":
        # Nanorods: length can vary; width/aspect and elongation matter most.
        geom = (
            0.40 * _rel(a.aspect_ratio, b.aspect_ratio)
            + 0.25 * abs(a.eccentricity - b.eccentricity)
            + 0.15 * _rel(a.feret_ratio, b.feret_ratio)
            + 0.10 * abs(a.circularity - b.circularity)
            + 0.05 * abs(a.solidity - b.solidity)
            + 0.05 * abs(a.extent - b.extent)
        )
        area = area_weight * _rel(a.area_px, b.area_px)
        return 0.20 * hu_dist + 0.80 * geom + area

    geom = (
        0.20 * _rel(a.aspect_ratio, b.aspect_ratio)
        + 0.15 * abs(a.eccentricity - b.eccentricity)
        + 0.30 * abs(a.circularity - b.circularity)
        + 0.15 * _rel(a.feret_ratio, b.feret_ratio)
        + 0.10 * abs(a.solidity - b.solidity)
        + 0.10 * abs(a.extent - b.extent)
    )
    area = area_weight * _rel(a.area_px, b.area_px)
    return 0.40 * hu_dist + 0.60 * geom + area


def find_similar_in_labels(
    labels: np.ndarray,
    template_mask: np.ndarray,
    *,
    max_score: float = 0.70,
    min_area_ratio: float = 0.12,
    max_area_ratio: float = 8.0,
    min_aspect_ratio_frac: float = 0.45,
    exclude_template_overlap: bool = True,
    mode: str = "auto",
) -> tuple[ShapeFeatures, list[ShapeMatch], dict[str, int]]:
    """
    Score every labeled blob against a template mask.

    Returns (template_features, matches sorted by ascending score, diagnostics).
    """
    template = features_from_mask(template_mask)
    if mode == "auto":
        mode = "rods" if template.aspect_ratio >= 2.0 else "dots"

    matches: list[ShapeMatch] = []
    n_regions = 0
    n_area_rejected = 0
    n_aspect_rejected = 0
    n_score_rejected = 0

    for region in regionprops(labels):
        if region.area <= 0:
            continue
        n_regions += 1
        area_ratio = float(region.area) / max(template.area_px, 1.0)
        if area_ratio < min_area_ratio or area_ratio > max_area_ratio:
            n_area_rejected += 1
            continue

        feat = features_from_region(region)

        # For rods: reject blobs that are much rounder than the template.
        if mode == "rods" and template.aspect_ratio >= 2.0:
            if feat.aspect_ratio < template.aspect_ratio * min_aspect_ratio_frac:
                n_aspect_rejected += 1
                continue

        score = shape_distance(template, feat, mode=mode)
        if score <= max_score:
            matches.append(
                ShapeMatch(
                    label_id=int(region.label),
                    score=float(score),
                    features=feat,
                    centroid_y=float(region.centroid[0]),
                    centroid_x=float(region.centroid[1]),
                    area_px=int(region.area),
                )
            )
        else:
            n_score_rejected += 1

    matches.sort(key=lambda m: m.score)
    diagnostics = {
        "n_segmented": n_regions,
        "n_matched": len(matches),
        "n_area_rejected": n_area_rejected,
        "n_aspect_rejected": n_aspect_rejected,
        "n_score_rejected": n_score_rejected,
    }
    return template, matches, diagnostics


def stroke_image_to_mask(
    stroke_rgba: np.ndarray,
    target_shape: tuple[int, int],
    *,
    alpha_threshold: int = 10,
    close_radius: int = 4,
) -> np.ndarray:
    """
    Convert a freehand canvas stroke (RGBA) into a filled boolean mask.

    Resizes the stroke to ``target_shape`` (row, col), closes small gaps, and
    flood-fills the interior so a drawn outline becomes a solid particle template.
    """
    from scipy import ndimage as ndi
    from skimage.morphology import binary_closing, disk
    from skimage.transform import resize

    if stroke_rgba is None or stroke_rgba.size == 0:
        raise ValueError("No drawing found — drag the mouse around a particle first.")

    if stroke_rgba.ndim == 3 and stroke_rgba.shape[2] >= 4:
        stroke = stroke_rgba[:, :, 3] > alpha_threshold
    elif stroke_rgba.ndim == 3:
        stroke = stroke_rgba.max(axis=2) > alpha_threshold
    else:
        stroke = stroke_rgba > alpha_threshold

    if not stroke.any():
        raise ValueError("No drawing found — drag the mouse around a particle first.")

    stroke_full = resize(
        stroke.astype(float),
        target_shape,
        order=0,
        preserve_range=True,
        anti_aliasing=False,
    ) > 0.5
    if close_radius > 0:
        stroke_full = binary_closing(stroke_full, disk(close_radius))
    filled = ndi.binary_fill_holes(stroke_full)
    # If the loop was not closed, fill_holes barely expands — use convex hull fallback.
    if filled.sum() <= stroke_full.sum() * 1.15:
        from skimage.morphology import convex_hull_image

        filled = convex_hull_image(stroke_full)
    if not filled.any():
        raise ValueError("Could not turn the stroke into a closed outline. Try again.")
    return filled.astype(bool)


def fabric_paths_to_points(
    json_data: dict | None,
    *,
    scale_x: float = 1.0,
    scale_y: float = 1.0,
) -> list[tuple[float, float]]:
    """Extract ordered (x, y) points from streamlit-drawable-canvas JSON."""
    if not json_data or "objects" not in json_data:
        return []
    points: list[tuple[float, float]] = []
    for obj in json_data.get("objects", []):
        left = float(obj.get("left", 0.0))
        top = float(obj.get("top", 0.0))
        sx = float(obj.get("scaleX", 1.0))
        sy = float(obj.get("scaleY", 1.0))
        path = obj.get("path")
        if path:
            for cmd in path:
                if not cmd:
                    continue
                op = cmd[0]
                if op in ("M", "L") and len(cmd) >= 3:
                    points.append(((left + cmd[1] * sx) * scale_x, (top + cmd[2] * sy) * scale_y))
                elif op == "Q" and len(cmd) >= 5:
                    points.append(((left + cmd[3] * sx) * scale_x, (top + cmd[4] * sy) * scale_y))
                elif op == "C" and len(cmd) >= 7:
                    points.append(((left + cmd[5] * sx) * scale_x, (top + cmd[6] * sy) * scale_y))
        raw_pts = obj.get("points")
        if raw_pts:
            for p in raw_pts:
                if isinstance(p, dict):
                    points.append(((left + float(p["x"]) * sx) * scale_x, (top + float(p["y"]) * sy) * scale_y))
    return points


def render_trace_overlay(
    image: np.ndarray,
    points_xy: list[tuple[float, float]],
    *,
    match_labels: np.ndarray | None = None,
    matched_ids: set[int] | None = None,
    template_mask: np.ndarray | None = None,
) -> np.ndarray:
    """RGB overlay showing the in-progress trace and optional matches."""
    from PIL import Image as PILImage, ImageDraw

    if image.ndim == 3:
        gray = image.astype(np.float64).mean(axis=2)
    else:
        gray = image.astype(np.float64)
    if gray.max() > 1.5:
        gray = gray / 255.0
    rgb = (np.stack([gray, gray, gray], axis=-1) * 255).astype(np.uint8)
    pil = PILImage.fromarray(rgb)
    draw = ImageDraw.Draw(pil)

    if template_mask is not None:
        from skimage.measure import find_contours

        for contour in find_contours(template_mask.astype(float), 0.5):
            pts = [(float(c[1]), float(c[0])) for c in contour]
            if len(pts) >= 2:
                draw.line(pts + [pts[0]], fill=(0, 200, 255), width=2)

    if match_labels is not None and matched_ids:
        from skimage.measure import find_contours

        for region in regionprops(match_labels):
            if region.label not in matched_ids:
                continue
            mask = match_labels == region.label
            for contour in find_contours(mask.astype(float), 0.5):
                pts = [(float(c[1]), float(c[0])) for c in contour]
                if len(pts) >= 2:
                    draw.line(pts + [pts[0]], fill=(0, 220, 100), width=2)

    if len(points_xy) >= 1:
        xy = [(float(x), float(y)) for x, y in points_xy]
        if len(xy) >= 2:
            draw.line(xy, fill=(255, 80, 80), width=2)
        r = 3
        for x, y in xy:
            draw.ellipse((x - r, y - r, x + r, y + r), fill=(255, 40, 40))

    return np.asarray(pil)

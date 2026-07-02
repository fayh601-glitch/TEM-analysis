"""
Scale Bar Detector — automatically find the scale bar in a TEM image
=====================================================================

Published TEM figures usually place a dark horizontal scale bar in the bottom-left
corner. This file finds the thin bar line (not the "200 nm" label text), estimates
pixel length, and optionally reads the nm value from the label or filename.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image
from skimage.measure import label, regionprops

from tem_rods.calibrate import nm_per_pixel_from_scale_bar, validate_nm_per_pixel


@dataclass(frozen=True)
class ScaleBarDetection:
    """Result of automatic scale bar detection."""

    bar_pixels: float
    bar_nm: float
    nm_per_pixel: float
    bbox: tuple[int, int, int, int]
    confidence: float = 1.0


def detect_scale_bar(
    image_path: str | Path,
    *,
    scale_bar_nm: float | None = None,
    search_bottom_fraction: float = 0.25,
    search_left_fraction: float = 0.55,
    max_dark_value: float | None = None,
    max_bar_height_px: int = 8,
    min_bar_width_px: int = 18,
    min_bar_aspect: float = 6.0,
    bbox_pad_px: int = 8,
) -> ScaleBarDetection:
    """
    Detect a thin horizontal scale bar in the bottom-left of a TEM micrograph.

    Returns calibrated nm/pixel plus a bounding box for masking during segmentation.
    """
    image_path = Path(image_path)
    im = np.array(Image.open(image_path).convert("L"), dtype=float)
    if im.max() > 1.5:
        im = im / 255.0

    h, w = im.shape
    row_start = int(h * (1.0 - search_bottom_fraction))
    col_end = int(w * search_left_fraction)
    roi = im[row_start:, :col_end]
    dark_threshold = max_dark_value if max_dark_value is not None else 0.25
    if max_dark_value is not None and max_dark_value > 1.5:
        dark_threshold = max_dark_value / 255.0

    binary = roi < dark_threshold
    labels = label(binary)
    candidates: list[tuple[float, object]] = []

    for region in regionprops(labels):
        min_row, min_col, max_row, max_col = region.bbox
        height = max_row - min_row
        width = max_col - min_col
        if height > max_bar_height_px:
            continue
        if width < min_bar_width_px:
            continue
        if width / max(height, 1) < min_bar_aspect:
            continue
        if width > w * 0.4:
            continue

        row_scores = []
        for rel_row in range(min_row, max_row):
            row = binary[rel_row]
            dark = np.where(row)[0]
            if len(dark) < min_bar_width_px:
                continue
            breaks = np.where(np.diff(dark) > 2)[0]
            starts = [0] + (breaks + 1).tolist()
            ends = breaks.tolist() + [len(dark) - 1]
            row_scores.append(max(float(dark[end] - dark[start]) for start, end in zip(starts, ends)))

        bar_width = float(np.median(row_scores)) if row_scores else float(width)
        if bar_width < min_bar_width_px:
            continue

        abs_row = row_start + (min_row + max_row) / 2.0
        score = bar_width + 0.25 * abs_row + 0.02 * (width / max(height, 1))
        candidates.append((score, region, bar_width))

    if not candidates:
        raise ValueError(f"Could not detect scale bar in {image_path}")

    _, best_region, bar_pixels = max(candidates, key=lambda item: item[0])
    min_row, min_col, max_row, max_col = best_region.bbox
    bbox = (
        max(0, row_start + min_row - bbox_pad_px),
        max(0, min_col - bbox_pad_px),
        min(h, row_start + max_row + bbox_pad_px),
        min(w, max_col + bbox_pad_px),
    )

    bar_nm = scale_bar_nm
    if bar_nm is None:
        bar_nm = _parse_scale_bar_nm(image_path, im, bbox)
    if bar_nm is None:
        raise ValueError(
            f"Could not determine scale bar length in nm for {image_path}. "
            "Pass scale_bar_nm or use a filename like sample_200nm.png."
        )

    nm_per_pixel = validate_nm_per_pixel(
        validate_scale_bar_calibration(bar_pixels, bar_nm, image_width=w)
    )
    confidence = min(1.0, bar_pixels / min_bar_width_px / 3.0)

    return ScaleBarDetection(
        bar_pixels=bar_pixels,
        bar_nm=bar_nm,
        nm_per_pixel=nm_per_pixel,
        bbox=bbox,
        confidence=confidence,
    )


def detect_scale_bar_pixels(
    image_path: str | Path,
    *,
    scale_bar_nm: float = 20.0,
    search_bottom_fraction: float = 0.22,
    search_left_fraction: float = 0.4,
    max_dark_value: float = 60.0,
) -> tuple[float, float]:
    """
    Backward-compatible helper returning (scale_bar_pixels, nm_per_pixel).
    """
    detection = detect_scale_bar(
        image_path,
        scale_bar_nm=scale_bar_nm,
        search_bottom_fraction=search_bottom_fraction,
        search_left_fraction=search_left_fraction,
        max_dark_value=max_dark_value,
    )
    return detection.bar_pixels, detection.nm_per_pixel


def validate_scale_bar_calibration(
    bar_pixels: float,
    bar_nm: float,
    *,
    image_width: int,
    min_bar_pixels: float = 12.0,
    max_bar_fraction: float = 0.45,
    min_nm_per_pixel: float = 0.02,
    max_nm_per_pixel: float = 8.0,
) -> float:
    """Validate detected scale bar and return nm/pixel."""
    if bar_pixels < min_bar_pixels:
        raise ValueError(f"Scale bar too short ({bar_pixels:.1f} px)")
    if bar_pixels > image_width * max_bar_fraction:
        raise ValueError(f"Scale bar too long ({bar_pixels:.1f} px)")
    nm_per_pixel = nm_per_pixel_from_scale_bar(bar_nm, bar_pixels)
    if nm_per_pixel < min_nm_per_pixel or nm_per_pixel > max_nm_per_pixel:
        raise ValueError(
            f"Implausible calibration {nm_per_pixel:.4f} nm/pixel "
            f"({bar_nm:g} nm / {bar_pixels:.1f} px)"
        )
    return nm_per_pixel


def _parse_scale_bar_nm(
    image_path: Path,
    image: np.ndarray,
    bar_bbox: tuple[int, int, int, int],
) -> float | None:
    """Try OCR, then filename, then label-region heuristics."""
    from_filename = _nm_from_filename(image_path)
    if from_filename is not None:
        return from_filename

    ocr_nm = _ocr_scale_bar_nm(image, bar_bbox)
    if ocr_nm is not None:
        return ocr_nm

    return None


def _nm_from_filename(path: Path) -> float | None:
    match = re.search(r"(\d+(?:\.\d+)?)\s*nm", path.stem, flags=re.IGNORECASE)
    if match:
        return float(match.group(1))
    return None


def _ocr_scale_bar_nm(
    image: np.ndarray,
    bar_bbox: tuple[int, int, int, int],
) -> float | None:
    """Optional OCR via pytesseract when installed."""
    try:
        import pytesseract
    except ImportError:
        return None

    row_min, col_min, row_max, col_max = bar_bbox
    h, w = image.shape
    text_row_min = max(0, row_min - 6)
    text_row_max = min(h, row_max + 20)
    text_col_min = col_min
    text_col_max = min(w, col_max + int(w * 0.18))
    patch = (image[text_row_min:text_row_max, text_col_min:text_col_max] * 255.0).astype(np.uint8)

    text = pytesseract.image_to_string(patch, config="--psm 7")
    match = re.search(r"(\d+(?:\.\d+)?)\s*nm", text, flags=re.IGNORECASE)
    if match:
        return float(match.group(1))
    return None

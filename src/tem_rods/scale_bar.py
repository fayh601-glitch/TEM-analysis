"""
Scale Bar Detector — automatically find the scale bar in a TEM image
=====================================================================

Published TEM figures usually place a dark horizontal scale bar in the bottom-left
corner. This file searches that region and estimates how many pixels long the bar
is, so you do not have to measure it by hand every time.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image


def detect_scale_bar_pixels(
    image_path: str | Path,
    *,
    scale_bar_nm: float = 20.0,
    search_bottom_fraction: float = 0.22,
    search_left_fraction: float = 0.4,
    max_dark_value: float = 60.0,
) -> tuple[float, float]:
    """
    Detect a dark horizontal scale bar in the bottom-left of a TEM micrograph.

    Returns (scale_bar_pixels, nm_per_pixel).
    """
    im = np.array(Image.open(image_path).convert("L"), dtype=float)
    h, w = im.shape
    roi = im[int(h * (1 - search_bottom_fraction)) :, : int(w * search_left_fraction)]
    rh, _ = roi.shape

    best_length = 0.0
    for row_idx in range(rh):
        row = roi[row_idx]
        dark = row < max_dark_value
        if dark.sum() < 15:
            continue
        indices = np.where(dark)[0]
        breaks = np.where(np.diff(indices) > 2)[0]
        starts = [0] + (breaks + 1).tolist()
        ends = breaks.tolist() + [len(indices) - 1]
        for start, end in zip(starts, ends):
            length = float(indices[end] - indices[start])
            if length > best_length:
                best_length = length

    if best_length <= 0:
        raise ValueError(f"Could not detect scale bar in {image_path}")

    return best_length, scale_bar_nm / best_length

"""
Image Preprocessor — clean up a TEM image before particle detection
=====================================================================

Microscopy images often have uneven brightness and noise. This file normalizes
contrast and applies a gentle blur so the segmentation step can find particles
more reliably without chasing every speck of grain.
"""

from __future__ import annotations

import numpy as np
from skimage import exposure, filters


def preprocess(
    image: np.ndarray,
    *,
    gaussian_sigma: float = 1.0,
    crop_margins: bool = False,
    use_clahe: bool = False,
) -> np.ndarray:
    """
    Enhance contrast for TEM images where particles are darker than background.

    Returns float image in [0, 1].
    """
    if image.ndim != 2:
        raise ValueError("preprocess expects a 2D grayscale image")

    img = image.astype(np.float64)
    if img.max() > 1.5:
        img = img / 255.0

    if crop_margins:
        img = crop_white_margins(img)

    img = exposure.rescale_intensity(img, in_range="image", out_range=(0.0, 1.0))
    if use_clahe:
        img = exposure.equalize_adapthist(img, clip_limit=0.02)
    if gaussian_sigma > 0:
        img = filters.gaussian(img, sigma=gaussian_sigma, preserve_range=True)
    return np.clip(img, 0.0, 1.0)


def crop_white_margins(image: np.ndarray, *, threshold: float = 0.92) -> np.ndarray:
    """
    Remove uniform bright bands attached to the image border (screenshot/PDF margins).

    Only rows/columns that are entirely brighter than ``threshold`` are removed.
    Normal TEM micrographs are not changed.
    """
    img = image
    h, w = img.shape

    top = 0
    while top < h and np.all(img[top, :] >= threshold):
        top += 1
    bottom = h
    while bottom > top and np.all(img[bottom - 1, :] >= threshold):
        bottom -= 1
    left = 0
    while left < w and np.all(img[:, left] >= threshold):
        left += 1
    right = w
    while right > left and np.all(img[:, right - 1] >= threshold):
        right -= 1

    if top >= bottom or left >= right:
        return image
    return img[top:bottom, left:right]


def detect_bottom_info_banner(
    image: np.ndarray,
    *,
    bright_threshold: float = 0.82,
    min_bright_fraction: float = 0.50,
    min_height_fraction: float = 0.04,
    max_height_fraction: float = 0.35,
    min_height_px: int = 10,
    min_banner_median: float = 0.80,
) -> int | None:
    """
    Find a bright instrument info bar glued to the bottom of a TEM micrograph.

    AMT/Gatan exports burn in a white strip with filename, ``50 nm`` scale bar,
    and ``Cal: … µm/pix``. Those rows are mostly white but not *entirely* white
    (because of black text), so :func:`crop_white_margins` leaves them in place.

    Returns the first row of the banner, or ``None`` if none is found.
    """
    img = np.asarray(image, dtype=np.float64)
    if img.ndim != 2 or img.size == 0:
        return None
    if img.max() > 1.5:
        img = img / 255.0

    h, _w = img.shape
    max_h = max(min_height_px, int(round(h * max_height_fraction)))
    min_h = max(min_height_px, int(round(h * min_height_fraction)))
    if h < min_h + 8:
        return None

    # Median stays high even when a row has black filename/Cal text.
    row_median = np.median(img, axis=1)
    bright_frac = (img >= bright_threshold).mean(axis=1)
    start = h
    row = h - 1
    while row >= h - max_h and (
        row_median[row] >= min_banner_median or bright_frac[row] >= min_bright_fraction
    ):
        start = row
        row -= 1

    banner_h = h - start
    if banner_h < min_h:
        return None
    banner_med = float(np.median(img[start:]))
    if banner_med < min_banner_median:
        return None
    # Require a darker micrograph just above the strip (relative drop, not
    # an absolute cutoff — light carbon films are often ~0.7–0.85).
    above = img[max(0, start - 12) : start]
    if above.size and banner_med - float(np.median(above)) < 0.06:
        return None
    return start


def crop_bottom_info_banner(image: np.ndarray, **kwargs) -> tuple[np.ndarray, int | None]:
    """Return ``(cropped_micrograph, banner_start_row_or_None)``."""
    start = detect_bottom_info_banner(image, **kwargs)
    if start is None:
        return image, None
    return image[:start], start

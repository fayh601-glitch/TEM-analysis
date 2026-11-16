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


def subtract_smooth_background(image: np.ndarray, *, sigma: float) -> np.ndarray:
    """
    Remove slow illumination falloff (TEM vignette) without stretching grain.

    AMT frames are often darker on one side. Global dark-thresholding then
    misses real particles on the bright side. Subtracting a large Gaussian
    leaves local dark dots intact.
    """
    if sigma is None or sigma <= 0:
        return image
    img = np.asarray(image, dtype=np.float64)
    background = filters.gaussian(img, sigma=float(sigma), preserve_range=True)
    return np.clip(img - background + float(np.median(img)), 0.0, 1.0)


def preprocess(
    image: np.ndarray,
    *,
    gaussian_sigma: float = 1.0,
    crop_margins: bool = False,
    use_clahe: bool = False,
    background_sigma: float = 0.0,
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

    # Do not min–max stretch. Stretching a grainy carbon film maps specks to 0 and 1,
    # so later thresholding treats background texture as particles.
    img = np.clip(img, 0.0, 1.0)
    if background_sigma and background_sigma > 0:
        img = subtract_smooth_background(img, sigma=background_sigma)
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


def _row_looks_like_info_bar(
    row: np.ndarray,
    *,
    bright_threshold: float,
    min_bright_fraction: float,
    min_body_median: float,
) -> bool:
    """True for a white/gray instrument row even when black letters are present.

    Circular glyphs (0, O, 8, C) are dark, so a raw row median can look like TEM
    film. Use the brighter part of the row (the paper/background of the strip).
    """
    if row.size == 0:
        return False
    bright_frac = float((row >= bright_threshold).mean())
    if bright_frac >= min_bright_fraction:
        return True
    cutoff = max(1, int(0.30 * row.size))
    body_median = float(np.median(np.sort(row)[cutoff:]))
    return body_median >= min_body_median


def detect_bottom_info_banner(
    image: np.ndarray,
    *,
    bright_threshold: float = 0.82,
    min_bright_fraction: float = 0.45,
    min_height_fraction: float = 0.04,
    max_height_fraction: float = 0.42,
    min_height_px: int = 10,
    min_banner_median: float = 0.80,
    pad_px: int = 6,
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

    start = h
    row = h - 1
    while row >= h - max_h and _row_looks_like_info_bar(
        img[row],
        bright_threshold=bright_threshold,
        min_bright_fraction=min_bright_fraction,
        min_body_median=min_banner_median,
    ):
        start = row
        row -= 1

    start = max(0, start - max(0, int(pad_px)))
    banner_h = h - start
    if banner_h < min_h:
        return None

    banner = img[start:]
    cutoff = max(1, int(0.30 * banner.size))
    banner_body = float(np.median(np.sort(banner.ravel())[cutoff:]))
    if banner_body < min_banner_median - 0.04:
        return None
    # Require a darker micrograph just above the strip.
    above = img[max(0, start - 12) : start]
    if above.size and banner_body - float(np.median(above)) < 0.06:
        return None
    return start


def crop_bottom_info_banner(image: np.ndarray, **kwargs) -> tuple[np.ndarray, int | None]:
    """Return ``(cropped_micrograph, banner_start_row_or_None)``."""
    start = detect_bottom_info_banner(image, **kwargs)
    if start is None:
        return image, None
    return image[:start], start

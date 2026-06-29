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


def preprocess(image: np.ndarray, *, gaussian_sigma: float = 1.0) -> np.ndarray:
    """
    Enhance contrast for TEM images where particles are darker than background.

    Returns float image in [0, 1].
    """
    if image.ndim != 2:
        raise ValueError("preprocess expects a 2D grayscale image")

    img = exposure.rescale_intensity(
        image.astype(np.float64), in_range="image", out_range=(0.0, 1.0)
    )
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

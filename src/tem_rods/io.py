"""
Image Loader — read TEM image files into the program
=====================================================

This file opens PNG, TIFF, or JPG microscopy images and converts them to a
grayscale number grid the rest of the pipeline can work with. It handles both
black-and-white and color inputs automatically.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from skimage import color, io


def load_grayscale(path: str | Path) -> np.ndarray:
    """Load a TEM image as a float64 grayscale array in [0, 1]."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {path}")

    img = io.imread(path)
    if img.ndim == 3:
        img = color.rgb2gray(img)
    elif img.ndim != 2:
        raise ValueError(f"Expected 2D or 3D image, got shape {img.shape}")

    img = img.astype(np.float64)
    if img.max() > 1.0:
        img = img / 255.0
    return img

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

    img = exposure.rescale_intensity(image.astype(np.float64), in_range="image", out_range=(0.0, 1.0))
    if gaussian_sigma > 0:
        img = filters.gaussian(img, sigma=gaussian_sigma, preserve_range=True)
    return np.clip(img, 0.0, 1.0)

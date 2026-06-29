"""
Preprocess Tests — verify screenshot margin cropping works
==========================================================

Screenshots often include white borders that break global Otsu thresholding.
These tests ensure preprocessing trims margins without affecting normal TEM PNGs.
"""

import numpy as np
from skimage.filters import threshold_otsu

from tem_rods.preprocess import crop_white_margins, preprocess


def test_crop_white_margins_trims_borders():
    image = np.ones((100, 120), dtype=np.float64) * 0.45
    image[20:80, 25:95] = 0.35
    image[:10, :] = 1.0
    image[-8:, :] = 1.0
    image[:, :6] = 1.0
    image[:, -5:] = 1.0

    cropped = crop_white_margins(image, threshold=0.92)
    assert cropped.shape[0] < 100
    assert cropped.shape[1] < 120
    assert threshold_otsu(cropped) < 0.7


def test_preprocess_keeps_normal_tem_image_shape():
    image = np.random.default_rng(0).uniform(0.3, 0.8, (200, 200))
    processed = preprocess(image, gaussian_sigma=0.0)
    assert processed.shape == image.shape

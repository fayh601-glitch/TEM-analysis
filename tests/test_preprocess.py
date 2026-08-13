"""
Preprocess Tests — verify screenshot margin cropping works
==========================================================

Screenshots often include white borders that break global Otsu thresholding.
These tests ensure preprocessing trims margins without affecting normal TEM PNGs.
"""

import numpy as np
from skimage.filters import threshold_otsu

from tem_rods.preprocess import crop_bottom_info_banner, crop_white_margins, detect_bottom_info_banner, preprocess


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


def test_preprocess_does_not_stretch_film_grain():
    """Min–max stretch used to map grain specks to 0 and 1, creating fake particles."""
    rng = np.random.default_rng(0)
    image = np.clip(rng.normal(0.70, 0.03, (80, 80)), 0.60, 0.80)
    processed = preprocess(image, gaussian_sigma=0.0)
    assert processed.min() >= 0.55
    assert processed.max() <= 0.85
    assert abs(float(np.median(processed)) - 0.70) < 0.05


def test_detect_bottom_info_banner_finds_amt_strip():
    rng = np.random.default_rng(2)
    micro = rng.uniform(0.45, 0.70, (320, 480))
    banner = np.ones((80, 480), dtype=np.float64) * 0.98
    # Black filename / Cal text in the banner (not a full-white row).
    banner[10:22, 20:220] = 0.05
    banner[40:46, 30:150] = 0.0
    image = np.vstack([micro, banner])

    start = detect_bottom_info_banner(image)
    assert start is not None
    assert 310 <= start <= 330

    cropped, row = crop_bottom_info_banner(image)
    assert row == start
    assert cropped.shape == (start, 480)
    assert cropped.shape[0] == 320 or abs(cropped.shape[0] - 320) <= 10


def test_detect_bottom_info_banner_ignores_normal_tem():
    image = np.random.default_rng(3).uniform(0.35, 0.80, (240, 300))
    assert detect_bottom_info_banner(image) is None
    cropped, row = crop_bottom_info_banner(image)
    assert row is None
    assert cropped.shape == image.shape

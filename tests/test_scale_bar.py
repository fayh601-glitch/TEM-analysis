"""
Scale Bar Tests — verify automatic scale bar detection
======================================================

Synthetic images with a thin horizontal bar check that label text is not
included in the measured bar length.
"""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from tem_rods.scale_bar import detect_scale_bar, validate_scale_bar_calibration


def _write_test_image(path, *, bar_width: int = 55, bar_row: int = 190) -> None:
    image = np.ones((200, 220), dtype=np.uint8) * 210
    image[bar_row : bar_row + 3, 20 : 20 + bar_width] = 0
    # Simulate label text to the right of the bar (should not affect bar length).
    image[bar_row - 8 : bar_row + 14, 20 + bar_width + 4 : 20 + bar_width + 48] = 0
    Image.fromarray(image).save(path)


def test_detect_scale_bar_ignores_label_text(tmp_path):
    image_path = tmp_path / "sample_200nm.png"
    _write_test_image(image_path, bar_width=55)

    detection = detect_scale_bar(image_path)

    assert detection.bar_nm == pytest.approx(200.0)
    assert detection.bar_pixels == pytest.approx(55.0, abs=3.0)
    assert detection.nm_per_pixel == pytest.approx(200.0 / 55.0, rel=0.05)


def test_detect_bright_scale_bar(tmp_path):
    image_path = tmp_path / "bright_50nm.png"
    image = np.ones((200, 220), dtype=np.uint8) * 120
    image[190:193, 25:25 + 80] = 255
    Image.fromarray(image).save(image_path)

    detection = detect_scale_bar(image_path, scale_bar_nm=50.0)
    assert detection.polarity == "bright"
    assert detection.bar_pixels == pytest.approx(80.0, abs=4.0)
    assert detection.nm_per_pixel == pytest.approx(50.0 / detection.bar_pixels, rel=0.05)

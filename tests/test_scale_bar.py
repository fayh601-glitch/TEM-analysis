"""
Scale Bar Tests — verify automatic scale bar detection
======================================================

Synthetic images with a thin horizontal bar check that label text is not
included in the measured bar length.
"""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image, ImageDraw

from tem_rods.scale_bar import detect_scale_bar, parse_embedded_nm_per_pixel, validate_scale_bar_calibration


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


def _write_amt_style_image(path, *, bar_width: int = 120) -> None:
    """Grainy micrograph + white AMT info bar with a dark 50 nm scale line."""
    rng = np.random.default_rng(4)
    micro_h, banner_h, w = 320, 80, 520
    micro = (rng.uniform(0.50, 0.72, (micro_h, w)) * 255).astype(np.uint8)
    banner = np.full((banner_h, w), 250, dtype=np.uint8)
    image = np.vstack([micro, banner])
    pil = Image.fromarray(image, mode="L")
    draw = ImageDraw.Draw(pil)
    y = micro_h + 42
    x0 = 36
    draw.line([(x0, y), (x0 + bar_width, y)], fill=0, width=3)
    draw.text((x0 + bar_width + 10, y - 8), "50 nm", fill=0)
    draw.text((12, micro_h + 8), "CT01_CdSeRods_45minrxn_003.tif", fill=0)
    draw.text((12, micro_h + 58), "Cal: 0.000231 um/pix  HV=80kV  Mag: 44000 x", fill=0)
    pil.save(path)


def test_detect_scale_bar_in_amt_info_banner(tmp_path):
    image_path = tmp_path / "CT01_CdSeRods_45minrxn_003.tif"
    _write_amt_style_image(image_path, bar_width=120)

    detection = detect_scale_bar(image_path, scale_bar_nm=50.0)
    assert detection.polarity == "dark"
    assert detection.bar_pixels == pytest.approx(120.0, abs=6.0)
    assert detection.nm_per_pixel == pytest.approx(50.0 / detection.bar_pixels, rel=0.08)


def test_parse_embedded_nm_per_pixel_from_tiff_tag(tmp_path):
    from PIL.TiffImagePlugin import ImageFileDirectory_v2

    path = tmp_path / "amt_tagged.tif"
    arr = np.full((40, 50), 180, dtype=np.uint8)
    info = ImageFileDirectory_v2()
    info[270] = "Cal: 0.000231 um/pix"
    Image.fromarray(arr, mode="L").save(path, format="TIFF", tiffinfo=info)

    parsed = parse_embedded_nm_per_pixel(path)
    assert parsed == pytest.approx(0.231)

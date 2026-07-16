"""
Image I/O Tests — verify loading PNG, RGBA, and grayscale images
===================================================================

Screenshots and exported figures often use RGBA PNGs. These tests make sure
load_grayscale handles them without crashing outside the IDE.
"""

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from tem_rods.io import load_grayscale


def test_load_rgba_png(tmp_path: Path):
    path = tmp_path / "rgba.png"
    rgba = np.zeros((40, 50, 4), dtype=np.uint8)
    rgba[..., :3] = 120
    rgba[..., 3] = 255
    Image.fromarray(rgba, mode="RGBA").save(path)

    gray = load_grayscale(path)
    assert gray.ndim == 2
    assert gray.shape == (40, 50)
    assert gray.max() <= 1.0


def test_load_grayscale_png(tmp_path: Path):
    path = tmp_path / "gray.png"
    Image.fromarray(np.full((30, 40), 200, dtype=np.uint8), mode="L").save(path)
    gray = load_grayscale(path)
    assert gray.shape == (30, 40)
    assert gray[0, 0] == pytest.approx(200 / 255.0)


def test_load_grayscale_bytes_jpeg(tmp_path: Path):
    from tem_rods.io import load_grayscale_bytes, save_grayscale_png

    path = tmp_path / "x.jpg"
    Image.fromarray(np.full((20, 25), 100, dtype=np.uint8), mode="L").save(path, format="JPEG")
    gray = load_grayscale_bytes(path.read_bytes(), name="x.jpg")
    assert gray.shape == (20, 25)
    out = tmp_path / "clean.png"
    save_grayscale_png(gray, out)
    assert out.exists()


def test_load_grayscale_bytes_rejects_garbage():
    from tem_rods.io import load_grayscale_bytes

    with pytest.raises(ValueError, match="Could not read"):
        load_grayscale_bytes(b"not-an-image", name="bad.png")

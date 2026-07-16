"""
Image Loader — read TEM image files into the program
=====================================================

This file opens PNG, TIFF, or JPG microscopy images and converts them to a
grayscale number grid the rest of the pipeline can work with. It handles color,
RGBA, and black-and-white inputs automatically.
"""

from __future__ import annotations

from io import BytesIO
from pathlib import Path

import numpy as np
from PIL import Image, ImageFile

# Allow slightly truncated exports from some microscope/screenshot tools.
ImageFile.LOAD_TRUNCATED_IMAGES = True


def _pil_to_gray01(img: Image.Image) -> np.ndarray:
    gray = np.array(img.convert("L"), dtype=np.float64)
    if gray.size == 0:
        raise ValueError("Image is empty.")
    if gray.max() > 1.0:
        gray = gray / 255.0
    return gray


def load_grayscale_bytes(data: bytes, *, name: str = "upload") -> np.ndarray:
    """
    Decode image bytes (PNG/JPEG/TIFF/etc.) to float64 grayscale in [0, 1].

    Prefer this for Streamlit uploads — avoids writing a possibly corrupt file
    before PIL can sniff the real format.
    """
    if not data:
        raise ValueError(f"Empty file: {name}")
    try:
        with Image.open(BytesIO(data)) as img:
            img.load()
            return _pil_to_gray01(img)
    except Exception as exc:
        raise ValueError(
            f"Could not read image '{name}' ({len(data)} bytes). "
            "Re-save/export as PNG, JPG, or TIFF and try again. "
            f"Details: {exc}"
        ) from exc


def load_grayscale(path: str | Path) -> np.ndarray:
    """Load a TEM image as a float64 grayscale array in [0, 1]."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {path}")
    try:
        return load_grayscale_bytes(path.read_bytes(), name=path.name)
    except ValueError:
        # Fall back to path-based open (helps with some multi-frame TIFFs).
        try:
            with Image.open(path) as img:
                img.load()
                return _pil_to_gray01(img)
        except Exception as exc:
            raise ValueError(
                f"Could not read image '{path.name}'. "
                "Re-save/export as PNG, JPG, or TIFF and try again. "
                f"Details: {exc}"
            ) from exc


def save_grayscale_png(image: np.ndarray, path: str | Path) -> Path:
    """Write a [0,1] or uint8 grayscale array as a clean PNG (for downstream tools)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.asarray(image)
    if arr.ndim != 2:
        raise ValueError("save_grayscale_png expects a 2D array")
    if arr.dtype != np.uint8:
        if arr.max() <= 1.5:
            arr = (np.clip(arr, 0, 1) * 255.0).astype(np.uint8)
        else:
            arr = np.clip(arr, 0, 255).astype(np.uint8)
    Image.fromarray(arr, mode="L").save(path)
    return path

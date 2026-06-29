"""
Scale Calibration — convert pixels to real-world nanometers
============================================================

TEM images include a scale bar (e.g. "20 nm"). This file does the simple math
that turns "how many pixels wide is the bar?" into "how many nanometers is each
pixel?" so all reported sizes are in physical units.
"""

from __future__ import annotations


def nm_per_pixel_from_scale_bar(scale_bar_nm: float, scale_bar_pixels: float) -> float:
    """Convert scale bar length in nm and pixels to nm per pixel."""
    if scale_bar_pixels <= 0:
        raise ValueError("scale_bar_pixels must be positive")
    if scale_bar_nm <= 0:
        raise ValueError("scale_bar_nm must be positive")
    return scale_bar_nm / scale_bar_pixels


def validate_nm_per_pixel(nm_per_pixel: float) -> float:
    if nm_per_pixel <= 0:
        raise ValueError("nm_per_pixel must be positive")
    return nm_per_pixel

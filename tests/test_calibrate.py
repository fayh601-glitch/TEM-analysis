"""
Scale Calibration Tests — verify pixel-to-nanometer math
=========================================================

These small tests make sure the scale-bar conversion formulas reject bad input
and return the correct nm/pixel value for known examples.
"""

from tem_rods.calibrate import (
    nm_per_pixel_from_cal_text,
    nm_per_pixel_from_scale_bar,
    validate_nm_per_pixel,
)
import pytest


def test_nm_per_pixel_from_scale_bar():
    assert nm_per_pixel_from_scale_bar(50, 100) == pytest.approx(0.5)


def test_validate_nm_per_pixel_rejects_zero():
    with pytest.raises(ValueError):
        validate_nm_per_pixel(0)


def test_nm_per_pixel_from_amt_cal_text():
    assert nm_per_pixel_from_cal_text("Cal: 0.000231 µm/pix") == pytest.approx(0.231)
    assert nm_per_pixel_from_cal_text("Cal: 0.000231 um/pix") == pytest.approx(0.231)
    assert nm_per_pixel_from_cal_text("filename Cal: 2.31e-4 µm/pix HV=80kV") == pytest.approx(
        0.231, rel=1e-6
    )
    assert nm_per_pixel_from_cal_text("no calibration here") is None

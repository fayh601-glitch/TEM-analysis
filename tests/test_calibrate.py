from tem_rods.calibrate import nm_per_pixel_from_scale_bar, validate_nm_per_pixel
import pytest


def test_nm_per_pixel_from_scale_bar():
    assert nm_per_pixel_from_scale_bar(50, 100) == pytest.approx(0.5)


def test_validate_nm_per_pixel_rejects_zero():
    with pytest.raises(ValueError):
        validate_nm_per_pixel(0)

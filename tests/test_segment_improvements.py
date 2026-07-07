"""Tests for segmentation improvements (hole filling, QC warnings)."""

from __future__ import annotations

import numpy as np
from skimage.draw import ellipse

from tem_rods.models import AnalysisConfig, ParticleClass, ParticleMeasurement
from tem_rods.pipeline import _quality_warnings
from tem_rods.preprocess import preprocess
from tem_rods.segment import segment_particles


def _hollow_rod_image() -> np.ndarray:
    """Rod-like ellipse with a bright center (common TEM diffraction contrast)."""
    image = np.ones((120, 220), dtype=np.float64) * 0.9
    rr, cc = ellipse(60, 110, 10, 35, rotation=0)
    image[rr, cc] = 0.08
    rr2, cc2 = ellipse(60, 110, 4, 14, rotation=0)
    image[rr2, cc2] = 0.88
    return image


def _segment_kw(**overrides):
    base = dict(
        min_particle_area_px=40,
        split_touching_particles=False,
        morphology_closing_radius=1,
        min_local_contrast=0.0,
        min_solidity=0.0,
        min_extent=0.0,
        mask_bottom_fraction=0.0,
        use_watershed=False,
    )
    base.update(overrides)
    return base


def test_fill_holes_reduces_split_rod_labels():
    processed = preprocess(_hollow_rod_image(), gaussian_sigma=0.5)
    without = segment_particles(processed, fill_holes=False, **_segment_kw())
    with_fill = segment_particles(processed, fill_holes=True, **_segment_kw())
    assert with_fill.max() < without.max()


def test_merge_warning_when_mean_exceeds_median():
    particles = [
        ParticleMeasurement(
            i,
            ParticleClass.ROD,
            length,
            5.0,
            3.0,
            0.9,
            100.0,
            0.0,
            0.0,
            10.0,
            3.0,
            50,
        )
        for i, length in enumerate([30.0, 32.0, 28.0, 400.0], start=1)
    ]
    warnings = _quality_warnings(particles, 0.5, None, AnalysisConfig())
    assert any("median" in w.lower() for w in warnings)


def test_dense_rods_preset_registered():
    from tem_rods.presets import get_preset

    preset = get_preset("dense_rods_50nm")
    assert preset.config.fill_holes is True
    assert preset.config.mask_bottom_fraction >= 0.12

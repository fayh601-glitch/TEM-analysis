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


def test_fill_holes_fills_hollow_rod_interior():
    """Hole-filling solidifies a connected ring; it does not merge two separate blobs."""
    from skimage.measure import regionprops

    image = np.ones((120, 220), dtype=np.float64) * 0.9
    rr, cc = ellipse(60, 110, 10, 35, rotation=0)
    image[rr, cc] = 0.08
    rr2, cc2 = ellipse(60, 110, 7, 18, rotation=0)
    image[rr2, cc2] = 0.88
    processed = preprocess(image, gaussian_sigma=0.0)
    kw = _segment_kw(morphology_closing_radius=2, min_particle_area_px=20)
    without = segment_particles(processed, fill_holes=False, **kw)
    with_fill = segment_particles(processed, fill_holes=True, **kw)
    area_without = sum(p.area for p in regionprops(without))
    area_with = sum(p.area for p in regionprops(with_fill))
    assert area_with > area_without
    assert with_fill.max() <= without.max()


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
    from tem_rods.models import ThresholdMode

    preset = get_preset("dense_rods_50nm")
    assert preset.config.fill_holes is True
    assert preset.config.mask_bottom_fraction >= 0.12
    assert preset.config.threshold_mode == ThresholdMode.DARK
    assert preset.config.min_local_contrast >= 0.04
    assert preset.config.min_darkness >= 0.10
    assert preset.config.min_length_nm is not None
    assert preset.config.max_width_nm is not None
    assert preset.config.max_width_nm <= 10.0


def test_dense_rods_alias_shares_config():
    from tem_rods.presets import get_preset

    assert get_preset("dense_rods").config is get_preset("dense_rods_50nm").config


def test_grainy_film_without_rods_is_not_called_rods():
    """Film grain must not be labeled as rods, even when it looks speckled."""
    from tem_rods.measure import measure_particles
    from tem_rods.models import ParticleClass
    from tem_rods.presets import get_preset
    from tem_rods.segment import segment_particles_from_config

    rng = np.random.default_rng(0)
    image = np.clip(rng.normal(0.68, 0.05, (400, 480)), 0.48, 0.95)
    preset = get_preset("dense_rods_50nm")
    processed = preprocess(image, gaussian_sigma=preset.config.gaussian_sigma)
    labels = segment_particles_from_config(processed, preset.config)
    particles = measure_particles(labels, nm_per_pixel=0.23, config=preset.config)
    rods = [p for p in particles if p.particle_class == ParticleClass.ROD]
    assert int(labels.max()) < 5
    assert len(rods) == 0


def test_dense_rods_preset_keeps_clear_rods_on_grain():
    from tem_rods.measure import measure_particles
    from tem_rods.models import ParticleClass
    from tem_rods.presets import get_preset
    from tem_rods.segment import segment_particles_from_config

    rng = np.random.default_rng(1)
    image = np.clip(rng.normal(0.70, 0.04, (360, 420)), 0.50, 0.92)
    for cy, cx, rot in ((80, 110, 0.2), (170, 250, 0.9), (260, 160, -0.5)):
        rr, cc = ellipse(cy, cx, 7, 30, rotation=rot)
        image[rr, cc] = 0.10
    preset = get_preset("dense_rods_50nm")
    processed = preprocess(image, gaussian_sigma=preset.config.gaussian_sigma)
    labels = segment_particles_from_config(processed, preset.config)
    particles = measure_particles(labels, nm_per_pixel=0.23, config=preset.config)
    rods = [p for p in particles if p.particle_class == ParticleClass.ROD]
    assert 2 <= len(rods) < 8


def test_bright_film_specks_are_not_particles():
    """Bright grain peaks on the carbon film must stay background."""
    from tem_rods.models import ThresholdMode
    from tem_rods.segment import _threshold_dark_particles, segment_particles

    rng = np.random.default_rng(5)
    image = np.clip(rng.normal(0.66, 0.03, (240, 280)), 0.50, 0.85)
    # Bright specks (the opposite of nanorods).
    for _ in range(60):
        y = int(rng.integers(8, 232))
        x = int(rng.integers(8, 272))
        image[y : y + 2, x : x + 2] = 0.95
    # One real dark rod.
    rr, cc = ellipse(120, 140, 8, 28, rotation=0.4)
    image[rr, cc] = 0.11
    processed = preprocess(image, gaussian_sigma=1.5)
    dark = _threshold_dark_particles(processed, min_darkness=0.12)
    assert float(processed[dark].mean()) < float(np.median(processed)) - 0.10
    labels = segment_particles(
        processed,
        min_particle_area_px=80,
        split_touching_particles=False,
        mask_bottom_fraction=0.0,
        threshold_mode=ThresholdMode.DARK,
        min_darkness=0.12,
        min_local_contrast=0.05,
    )
    assert 1 <= int(labels.max()) <= 4


def test_segmentation_finishes_quickly_on_grainy_field():
    """Full-image per-particle scans used to take minutes on camera TIFFs."""
    import time

    from tem_rods.models import ThresholdMode

    rng = np.random.default_rng(1)
    image = rng.uniform(0.55, 0.85, (640, 720))
    for i in range(40):
        rr, cc = ellipse(40 + (i % 8) * 70, 50 + (i // 8) * 85, 6, 18, rotation=0.2 * i)
        rr = np.clip(rr, 0, image.shape[0] - 1)
        cc = np.clip(cc, 0, image.shape[1] - 1)
        image[rr, cc] = 0.12
    t0 = time.perf_counter()
    labels = segment_particles(
        image,
        min_particle_area_px=25,
        split_touching_particles=True,
        mask_bottom_fraction=0.0,
        min_local_contrast=0.02,
        min_solidity=0.3,
        min_extent=0.1,
        threshold_mode=ThresholdMode.LOCAL,
        local_threshold_block_size=35,
    )
    elapsed = time.perf_counter() - t0
    assert elapsed < 8.0
    assert int(labels.max()) >= 10

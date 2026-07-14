"""Tests for Aviles & Lear–style morphometrics and log-normal summaries."""

from __future__ import annotations

import numpy as np
from skimage.measure import label, regionprops

from tem_rods.distributions import fit_lognormal, sample_size_note
from tem_rods.morphometrics import (
    circularity_from_area_perimeter,
    equivalent_diameter_px,
    feret_diameters_px,
    region_morphometrics,
)
from tem_rods.measure import measure_particles
from tem_rods.models import AnalysisConfig, AnalysisMode, ParticleClass


def test_circularity_circle_near_one():
    # Disc of radius ~20
    yy, xx = np.ogrid[-30:31, -30:31]
    img = (xx * xx + yy * yy) <= 20 * 20
    region = regionprops(label(img.astype(np.uint8)))[0]
    morph = region_morphometrics(region)
    assert morph["circularity"] > 0.85
    assert morph["feret_max_px"] > morph["feret_min_px"] * 0.8


def test_feret_rod_max_gt_min():
    img = np.zeros((60, 40), dtype=np.uint8)
    img[5:55, 15:25] = 1
    coords = regionprops(label(img))[0].coords
    fmax, fmin = feret_diameters_px(coords)
    assert fmax > fmin
    assert fmax >= 49  # long axis ~50


def test_equiv_diameter():
    # area 100 → d = 2*sqrt(100/pi) ≈ 11.28
    assert abs(equivalent_diameter_px(100) - 2 * np.sqrt(100 / np.pi)) < 1e-6
    assert circularity_from_area_perimeter(100, 0) == 0.0


def test_lognormal_fit_and_sample_note():
    rng = np.random.default_rng(0)
    # Log-normal sample with geometric mean ~exp(3) ≈ 20
    values = np.exp(rng.normal(3.0, 0.2, size=80))
    fit = fit_lognormal(values)
    assert fit is not None
    assert 15 < fit.geometric_mean < 25
    assert fit.geometric_mean_se > 0
    assert fit.n == 80
    assert sample_size_note(50) is not None
    assert sample_size_note(250) is None


def test_measure_particles_includes_feret_fields():
    img = np.zeros((80, 80), dtype=np.uint8)
    img[20:60, 30:40] = 1
    labels = label(img)
    cfg = AnalysisConfig(analysis_mode=AnalysisMode.RODS)
    particles = measure_particles(labels, nm_per_pixel=0.5, config=cfg)
    assert len(particles) == 1
    p = particles[0]
    assert p.feret_max_nm > p.feret_min_nm > 0
    assert 0 < p.circularity <= 1
    assert p.equiv_diameter_nm > 0
    assert p.particle_class in (ParticleClass.ROD, ParticleClass.REJECT, ParticleClass.DOT)

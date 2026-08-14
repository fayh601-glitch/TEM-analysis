"""Tests for Aviles & Lear–style morphometrics and log-normal summaries."""

from __future__ import annotations

import numpy as np
import pytest
from skimage.measure import label, regionprops

from tem_rods.distributions import fit_lognormal, sample_size_note
from tem_rods.morphometrics import (
    circularity_from_area_perimeter,
    equivalent_diameter_px,
    feret_diameters_px,
    principal_caliper_px,
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


def test_principal_caliper_matches_rectangle_not_ellipse():
    """ImageJ line tools measure the rod box, not the 4√λ moment ellipse.

    For a filled rectangle, the moment ellipse is ~15% longer/wider.
    """
    img = np.zeros((80, 50), dtype=np.uint8)
    img[10:60, 18:28] = 1  # 50 px long, 10 px wide
    region = regionprops(label(img))[0]
    length_px, width_px = principal_caliper_px(region.coords)
    assert 49 <= length_px <= 52
    assert 9 <= width_px <= 12
    # Ellipse fit is systematically larger than the ImageJ-style lines.
    assert region.major_axis_length > length_px + 3
    assert region.minor_axis_length > width_px + 0.5


def test_reported_length_uses_ellipse_not_tight_caliper():
    """Dark-core calipers run short on TEM rods; ellipse axes matched ImageJ ~15.5 nm."""
    img = np.zeros((90, 80), dtype=np.uint8)
    img[11:79, 30:50] = 1  # 68 px long, 20 px wide
    labels = label(img)
    region = regionprops(labels)[0]
    particles = measure_particles(labels, nm_per_pixel=0.231)
    p = particles[0]
    assert p.length_nm == pytest.approx(region.major_axis_length * 0.231, abs=0.05)
    assert p.width_nm == pytest.approx(region.minor_axis_length * 0.231, abs=0.05)
    cal_len, _cal_w = principal_caliper_px(region.coords)
    # Tight caliper is shorter than the ellipse length we report.
    assert p.length_px > cal_len


def test_size_gates_reject_too_wide_blobs():
    img = np.zeros((80, 80), dtype=np.uint8)
    img[20:60, 10:70] = 1  # ~40×60 blob, too wide to be one nanorod
    labels = label(img)
    cfg = AnalysisConfig(
        analysis_mode=AnalysisMode.RODS,
        min_length_nm=8.0,
        max_length_nm=32.0,
        min_width_nm=2.0,
        max_width_nm=9.0,
    )
    particles = measure_particles(labels, nm_per_pixel=0.231, config=cfg)
    assert particles[0].particle_class == ParticleClass.REJECT


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

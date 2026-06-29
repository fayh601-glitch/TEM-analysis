"""
Pipeline Tests — check that analysis works on synthetic TEM-like images
========================================================================

These tests build fake images with one rod and one dot, then verify that the
pipeline finds them, classifies them correctly, and writes output files. They
run automatically with `pytest` to catch regressions after code changes.
"""

from __future__ import annotations

import numpy as np
import pytest
from skimage.draw import disk, ellipse

from tem_rods.classify import classify_shape
from tem_rods.measure import measure_particles, summarize_by_class
from tem_rods.models import AnalysisConfig, ParticleClass
from tem_rods.pipeline import analyze_image
from tem_rods.preprocess import preprocess
from tem_rods.segment import segment_particles


def _synthetic_tem_image() -> tuple[np.ndarray, float]:
    """Create image with one rod-like ellipse and one dot. Returns (image, nm_per_pixel)."""
    image = np.ones((200, 200), dtype=np.float64) * 0.85
    # Dot at (50, 50), radius ~8 px
    rr, cc = disk((50, 50), 8)
    image[rr, cc] = 0.15
    # Rod-like ellipse at (140, 100): major ~40 px, minor ~8 px
    rr, cc = ellipse(100, 140, 20, 4, rotation=np.deg2rad(30))
    image[rr, cc] = 0.12

    nm_per_pixel = 0.5  # 0.5 nm/px for easy mental math
    return image, nm_per_pixel


def test_classify_rod_dot_and_reject():
    cfg = AnalysisConfig()
    assert classify_shape(eccentricity=0.95, aspect_ratio=5.0, config=cfg) == ParticleClass.ROD
    assert classify_shape(eccentricity=0.2, aspect_ratio=1.1, config=cfg) == ParticleClass.DOT
    assert classify_shape(eccentricity=0.80, aspect_ratio=2.0, config=cfg) == ParticleClass.REJECT


def test_measure_synthetic_regions():
    image, nm_per_pixel = _synthetic_tem_image()
    processed = preprocess(image, gaussian_sigma=0.5)
    labels = segment_particles(processed, min_particle_area_px=20, use_watershed=True)
    particles = measure_particles(labels, nm_per_pixel=nm_per_pixel)

    assert len(particles) >= 2
    classes = {p.particle_class for p in particles}
    assert ParticleClass.ROD in classes
    assert ParticleClass.DOT in classes

    rods = [p for p in particles if p.particle_class == ParticleClass.ROD]
    assert rods[0].length_nm > rods[0].width_nm
    assert rods[0].aspect_ratio >= 1.5


def test_summarize_by_class():
    image, nm_per_pixel = _synthetic_tem_image()
    labels = segment_particles(preprocess(image), min_particle_area_px=20)
    particles = measure_particles(labels, nm_per_pixel=nm_per_pixel)
    rod_stats = summarize_by_class(particles, ParticleClass.ROD)
    assert rod_stats["count"] >= 1
    assert rod_stats["mean_length_nm"] > rod_stats["mean_width_nm"]


def test_analyze_image_writes_outputs(tmp_path):
    image, nm_per_pixel = _synthetic_tem_image()
    image_path = tmp_path / "synthetic.png"
    from skimage import io

    io.imsave(image_path, (image * 255).astype(np.uint8))

    out_dir = tmp_path / "outputs"
    result = analyze_image(image_path, nm_per_pixel, output_dir=out_dir)

    assert len(result.particles) >= 2
    assert result.csv_path is not None
    assert result.overlay_path is not None
    assert result.csv_path.exists()
    assert result.overlay_path.exists()

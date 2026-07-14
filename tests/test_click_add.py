"""Tests for click-to-add missed particles."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from skimage.draw import ellipse

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))

from particle_review import add_particle_at_click  # noqa: E402
from tem_rods.models import ParticleClass


def test_add_particle_grows_dark_ellipse():
    image = np.ones((120, 160), dtype=np.float64) * 0.9
    rr, cc = ellipse(60, 80, 8, 28, rotation=0)
    image[rr, cc] = 0.1
    labels = np.zeros((120, 160), dtype=np.int32)

    particles, new_labels, msg = add_particle_at_click(
        image,
        labels,
        [],
        click_y=60,
        click_x=80,
        nm_per_pixel=0.5,
        preferred_class=ParticleClass.ROD,
    )
    assert len(particles) == 1
    assert particles[0].particle_class == ParticleClass.ROD
    assert particles[0].length_nm > particles[0].width_nm
    assert new_labels.max() == particles[0].particle_id
    assert "Added particle" in msg

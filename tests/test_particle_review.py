"""Unit tests for interactive particle review helpers."""

from __future__ import annotations

from tem_rods.models import ParticleClass, ParticleMeasurement

# Import from app package path via repo-relative import in tests.
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))

from particle_review import (  # noqa: E402
    default_approved_ids,
    filter_particles,
    particle_id_from_plotly_selection,
    summarize_approved,
    toggle_particle,
)


def _p(pid: int, cls: ParticleClass, length: float = 30.0) -> ParticleMeasurement:
    return ParticleMeasurement(
        pid,
        cls,
        length,
        5.0,
        6.0,
        0.9,
        100.0,
        10.0,
        20.0,
        60.0,
        10.0,
        50,
    )


def test_default_approved_skips_rejects():
    particles = [
        _p(1, ParticleClass.ROD),
        _p(2, ParticleClass.DOT),
        _p(3, ParticleClass.REJECT),
    ]
    assert default_approved_ids(particles) == {1, 2}
    assert default_approved_ids(particles, include_rejects=True) == {1, 2, 3}


def test_toggle_and_filter():
    approved = {1, 2}
    approved = toggle_particle(approved, 2)
    assert approved == {1}
    approved = toggle_particle(approved, 2)
    assert approved == {1, 2}
    particles = [_p(1, ParticleClass.ROD), _p(2, ParticleClass.ROD, 40.0)]
    kept = filter_particles(particles, {1})
    assert len(kept) == 1 and kept[0].particle_id == 1


def test_summarize_approved():
    particles = [
        _p(1, ParticleClass.ROD, 20.0),
        _p(2, ParticleClass.ROD, 40.0),
        _p(3, ParticleClass.REJECT),
    ]
    stats = summarize_approved(particles, {1, 2})
    assert stats["approved_rods"] == 2
    assert stats["mean_rod_length_nm"] == 30.0


def test_filter_approved_by_length():
    from particle_review import filter_approved_by_length

    particles = [
        ParticleMeasurement(1, ParticleClass.ROD, 30.0, 5.0, 6.0, 0.9, 100.0, 0, 0, 10, 3, 50),
        ParticleMeasurement(2, ParticleClass.ROD, 100.0, 5.0, 6.0, 0.9, 100.0, 0, 0, 10, 3, 50),
        ParticleMeasurement(3, ParticleClass.ROD, 250.0, 5.0, 6.0, 0.9, 100.0, 0, 0, 10, 3, 50),
    ]
    kept, n = filter_approved_by_length(
        particles, {1, 2, 3}, min_length_nm=50.0, max_length_nm=120.0
    )
    assert kept == {2}
    assert n == 2


def test_particle_id_from_selection():
    class Sel:
        class selection:
            points = [{"customdata": 7}]

    assert particle_id_from_plotly_selection(Sel()) == 7
    assert particle_id_from_plotly_selection({"selection": {"points": [{"customdata": 3}]}}) == 3
    assert particle_id_from_plotly_selection(None) is None

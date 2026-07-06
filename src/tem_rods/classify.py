"""
Shape Classifier — decide if a blob is a rod, dot, or background noise
=======================================================================

After segmentation finds dark blobs, this file labels each one as a nanorod,
a round dot, or "reject" (background texture that should be ignored). Rods
must be clearly elongated; dots must be roughly round; everything ambiguous
in between is rejected instead of forced into the wrong category.

The ``analysis_mode`` setting (rods / dots / both) filters labels for samples
where you know only one shape is present — round fragments are not called
"dots" in rods-only mode; they become rejects instead.
"""

from __future__ import annotations

from tem_rods.models import AnalysisConfig, AnalysisMode, ParticleClass


def classify_shape_raw(
    *,
    eccentricity: float,
    aspect_ratio: float,
    config: AnalysisConfig | None = None,
) -> ParticleClass:
    """
    Classify by shape only (rod / dot / reject), ignoring analysis_mode.

    Used internally so borderline promotion can run before mode filtering.
    """
    cfg = config or AnalysisConfig()
    if (
        eccentricity >= cfg.min_eccentricity_rod
        and aspect_ratio >= cfg.min_aspect_ratio_rod
    ):
        return ParticleClass.ROD
    if (
        eccentricity <= cfg.max_eccentricity_dot
        and aspect_ratio <= cfg.max_aspect_ratio_dot
    ):
        return ParticleClass.DOT
    return ParticleClass.REJECT


def apply_analysis_mode(
    particle_class: ParticleClass,
    mode: AnalysisMode,
) -> ParticleClass:
    """
    Restrict output to rods-only or dots-only when the sample type is known.

    In rods-only mode, round blobs (would-be dots) become rejects because
    they are usually noise or rod fragments, not real spherical particles.
    In dots-only mode, elongated blobs become rejects for the same reason.
    """
    if mode == AnalysisMode.BOTH:
        return particle_class
    if mode == AnalysisMode.RODS and particle_class == ParticleClass.DOT:
        return ParticleClass.REJECT
    if mode == AnalysisMode.DOTS and particle_class == ParticleClass.ROD:
        return ParticleClass.REJECT
    return particle_class


def classify_shape(
    *,
    eccentricity: float,
    aspect_ratio: float,
    config: AnalysisConfig | None = None,
) -> ParticleClass:
    """Classify a particle and apply the configured analysis_mode filter."""
    cfg = config or AnalysisConfig()
    base = classify_shape_raw(
        eccentricity=eccentricity,
        aspect_ratio=aspect_ratio,
        config=cfg,
    )
    return apply_analysis_mode(base, cfg.analysis_mode)

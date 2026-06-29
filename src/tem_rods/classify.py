"""
Shape Classifier — decide if a blob is a rod, dot, or background noise
=======================================================================

After segmentation finds dark blobs, this file labels each one as a nanorod,
a round dot, or "reject" (background texture that should be ignored). Rods
must be clearly elongated; dots must be roughly round; everything ambiguous
in between is rejected instead of forced into the wrong category.
"""

from __future__ import annotations

from tem_rods.models import AnalysisConfig, ParticleClass


def classify_shape(
    *,
    eccentricity: float,
    aspect_ratio: float,
    config: AnalysisConfig | None = None,
) -> ParticleClass:
    """
    Classify a particle as rod, dot, or reject.

    Rods: high eccentricity AND elongated (aspect ratio >= threshold).
    Dots: low eccentricity AND roughly round.
    Reject: ambiguous elongated smudges and other unreliable detections.
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

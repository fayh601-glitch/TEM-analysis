from __future__ import annotations

from tem_rods.models import AnalysisConfig, ParticleClass


def classify_shape(
    *,
    eccentricity: float,
    aspect_ratio: float,
    config: AnalysisConfig | None = None,
) -> ParticleClass:
    """
    Classify a particle as rod or dot (sphere-like).

    Rods: high eccentricity AND elongated (aspect ratio >= threshold).
    Dots: everything else among detected particles.
    """
    cfg = config or AnalysisConfig()
    is_rod = (
        eccentricity >= cfg.min_eccentricity_rod
        and aspect_ratio >= cfg.min_aspect_ratio_rod
    )
    return ParticleClass.ROD if is_rod else ParticleClass.DOT

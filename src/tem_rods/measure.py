from __future__ import annotations

import numpy as np
from skimage.measure import regionprops

from tem_rods.classify import classify_shape
from tem_rods.models import AnalysisConfig, ParticleClass, ParticleMeasurement


def _length_width_px(region) -> tuple[float, float]:
    """Extract length (long axis) and width (short axis) in pixels from a region."""
    length_px = float(region.feret_diameter_max)
    if hasattr(region, "feret_diameter_min"):
        width_px = float(region.feret_diameter_min)
    else:
        width_px = float(region.axis_minor_length)
    return length_px, width_px


def measure_particles(
    labels: np.ndarray,
    *,
    nm_per_pixel: float,
    config: AnalysisConfig | None = None,
) -> list[ParticleMeasurement]:
    """Measure length, width, and shape metrics for each labeled particle."""
    cfg = config or AnalysisConfig()
    measurements: list[ParticleMeasurement] = []

    for idx, region in enumerate(regionprops(labels), start=1):
        length_px, width_px = _length_width_px(region)
        if width_px <= 0:
            width_px = 1e-6

        aspect_ratio = length_px / width_px
        eccentricity = float(region.eccentricity)
        particle_class = classify_shape(
            eccentricity=eccentricity,
            aspect_ratio=aspect_ratio,
            config=cfg,
        )

        area_px = int(region.area)
        cy, cx = region.centroid

        measurements.append(
            ParticleMeasurement(
                particle_id=idx,
                particle_class=particle_class,
                length_nm=length_px * nm_per_pixel,
                width_nm=width_px * nm_per_pixel,
                aspect_ratio=aspect_ratio,
                eccentricity=eccentricity,
                area_nm2=area_px * (nm_per_pixel**2),
                centroid_y=float(cy),
                centroid_x=float(cx),
                length_px=length_px,
                width_px=width_px,
                area_px=area_px,
            )
        )

    return measurements


def summarize_by_class(
    particles: list[ParticleMeasurement],
    particle_class: ParticleClass,
) -> dict[str, float]:
    """Return mean length/width for rods or dots."""
    subset = [p for p in particles if p.particle_class == particle_class]
    if not subset:
        return {"count": 0, "mean_length_nm": float("nan"), "mean_width_nm": float("nan")}

    return {
        "count": len(subset),
        "mean_length_nm": float(np.mean([p.length_nm for p in subset])),
        "mean_width_nm": float(np.mean([p.width_nm for p in subset])),
        "std_length_nm": float(np.std([p.length_nm for p in subset])),
        "std_width_nm": float(np.std([p.width_nm for p in subset])),
    }

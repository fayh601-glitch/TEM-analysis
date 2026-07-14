"""
Particle Measurer — turn each labeled blob into length, width, and shape stats
==============================================================================

This file walks through every numbered region in a segmented image and computes
its size in pixels and nanometers. Length and width come from the fitted ellipse
axes so they match what you see drawn on the overlay. Feret diameters,
circularity, and equivalent-area diameter follow Aviles & Lear TEM reporting.
"""

from __future__ import annotations

import numpy as np
from skimage.measure import regionprops

from tem_rods.classify import apply_analysis_mode, classify_shape_raw
from tem_rods.distributions import fit_lognormal
from tem_rods.models import AnalysisConfig, ParticleClass, ParticleMeasurement
from tem_rods.morphometrics import region_morphometrics


def _length_width_px(region) -> tuple[float, float]:
    """Extract length (long axis) and width (short axis) in pixels from a region."""
    length_px = float(region.major_axis_length)
    width_px = float(region.minor_axis_length)
    if width_px <= 0:
        width_px = float(getattr(region, "feret_diameter_min", 0.0) or region.axis_minor_length)
    return length_px, width_px


def major_axis_angle_deg(region) -> float:
    """
    Angle of the major axis from +x (columns), for matplotlib Ellipse.

    Matplotlib treats ``width`` as the ellipse diameter before rotation, so this
    must match the direction of ``major_axis_length``. Skimage's ``orientation``
    property is measured from the y-axis and does not map directly.
    """
    coords = region.coords.astype(float)
    cy, cx = region.centroid
    y = coords[:, 0] - cy
    x = coords[:, 1] - cx
    cov = np.cov(x, y)
    evals, evecs = np.linalg.eigh(cov)
    major = evecs[:, int(np.argmax(evals))]
    return float(np.degrees(np.arctan2(major[1], major[0])))


def measure_from_region(
    region,
    *,
    particle_id: int,
    nm_per_pixel: float,
    particle_class: ParticleClass,
) -> ParticleMeasurement:
    """Build a ParticleMeasurement from a skimage region (shared by pipeline + click-add)."""
    length_px, width_px = _length_width_px(region)
    if width_px <= 0:
        width_px = 1e-6
    morph = region_morphometrics(region)
    area_px = int(region.area)
    cy, cx = region.centroid
    return ParticleMeasurement(
        particle_id=particle_id,
        particle_class=particle_class,
        length_nm=length_px * nm_per_pixel,
        width_nm=width_px * nm_per_pixel,
        aspect_ratio=length_px / width_px,
        eccentricity=float(region.eccentricity),
        area_nm2=area_px * (nm_per_pixel**2),
        centroid_y=float(cy),
        centroid_x=float(cx),
        length_px=length_px,
        width_px=width_px,
        area_px=area_px,
        feret_max_nm=morph["feret_max_px"] * nm_per_pixel,
        feret_min_nm=morph["feret_min_px"] * nm_per_pixel,
        circularity=morph["circularity"],
        equiv_diameter_nm=morph["equiv_diameter_px"] * nm_per_pixel,
    )


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
        particle_class = classify_shape_raw(
            eccentricity=eccentricity,
            aspect_ratio=aspect_ratio,
            config=cfg,
        )
        # Borderline elongated blobs can be promoted to rods when tuning Enright panels.
        if (
            particle_class == ParticleClass.REJECT
            and cfg.promote_borderline_rejects
            and eccentricity >= cfg.borderline_min_eccentricity
            and aspect_ratio >= cfg.borderline_min_aspect_ratio
        ):
            particle_class = ParticleClass.ROD
        particle_class = apply_analysis_mode(particle_class, cfg.analysis_mode)

        measurements.append(
            measure_from_region(
                region,
                particle_id=idx,
                nm_per_pixel=nm_per_pixel,
                particle_class=particle_class,
            )
        )

    return measurements


def summarize_by_class(
    particles: list[ParticleMeasurement],
    particle_class: ParticleClass,
) -> dict[str, float]:
    """Return mean length/width/Feret/circularity for rods or dots."""
    subset = [p for p in particles if p.particle_class == particle_class]
    empty = {
        "count": 0,
        "mean_length_nm": float("nan"),
        "mean_width_nm": float("nan"),
        "std_length_nm": float("nan"),
        "std_width_nm": float("nan"),
        "mean_feret_max_nm": float("nan"),
        "mean_feret_min_nm": float("nan"),
        "std_feret_max_nm": float("nan"),
        "mean_circularity": float("nan"),
        "mean_equiv_diameter_nm": float("nan"),
        "lognormal_feret_max_nm": float("nan"),
        "lognormal_feret_max_se_nm": float("nan"),
        "lognormal_length_nm": float("nan"),
        "lognormal_length_se_nm": float("nan"),
        "lognormal_equiv_diameter_nm": float("nan"),
        "lognormal_equiv_diameter_se_nm": float("nan"),
    }
    if not subset:
        return empty

    out: dict[str, float] = {
        "count": len(subset),
        "mean_length_nm": float(np.mean([p.length_nm for p in subset])),
        "mean_width_nm": float(np.mean([p.width_nm for p in subset])),
        "std_length_nm": float(np.std([p.length_nm for p in subset])),
        "std_width_nm": float(np.std([p.width_nm for p in subset])),
        "mean_feret_max_nm": float(np.mean([p.feret_max_nm for p in subset])),
        "mean_feret_min_nm": float(np.mean([p.feret_min_nm for p in subset])),
        "std_feret_max_nm": float(np.std([p.feret_max_nm for p in subset])),
        "mean_circularity": float(np.mean([p.circularity for p in subset])),
        "mean_equiv_diameter_nm": float(np.mean([p.equiv_diameter_nm for p in subset])),
        "lognormal_feret_max_nm": float("nan"),
        "lognormal_feret_max_se_nm": float("nan"),
        "lognormal_length_nm": float("nan"),
        "lognormal_length_se_nm": float("nan"),
        "lognormal_equiv_diameter_nm": float("nan"),
        "lognormal_equiv_diameter_se_nm": float("nan"),
    }
    for key, values in (
        ("feret_max", [p.feret_max_nm for p in subset]),
        ("length", [p.length_nm for p in subset]),
        ("equiv_diameter", [p.equiv_diameter_nm for p in subset]),
    ):
        fit = fit_lognormal(values)
        if fit is not None:
            out[f"lognormal_{key}_nm"] = fit.geometric_mean
            out[f"lognormal_{key}_se_nm"] = fit.geometric_mean_se
    return out

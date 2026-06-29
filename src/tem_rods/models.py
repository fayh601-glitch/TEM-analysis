from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path


class ParticleClass(str, Enum):
    ROD = "rod"
    DOT = "dot"


@dataclass(frozen=True)
class AnalysisConfig:
    """Tunable parameters for segmentation and rod/dot classification."""

    gaussian_sigma: float = 1.0
    min_particle_area_px: int = 30
    max_particle_area_px: int | None = None
    min_eccentricity_rod: float = 0.85
    min_aspect_ratio_rod: float = 1.5
    watershed_min_distance: int = 10
    use_watershed: bool = True
    exclude_border: bool = True


@dataclass
class ParticleMeasurement:
    particle_id: int
    particle_class: ParticleClass
    length_nm: float
    width_nm: float
    aspect_ratio: float
    eccentricity: float
    area_nm2: float
    centroid_y: float
    centroid_x: float
    length_px: float
    width_px: float
    area_px: int


@dataclass
class AnalysisResult:
    image_path: Path
    nm_per_pixel: float
    particles: list[ParticleMeasurement] = field(default_factory=list)
    overlay_path: Path | None = None
    csv_path: Path | None = None

    @property
    def rods(self) -> list[ParticleMeasurement]:
        return [p for p in self.particles if p.particle_class == ParticleClass.ROD]

    @property
    def dots(self) -> list[ParticleMeasurement]:
        return [p for p in self.particles if p.particle_class == ParticleClass.DOT]

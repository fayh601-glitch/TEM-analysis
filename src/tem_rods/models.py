"""
Shared Data Types — settings and results used across the project
=================================================================

This file defines the "vocabulary" the rest of the code uses: what a rod, dot,
or rejected blob looks like in data, and which knobs (area thresholds, etc.)
control the analysis. Think of it as the project's dictionary — no image
processing happens here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path


class ParticleClass(str, Enum):
    """How each detected blob is categorized."""

    ROD = "rod"
    DOT = "dot"
    REJECT = "reject"


class ThresholdMode(str, Enum):
    """How to binarize particles vs background."""

    OTSU = "otsu"
    PERCENTILE = "percentile"
    LOCAL = "local"
    AUTO = "auto"


class AnalysisMode(str, Enum):
    """
    What particle shapes to report for this image.

    Use RODS for nanorod-only samples (Enright S2A); DOTS for spherical QDs;
    BOTH when the sample may contain either shape.
    """

    BOTH = "both"
    RODS = "rods"
    DOTS = "dots"


@dataclass(frozen=True)
class AnalysisConfig:
    """Tunable parameters for segmentation and rod/dot classification."""

    gaussian_sigma: float = 1.0
    min_particle_area_px: int = 30
    max_particle_area_px: int | None = None
    min_eccentricity_rod: float = 0.85
    min_aspect_ratio_rod: float = 1.5
    max_eccentricity_dot: float = 0.75
    max_aspect_ratio_dot: float = 1.35
    min_solidity: float = 0.48
    min_extent: float = 0.18
    min_local_contrast: float = 0.025
    mask_bottom_fraction: float = 0.12
    use_scale_bar_bbox_mask: bool = True
    expected_scale_bar_nm: float | None = None
    morphology_closing_radius: int = 2
    fill_holes: bool = True
    watershed_min_distance: int = 10
    use_watershed: bool = False
    split_touching_particles: bool = True
    split_min_area_px: int = 500
    split_max_aspect_ratio: float = 4.5
    split_min_width_px: float = 22.0
    split_watershed_min_distance: int = 5
    exclude_border: bool = True
    threshold_mode: ThresholdMode = ThresholdMode.AUTO
    percentile_threshold: float = 40.0
    local_threshold_block_size: int = 35
    local_threshold_offset: float = 0.01
    crop_margins: bool = False
    use_clahe: bool = False
    show_rejected_on_overlay: bool = True
    write_segmentation_debug: bool = False
    promote_borderline_rejects: bool = False
    borderline_min_eccentricity: float = 0.78
    borderline_min_aspect_ratio: float = 1.35
    analysis_mode: AnalysisMode = AnalysisMode.BOTH
    max_rods: int | None = None
    max_dots: int | None = None
    sample_seed: int = 42
    merge_warning_mean_median_ratio: float = 2.5


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
    scale_bar_pixels: float | None = None
    scale_bar_nm: float | None = None
    warnings: list[str] = field(default_factory=list)
    show_rejected_on_overlay: bool = True
    analysis_mode: AnalysisMode = AnalysisMode.BOTH
    selected_rod_ids: set[int] | None = None
    selected_dot_ids: set[int] | None = None

    @property
    def rods(self) -> list[ParticleMeasurement]:
        return [p for p in self.particles if p.particle_class == ParticleClass.ROD]

    @property
    def reported_rods(self) -> list[ParticleMeasurement]:
        """Rods included in CSV/overlay after optional --max-rods subsampling."""
        rods = self.rods
        if self.selected_rod_ids is None:
            return rods
        return [p for p in rods if p.particle_id in self.selected_rod_ids]

    @property
    def reported_dots(self) -> list[ParticleMeasurement]:
        """Dots included in CSV/overlay after optional --max-dots subsampling."""
        dots = self.dots
        if self.selected_dot_ids is None:
            return dots
        return [p for p in dots if p.particle_id in self.selected_dot_ids]

    @property
    def dots(self) -> list[ParticleMeasurement]:
        return [p for p in self.particles if p.particle_class == ParticleClass.DOT]

    @property
    def rejected(self) -> list[ParticleMeasurement]:
        return [p for p in self.particles if p.particle_class == ParticleClass.REJECT]

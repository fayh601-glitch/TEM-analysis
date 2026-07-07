"""
Image Presets — tuned AnalysisConfig bundles for common TEM figure types
=========================================================================

Use presets when images differ in scale bar size, density, or export format
(Enright SI panels vs screenshots vs large sparse cluster micrographs).
"""

from __future__ import annotations

from dataclasses import dataclass

from tem_rods.models import AnalysisConfig, AnalysisMode, ThresholdMode


@dataclass(frozen=True)
class ImagePreset:
    name: str
    config: AnalysisConfig
    default_scale_bar_nm: float | None = None
    description: str = ""


PRESETS: dict[str, ImagePreset] = {
    "enright_rods": ImagePreset(
        name="enright_rods",
        description="Dense Enright SI nanorod panels (20 nm scale bar).",
        default_scale_bar_nm=20.0,
        config=AnalysisConfig(
            min_particle_area_px=110,
            min_eccentricity_rod=0.82,
            min_aspect_ratio_rod=1.38,
            max_eccentricity_dot=0.72,
            max_aspect_ratio_dot=1.28,
            promote_borderline_rejects=True,
            borderline_min_eccentricity=0.75,
            borderline_min_aspect_ratio=1.30,
            # Splitting Enright clusters often creates short fragments; keep off for SI panels.
            split_touching_particles=False,
            split_min_area_px=520,
            split_min_width_px=24.0,
            split_watershed_min_distance=5,
            split_max_aspect_ratio=4.5,
            mask_bottom_fraction=0.10,
            use_scale_bar_bbox_mask=False,
            fill_holes=True,
            threshold_mode=ThresholdMode.OTSU,
            crop_margins=False,
            show_rejected_on_overlay=True,
            write_segmentation_debug=True,
            analysis_mode=AnalysisMode.RODS,
        ),
    ),
    "dots_only": ImagePreset(
        name="dots_only",
        description="Spherical quantum dots / round nanoparticles only.",
        default_scale_bar_nm=20.0,
        config=AnalysisConfig(
            min_particle_area_px=80,
            max_eccentricity_dot=0.78,
            max_aspect_ratio_dot=1.35,
            min_eccentricity_rod=0.90,
            min_aspect_ratio_rod=1.8,
            split_touching_particles=False,
            mask_bottom_fraction=0.10,
            threshold_mode=ThresholdMode.OTSU,
            show_rejected_on_overlay=True,
            write_segmentation_debug=True,
            analysis_mode=AnalysisMode.DOTS,
        ),
    ),
    "screenshot": ImagePreset(
        name="screenshot",
        description="Paper figure screenshots with white margins.",
        default_scale_bar_nm=20.0,
        config=AnalysisConfig(
            min_particle_area_px=80,
            split_touching_particles=True,
            split_min_area_px=400,
            mask_bottom_fraction=0.12,
            use_scale_bar_bbox_mask=False,
            threshold_mode=ThresholdMode.PERCENTILE,
            crop_margins=True,
            gaussian_sigma=1.0,
        ),
    ),
    "sparse_cluster": ImagePreset(
        name="sparse_cluster",
        description="Large sparse cluster images with labeled scale bars (e.g. 200 nm).",
        default_scale_bar_nm=200.0,
        config=AnalysisConfig(
            min_particle_area_px=300,
            split_touching_particles=False,
            mask_bottom_fraction=0.12,
            use_scale_bar_bbox_mask=True,
            expected_scale_bar_nm=200.0,
            fill_holes=True,
            morphology_closing_radius=2,
            threshold_mode=ThresholdMode.LOCAL,
            crop_margins=False,
            min_local_contrast=0.02,
            gaussian_sigma=1.5,
            local_threshold_block_size=51,
            analysis_mode=AnalysisMode.RODS,
        ),
    ),
    "dense_rods_50nm": ImagePreset(
        name="dense_rods_50nm",
        description="Dense nanorod fields (~50 nm scale bar); reduces light-center rod splitting.",
        default_scale_bar_nm=50.0,
        config=AnalysisConfig(
            min_particle_area_px=120,
            min_eccentricity_rod=0.78,
            min_aspect_ratio_rod=1.35,
            split_touching_particles=True,
            split_min_area_px=350,
            mask_bottom_fraction=0.15,
            use_scale_bar_bbox_mask=True,
            expected_scale_bar_nm=50.0,
            fill_holes=True,
            morphology_closing_radius=2,
            threshold_mode=ThresholdMode.LOCAL,
            local_threshold_block_size=41,
            gaussian_sigma=1.2,
            min_local_contrast=0.015,
            analysis_mode=AnalysisMode.RODS,
        ),
    ),
    "dense_rods": ImagePreset(
        name="dense_rods",
        description="Alias for dense_rods_50nm (dense fields, hole-filling enabled).",
        default_scale_bar_nm=50.0,
        config=AnalysisConfig(
            min_particle_area_px=120,
            min_eccentricity_rod=0.78,
            min_aspect_ratio_rod=1.35,
            split_touching_particles=True,
            split_min_area_px=350,
            mask_bottom_fraction=0.15,
            use_scale_bar_bbox_mask=True,
            expected_scale_bar_nm=50.0,
            fill_holes=True,
            morphology_closing_radius=2,
            threshold_mode=ThresholdMode.LOCAL,
            local_threshold_block_size=41,
            gaussian_sigma=1.2,
            min_local_contrast=0.015,
            analysis_mode=AnalysisMode.RODS,
        ),
    ),
}


def get_preset(name: str) -> ImagePreset:
    key = name.lower().replace("-", "_")
    if key not in PRESETS:
        known = ", ".join(sorted(PRESETS))
        raise ValueError(f"Unknown preset {name!r}. Choose from: {known}")
    return PRESETS[key]

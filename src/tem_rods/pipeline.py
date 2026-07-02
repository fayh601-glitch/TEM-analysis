"""
Analysis Pipeline — run the full TEM image workflow end to end
===============================================================

This is the main "conductor" file: it loads an image, finds particles, measures
them, classifies rods vs dots, and saves a CSV plus an annotated overlay PNG.
Most other files are helpers that this one calls in order.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Ellipse
from skimage.measure import find_contours, regionprops

from tem_rods.calibrate import validate_nm_per_pixel
from tem_rods.io import load_grayscale
from tem_rods.measure import major_axis_angle_deg, measure_particles, summarize_by_class
from tem_rods.models import AnalysisConfig, AnalysisResult, ParticleClass
from tem_rods.preprocess import preprocess
from tem_rods.scale_bar import ScaleBarDetection
from tem_rods.segment import segment_particles_from_config


def analyze_image(
    image_path: str | Path,
    nm_per_pixel: float,
    *,
    output_dir: str | Path | None = None,
    config: AnalysisConfig | None = None,
    scale_bar: ScaleBarDetection | None = None,
    save_outputs: bool = True,
) -> AnalysisResult:
    """
    Full pipeline: load → preprocess → segment → classify → measure → export.
    """
    cfg = config or AnalysisConfig()
    image_path = Path(image_path)
    nm_per_pixel = validate_nm_per_pixel(nm_per_pixel)
    warnings: list[str] = []

    image = load_grayscale(image_path)
    processed = preprocess(
        image,
        gaussian_sigma=cfg.gaussian_sigma,
        crop_margins=cfg.crop_margins,
        use_clahe=cfg.use_clahe,
    )
    exclude_bbox = scale_bar.bbox if scale_bar is not None and cfg.use_scale_bar_bbox_mask else None
    labels = segment_particles_from_config(processed, cfg, exclude_bbox=exclude_bbox)
    particles = measure_particles(labels, nm_per_pixel=nm_per_pixel, config=cfg)
    warnings.extend(_quality_warnings(particles, nm_per_pixel, scale_bar))

    result = AnalysisResult(
        image_path=image_path,
        nm_per_pixel=nm_per_pixel,
        particles=particles,
        scale_bar_pixels=scale_bar.bar_pixels if scale_bar else None,
        scale_bar_nm=scale_bar.bar_nm if scale_bar else None,
        warnings=warnings,
        show_rejected_on_overlay=cfg.show_rejected_on_overlay,
    )

    if save_outputs:
        out_dir = Path(output_dir) if output_dir else Path("outputs")
        out_dir.mkdir(parents=True, exist_ok=True)
        stem = image_path.stem
        result.csv_path = out_dir / f"{stem}_measurements.csv"
        result.overlay_path = out_dir / f"{stem}_overlay.png"
        _write_csv(result)
        _write_overlay(image, labels, result, scale_bar=scale_bar)
        if cfg.write_segmentation_debug:
            debug_path = out_dir / f"{stem}_segments_debug.png"
            _write_segments_debug(image, labels, debug_path)
            print(f"Segment debug: {debug_path}")

    return result


def _quality_warnings(
    particles,
    nm_per_pixel: float,
    scale_bar: ScaleBarDetection | None,
) -> list[str]:
    warnings: list[str] = []
    if scale_bar is not None and scale_bar.confidence < 0.5:
        warnings.append("Scale bar detection confidence is low; verify calibration manually.")

    if not particles:
        warnings.append("No particles detected.")
        return warnings

    rejected_frac = sum(1 for p in particles if p.particle_class == ParticleClass.REJECT) / len(
        particles
    )
    if rejected_frac > 0.35:
        warnings.append(
            f"High reject rate ({rejected_frac:.0%}); segmentation or classification may need tuning."
        )

    rods = [p for p in particles if p.particle_class == ParticleClass.ROD]
    if rods:
        mean_len = float(np.mean([p.length_nm for p in rods]))
        if mean_len < 5 or mean_len > 500:
            warnings.append(
                f"Rod mean length ({mean_len:.1f} nm) looks unusual for nm/pixel={nm_per_pixel:.3f}."
            )
    return warnings


def _write_csv(result: AnalysisResult) -> None:
    rows = [
        {
            "particle_id": p.particle_id,
            "class": p.particle_class.value,
            "length_nm": round(p.length_nm, 2),
            "width_nm": round(p.width_nm, 2),
            "aspect_ratio": round(p.aspect_ratio, 3),
            "eccentricity": round(p.eccentricity, 3),
            "area_nm2": round(p.area_nm2, 2),
            "centroid_x": round(p.centroid_x, 1),
            "centroid_y": round(p.centroid_y, 1),
        }
        for p in result.particles
    ]
    df = pd.DataFrame(rows)
    assert result.csv_path is not None
    df.to_csv(result.csv_path, index=False)


def _write_overlay(
    image: np.ndarray,
    labels: np.ndarray,
    result: AnalysisResult,
    *,
    scale_bar: ScaleBarDetection | None = None,
) -> None:
    assert result.overlay_path is not None
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(image, cmap="gray")

    color_map = {
        ParticleClass.ROD: "#00ff88",
        ParticleClass.DOT: "#4488ff",
        ParticleClass.REJECT: "#ff6644",
    }

    show_rejected = result.show_rejected_on_overlay

    for region, particle in zip(regionprops(labels), result.particles):
        if particle.particle_class == ParticleClass.REJECT and not show_rejected:
            continue

        cy, cx = region.centroid
        color = color_map[particle.particle_class]
        is_reject = particle.particle_class == ParticleClass.REJECT

        particle_mask = labels == region.label
        line_style = ":" if is_reject else "-"
        line_width = 1.2 if is_reject else 1.5
        for contour in find_contours(particle_mask.astype(float), 0.5):
            ax.plot(
                contour[:, 1],
                contour[:, 0],
                color=color,
                linewidth=line_width,
                linestyle=line_style,
                alpha=0.9 if is_reject else 1.0,
            )

        if not is_reject:
            angle_deg = major_axis_angle_deg(region)
            ell = Ellipse(
                (cx, cy),
                width=region.major_axis_length,
                height=region.minor_axis_length,
                angle=angle_deg,
                fill=False,
                edgecolor=color,
                linewidth=1.0,
                linestyle="--",
                alpha=0.85,
            )
            ax.add_patch(ell)

        prefix = "X" if is_reject else particle.particle_class.value[0].upper()
        label_offset = region.major_axis_length / 2 + 4
        ax.text(
            cx,
            cy - label_offset,
            f"{prefix} {particle.length_nm:.1f}×{particle.width_nm:.1f} nm",
            color=color,
            fontsize=6 if is_reject else 7,
            ha="center",
            va="bottom",
            alpha=0.85 if is_reject else 1.0,
        )

    rod_stats = summarize_by_class(result.particles, ParticleClass.ROD)
    dot_stats = summarize_by_class(result.particles, ParticleClass.DOT)
    reject_count = len(result.rejected)
    calib = f"{result.nm_per_pixel:.3f} nm/px"
    if result.scale_bar_pixels and result.scale_bar_nm:
        calib = (
            f"{result.scale_bar_nm:g} nm / {result.scale_bar_pixels:.0f} px "
            f"= {result.nm_per_pixel:.3f} nm/px"
        )
    title = (
        f"{result.image_path.name} | {calib} | "
        f"rods: {rod_stats['count']} | dots: {dot_stats['count']} | "
        f"rejected: {reject_count} | "
        f"green=rod blue=dot orange=reject"
    )
    ax.set_title(title, fontsize=11)

    if scale_bar is not None:
        row_min, col_min, row_max, col_max = scale_bar.bbox
        ax.plot(
            [col_min, col_max, col_max, col_min, col_min],
            [row_min, row_min, row_max, row_max, row_min],
            color="#ffcc00",
            linewidth=1.0,
            linestyle=":",
            alpha=0.9,
        )

    ax.axis("off")
    fig.tight_layout()
    fig.savefig(result.overlay_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _write_segments_debug(
    image: np.ndarray,
    labels: np.ndarray,
    output_path: Path,
) -> None:
    """
    Draw every segmented blob (before rod/dot/reject classification) with numeric IDs.

    Use this to see whether missing rods failed at segmentation vs classification.
    """
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(image, cmap="gray")

    cmap = plt.colormaps["nipy_spectral"]
    n_labels = int(labels.max())
    for region in regionprops(labels):
        mask = labels == region.label
        color = cmap(region.label / max(n_labels, 1))
        for contour in find_contours(mask.astype(float), 0.5):
            ax.plot(contour[:, 1], contour[:, 0], color=color, linewidth=1.2)
        cy, cx = region.centroid
        ax.text(
            cx,
            cy,
            str(region.label),
            color="white",
            fontsize=6,
            ha="center",
            va="center",
            bbox=dict(boxstyle="round,pad=0.1", facecolor="black", alpha=0.5),
        )

    ax.set_title(
        f"Segmentation debug ({n_labels} blobs) — compare to overlay for classification",
        fontsize=11,
    )
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def print_summary(result: AnalysisResult) -> None:
    """Print human-readable summary for CLI."""
    rod_stats = summarize_by_class(result.particles, ParticleClass.ROD)
    dot_stats = summarize_by_class(result.particles, ParticleClass.DOT)
    reject_count = len(result.rejected)

    print(f"\nImage: {result.image_path}")
    print(f"Calibration: {result.nm_per_pixel:.4f} nm/pixel")
    if result.scale_bar_pixels and result.scale_bar_nm:
        print(
            f"Scale bar: {result.scale_bar_nm:g} nm / "
            f"{result.scale_bar_pixels:.1f} px"
        )
    print(f"Total particles: {len(result.particles)}")
    print(f"  Rods: {rod_stats['count']}")
    print(f"  Dots: {dot_stats['count']}")
    print(f"  Rejected: {reject_count}")

    if rod_stats["count"] > 0:
        print(
            f"  Rod mean length: {rod_stats['mean_length_nm']:.1f} ± "
            f"{rod_stats['std_length_nm']:.1f} nm"
        )
        print(
            f"  Rod mean width:  {rod_stats['mean_width_nm']:.1f} ± "
            f"{rod_stats['std_width_nm']:.1f} nm"
        )
    if dot_stats["count"] > 0:
        print(
            f"  Dot mean diameter (major axis): {dot_stats['mean_length_nm']:.1f} ± "
            f"{dot_stats['std_length_nm']:.1f} nm"
        )

    if result.csv_path:
        print(f"\nCSV: {result.csv_path}")
    if result.overlay_path:
        print(f"Overlay: {result.overlay_path}")
    for warning in result.warnings:
        print(f"Warning: {warning}")

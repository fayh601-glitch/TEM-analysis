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
from tem_rods.segment import segment_particles_from_config


def analyze_image(
    image_path: str | Path,
    nm_per_pixel: float,
    *,
    output_dir: str | Path | None = None,
    config: AnalysisConfig | None = None,
    save_outputs: bool = True,
) -> AnalysisResult:
    """
    Full pipeline: load → preprocess → segment → classify → measure → export.
    """
    cfg = config or AnalysisConfig()
    image_path = Path(image_path)
    nm_per_pixel = validate_nm_per_pixel(nm_per_pixel)

    image = load_grayscale(image_path)
    processed = preprocess(image, gaussian_sigma=cfg.gaussian_sigma)
    labels = segment_particles_from_config(processed, cfg)
    particles = measure_particles(labels, nm_per_pixel=nm_per_pixel, config=cfg)

    result = AnalysisResult(
        image_path=image_path,
        nm_per_pixel=nm_per_pixel,
        particles=particles,
    )

    if save_outputs:
        out_dir = Path(output_dir) if output_dir else Path("outputs")
        out_dir.mkdir(parents=True, exist_ok=True)
        stem = image_path.stem
        result.csv_path = out_dir / f"{stem}_measurements.csv"
        result.overlay_path = out_dir / f"{stem}_overlay.png"
        _write_csv(result)
        _write_overlay(image, labels, result)

    return result


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
) -> None:
    assert result.overlay_path is not None
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(image, cmap="gray")

    color_map = {
        ParticleClass.ROD: "#00ff88",
        ParticleClass.DOT: "#4488ff",
        ParticleClass.REJECT: "#ff6644",
    }

    for region, particle in zip(regionprops(labels), result.particles):
        if particle.particle_class == ParticleClass.REJECT:
            continue

        cy, cx = region.centroid
        color = color_map[particle.particle_class]

        particle_mask = labels == region.label
        for contour in find_contours(particle_mask.astype(float), 0.5):
            ax.plot(contour[:, 1], contour[:, 0], color=color, linewidth=1.5)

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

        label_offset = region.major_axis_length / 2 + 4
        ax.text(
            cx,
            cy - label_offset,
            f"{particle.particle_class.value[0].upper()} "
            f"{particle.length_nm:.1f}×{particle.width_nm:.1f} nm",
            color=color,
            fontsize=7,
            ha="center",
            va="bottom",
        )

    rod_stats = summarize_by_class(result.particles, ParticleClass.ROD)
    dot_stats = summarize_by_class(result.particles, ParticleClass.DOT)
    reject_count = len(result.rejected)
    title = (
        f"{result.image_path.name} | "
        f"rods: {rod_stats['count']} | dots: {dot_stats['count']} | "
        f"rejected: {reject_count}"
    )
    ax.set_title(title, fontsize=11)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(result.overlay_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def print_summary(result: AnalysisResult) -> None:
    """Print human-readable summary for CLI."""
    rod_stats = summarize_by_class(result.particles, ParticleClass.ROD)
    dot_stats = summarize_by_class(result.particles, ParticleClass.DOT)
    reject_count = len(result.rejected)

    print(f"\nImage: {result.image_path}")
    print(f"Calibration: {result.nm_per_pixel:.4f} nm/pixel")
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

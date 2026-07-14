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
from tem_rods.scale_bar import ScaleBarDetection, detect_scale_bar
from tem_rods.segment import segment_particles_from_config


def _resolve_exclude_bbox(
    image_path: Path,
    scale_bar: ScaleBarDetection | None,
    cfg: AnalysisConfig,
    *,
    scale_bar_nm_hint: float | None = None,
) -> tuple[tuple[int, int, int, int] | None, list[str]]:
    """
    Return a bbox to mask during segmentation (scale bar + label text).

    Uses the detected scale bar when available; otherwise tries auto-detection
    when use_scale_bar_bbox_mask is enabled.
    """
    notes: list[str] = []
    if not cfg.use_scale_bar_bbox_mask:
        return None, notes

    if scale_bar is not None:
        return scale_bar.bbox, notes

    hint = scale_bar_nm_hint or cfg.expected_scale_bar_nm
    try:
        detected = detect_scale_bar(image_path, scale_bar_nm=hint)
        notes.append("Auto-detected scale bar region for masking (label text excluded).")
        return detected.bbox, notes
    except ValueError:
        notes.append(
            "Could not auto-detect scale bar bbox; bottom strip mask still applied."
        )
        return None, notes


def _select_class_ids(
    particles: list,
    particle_class: ParticleClass,
    max_count: int,
    seed: int,
) -> set[int]:
    """Randomly pick up to max_count particle_ids of the given class."""
    subset = [p for p in particles if p.particle_class == particle_class]
    if len(subset) <= max_count:
        return {p.particle_id for p in subset}
    rng = np.random.default_rng(seed)
    pick = rng.choice(len(subset), size=max_count, replace=False)
    return {subset[int(i)].particle_id for i in pick}


def _select_rod_ids(
    particles: list,
    max_rods: int,
    seed: int,
) -> set[int]:
    return _select_class_ids(particles, ParticleClass.ROD, max_rods, seed)


def _select_dot_ids(
    particles: list,
    max_dots: int,
    seed: int,
) -> set[int]:
    return _select_class_ids(particles, ParticleClass.DOT, max_dots, seed)


def _particle_is_drawn(particle, result: AnalysisResult) -> bool:
    """Whether a particle appears on the overlay and in the exported CSV."""
    if (
        particle.particle_class == ParticleClass.ROD
        and result.selected_rod_ids is not None
        and particle.particle_id not in result.selected_rod_ids
    ):
        return False
    if (
        particle.particle_class == ParticleClass.DOT
        and result.selected_dot_ids is not None
        and particle.particle_id not in result.selected_dot_ids
    ):
        return False
    return True


def analyze_image(
    image_path: str | Path,
    nm_per_pixel: float,
    *,
    output_dir: str | Path | None = None,
    config: AnalysisConfig | None = None,
    scale_bar: ScaleBarDetection | None = None,
    save_outputs: bool = True,
    scale_bar_nm_hint: float | None = None,
) -> AnalysisResult:
    """
    Full pipeline: load → preprocess → segment → classify → measure → export.
    """
    cfg = config or AnalysisConfig()
    image_path = Path(image_path)
    nm_per_pixel = validate_nm_per_pixel(nm_per_pixel)
    warnings: list[str] = []

    image = load_grayscale(image_path)
    # Preprocess reduces film grain so thresholding does not count every speck as a particle.
    processed = preprocess(
        image,
        gaussian_sigma=cfg.gaussian_sigma,
        crop_margins=cfg.crop_margins,
        use_clahe=cfg.use_clahe,
    )
    exclude_bbox, mask_notes = _resolve_exclude_bbox(
        image_path,
        scale_bar,
        cfg,
        scale_bar_nm_hint=scale_bar_nm_hint,
    )
    warnings.extend(mask_notes)
    # Segmentation is the hardest step: find dark blobs and drop scale-bar strip / noise.
    labels = segment_particles_from_config(processed, cfg, exclude_bbox=exclude_bbox)
    particles = measure_particles(labels, nm_per_pixel=nm_per_pixel, config=cfg)
    selected_rod_ids: set[int] | None = None
    selected_dot_ids: set[int] | None = None
    if cfg.max_rods is not None and cfg.max_rods > 0:
        selected_rod_ids = _select_rod_ids(particles, cfg.max_rods, cfg.sample_seed)
    if cfg.max_dots is not None and cfg.max_dots > 0:
        selected_dot_ids = _select_dot_ids(particles, cfg.max_dots, cfg.sample_seed)
    warnings.extend(_quality_warnings(particles, nm_per_pixel, scale_bar, cfg))
    if selected_rod_ids is not None:
        total_rods = sum(1 for p in particles if p.particle_class == ParticleClass.ROD)
        if total_rods > len(selected_rod_ids):
            warnings.append(
                f"Subsampled rods for report: {len(selected_rod_ids)} of {total_rods} detected "
                f"(seed={cfg.sample_seed})."
            )
    if selected_dot_ids is not None:
        total_dots = sum(1 for p in particles if p.particle_class == ParticleClass.DOT)
        if total_dots > len(selected_dot_ids):
            warnings.append(
                f"Subsampled dots for report: {len(selected_dot_ids)} of {total_dots} detected "
                f"(seed={cfg.sample_seed})."
            )

    result = AnalysisResult(
        image_path=image_path,
        nm_per_pixel=nm_per_pixel,
        particles=particles,
        scale_bar_pixels=scale_bar.bar_pixels if scale_bar else None,
        scale_bar_nm=scale_bar.bar_nm if scale_bar else None,
        warnings=warnings,
        show_rejected_on_overlay=cfg.show_rejected_on_overlay,
        analysis_mode=cfg.analysis_mode,
        selected_rod_ids=selected_rod_ids,
        selected_dot_ids=selected_dot_ids,
        labels=labels,
        image=image,
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
    cfg: AnalysisConfig,
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
        lengths = [p.length_nm for p in rods]
        mean_len = float(np.mean(lengths))
        median_len = float(np.median(lengths))
        if mean_len < 5 or mean_len > 500:
            warnings.append(
                f"Rod mean length ({mean_len:.1f} nm) looks unusual for nm/pixel={nm_per_pixel:.3f}."
            )
        ratio = cfg.merge_warning_mean_median_ratio
        if median_len > 0 and mean_len / median_len >= ratio:
            warnings.append(
                f"Mean rod length ({mean_len:.0f} nm) is much larger than median ({median_len:.0f} nm) "
                f"— likely merged clusters or split fragments; interpret counts with caution."
            )
    return warnings


def _write_csv(result: AnalysisResult) -> None:
    rows = [
        {
            "particle_id": p.particle_id,
            "class": p.particle_class.value,
            "length_nm_ellipse": round(p.length_nm, 2),
            "width_nm_ellipse": round(p.width_nm, 2),
            "feret_max_nm": round(p.feret_max_nm, 2),
            "feret_min_nm": round(p.feret_min_nm, 2),
            "equiv_diameter_nm": round(p.equiv_diameter_nm, 2),
            "circularity": round(p.circularity, 3),
            "aspect_ratio": round(p.aspect_ratio, 3),
            "eccentricity": round(p.eccentricity, 3),
            "area_nm2": round(p.area_nm2, 2),
            "centroid_x": round(p.centroid_x, 1),
            "centroid_y": round(p.centroid_y, 1),
            # Legacy aliases (same as ellipse axes)
            "length_nm": round(p.length_nm, 2),
            "width_nm": round(p.width_nm, 2),
        }
        for p in result.particles
        if _particle_is_drawn(p, result)
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
        if not _particle_is_drawn(particle, result):
            continue
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

    rod_stats = summarize_by_class(result.reported_rods, ParticleClass.ROD)
    dot_stats = summarize_by_class(result.reported_dots, ParticleClass.DOT)
    reject_count = len(result.rejected)
    calib = f"{result.nm_per_pixel:.3f} nm/px"
    if result.scale_bar_pixels and result.scale_bar_nm:
        calib = (
            f"{result.scale_bar_nm:g} nm / {result.scale_bar_pixels:.0f} px "
            f"= {result.nm_per_pixel:.3f} nm/px"
        )
    title = (
        f"{result.image_path.name} | mode={result.analysis_mode.value} | {calib} | "
        f"rods: {rod_stats['count']} | dots: {dot_stats['count']} | "
        f"rejected: {reject_count} | "
        f"green=rod blue=dot orange=reject"
    )
    if result.selected_rod_ids is not None and len(result.selected_rod_ids) < len(result.rods):
        title += f" | reported rods: {len(result.selected_rod_ids)}"
    if result.selected_dot_ids is not None and len(result.selected_dot_ids) < len(result.dots):
        title += f" | reported dots: {len(result.selected_dot_ids)}"
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
    rod_stats = summarize_by_class(result.reported_rods, ParticleClass.ROD)
    dot_stats = summarize_by_class(result.reported_dots, ParticleClass.DOT)
    reject_count = len(result.rejected)
    total_rods = len(result.rods)

    print(f"\nImage: {result.image_path}")
    print(f"Analysis mode: {result.analysis_mode.value}")
    if result.selected_rod_ids is not None and total_rods > len(result.selected_rod_ids):
        print(f"Rods detected: {total_rods} (reporting {len(result.selected_rod_ids)} via --max-rods)")
    total_dots = len(result.dots)
    if result.selected_dot_ids is not None and total_dots > len(result.selected_dot_ids):
        print(f"Dots detected: {total_dots} (reporting {len(result.selected_dot_ids)} via --max-dots)")
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
            f"  Rod mean length (ellipse): {rod_stats['mean_length_nm']:.1f} ± "
            f"{rod_stats['std_length_nm']:.1f} nm"
        )
        print(
            f"  Rod mean width (ellipse):  {rod_stats['mean_width_nm']:.1f} ± "
            f"{rod_stats['std_width_nm']:.1f} nm"
        )
        print(
            f"  Rod mean Feret max/min:    {rod_stats['mean_feret_max_nm']:.1f} / "
            f"{rod_stats['mean_feret_min_nm']:.1f} nm"
        )
        if not np.isnan(rod_stats.get("lognormal_feret_max_nm", float("nan"))):
            print(
                f"  Rod Feret max (log-normal geom.): "
                f"{rod_stats['lognormal_feret_max_nm']:.1f} ± "
                f"{rod_stats['lognormal_feret_max_se_nm']:.1f} nm"
            )
        print(f"  Rod mean circularity:     {rod_stats['mean_circularity']:.3f}")
    if dot_stats["count"] > 0:
        print(
            f"  Dot mean equiv. diameter:  {dot_stats['mean_equiv_diameter_nm']:.1f} nm"
        )
        print(
            f"  Dot mean Feret max:        {dot_stats['mean_feret_max_nm']:.1f} nm"
        )
        if not np.isnan(dot_stats.get("lognormal_equiv_diameter_nm", float("nan"))):
            print(
                f"  Dot diam. (log-normal geom.): "
                f"{dot_stats['lognormal_equiv_diameter_nm']:.1f} ± "
                f"{dot_stats['lognormal_equiv_diameter_se_nm']:.1f} nm"
            )
        print(f"  Dot mean circularity:     {dot_stats['mean_circularity']:.3f}")

    if result.csv_path:
        print(f"\nCSV: {result.csv_path}")
    if result.overlay_path:
        print(f"Overlay: {result.overlay_path}")
    for warning in result.warnings:
        print(f"Warning: {warning}")

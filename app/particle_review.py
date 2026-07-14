"""
Interactive particle review helpers for the Streamlit app.

Build a clickable Plotly overlay and filter exports to human-approved IDs.
"""

from __future__ import annotations

from dataclasses import asdict
from typing import Iterable

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from skimage.measure import find_contours, regionprops

from tem_rods.models import ParticleClass, ParticleMeasurement


def default_approved_ids(
    particles: Iterable[ParticleMeasurement],
    *,
    include_rejects: bool = False,
) -> set[int]:
    """Start with all rods/dots approved; rejects stay discarded unless requested."""
    approved: set[int] = set()
    for p in particles:
        if p.particle_class == ParticleClass.REJECT and not include_rejects:
            continue
        approved.add(p.particle_id)
    return approved


def particles_to_rows(
    particles: list[ParticleMeasurement],
    approved_ids: set[int],
) -> list[dict]:
    rows = []
    for p in particles:
        rows.append(
            {
                "particle_id": p.particle_id,
                "class": p.particle_class.value,
                "length_nm": round(p.length_nm, 2),
                "width_nm": round(p.width_nm, 2),
                "aspect_ratio": round(p.aspect_ratio, 2),
                "approved": p.particle_id in approved_ids,
                "centroid_x": round(p.centroid_x, 1),
                "centroid_y": round(p.centroid_y, 1),
            }
        )
    return rows


def filter_particles(
    particles: list[ParticleMeasurement],
    approved_ids: set[int],
) -> list[ParticleMeasurement]:
    return [p for p in particles if p.particle_id in approved_ids]


def summarize_approved(
    particles: list[ParticleMeasurement],
    approved_ids: set[int],
) -> dict[str, float | int]:
    kept = filter_particles(particles, approved_ids)
    rods = [p for p in kept if p.particle_class == ParticleClass.ROD]
    dots = [p for p in kept if p.particle_class == ParticleClass.DOT]
    out: dict[str, float | int] = {
        "approved_count": len(kept),
        "approved_rods": len(rods),
        "approved_dots": len(dots),
        "discarded_count": len(particles) - len(kept),
    }
    if rods:
        out["mean_rod_length_nm"] = round(sum(p.length_nm for p in rods) / len(rods), 2)
        out["mean_rod_width_nm"] = round(sum(p.width_nm for p in rods) / len(rods), 2)
    if dots:
        out["mean_dot_length_nm"] = round(sum(p.length_nm for p in dots) / len(dots), 2)
        out["mean_dot_width_nm"] = round(sum(p.width_nm for p in dots) / len(dots), 2)
    return out


def approved_csv_bytes(
    particles: list[ParticleMeasurement],
    approved_ids: set[int],
) -> bytes:
    kept = filter_particles(particles, approved_ids)
    rows = [asdict(p) for p in kept]
    for row in rows:
        row["particle_class"] = row["particle_class"].value
    df = pd.DataFrame(rows)
    return df.to_csv(index=False).encode("utf-8")


def toggle_particle(approved_ids: set[int], particle_id: int) -> set[int]:
    updated = set(approved_ids)
    if particle_id in updated:
        updated.remove(particle_id)
    else:
        updated.add(particle_id)
    return updated


def build_review_figure(
    image: np.ndarray,
    particles: list[ParticleMeasurement],
    approved_ids: set[int],
    labels: np.ndarray | None = None,
    *,
    show_rejects: bool = True,
) -> go.Figure:
    """
    Grayscale TEM image with contour outlines and clickable centroid markers.

    Green markers = approved, red = discarded. Click a marker to toggle.
    """
    if image.ndim == 2:
        rgb = np.stack([image, image, image], axis=-1)
    else:
        rgb = image
    if rgb.dtype != np.uint8:
        scaled = rgb.astype(np.float64)
        scaled = scaled - scaled.min()
        vmax = scaled.max() or 1.0
        rgb = (255 * scaled / vmax).astype(np.uint8)

    fig = go.Figure()
    fig.add_trace(go.Image(z=rgb))

    # Contours (visual only — not the click target).
    if labels is not None:
        label_to_particle = {p.particle_id: p for p in particles}
        for region in regionprops(labels):
            particle = label_to_particle.get(region.label)
            if particle is None:
                continue
            if particle.particle_class == ParticleClass.REJECT and not show_rejects:
                continue
            approved = particle.particle_id in approved_ids
            if particle.particle_class == ParticleClass.REJECT:
                color = "orange"
            elif approved:
                color = "#00cc66"
            else:
                color = "#cc3333"
            mask = labels == region.label
            for contour in find_contours(mask.astype(float), 0.5):
                fig.add_trace(
                    go.Scatter(
                        x=contour[:, 1],
                        y=contour[:, 0],
                        mode="lines",
                        line=dict(color=color, width=1.5),
                        hoverinfo="skip",
                        showlegend=False,
                    )
                )

    # Clickable centroids — customdata carries particle_id.
    reviewable = [
        p
        for p in particles
        if show_rejects or p.particle_class != ParticleClass.REJECT
    ]
    if reviewable:
        xs = [p.centroid_x for p in reviewable]
        ys = [p.centroid_y for p in reviewable]
        colors = []
        texts = []
        custom = []
        for p in reviewable:
            approved = p.particle_id in approved_ids
            if p.particle_class == ParticleClass.REJECT:
                colors.append("orange")
            elif approved:
                colors.append("#00cc66")
            else:
                colors.append("#cc3333")
            status = "approved" if approved else "discarded"
            texts.append(
                f"#{p.particle_id} {p.particle_class.value}<br>"
                f"{p.length_nm:.1f}×{p.width_nm:.1f} nm<br>"
                f"<b>{status}</b> — click to toggle"
            )
            custom.append(p.particle_id)

        fig.add_trace(
            go.Scatter(
                x=xs,
                y=ys,
                mode="markers+text",
                marker=dict(size=12, color=colors, line=dict(width=1, color="white")),
                text=[str(p.particle_id) for p in reviewable],
                textposition="top center",
                textfont=dict(size=9, color="white"),
                customdata=custom,
                hovertext=texts,
                hoverinfo="text",
                name="particles",
                showlegend=False,
            )
        )

    fig.update_layout(
        title="Click a numbered marker to approve (green) or discard (red)",
        margin=dict(l=10, r=10, t=40, b=10),
        xaxis=dict(visible=False, scaleanchor="y", scaleratio=1),
        yaxis=dict(visible=False, autorange="reversed"),
        height=700,
        dragmode=False,
        clickmode="event+select",
    )
    return fig


def particle_id_from_plotly_selection(selection) -> int | None:
    """Extract particle_id from a Streamlit plotly chart selection event."""
    if selection is None:
        return None
    points = None
    if hasattr(selection, "selection"):
        points = getattr(selection.selection, "points", None)
    elif isinstance(selection, dict):
        points = selection.get("selection", {}).get("points") or selection.get("points")
    if not points:
        return None

    point = points[-1]
    custom = point.get("customdata") if isinstance(point, dict) else None
    if custom is None and hasattr(point, "get"):
        custom = point.get("customdata")
    if isinstance(custom, (list, tuple)) and custom:
        custom = custom[0]
    if custom is None:
        return None
    try:
        return int(custom)
    except (TypeError, ValueError):
        return None


def _as_float_gray(image: np.ndarray) -> np.ndarray:
    if image.ndim == 3:
        gray = image.astype(np.float64).mean(axis=2)
    else:
        gray = image.astype(np.float64)
    if gray.max() > 1.5:
        gray = gray / 255.0
    return gray


def _measure_region(
    region,
    *,
    particle_id: int,
    nm_per_pixel: float,
    forced_class: ParticleClass,
) -> ParticleMeasurement:
    length_px = float(region.major_axis_length)
    width_px = float(region.minor_axis_length)
    if width_px <= 0:
        width_px = 1e-6
    return ParticleMeasurement(
        particle_id=particle_id,
        particle_class=forced_class,
        length_nm=length_px * nm_per_pixel,
        width_nm=width_px * nm_per_pixel,
        aspect_ratio=length_px / width_px,
        eccentricity=float(region.eccentricity),
        area_nm2=int(region.area) * (nm_per_pixel**2),
        centroid_y=float(region.centroid[0]),
        centroid_x=float(region.centroid[1]),
        length_px=length_px,
        width_px=width_px,
        area_px=int(region.area),
    )


def add_particle_at_click(
    image: np.ndarray,
    labels: np.ndarray,
    particles: list[ParticleMeasurement],
    *,
    click_y: float,
    click_x: float,
    nm_per_pixel: float,
    preferred_class: ParticleClass,
    min_area_px: int = 25,
) -> tuple[list[ParticleMeasurement], np.ndarray, str]:
    """
    Add or promote a particle under a user click.

    - If the click lands on an existing segmented blob, promote/keep it.
    - Otherwise grow a dark connected region from the click and measure it.
    """
    from skimage.filters import threshold_local, threshold_otsu
    from skimage.measure import label as sk_label
    from skimage.morphology import binary_closing, disk

    if labels is None:
        labels = np.zeros(image.shape[:2], dtype=np.int32)
    else:
        labels = np.asarray(labels).copy()

    h, w = labels.shape[:2]
    y = int(np.clip(round(click_y), 0, h - 1))
    x = int(np.clip(round(click_x), 0, w - 1))
    by_id = {p.particle_id: p for p in particles}

    existing_id = int(labels[y, x])
    if existing_id > 0:
        if existing_id in by_id:
            p = by_id[existing_id]
            if p.particle_class != preferred_class:
                updated = []
                for item in particles:
                    if item.particle_id == existing_id:
                        updated.append(
                            ParticleMeasurement(
                                particle_id=item.particle_id,
                                particle_class=preferred_class,
                                length_nm=item.length_nm,
                                width_nm=item.width_nm,
                                aspect_ratio=item.aspect_ratio,
                                eccentricity=item.eccentricity,
                                area_nm2=item.area_nm2,
                                centroid_y=item.centroid_y,
                                centroid_x=item.centroid_x,
                                length_px=item.length_px,
                                width_px=item.width_px,
                                area_px=item.area_px,
                            )
                        )
                    else:
                        updated.append(item)
                return (
                    updated,
                    labels,
                    f"Promoted particle #{existing_id} to {preferred_class.value} and added it.",
                )
            return particles, labels, f"Particle #{existing_id} was already detected — keep it approved."
        # Label exists but no measurement (orphan): measure it now.
        props = [r for r in regionprops(labels) if r.label == existing_id]
        if props and props[0].area >= min_area_px:
            measured = _measure_region(
                props[0],
                particle_id=existing_id,
                nm_per_pixel=nm_per_pixel,
                forced_class=preferred_class,
            )
            return (
                particles + [measured],
                labels,
                f"Added existing blob as particle #{existing_id}.",
            )

    gray = _as_float_gray(image)
    try:
        local_t = threshold_local(gray, block_size=51, offset=0.01)
        binary = gray < local_t
    except ValueError:
        binary = gray < threshold_otsu(gray)

    if not binary[y, x]:
        # Seed is bright; still try a small neighborhood for a dark pixel.
        yy, xx = np.ogrid[-4:5, -4:5]
        nearby = (y + yy, x + xx)
        dark_hits = []
        for dy in range(-4, 5):
            for dx in range(-4, 5):
                ny, nx = y + dy, x + dx
                if 0 <= ny < h and 0 <= nx < w and binary[ny, nx]:
                    dark_hits.append((ny, nx))
        if not dark_hits:
            binary = binary_closing(binary, disk(2))
            if not binary[y, x]:
                return (
                    particles,
                    labels,
                    "No dark particle found at that click. Aim at the center of a dark rod/dot.",
                )
        else:
            y, x = min(dark_hits, key=lambda p: (p[0] - y) ** 2 + (p[1] - x) ** 2)

    lab = sk_label(binary)
    lid = int(lab[y, x])
    if lid == 0:
        binary = binary_closing(binary, disk(2))
        lab = sk_label(binary)
        lid = int(lab[y, x])
    if lid == 0:
        return particles, labels, "Could not grow a particle from that click."

    mask = lab == lid
    area = int(mask.sum())
    if area < min_area_px:
        return particles, labels, f"Region too small ({area} px). Try a clearer click on a particle."
    if area > (h * w) * 0.15:
        return particles, labels, "Region too large — click may have leaked into background."

    new_id = int(labels.max()) + 1 if labels.max() > 0 else 1
    # Avoid id collision with particle_id namespace that might not match label max.
    if particles:
        new_id = max(new_id, max(p.particle_id for p in particles) + 1)
    labels[mask] = new_id
    props = [r for r in regionprops(labels) if r.label == new_id]
    measured = _measure_region(
        props[0],
        particle_id=new_id,
        nm_per_pixel=nm_per_pixel,
        forced_class=preferred_class,
    )
    return (
        particles + [measured],
        labels,
        f"Added particle #{new_id} ({measured.length_nm:.1f}×{measured.width_nm:.1f} nm).",
    )


def render_annotated_rgb(
    image: np.ndarray,
    particles: list[ParticleMeasurement],
    approved_ids: set[int],
    labels: np.ndarray | None = None,
    *,
    show_rejects: bool = True,
) -> np.ndarray:
    """RGB overlay image for click-to-add interaction."""
    gray = _as_float_gray(image)
    rgb = (np.stack([gray, gray, gray], axis=-1) * 255).astype(np.uint8)
    if labels is None:
        return rgb

    from PIL import Image as PILImage, ImageDraw

    pil = PILImage.fromarray(rgb)
    draw = ImageDraw.Draw(pil)
    by_id = {p.particle_id: p for p in particles}

    for region in regionprops(labels):
        particle = by_id.get(region.label)
        if particle is None:
            continue
        if particle.particle_class == ParticleClass.REJECT and not show_rejects:
            continue
        approved = particle.particle_id in approved_ids
        if particle.particle_class == ParticleClass.REJECT:
            color = (255, 140, 0)
        elif approved:
            color = (0, 200, 100)
        else:
            color = (200, 50, 50)
        mask = labels == region.label
        for contour in find_contours(mask.astype(float), 0.5):
            pts = [(float(c[1]), float(c[0])) for c in contour]
            if len(pts) >= 2:
                draw.line(pts, fill=color, width=2)
        cy, cx = region.centroid
        draw.ellipse((cx - 4, cy - 4, cx + 4, cy + 4), fill=color)
    return np.asarray(pil)

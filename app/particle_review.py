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

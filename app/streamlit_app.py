"""
TEM Particle Analyzer — public web UI with human review
=======================================================

Run locally::

    streamlit run app/streamlit_app.py

Or deploy on Streamlit Community Cloud (see docs/DEPLOY_WEBSITE.md).

After analysis, click particle markers to keep (green) or discard (red)
before downloading the final CSV.
"""

from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

import pandas as pd
import numpy as np
import streamlit as st

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))
# Package lives under src/ (needed on Streamlit Cloud without `pip install -e .`)
if str(_REPO / "src") not in sys.path:
    sys.path.insert(0, str(_REPO / "src"))
if str(_REPO / "app") not in sys.path:
    sys.path.insert(0, str(_REPO / "app"))

# Bump when Cloud keeps stale tem_rods modules after a deploy (forces reload).
_APP_BUILD = "2026-07-16-robust-image-load-1"
for _mod in list(sys.modules):
    if _mod == "tem_rods" or _mod.startswith("tem_rods."):
        del sys.modules[_mod]

from particle_review import (  # noqa: E402
    add_particle_at_click,
    approved_csv_bytes,
    build_review_figure,
    default_approved_ids,
    filter_approved_by_length,
    particle_id_from_plotly_selection,
    particles_to_rows,
    render_annotated_rgb,
    summarize_approved,
    toggle_particle,
)
from tem_rods.measure import measure_from_region  # noqa: E402
from tem_rods.models import AnalysisMode, ParticleClass, ParticleMeasurement  # noqa: E402
from tem_rods.pipeline import analyze_image  # noqa: E402
from tem_rods.presets import PRESETS, get_preset  # noqa: E402
from tem_rods.preprocess import preprocess  # noqa: E402
from tem_rods.scale_bar import ScaleBarDetection, detect_scale_bar  # noqa: E402
from tem_rods.segment import segment_particles_from_config  # noqa: E402
from tem_rods.shape_match import (  # noqa: E402
    find_similar_in_labels,
    render_trace_overlay,
    stroke_image_to_mask,
)
from skimage.measure import regionprops  # noqa: E402
from PIL import Image as PILImage  # noqa: E402

st.set_page_config(
    page_title="Python Based Geometric Analysis for TEM Images",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <style>
    html, body, [class*="css"], .stApp, .stMarkdown, .stText,
    .stButton > button, .stSelectbox, .stRadio, .stNumberInput,
    .stCaption, label, p, h1, h2, h3, h4, h5, h6 {
        font-family: "Times New Roman", Times, serif !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("Python Based Geometric Analysis for TEM Images")
st.caption(
    "Choose a workspace: auto-detect particles, or trace one outline to find similar shapes. "
    "Enter the scale bar (nm), approve outlines, download results. "
    "Reference: Enright et al. 2018."
)


def _init_session() -> None:
    defaults = {
        "particles": None,
        "approved_ids": set(),
        "image": None,
        "labels": None,
        "overlay_bytes": None,
        "stem": "analysis",
        "warnings": [],
        "last_clicked_id": None,
        "analysis_done": False,
        "calibration_note": None,
        "nm_per_pixel": None,
        "analysis_mode": AnalysisMode.RODS.value,
        "add_message": None,
        "last_add_click": None,
        "show_feret_histogram": False,
        "trace_points": [],
        "trace_last_click": None,
        "shape_match_ids": set(),
        "shape_match_labels": None,
        "shape_match_scores": {},
        "shape_match_image": None,
        "shape_match_message": None,
        "shape_match_particles": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def _dicts_to_particles(rows: list[dict]) -> list[ParticleMeasurement]:
    out: list[ParticleMeasurement] = []
    for row in rows:
        out.append(
            ParticleMeasurement(
                particle_id=int(row["particle_id"]),
                particle_class=ParticleClass(row["particle_class"]),
                length_nm=float(row["length_nm"]),
                width_nm=float(row["width_nm"]),
                aspect_ratio=float(row["aspect_ratio"]),
                eccentricity=float(row["eccentricity"]),
                area_nm2=float(row["area_nm2"]),
                centroid_y=float(row["centroid_y"]),
                centroid_x=float(row["centroid_x"]),
                length_px=float(row["length_px"]),
                width_px=float(row["width_px"]),
                area_px=int(row["area_px"]),
                feret_max_nm=float(row.get("feret_max_nm", 0.0) or 0.0),
                feret_min_nm=float(row.get("feret_min_nm", 0.0) or 0.0),
                circularity=float(row.get("circularity", 0.0) or 0.0),
                equiv_diameter_nm=float(row.get("equiv_diameter_nm", 0.0) or 0.0),
            )
        )
    return out


def _particles_to_dicts(particles: list[ParticleMeasurement]) -> list[dict]:
    """Serialize particles for session_state (tolerant of older ParticleMeasurement)."""
    rows = []
    for p in particles:
        rows.append(
            {
                "particle_id": p.particle_id,
                "particle_class": p.particle_class.value,
                "length_nm": p.length_nm,
                "width_nm": p.width_nm,
                "aspect_ratio": p.aspect_ratio,
                "eccentricity": p.eccentricity,
                "area_nm2": p.area_nm2,
                "centroid_y": p.centroid_y,
                "centroid_x": p.centroid_x,
                "length_px": p.length_px,
                "width_px": p.width_px,
                "area_px": p.area_px,
                "feret_max_nm": float(getattr(p, "feret_max_nm", 0.0) or 0.0),
                "feret_min_nm": float(getattr(p, "feret_min_nm", 0.0) or 0.0),
                "circularity": float(getattr(p, "circularity", 0.0) or 0.0),
                "equiv_diameter_nm": float(getattr(p, "equiv_diameter_nm", 0.0) or 0.0),
            }
        )
    return rows


def _ensure_session_dir() -> Path:
    if "_session_folder" not in st.session_state:
        import uuid

        st.session_state._session_folder = uuid.uuid4().hex[:12]
    session_dir = _REPO / "outputs" / "streamlit_sessions" / st.session_state._session_folder
    session_dir.mkdir(parents=True, exist_ok=True)
    return session_dir


def _store_analysis_result(
    result,
    stem: str,
    *,
    calibration_note: str | None = None,
    analysis_mode: AnalysisMode | None = None,
) -> None:
    st.session_state.particles = _particles_to_dicts(result.particles)
    st.session_state.approved_ids = default_approved_ids(result.particles)
    st.session_state.image = result.image
    st.session_state.labels = result.labels
    st.session_state.stem = stem
    st.session_state.warnings = list(result.warnings)
    st.session_state.analysis_done = True
    st.session_state.last_clicked_id = None
    st.session_state.plot_nonce = st.session_state.get("plot_nonce", 0) + 1
    st.session_state.calibration_note = calibration_note
    st.session_state.nm_per_pixel = float(result.nm_per_pixel)
    st.session_state.analysis_mode = (
        analysis_mode.value if analysis_mode is not None else result.analysis_mode.value
    )
    st.session_state.add_message = None
    st.session_state.last_add_click = None
    if result.overlay_path and result.overlay_path.exists():
        st.session_state.overlay_bytes = result.overlay_path.read_bytes()
    else:
        st.session_state.overlay_bytes = None


def _resolve_calibration(
    image_path: Path,
    *,
    scale_bar_nm: float,
    manual_scale_bar_px: float | None,
) -> tuple[float, ScaleBarDetection | None, str]:
    """
    Return (nm_per_pixel, scale_bar_detection_or_None, user-facing note).

    Prefers automatic pixel measurement; falls back to manual px override.
    """
    if manual_scale_bar_px is not None and manual_scale_bar_px > 0:
        nm_per_pixel = scale_bar_nm / manual_scale_bar_px
        note = (
            f"Manual scale bar: {scale_bar_nm:g} nm / {manual_scale_bar_px:.1f} px "
            f"= {nm_per_pixel:.4f} nm/px"
        )
        return nm_per_pixel, None, note

    detection = detect_scale_bar(image_path, scale_bar_nm=scale_bar_nm)
    note = (
        f"Auto-detected {detection.polarity} scale bar: "
        f"{detection.bar_nm:g} nm / {detection.bar_pixels:.1f} px "
        f"= {detection.nm_per_pixel:.4f} nm/px"
    )
    return detection.nm_per_pixel, detection, note


def _run_analysis(
    image_path: Path,
    *,
    scale_bar_nm: float,
    preset_name: str,
    analysis_mode: AnalysisMode,
    show_rejected: bool,
    manual_scale_bar_px: float | None = None,
) -> None:
    session_dir = _ensure_session_dir()
    out_dir = session_dir / "outputs"
    out_dir.mkdir(exist_ok=True)
    preset = get_preset(preset_name)
    config = replace(
        preset.config,
        analysis_mode=analysis_mode,
        show_rejected_on_overlay=show_rejected,
        write_segmentation_debug=True,
    )
    with st.spinner("Detecting scale bar and analyzing..."):
        nm_per_pixel, scale_bar, calib_note = _resolve_calibration(
            image_path,
            scale_bar_nm=scale_bar_nm,
            manual_scale_bar_px=manual_scale_bar_px,
        )
        result = analyze_image(
            image_path,
            nm_per_pixel,
            output_dir=out_dir,
            config=config,
            scale_bar=scale_bar,
            scale_bar_nm_hint=scale_bar_nm,
        )
    _store_analysis_result(
        result,
        image_path.stem,
        calibration_note=calib_note,
        analysis_mode=analysis_mode,
    )


def _default_preset_for_mode(mode: AnalysisMode) -> str:
    if mode == AnalysisMode.DOTS:
        return "dots_only"
    if mode == AnalysisMode.BOTH:
        return "screenshot"
    if "dense_rods_50nm" in PRESETS:
        return "dense_rods_50nm"
    return "enright_rods"


def _as_float_gray(image: np.ndarray) -> np.ndarray:
    if image.ndim == 3:
        gray = image.astype(np.float64).mean(axis=2)
    else:
        gray = image.astype(np.float64)
    if gray.max() > 1.5:
        gray = gray / 255.0
    return gray


def _patch_drawable_canvas_for_new_streamlit() -> None:
    """
    streamlit-drawable-canvas 0.9.3 calls the old
    ``streamlit.elements.image.image_to_url(image, width, clamp, channels, format, id)``
    API. Streamlit ≥1.41 moved/changed that function, so install a shim that
    matches the canvas calling convention.
    """
    import base64
    from io import BytesIO

    import streamlit.elements.image as st_image

    def image_to_url(image, width, clamp, channels, output_format, image_id=None, *args, **kwargs):  # noqa: ARG001
        if not isinstance(image, PILImage.Image):
            arr = np.asarray(image)
            image = PILImage.fromarray(arr)

        # Old Streamlit API: ``width`` was an int pixel cap (or -1 for native).
        if isinstance(width, (int, float)) and int(width) > 0 and image.width > int(width):
            new_w = int(width)
            new_h = max(1, int(round(image.height * (new_w / image.width))))
            image = image.resize((new_w, new_h), getattr(PILImage, "Resampling", PILImage).BILINEAR)

        channel = (channels or "RGB").upper()
        if channel == "RGB" and image.mode != "RGB":
            image = image.convert("RGB")
        elif channel == "RGBA" and image.mode != "RGBA":
            image = image.convert("RGBA")

        fmt = str(output_format or "PNG").upper()
        if fmt in ("AUTO", "", "None"):
            fmt = "PNG"
        if fmt not in ("PNG", "JPEG", "JPG"):
            fmt = "PNG"
        if fmt in ("JPEG", "JPG") and image.mode not in ("RGB", "L"):
            image = image.convert("RGB")
            fmt = "JPEG"

        buf = BytesIO()
        image.save(buf, format="JPEG" if fmt in ("JPEG", "JPG") else "PNG")
        mime = "image/jpeg" if fmt in ("JPEG", "JPG") else "image/png"
        b64 = base64.b64encode(buf.getvalue()).decode("ascii")
        return f"data:{mime};base64,{b64}"

    # Always override — even if Streamlit exposes a new image_to_url, its
    # signature no longer matches what drawable-canvas passes.
    st_image.image_to_url = image_to_url  # type: ignore[attr-defined]


def _render_trace_and_match_tab() -> None:
    """Freehand-trace one particle outline, then find similarly shaped particles."""
    st.subheader("Trace one particle → find similar shapes")
    st.caption(
        "Hold the mouse button and **draw a closed loop** around one particle. "
        "Then click **Find similar particles**. "
        "The app segments the image and keeps blobs whose shape matches your tracing."
    )

    scale_nm = st.number_input(
        "Scale bar length (nm)",
        min_value=1.0,
        value=float(st.session_state.get("trace_scale_nm", 50.0)),
        step=1.0,
        key="trace_scale_nm",
    )
    preset_names = sorted(PRESETS.keys())
    default_preset = (
        "dense_rods_50nm" if "dense_rods_50nm" in PRESETS else "enright_rods"
    )
    preset_name = st.selectbox(
        "Segmentation preset",
        preset_names,
        index=preset_names.index(default_preset)
        if default_preset in preset_names
        else 0,
        key="trace_preset",
    )
    max_score = st.slider(
        "Similarity tolerance (lower = stricter)",
        min_value=0.10,
        max_value=0.80,
        value=0.35,
        step=0.05,
        help="Maximum shape-distance score to count as a match.",
        key="trace_max_score",
    )
    stroke_width = st.slider(
        "Brush thickness",
        min_value=1,
        max_value=8,
        value=3,
        key="trace_brush",
    )

    uploaded = st.file_uploader(
        "Upload TEM image for shape matching",
        type=["png", "jpg", "jpeg", "tif", "tiff"],
        key="trace_uploader",
    )

    image = None
    image_path: Path | None = None
    if uploaded is not None:
        from tem_rods.io import load_grayscale_bytes, save_grayscale_png

        session_dir = _ensure_session_dir()
        raw = uploaded.getvalue()
        try:
            image = load_grayscale_bytes(raw, name=uploaded.name)
        except ValueError as exc:
            st.error(str(exc))
            return
        # Re-encode a clean PNG so scale-bar helpers never see a corrupt upload.
        stem = Path(uploaded.name).stem or "trace_upload"
        image_path = session_dir / f"trace_{stem}.png"
        save_grayscale_png(image, image_path)
        st.session_state.shape_match_image = image
        st.session_state.shape_match_image_path = str(image_path)
        if st.session_state.get("trace_upload_name") != uploaded.name:
            st.session_state.trace_upload_name = uploaded.name
            st.session_state.shape_match_ids = set()
            st.session_state.shape_match_labels = None
            st.session_state.shape_match_scores = {}
            st.session_state.shape_match_particles = None
            st.session_state.shape_match_message = None
            st.session_state.trace_canvas_nonce = (
                st.session_state.get("trace_canvas_nonce", 0) + 1
            )
    elif st.session_state.get("shape_match_image") is not None:
        image = st.session_state.shape_match_image
        if st.session_state.get("shape_match_image_path"):
            image_path = Path(st.session_state.shape_match_image_path)
    elif st.session_state.get("image") is not None:
        image = st.session_state.image
        st.session_state.shape_match_image = image
        st.info("Using the image from Auto detect & review.")

    if image is None:
        st.warning("Upload a TEM image (or run Auto detect first) to start tracing.")
        return

    try:
        _patch_drawable_canvas_for_new_streamlit()
        from streamlit_drawable_canvas import st_canvas
    except ImportError:
        st.error(
            "Missing package `streamlit-drawable-canvas`. "
            "Install with: pip install streamlit-drawable-canvas"
        )
        return

    gray = _as_float_gray(image)
    h, w = gray.shape[:2]
    rgb = (np.stack([gray, gray, gray], axis=-1) * 255).astype(np.uint8)

    # Draw matches on the background so the canvas stays freehand-only.
    matched_ids: set[int] = set(st.session_state.shape_match_ids or set())
    match_labels = st.session_state.shape_match_labels
    if match_labels is not None and matched_ids:
        bg_arr = render_trace_overlay(
            gray,
            [],
            match_labels=match_labels,
            matched_ids=matched_ids,
            template_mask=None,
        )
    else:
        bg_arr = rgb

    bg_pil = PILImage.fromarray(bg_arr)
    max_canvas_w = 900
    display_scale = min(1.0, max_canvas_w / float(w))
    canvas_w = max(1, int(round(w * display_scale)))
    canvas_h = max(1, int(round(h * display_scale)))
    bg_display = bg_pil.resize((canvas_w, canvas_h), getattr(PILImage, "Resampling", PILImage).BILINEAR)

    c1, c2 = st.columns([1, 3])
    with c1:
        clear = st.button("Clear drawing")
        find = st.button("Find similar particles", type="primary")
    with c2:
        st.caption(
            "Drawing mode: **freehand**. Close the loop around the particle as best you can."
        )

    if clear:
        st.session_state.shape_match_ids = set()
        st.session_state.shape_match_labels = None
        st.session_state.shape_match_scores = {}
        st.session_state.shape_match_particles = None
        st.session_state.shape_match_message = None
        st.session_state.trace_canvas_nonce = (
            st.session_state.get("trace_canvas_nonce", 0) + 1
        )
        st.rerun()

    canvas = st_canvas(
        fill_color="rgba(0, 0, 0, 0)",
        stroke_width=int(stroke_width),
        stroke_color="#ff3333",
        background_image=bg_display,
        update_streamlit=True,
        height=canvas_h,
        width=canvas_w,
        drawing_mode="freedraw",
        key=f"trace_canvas_{st.session_state.get('trace_canvas_nonce', 0)}",
        display_toolbar=True,
    )

    if find:
        stroke = None if canvas is None else canvas.image_data
        try:
            template_mask = stroke_image_to_mask(stroke, (h, w))
        except ValueError as exc:
            st.error(str(exc))
            return

        preset = get_preset(preset_name)
        cfg = replace(preset.config)
        processed = preprocess(
            gray,
            gaussian_sigma=cfg.gaussian_sigma,
            crop_margins=cfg.crop_margins,
            use_clahe=cfg.use_clahe,
        )
        labels = segment_particles_from_config(processed, cfg)
        _feat, matches = find_similar_in_labels(
            labels,
            template_mask,
            max_score=float(max_score),
        )
        matched_ids = {m.label_id for m in matches}
        scores = {m.label_id: m.score for m in matches}

        nm_pp = float(st.session_state.nm_per_pixel or 0.0)
        calib_note = None
        if nm_pp <= 0 and image_path is not None:
            try:
                sb = detect_scale_bar(image_path, scale_bar_nm=scale_nm)
                nm_pp = float(sb.nm_per_pixel)
                calib_note = (
                    f"Scale bar auto: {scale_nm:g} nm / {sb.bar_pixels:.1f} px "
                    f"→ {nm_pp:.4f} nm/px"
                )
            except Exception:
                nm_pp = 1.0
                calib_note = (
                    "Could not auto-read scale bar — sizes use a 1 nm/px placeholder. "
                    "Run Auto detect first for accurate nm calibration."
                )
        if nm_pp <= 0:
            nm_pp = 1.0

        preferred = (
            ParticleClass.DOT
            if "dot" in preset_name.lower()
            else ParticleClass.ROD
        )
        st.session_state.shape_match_preferred_class = preferred.value
        particles: list[ParticleMeasurement] = []
        for region in regionprops(labels):
            if region.label not in matched_ids:
                continue
            particles.append(
                measure_from_region(
                    region,
                    particle_id=int(region.label),
                    nm_per_pixel=nm_pp,
                    particle_class=preferred,
                )
            )

        st.session_state.shape_match_labels = labels
        st.session_state.shape_match_ids = matched_ids
        st.session_state.shape_match_scores = scores
        st.session_state.shape_match_particles = _particles_to_dicts(particles)
        st.session_state.nm_per_pixel = nm_pp
        st.session_state.shape_match_template_mask = template_mask
        st.session_state.shape_match_message = (
            f"Found {len(matches)} similar particle(s) "
            f"(tolerance ≤ {max_score:.2f})."
            + (f" {calib_note}" if calib_note else "")
        )
        st.rerun()

    if st.session_state.shape_match_message:
        st.success(st.session_state.shape_match_message)

    particles_rows = st.session_state.shape_match_particles
    if particles_rows:
        particles = _dicts_to_particles(particles_rows)
        scores = st.session_state.shape_match_scores or {}
        rows = []
        for p in particles:
            rows.append(
                {
                    "particle_id": p.particle_id,
                    "class": p.particle_class.value,
                    "similarity_score": round(float(scores.get(p.particle_id, 0.0)), 3),
                    "length_nm": round(p.length_nm, 2),
                    "width_nm": round(p.width_nm, 2),
                    "feret_max_nm": round(
                        float(getattr(p, "feret_max_nm", 0.0) or 0.0), 2
                    ),
                    "circularity": round(
                        float(getattr(p, "circularity", 0.0) or 0.0), 3
                    ),
                    "aspect_ratio": round(p.aspect_ratio, 2),
                }
            )
        table = pd.DataFrame(rows).sort_values("similarity_score")
        st.dataframe(table, use_container_width=True, hide_index=True)
        csv_bytes = table.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download matched particles CSV",
            data=csv_bytes,
            file_name="shape_match_particles.csv",
            mime="text/csv",
        )

        if st.button("Send matches to Auto detect review"):
            preferred_val = st.session_state.get(
                "shape_match_preferred_class", AnalysisMode.RODS.value
            )
            try:
                preferred_cls = ParticleClass(preferred_val)
            except ValueError:
                preferred_cls = ParticleClass.ROD
            st.session_state.particles = _particles_to_dicts(particles)
            st.session_state.labels = st.session_state.shape_match_labels
            st.session_state.image = gray
            st.session_state.approved_ids = {p.particle_id for p in particles}
            st.session_state.analysis_done = True
            st.session_state.stem = "shape_match"
            st.session_state.analysis_mode = (
                AnalysisMode.DOTS.value
                if preferred_cls == ParticleClass.DOT
                else AnalysisMode.RODS.value
            )
            st.session_state.add_message = (
                f"Imported {len(particles)} shape-matched particles into review."
            )
            st.info("Switch to **Auto detect & review** to edit keep/discard.")
            st.rerun()


_init_session()

SAMPLE_50NM = _REPO / "data" / "curated" / "user_rods_50nm_jul13.png"

workspace = st.radio(
    "Workspace",
    options=["Auto detect & review", "Trace & find similar"],
    horizontal=True,
    help="Auto detect runs the full pipeline. Trace lets you outline one particle and find look-alikes.",
)

if workspace == "Trace & find similar":
    _render_trace_and_match_tab()
    st.markdown("---")
    st.markdown(
        "Draw a closed freehand loop around one particle · green = similar matches  \n"
        "Repo: [github.com/fayh601-glitch/TEM-analysis](https://github.com/fayh601-glitch/TEM-analysis)"
    )
    st.stop()

# --- Primary controls (always visible before analyze) ---
st.subheader("1. Particle morphology")
mode_label = st.radio(
    "Particle type",
    options=["Rods", "Dots", "Both"],
    index=0,
    horizontal=True,
    help="Rods-only ignores round fragments. Dots-only ignores elongated shapes.",
)
mode_map = {
    "Rods": AnalysisMode.RODS,
    "Dots": AnalysisMode.DOTS,
    "Both": AnalysisMode.BOTH,
}
analysis_mode = mode_map[mode_label]

st.subheader("2. Scale bar")
st.caption(
    "Enter only the printed number (e.g. **50** if the image says “50 nm”). "
    "The app finds the white or black scale-bar line and measures its length in pixels."
)
scale_bar_nm = st.number_input(
    "Scale bar length (nm)",
    min_value=1.0,
    value=50.0,
    step=1.0,
    help="The number printed next to the scale bar on the TEM image.",
)

with st.expander("Advanced settings", expanded=False):
    preset_names = sorted(PRESETS.keys())
    default_preset = _default_preset_for_mode(analysis_mode)
    preset_name = st.selectbox(
        "Image preset",
        preset_names,
        index=preset_names.index(default_preset) if default_preset in preset_names else 0,
        help="Tuned thresholds for dense rods, paper screenshots, dots, etc.",
    )
    show_rejected = st.checkbox("Show rejected particles (orange)", value=True)
    override_px = st.checkbox(
        "Manually override scale-bar pixels (only if auto-detect fails)",
        value=False,
    )
    manual_scale_bar_px: float | None = None
    if override_px:
        manual_scale_bar_px = st.number_input(
            "Scale bar length (pixels)",
            min_value=1.0,
            value=98.0,
            step=1.0,
        )
    st.caption("Green = keep · red = discard · click a numbered marker to toggle.")

st.subheader("3. Image")
if SAMPLE_50NM.exists() and not st.session_state.analysis_done:
    st.info("Demo image available: dense nanorods with a 50 nm scale bar.")
    if st.button("Analyze demo (50 nm rods)", type="secondary"):
        try:
            _run_analysis(
                SAMPLE_50NM,
                scale_bar_nm=scale_bar_nm,
                preset_name="dense_rods_50nm" if "dense_rods_50nm" in PRESETS else preset_name,
                analysis_mode=analysis_mode,
                show_rejected=show_rejected,
                manual_scale_bar_px=manual_scale_bar_px,
            )
            st.rerun()
        except ValueError as exc:
            st.error(f"Scale bar / analysis failed: {exc}")

uploaded = st.file_uploader(
    "Upload TEM image (PNG, JPG, TIF)",
    type=["png", "jpg", "jpeg", "tif", "tiff"],
)

if uploaded is not None and not st.session_state.analysis_done:
    st.image(uploaded, caption="Uploaded image", use_column_width="always")

analyze_clicked = st.button("Analyze image", type="primary", disabled=uploaded is None)

if analyze_clicked:
    assert uploaded is not None
    from tem_rods.io import load_grayscale_bytes, save_grayscale_png

    session_dir = _ensure_session_dir()
    raw = uploaded.getvalue()
    try:
        gray_preview = load_grayscale_bytes(raw, name=uploaded.name)
    except ValueError as exc:
        st.error(str(exc))
    else:
        stem = Path(uploaded.name).stem or "upload"
        image_path = session_dir / f"{stem}.png"
        save_grayscale_png(gray_preview, image_path)
        try:
            _run_analysis(
                image_path,
                scale_bar_nm=scale_bar_nm,
                preset_name=preset_name,
                analysis_mode=analysis_mode,
                show_rejected=show_rejected,
                manual_scale_bar_px=manual_scale_bar_px,
            )
            st.rerun()
        except ValueError as exc:
            st.error(
                f"Could not measure the scale bar automatically: {exc}\n\n"
                "Open **Advanced settings** and enable the pixel override, "
                "or check that the scale bar is visible near the bottom of the image."
            )

# --- Review panel ---
if st.session_state.analysis_done and st.session_state.particles:
    st.markdown("---")
    st.subheader("4. Review detections")
    particles = _dicts_to_particles(st.session_state.particles)
    approved_ids: set[int] = set(st.session_state.approved_ids)
    stats = summarize_approved(particles, approved_ids)

    st.success(
        f"Approved {stats['approved_count']} particles "
        f"({stats['approved_rods']} rods, {stats['approved_dots']} dots) · "
        f"discarded {stats['discarded_count']}"
    )
    if st.session_state.calibration_note:
        st.info(st.session_state.calibration_note)

    st.markdown("##### Length filter (discard outliers)")
    st.caption(
        "After you see auto-detected lengths, set an allowed range. "
        "Particles shorter or longer than this range are discarded (shown red)."
    )
    lengths = [p.length_nm for p in particles if p.particle_class != ParticleClass.REJECT]
    if lengths:
        st.caption(
            f"Detected length range (non-reject): "
            f"{min(lengths):.1f}–{max(lengths):.1f} nm · "
            f"median {float(np.median(lengths)):.1f} nm"
        )
    lf1, lf2, lf3 = st.columns([1, 1, 1])
    with lf1:
        min_len = st.number_input(
            "Min length (nm)",
            min_value=0.0,
            value=float(st.session_state.get("filter_min_length_nm", 50.0)),
            step=1.0,
            help="Discard particles shorter than this.",
        )
    with lf2:
        max_len = st.number_input(
            "Max length (nm)",
            min_value=0.0,
            value=float(st.session_state.get("filter_max_length_nm", 200.0)),
            step=1.0,
            help="Discard particles longer than this.",
        )
    with lf3:
        st.write("")
        st.write("")
        apply_len = st.button("Apply length filter", type="primary")
    if apply_len:
        if max_len < min_len:
            st.error("Max length must be ≥ min length.")
        else:
            st.session_state.filter_min_length_nm = min_len
            st.session_state.filter_max_length_nm = max_len
            # Start from all rods/dots, then apply range (so re-applying is predictable).
            base = default_approved_ids(particles)
            filtered, n_out = filter_approved_by_length(
                particles,
                base,
                min_length_nm=min_len,
                max_length_nm=max_len,
            )
            st.session_state.approved_ids = filtered
            st.session_state.add_message = (
                f"Length filter applied ({min_len:g}–{max_len:g} nm): "
                f"kept {len(filtered)}, discarded {n_out} outliers."
            )
            st.session_state.plot_nonce = st.session_state.get("plot_nonce", 0) + 1
            st.rerun()

    click_mode = st.radio(
        "Click action",
        options=[
            "Toggle keep / discard (click numbered marker)",
            "Add missed particle (click dark particle on image)",
        ],
        index=0,
        horizontal=True,
        help=(
            "Use Add mode when auto-detect missed rods/dots. "
            "Click near the center of a dark particle to measure and keep it."
        ),
    )

    preferred_class = (
        ParticleClass.DOT
        if st.session_state.analysis_mode == AnalysisMode.DOTS.value
        else ParticleClass.ROD
    )
    if st.session_state.analysis_mode == AnalysisMode.BOTH.value:
        preferred_class = (
            ParticleClass.DOT if mode_label == "Dots" else ParticleClass.ROD
        )

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        if st.button("Approve all rods/dots"):
            st.session_state.approved_ids = default_approved_ids(particles)
            st.rerun()
    with c2:
        if st.button("Discard all"):
            st.session_state.approved_ids = set()
            st.rerun()
    with c3:
        if st.button("Approve all including rejects"):
            st.session_state.approved_ids = default_approved_ids(
                particles, include_rejects=True
            )
            st.rerun()
    with c4:
        if st.button("Clear / start over"):
            for key in (
                "particles",
                "approved_ids",
                "image",
                "labels",
                "overlay_bytes",
                "warnings",
                "last_clicked_id",
                "calibration_note",
                "nm_per_pixel",
                "add_message",
                "last_add_click",
            ):
                st.session_state[key] = None if key != "approved_ids" else set()
            st.session_state.analysis_done = False
            st.rerun()

    m1, m2, m3, m4 = st.columns(4)
    with m1:
        if "mean_rod_length_nm" in stats:
            st.metric(
                "Mean rod length (ellipse, nm)",
                f"{stats['mean_rod_length_nm']:.1f}",
                help="Fitted ellipse major axis — matches overlay drawing.",
            )
        elif "mean_dot_diameter_nm" in stats:
            st.metric(
                "Mean dot diameter (nm)",
                f"{stats['mean_dot_diameter_nm']:.1f}",
                help="Area-equivalent diameter 2√(A/π).",
            )
    with m2:
        if "mean_rod_width_nm" in stats:
            st.metric(
                "Mean rod width (ellipse, nm)",
                f"{stats['mean_rod_width_nm']:.1f}",
                help="Fitted ellipse minor axis.",
            )
        elif "mean_dot_feret_max_nm" in stats:
            st.metric(
                "Mean dot Feret max (nm)",
                f"{stats['mean_dot_feret_max_nm']:.1f}",
                help="Maximum caliper diameter.",
            )
    with m3:
        if "mean_rod_feret_max_nm" in stats:
            st.metric(
                "Mean rod Feret max (nm)",
                f"{stats['mean_rod_feret_max_nm']:.1f}",
                help="Maximum caliper diameter (Aviles & Lear size metric).",
            )
        elif "mean_dot_circularity" in stats:
            st.metric(
                "Mean dot circularity",
                f"{stats['mean_dot_circularity']:.3f}",
                help="4π·area/perimeter² (1 = perfect circle).",
            )
    with m4:
        if "lognormal_rod_feret_max_nm" in stats:
            st.metric(
                "Rod Feret max (log-normal)",
                f"{stats['lognormal_rod_feret_max_nm']:.1f}",
                delta=f"±{stats['lognormal_rod_feret_max_se_nm']:.1f} SE",
                delta_color="off",
                help="Geometric mean from log-normal fit; ± is SE of the mean, not sample SD.",
            )
        elif "lognormal_dot_diameter_nm" in stats:
            st.metric(
                "Dot diameter (log-normal)",
                f"{stats['lognormal_dot_diameter_nm']:.1f}",
                delta=f"±{stats['lognormal_dot_diameter_se_nm']:.1f} SE",
                delta_color="off",
                help="Geometric mean from log-normal fit; ± is SE of the mean, not sample SD.",
            )
        elif "mean_rod_feret_min_nm" in stats:
            st.metric(
                "Mean rod Feret min (nm)",
                f"{stats['mean_rod_feret_min_nm']:.1f}",
                help="Minimum caliper diameter (rod width proxy).",
            )

    if stats.get("sample_size_note"):
        st.info(stats["sample_size_note"])

    # Size / morphology histograms (opt-in — Feret chart is off by default)
    approved_particles = [p for p in particles if p.particle_id in approved_ids]
    rods_a = [p for p in approved_particles if p.particle_class == ParticleClass.ROD]
    dots_a = [p for p in approved_particles if p.particle_class == ParticleClass.DOT]
    if rods_a or dots_a:
        show_feret_hist = st.checkbox(
            "Show Feret / size histogram",
            value=bool(st.session_state.get("show_feret_histogram", False)),
            help="Optional Aviles & Lear–style size distribution chart. Off by default.",
            key="show_feret_histogram",
        )
        if show_feret_hist:
            st.subheader("Approved size distributions")
            st.caption(
                "Bars touch (continuous size). Log-normal geometric mean is preferred over "
                "arithmetic mean for nanoparticle sizes (Aviles & Lear)."
            )
            import plotly.express as px

            hist_cols = st.columns(2 if (rods_a and dots_a) else 1)
            col_i = 0
            if rods_a:
                with hist_cols[col_i]:
                    df_r = pd.DataFrame(
                        {
                            "Feret max (nm)": [
                                float(getattr(p, "feret_max_nm", 0.0) or 0.0) for p in rods_a
                            ],
                        }
                    )
                    fig_r = px.histogram(
                        df_r,
                        x="Feret max (nm)",
                        nbins=min(30, max(8, len(rods_a) // 2)),
                        title=f"Rods — Feret max (n={len(rods_a)})",
                    )
                    fig_r.update_layout(bargap=0, height=280, margin=dict(t=40, b=20))
                    st.plotly_chart(fig_r, use_container_width=True)
                    if "mean_rod_circularity" in stats:
                        st.caption(
                            f"Mean circularity: {stats['mean_rod_circularity']:.3f}"
                        )
                col_i += 1
            if dots_a:
                with hist_cols[col_i if rods_a and dots_a else 0]:
                    df_d = pd.DataFrame(
                        {
                            "Diameter (nm)": [
                                float(getattr(p, "equiv_diameter_nm", 0.0) or 0.0)
                                if float(getattr(p, "equiv_diameter_nm", 0.0) or 0.0) > 0
                                else 0.5 * (p.length_nm + p.width_nm)
                                for p in dots_a
                            ]
                        }
                    )
                    fig_d = px.histogram(
                        df_d,
                        x="Diameter (nm)",
                        nbins=min(30, max(8, len(dots_a) // 2)),
                        title=f"Dots — equiv. diameter (n={len(dots_a)})",
                    )
                    fig_d.update_layout(bargap=0, height=280, margin=dict(t=40, b=20))
                    st.plotly_chart(fig_d, use_container_width=True)
                    if "mean_dot_circularity" in stats:
                        st.caption(
                            f"Mean circularity: {stats['mean_dot_circularity']:.3f}"
                        )

    if st.session_state.add_message:
        st.success(st.session_state.add_message)

    if click_mode.startswith("Toggle"):
        fig = build_review_figure(
            st.session_state.image,
            particles,
            approved_ids,
            labels=st.session_state.labels,
            show_rejects=show_rejected,
        )
        plot_key = f"review_plot_{st.session_state.get('plot_nonce', 0)}"
        selection = st.plotly_chart(
            fig,
            use_container_width=True,
            on_select="rerun",
            selection_mode="points",
            key=plot_key,
        )
        clicked_id = particle_id_from_plotly_selection(selection)
        if clicked_id is not None:
            st.session_state.approved_ids = toggle_particle(approved_ids, clicked_id)
            st.session_state.plot_nonce = st.session_state.get("plot_nonce", 0) + 1
            st.rerun()
    else:
        st.caption(
            "Click the **dark center** of a missed particle. "
            f"New particles are saved as **{preferred_class.value}s**."
        )
        try:
            from streamlit_image_coordinates import streamlit_image_coordinates
        except ImportError:
            st.error(
                "Missing package `streamlit-image-coordinates`. "
                "Install with: pip install streamlit-image-coordinates"
            )
            streamlit_image_coordinates = None  # type: ignore

        if streamlit_image_coordinates is not None:
            annotated = render_annotated_rgb(
                st.session_state.image,
                particles,
                approved_ids,
                labels=st.session_state.labels,
                show_rejects=show_rejected,
            )
            click = streamlit_image_coordinates(
                annotated,
                key=f"add_click_{st.session_state.get('plot_nonce', 0)}",
            )
            if click and "x" in click and "y" in click:
                click_key = (int(click["x"]), int(click["y"]))
                if click_key != st.session_state.last_add_click:
                    nm_pp = float(st.session_state.nm_per_pixel or 1.0)
                    new_particles, new_labels, msg = add_particle_at_click(
                        st.session_state.image,
                        st.session_state.labels,
                        particles,
                        click_y=float(click["y"]),
                        click_x=float(click["x"]),
                        nm_per_pixel=nm_pp,
                        preferred_class=preferred_class,
                    )
                    old_ids = {p.particle_id for p in particles}
                    new_ids = {p.particle_id for p in new_particles} - old_ids
                    st.session_state.particles = _particles_to_dicts(new_particles)
                    st.session_state.labels = new_labels
                    approved = set(st.session_state.approved_ids) | new_ids
                    if "particle #" in msg:
                        try:
                            pid = int(msg.split("particle #")[1].split()[0].rstrip("."))
                            approved.add(pid)
                        except ValueError:
                            pass
                    st.session_state.approved_ids = approved
                    st.session_state.add_message = msg
                    st.session_state.last_add_click = click_key
                    st.session_state.plot_nonce = (
                        st.session_state.get("plot_nonce", 0) + 1
                    )
                    st.rerun()

    st.subheader("Particle list")
    st.caption(
        "Uncheck **approved** to discard. "
        "**length/width** = ellipse axes; **Feret** = caliper diameters; "
        "**equiv_diameter** = 2√(A/π); **circularity** = 4πA/P². "
        "Use **Add missed particle** mode above for missed detections."
    )
    table = pd.DataFrame(particles_to_rows(particles, approved_ids))
    edited = st.data_editor(
        table,
        hide_index=True,
        use_container_width=True,
        disabled=[c for c in table.columns if c != "approved"],
        column_config={
            "approved": st.column_config.CheckboxColumn("approved", default=True),
        },
        key="particle_table",
    )
    table_approved = set(int(i) for i in edited.loc[edited["approved"], "particle_id"])
    if table_approved != approved_ids:
        st.session_state.approved_ids = table_approved
        st.rerun()

    csv_bytes = approved_csv_bytes(particles, set(st.session_state.approved_ids))
    d1, d2 = st.columns(2)
    with d1:
        st.download_button(
            "Download approved measurements CSV",
            data=csv_bytes,
            file_name=f"{st.session_state.stem}_approved_measurements.csv",
            mime="text/csv",
        )
    with d2:
        if st.session_state.overlay_bytes:
            st.download_button(
                "Download original overlay PNG",
                data=st.session_state.overlay_bytes,
                file_name=f"{st.session_state.stem}_overlay.png",
                mime="image/png",
            )

    for warning in st.session_state.warnings or []:
        st.warning(warning)

st.markdown("---")
st.markdown(
    "**Legend:** green = keep · red = discard · orange = pipeline reject  \n"
    "**Add mode:** click a missed dark particle to measure and include it  \n"
    "Repo: [github.com/fayh601-glitch/TEM-analysis](https://github.com/fayh601-glitch/TEM-analysis) · "
    "Deploy guide: `docs/DEPLOY_WEBSITE.md`"
)

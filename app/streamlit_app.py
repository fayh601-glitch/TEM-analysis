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
import streamlit as st

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))
if str(_REPO / "app") not in sys.path:
    sys.path.insert(0, str(_REPO / "app"))

from particle_review import (  # noqa: E402
    approved_csv_bytes,
    build_review_figure,
    default_approved_ids,
    particle_id_from_plotly_selection,
    particles_to_rows,
    summarize_approved,
    toggle_particle,
)
from tem_rods.models import AnalysisMode, ParticleClass, ParticleMeasurement  # noqa: E402
from tem_rods.pipeline import analyze_image  # noqa: E402
from tem_rods.presets import PRESETS, get_preset  # noqa: E402

st.set_page_config(
    page_title="TEM Particle Analyzer",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.title("TEM Particle Analyzer")
st.caption(
    "Upload a TEM image → choose rods or dots → set the scale bar → "
    "approve or discard outlines → download nm measurements. "
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
            )
        )
    return out


def _particles_to_dicts(particles: list[ParticleMeasurement]) -> list[dict]:
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


def _store_analysis_result(result, stem: str) -> None:
    st.session_state.particles = _particles_to_dicts(result.particles)
    st.session_state.approved_ids = default_approved_ids(result.particles)
    st.session_state.image = result.image
    st.session_state.labels = result.labels
    st.session_state.stem = stem
    st.session_state.warnings = list(result.warnings)
    st.session_state.analysis_done = True
    st.session_state.last_clicked_id = None
    st.session_state.plot_nonce = st.session_state.get("plot_nonce", 0) + 1
    if result.overlay_path and result.overlay_path.exists():
        st.session_state.overlay_bytes = result.overlay_path.read_bytes()
    else:
        st.session_state.overlay_bytes = None


def _run_analysis(
    image_path: Path,
    *,
    nm_per_pixel: float,
    preset_name: str,
    analysis_mode: AnalysisMode,
    show_rejected: bool,
    scale_bar_nm_hint: float | None,
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
    with st.spinner("Analyzing..."):
        result = analyze_image(
            image_path,
            nm_per_pixel,
            output_dir=out_dir,
            config=config,
            scale_bar_nm_hint=scale_bar_nm_hint,
        )
    _store_analysis_result(result, image_path.stem)


def _default_preset_for_mode(mode: AnalysisMode) -> str:
    if mode == AnalysisMode.DOTS:
        return "dots_only"
    if mode == AnalysisMode.BOTH:
        return "screenshot"
    if "dense_rods_50nm" in PRESETS:
        return "dense_rods_50nm"
    return "enright_rods"


_init_session()

SAMPLE_50NM = _REPO / "data" / "curated" / "user_rods_50nm_jul13.png"

# --- Primary controls (always visible before analyze) ---
st.subheader("1. What are you measuring?")
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
    "Enter the printed scale-bar value (e.g. 50 if the image says “50 nm”), "
    "then the length of that white line in **pixels** (measure in Preview/Fiji or use a known screenshot value)."
)
c_nm, c_px, c_cal = st.columns(3)
with c_nm:
    scale_bar_nm = st.number_input(
        "Scale bar length (nm)",
        min_value=1.0,
        value=50.0,
        step=1.0,
        help="The number printed next to the scale bar on the TEM image.",
    )
with c_px:
    scale_bar_px = st.number_input(
        "Scale bar length (pixels)",
        min_value=1.0,
        value=98.0,
        step=1.0,
        help="How many pixels long the white bar is in your uploaded image.",
    )
with c_cal:
    nm_per_pixel = scale_bar_nm / scale_bar_px
    st.metric("Calibration", f"{nm_per_pixel:.4f} nm/px")

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
    st.caption("Green = keep · red = discard · click a numbered marker to toggle.")

st.subheader("3. Image")
if SAMPLE_50NM.exists() and not st.session_state.analysis_done:
    st.info("Demo image available: dense nanorods with a 50 nm scale bar.")
    if st.button("Analyze demo (50 nm rods)", type="secondary"):
        _run_analysis(
            SAMPLE_50NM,
            nm_per_pixel=scale_bar_nm / scale_bar_px,
            preset_name="dense_rods_50nm" if "dense_rods_50nm" in PRESETS else preset_name,
            analysis_mode=analysis_mode,
            show_rejected=show_rejected,
            scale_bar_nm_hint=scale_bar_nm,
        )
        st.rerun()

uploaded = st.file_uploader(
    "Upload TEM image (PNG, JPG, TIF)",
    type=["png", "jpg", "jpeg", "tif", "tiff"],
)

if uploaded is not None and not st.session_state.analysis_done:
    st.image(uploaded, caption="Uploaded image", use_column_width="always")

analyze_clicked = st.button("Analyze image", type="primary", disabled=uploaded is None)

if analyze_clicked:
    assert uploaded is not None
    session_dir = _ensure_session_dir()
    image_path = session_dir / uploaded.name
    image_path.write_bytes(uploaded.getvalue())
    _run_analysis(
        image_path,
        nm_per_pixel=nm_per_pixel,
        preset_name=preset_name,
        analysis_mode=analysis_mode,
        show_rejected=show_rejected,
        scale_bar_nm_hint=scale_bar_nm,
    )
    st.rerun()

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
            ):
                st.session_state[key] = None if key != "approved_ids" else set()
            st.session_state.analysis_done = False
            st.rerun()

    m1, m2 = st.columns(2)
    with m1:
        if "mean_rod_length_nm" in stats:
            st.metric(
                "Mean approved rod length (nm)",
                f"{stats['mean_rod_length_nm']:.1f}",
            )
    with m2:
        if "mean_dot_length_nm" in stats:
            st.metric(
                "Mean approved dot size (nm)",
                f"{stats['mean_dot_length_nm']:.1f}",
            )

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

    st.subheader("Particle list")
    st.caption("Uncheck **approved** to discard, or click markers on the plot.")
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
    "**Legend:** green marker = keep · red = discard · orange = pipeline reject  \n"
    "Repo: [github.com/fayh601-glitch/TEM-analysis](https://github.com/fayh601-glitch/TEM-analysis) · "
    "Deploy guide: `docs/DEPLOY_WEBSITE.md`"
)

"""
TEM Rod Analyzer — Web upload interface (Streamlit)
====================================================

Run from the project root::

    source .venv/bin/activate
    pip install -r requirements-optional.txt
    streamlit run app/streamlit_app.py

No Terminal commands are needed after the app is open — upload an image,
enter the scale bar, and download results.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import streamlit as st

from tem_rods.models import AnalysisMode
from tem_rods.pipeline import analyze_image, print_summary
from tem_rods.presets import PRESETS, get_preset

st.set_page_config(page_title="TEM Rod Analyzer", page_icon="🔬", layout="wide")

st.title("CdSe/CdS TEM Nanorod Analyzer")
st.caption(
    "Upload a TEM image, calibrate the scale bar, and get rod length/width in nanometers. "
    "Reference: Enright et al. 2018."
)

with st.sidebar:
    st.header("Settings")
    mode_label = st.radio(
        "Particle type in this image",
        options=["Nanorods only", "Round dots only", "Both rods and dots"],
        index=0,
    )
    mode_map = {
        "Nanorods only": AnalysisMode.RODS,
        "Round dots only": AnalysisMode.DOTS,
        "Both rods and dots": AnalysisMode.BOTH,
    }
    analysis_mode = mode_map[mode_label]

    preset_names = sorted(PRESETS.keys())
    default_preset = "enright_rods" if analysis_mode == AnalysisMode.RODS else "dots_only"
    if analysis_mode == AnalysisMode.BOTH:
        default_preset = "screenshot"
    preset_name = st.selectbox(
        "Image preset",
        preset_names,
        index=preset_names.index(default_preset) if default_preset in preset_names else 0,
    )

    scale_bar_nm = st.number_input("Scale bar (nm)", min_value=1.0, value=20.0, step=1.0)
    scale_bar_px = st.number_input(
        "Scale bar length (pixels)",
        min_value=1.0,
        value=45.0,
        step=1.0,
        help="Measure the white scale-bar line in your image.",
    )
    show_rejected = st.checkbox("Show rejected particles (orange)", value=True)

uploaded = st.file_uploader(
    "Upload TEM image",
    type=["png", "jpg", "jpeg", "tif", "tiff"],
)

if uploaded is not None:
    st.image(uploaded, caption="Uploaded image", use_container_width=True)

if st.button("Analyze", type="primary", disabled=uploaded is None):
    assert uploaded is not None
    nm_per_pixel = scale_bar_nm / scale_bar_px

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        image_path = tmp_path / uploaded.name
        image_path.write_bytes(uploaded.getvalue())
        out_dir = tmp_path / "outputs"
        out_dir.mkdir()

        preset = get_preset(preset_name)
        from dataclasses import replace

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
            )

        st.success(f"Found {len(result.rods)} rods, {len(result.dots)} dots, {len(result.rejected)} rejected.")

        col1, col2 = st.columns(2)
        with col1:
            if result.overlay_path and result.overlay_path.exists():
                st.subheader("Annotated overlay")
                st.image(str(result.overlay_path), use_container_width=True)
        with col2:
            debug_path = out_dir / f"{image_path.stem}_segments_debug.png"
            if debug_path.exists():
                st.subheader("Segmentation debug")
                st.caption("Numbered blobs before rod/dot classification.")
                st.image(str(debug_path), use_container_width=True)

        if result.csv_path and result.csv_path.exists():
            st.subheader("Measurements")
            st.dataframe(
                __import__("pandas").read_csv(result.csv_path),
                use_container_width=True,
            )
            st.download_button(
                "Download CSV",
                data=result.csv_path.read_bytes(),
                file_name=f"{image_path.stem}_measurements.csv",
                mime="text/csv",
            )
        if result.overlay_path and result.overlay_path.exists():
            st.download_button(
                "Download overlay PNG",
                data=result.overlay_path.read_bytes(),
                file_name=f"{image_path.stem}_overlay.png",
                mime="image/png",
            )

        for warning in result.warnings:
            st.warning(warning)

st.markdown("---")
st.markdown(
    "**Legend:** green = rod · blue = dot · orange dashed = reject  \n"
    "**Docs:** see `docs/GETTING_STARTED.md` and `docs/TEM-analysis-Manual.pdf`"
)

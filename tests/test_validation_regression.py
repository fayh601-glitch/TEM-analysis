"""
Validation regression tests — pipeline vs Enright 2018 published rod dimensions.

Runs on curated SI Figure S2 panels when present. S2A must stay within tolerance;
S2B/S2D are known failures until segmentation improves.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from tem_rods.pipeline import analyze_image
from tem_rods.presets import get_preset

REPO = Path(__file__).resolve().parents[1]
CURATED = REPO / "data" / "curated"
CALIBRATION = REPO / "data" / "calibration.csv"

S2_PANELS = [
    "s2_A_starting_rods.png",
    "s2_B_30min.png",
    "s2_D_65min.png",
]

S2A_LENGTH_TOLERANCE_PCT = 15.0


def _curated_images_available() -> bool:
    return CALIBRATION.exists() and all((CURATED / name).exists() for name in S2_PANELS)


def _paper_rows() -> pd.DataFrame:
    cal = pd.read_csv(CALIBRATION).set_index("filename")
    return cal.loc[S2_PANELS]


def _run_panel(image_name: str) -> tuple[float, float, int]:
    cal = pd.read_csv(CALIBRATION).set_index("filename").loc[image_name]
    preset = get_preset("enright_rods")
    result = analyze_image(
        CURATED / image_name,
        float(cal["nm_per_pixel"]),
        config=preset.config,
        save_outputs=False,
        scale_bar_nm_hint=float(cal["scale_bar_nm"]),
    )
    rods = result.rods
    assert len(rods) > 0, f"No rods detected in {image_name}"
    mean_l = sum(r.length_nm for r in rods) / len(rods)
    mean_w = sum(r.width_nm for r in rods) / len(rods)
    return mean_l, mean_w, len(rods)


@pytest.fixture(scope="module")
def panel_results() -> dict[str, tuple[float, float, int]]:
    if not _curated_images_available():
        pytest.skip("Curated S2 images or calibration.csv not found")
    return {name: _run_panel(name) for name in S2_PANELS}


@pytest.mark.validation
def test_pipeline_returns_rods_on_all_s2_panels(panel_results):
    for image_name, (_mean_l, _mean_w, n_rods) in panel_results.items():
        assert n_rods > 0, f"Expected rods in {image_name}"


@pytest.mark.validation
def test_s2a_mean_length_within_paper_tolerance(panel_results):
    paper = _paper_rows().loc["s2_A_starting_rods.png"]
    mean_l, _mean_w, _n = panel_results["s2_A_starting_rods.png"]
    error_pct = abs(100 * (mean_l - paper["paper_length_nm"]) / paper["paper_length_nm"])
    assert error_pct <= S2A_LENGTH_TOLERANCE_PCT, (
        f"S2A length error {error_pct:.1f}% exceeds {S2A_LENGTH_TOLERANCE_PCT}% "
        f"(pipeline={mean_l:.1f} nm, paper={paper['paper_length_nm']} nm)"
    )


@pytest.mark.validation
def test_s2b_mean_length_within_paper_tolerance(panel_results):
    paper = _paper_rows().loc["s2_B_30min.png"]
    mean_l, _mean_w, _n = panel_results["s2_B_30min.png"]
    error_pct = abs(100 * (mean_l - paper["paper_length_nm"]) / paper["paper_length_nm"])
    assert error_pct <= S2A_LENGTH_TOLERANCE_PCT, (
        f"S2B length error {error_pct:.1f}% exceeds {S2A_LENGTH_TOLERANCE_PCT}%"
    )


@pytest.mark.validation
@pytest.mark.xfail(
    reason="S2B width error ~85% vs paper — merged/over-wide rod measurements (known failure)",
    strict=False,
)
def test_s2b_mean_width_within_paper_tolerance(panel_results):
    paper = _paper_rows().loc["s2_B_30min.png"]
    _mean_l, mean_w, _n = panel_results["s2_B_30min.png"]
    error_pct = abs(100 * (mean_w - paper["paper_width_nm"]) / paper["paper_width_nm"])
    assert error_pct <= S2A_LENGTH_TOLERANCE_PCT


@pytest.mark.validation
@pytest.mark.xfail(
    reason="S2D under-counts long rods; ~78% length error vs paper (known failure)",
    strict=False,
)
def test_s2d_mean_length_matches_paper(panel_results):
    paper = _paper_rows().loc["s2_D_65min.png"]
    mean_l, _mean_w, _n = panel_results["s2_D_65min.png"]
    error_pct = abs(100 * (mean_l - paper["paper_length_nm"]) / paper["paper_length_nm"])
    assert error_pct <= S2A_LENGTH_TOLERANCE_PCT

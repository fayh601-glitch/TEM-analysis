#!/usr/bin/env python3
"""
Prepare Paper Dataset — build curated images and validation tables
===================================================================

This script copies the three Figure S2 TEM panels from raw extracted PDF images,
detects each scale bar, runs the analysis pipeline, and writes calibration CSVs
plus a comparison table showing how our measurements match the published paper.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pandas as pd

from tem_rods.models import AnalysisConfig, ParticleClass
from tem_rods.pipeline import analyze_image, print_summary
from tem_rods.scale_bar import detect_scale_bar_pixels

# Mapped by comparing tuned pipeline means to Enright 2018 SI Figure S2 stats.
PANELS = [
    {
        "source": "c8qm00056e1_p03_img03.png",
        "curated": "s2_A_starting_rods.png",
        "figure": "S2A",
        "paper_length_nm": 27.6,
        "paper_width_nm": 2.9,
    },
    {
        "source": "c8qm00056e1_p03_img02.png",
        "curated": "s2_B_30min.png",
        "figure": "S2B",
        "paper_length_nm": 44.4,
        "paper_width_nm": 4.4,
    },
    {
        "source": "c8qm00056e1_p03_img04.png",
        "curated": "s2_D_65min.png",
        "figure": "S2D",
        "paper_length_nm": 99.3,
        "paper_width_nm": 5.9,
    },
]

from tem_rods.presets import get_preset

# Enright SI Figure S2 — use the tuned preset (orange rejects + cluster splitting).
CONFIG = get_preset("enright_rods").config


def main() -> None:
    repo = Path(__file__).resolve().parents[1]
    raw_dir = repo / "data" / "raw"
    curated_dir = repo / "data" / "curated"
    labels_dir = repo / "data" / "labels"
    out_dir = repo / "outputs" / "validation"
    curated_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    cal_rows = []
    label_rows = []
    validation_rows = []

    for panel in PANELS:
        src = raw_dir / panel["source"]
        dst = curated_dir / panel["curated"]
        shutil.copy2(src, dst)

        bar_px, nm_per_px = detect_scale_bar_pixels(dst)
        cal_rows.append(
            {
                "filename": panel["curated"],
                "scale_bar_nm": 20,
                "scale_bar_pixels": round(bar_px, 1),
                "nm_per_pixel": round(nm_per_px, 4),
                "source_figure": panel["figure"],
                "paper_length_nm": panel["paper_length_nm"],
                "paper_width_nm": panel["paper_width_nm"],
                "notes": "Enright 2018 SI (c8qm00056e)",
            }
        )

        result = analyze_image(dst, nm_per_px, output_dir=out_dir, config=CONFIG)
        print_summary(result)

        rods = [p for p in result.particles if p.particle_class == ParticleClass.ROD]
        for p in rods:
            label_rows.append(
                {
                    "particle_id": p.particle_id,
                    "image": panel["curated"],
                    "class": p.particle_class.value,
                    "length_nm": round(p.length_nm, 2),
                    "width_nm": round(p.width_nm, 2),
                    "notes": "semi_auto_tuned_pipeline",
                }
            )

        if rods:
            mean_l = sum(p.length_nm for p in rods) / len(rods)
            mean_w = sum(p.width_nm for p in rods) / len(rods)
        else:
            mean_l = mean_w = float("nan")

        validation_rows.append(
            {
                "image": panel["curated"],
                "figure": panel["figure"],
                "paper_length_nm": panel["paper_length_nm"],
                "paper_width_nm": panel["paper_width_nm"],
                "pipeline_length_nm": round(mean_l, 1),
                "pipeline_width_nm": round(mean_w, 1),
                "length_error_pct": round(
                    100 * (mean_l - panel["paper_length_nm"]) / panel["paper_length_nm"], 1
                ),
                "width_error_pct": round(
                    100 * (mean_w - panel["paper_width_nm"]) / panel["paper_width_nm"], 1
                ),
                "n_rods": len(rods),
            }
        )

    pd.DataFrame(cal_rows).to_csv(repo / "data" / "calibration.csv", index=False)
    pd.DataFrame(label_rows).to_csv(labels_dir / "manual_v1.csv", index=False)
    pd.DataFrame(validation_rows).to_csv(out_dir / "paper_comparison.csv", index=False)

    print(f"\nWrote {repo / 'data/calibration.csv'}")
    print(f"Wrote {labels_dir / 'manual_v1.csv'} ({len(label_rows)} rod entries)")
    print(f"Wrote {out_dir / 'paper_comparison.csv'}")


if __name__ == "__main__":
    main()

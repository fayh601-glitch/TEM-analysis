#!/usr/bin/env python3
"""
Compare pipeline output to semi-manual labels in data/labels/manual_v1.csv.

Use this after segmentation or preset changes to see whether rod counts and mean
lengths move closer to the reference labels (not rigorous ground truth).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from tem_rods.pipeline import analyze_image
from tem_rods.presets import get_preset


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--labels",
        type=Path,
        default=None,
        help="Path to manual labels CSV (default: data/labels/manual_v1.csv).",
    )
    parser.add_argument(
        "--calibration",
        type=Path,
        default=None,
        help="Path to calibration CSV (default: data/calibration.csv).",
    )
    parser.add_argument(
        "--preset",
        default="enright_rods",
        help="Analysis preset (default: enright_rods).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Write comparison table here (default: outputs/validation/manual_comparison.csv).",
    )
    args = parser.parse_args()

    repo = Path(__file__).resolve().parents[1]
    labels_path = args.labels or repo / "data" / "labels" / "manual_v1.csv"
    cal_path = args.calibration or repo / "data" / "calibration.csv"
    out_path = args.output or repo / "outputs" / "validation" / "manual_comparison.csv"

    labels = pd.read_csv(labels_path)
    calibration = pd.read_csv(cal_path).set_index("filename")
    curated_dir = repo / "data" / "curated"
    preset = get_preset(args.preset)
    cfg = preset.config

    rows = []
    for image_name, group in labels.groupby("image"):
        image_path = curated_dir / image_name
        if not image_path.exists():
            print(f"Skip {image_name}: not found in {curated_dir}")
            continue
        if image_name not in calibration.index:
            print(f"Skip {image_name}: no calibration row")
            continue

        cal = calibration.loc[image_name]
        nm_per_pixel = float(cal["nm_per_pixel"])
        bar_nm = float(cal["scale_bar_nm"])

        result = analyze_image(
            image_path,
            nm_per_pixel,
            config=cfg,
            save_outputs=False,
            scale_bar_nm_hint=bar_nm,
        )
        rods = result.rods
        label_rods = group[group["class"] == "rod"]

        rows.append(
            {
                "image": image_name,
                "label_count": len(label_rods),
                "pipeline_count": len(rods),
                "count_delta": len(rods) - len(label_rods),
                "label_mean_length_nm": round(label_rods["length_nm"].mean(), 1),
                "pipeline_mean_length_nm": round(
                    sum(r.length_nm for r in rods) / len(rods), 1
                )
                if rods
                else None,
                "label_median_length_nm": round(label_rods["length_nm"].median(), 1),
                "pipeline_median_length_nm": round(
                    float(pd.Series([r.length_nm for r in rods]).median()), 1
                )
                if rods
                else None,
                "warnings": "; ".join(result.warnings),
            }
        )

    df = pd.DataFrame(rows)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(df.to_string(index=False))
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()

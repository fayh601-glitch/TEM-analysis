# Project status

**Repo:** [github.com/fayh601-glitch/TEM-analysis](https://github.com/fayh601-glitch/TEM-analysis)

**Goal:** Upload TEM image → classify rods vs dots → report length and width (nm).

## Completed (v0.1 → v0.3)

- [x] Classical CV pipeline (`src/tem_rods/`)
- [x] CLI: `tem-rods analyze`
- [x] PDF image extraction script
- [x] Auto scale-bar detection
- [x] Curated Enright 2018 SI Figure S2 panels in `data/curated/`
- [x] Calibration + label CSVs
- [x] Validation vs published rod dimensions
- [x] Unit tests (pytest)
- [x] **v0.2:** Segmentation quality filters (contrast, solidity, extent, scale-bar mask)
- [x] **v0.2:** Reject class for ambiguous background blobs
- [x] **v0.2:** Overlay draws actual contours + aligned ellipse axes
- [x] **v0.2:** Beginner documentation (`docs/HOW_IT_WORKS.md`, file-level comments)
- [x] **v0.2:** Analysis modes (rods/dots/both), presets, Streamlit UI, PDF manual
- [x] **v0.3:** Hole-filling after morphological closing (reduces light-center rod splits)
- [x] **v0.3:** Auto scale-bar bbox masking + default bottom strip mask (12–15%)
- [x] **v0.3:** QC warning when mean length ≫ median (merged clusters)
- [x] **v0.3:** `dense_rods` / `dense_rods_50nm` presets; fixed `sparse_cluster` masking
- [x] **v0.3:** `scripts/compare_manual_labels.py` for label-vs-pipeline checks

## Validation (Enright 2018 Figure S2)

Re-run after changes: `bash scripts/run_demo.sh` or
`python scripts/prepare_paper_dataset.py`

| Panel | Paper L (nm) | Pipeline L (nm) | Length err | Width err | Notes |
|---|---|---|---|---|---|
| S2A starting rods | 27.6 | ~28 | ~3% | ~62% | Length OK |
| S2B 30 min | 44.4 | 66.1 | ~49% | ~132% | Over-segmentation / merged rods |
| S2D 65 min | 99.3 | 52.9 | ~47% | ~15% | Under-counts long rods |

See `outputs/validation/paper_comparison.csv` for latest numbers (includes
`width_error_pct`).

Tuned settings: `min_particle_area_px=150`, `use_watershed=False`.

**Label files:** `data/labels/manual_v1.csv` is semi-automated pipeline output
(`notes=semi_auto_tuned_pipeline`), not Fiji hand labels. Real validation requires
`manual_v2.csv` per [docs/LABELING_GUIDE.md](LABELING_GUIDE.md).

## Not started

- [ ] Fiji hand labels (`manual_v2.csv`, 30–50 rods per panel)
- [ ] UV–Vis correlation module
- [ ] ML segmentation

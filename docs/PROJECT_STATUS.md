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

Re-run after changes: `bash scripts/run_demo.sh`

| Panel | Paper L (nm) | Pipeline L (nm) | Notes |
|---|---|---|---|
| S2A starting rods | 27.6 | ~28 | ✓ ~3% error (re-run demo for latest) |
| S2B 30 min | 44.4 | 66.1 | Over-segmentation / merged rods |
| S2D 65 min | 99.3 | 52.9 | Under-counts long rods |

Tuned settings: `min_particle_area_px=150`, `use_watershed=False`.

## Not started

- [ ] Fiji-quality hand labels
- [ ] UV–Vis correlation module
- [ ] ML segmentation

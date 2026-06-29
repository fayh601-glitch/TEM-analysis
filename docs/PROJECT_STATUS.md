# Project status

**Repo:** [github.com/fayh601-glitch/TEM-analysis](https://github.com/fayh601-glitch/TEM-analysis)

**Goal:** Upload TEM image → classify rods vs dots → report length and width (nm).

## Completed (v0.1)

- [x] Classical CV pipeline (`src/tem_rods/`)
- [x] CLI: `tem-rods analyze`
- [x] PDF image extraction script
- [x] Auto scale-bar detection
- [x] Curated Enright 2018 SI Figure S2 panels in `data/curated/`
- [x] Calibration + label CSVs
- [x] Validation vs published rod dimensions
- [x] Unit tests (pytest)

## Validation (Enright 2018 Figure S2)

| Panel | Paper L (nm) | Pipeline L (nm) | Notes |
|---|---|---|---|
| S2A starting rods | 27.6 | 28.4 | ✓ ~3% error |
| S2B 30 min | 44.4 | 66.1 | Over-segmentation / merged rods |
| S2D 65 min | 99.3 | 52.9 | Under-counts long rods |

Tuned settings: `min_particle_area_px=150`, `use_watershed=False`.

## Not started

- [ ] Streamlit upload UI
- [ ] Fiji-quality hand labels
- [ ] UV–Vis correlation module
- [ ] ML segmentation

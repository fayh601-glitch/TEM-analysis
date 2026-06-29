# CdSe/CdS TEM Rod & Dot Analyzer

Classical computer-vision pipeline to analyze **transmission electron microscopy (TEM)** images of **CdSe** and **CdS** nanoparticles. Detects particles, classifies **rods vs dots (spheres)**, and reports **length** and **width** in nanometers.

Validation reference: Enright et al., *Mater. Chem. Front.* **2018**, [DOI: 10.1039/c8qm00056e](https://doi.org/10.1039/c8qm00056e) — SI Figure S2 starting rods ~27.6 × 2.9 nm.

## Setup

```bash
cd TEM-analysis
python3.10 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

## Workflow

### 1. Extract TEM panels from the SI PDF

```bash
python scripts/extract_pdf_images.py ~/Desktop/c8qm00056e1.pdf --output data/raw
```

Crop individual TEM fields-of-view from extracted pages. Name files clearly (e.g. `s2_A_starting_rods.png`).

### 2. Record scale bar calibration

Copy `data/calibration.example.csv` → `data/calibration.csv` and fill in `scale_bar_nm` and `scale_bar_pixels` for each image (measure the scale bar in Fiji/ImageJ).

### 3. Run analysis

```bash
# Option A: direct nm/pixel
tem-rods analyze --image data/raw/s2_A_starting_rods.png --nm-per-pixel 0.42

# Option B: from scale bar
tem-rods analyze --image data/curated/s2_A_starting_rods.png --scale-bar-nm 20 --scale-bar-pixels 48

# Prepare curated SI panels + validation against Enright 2018 Figure S2
python scripts/prepare_paper_dataset.py
```

### 4. Outputs

Written to `outputs/`:

- `{stem}_measurements.csv` — per-particle length, width, class, aspect ratio
- `{stem}_overlay.png` — annotated image (green = rod, blue = dot)

## Manual validation labels

Use `data/labels/manual_v1.example.csv` as a template. Measure ~20–50 rods/dots in Fiji, then compare your manual means to pipeline output and to published SI values.

## Measurement definitions

| Quantity | Definition |
|---|---|
| **Length** | Feret maximum diameter (longest caliper) |
| **Width** | Feret minimum diameter |
| **Rod** | eccentricity ≥ 0.85 **and** aspect ratio ≥ 1.5 |
| **Dot** | all other detected particles |

## Project layout

```
TEM-analysis/
├── data/raw/           # TEM images (gitignored contents)
├── data/labels/        # manual ground-truth CSV
├── outputs/            # results (gitignored)
├── scripts/            # PDF extraction helper
├── src/tem_rods/       # analysis package
└── tests/
```

## Tests

```bash
pytest
```

## Next steps

1. Extract and crop SI TEM panels from Cossairt 2018
2. Fill `data/calibration.csv`
3. Label 50–100 particles in `data/labels/manual_v1.csv`
4. Run pipeline and compare rod means to SI Figure S2 (27.6 nm × 2.9 nm)
5. Tune `--min-area`, `--min-eccentricity-rod` if needed

## Limitations (v1)

- 2D projection — tilted rods appear shorter than true 3D length
- Touching/overlapping rods may merge (try without `--no-watershed` first)
- Tetrapods and branched shapes are **not** supported — rods and dots only

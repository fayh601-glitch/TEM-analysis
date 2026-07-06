# TEM-Analysis Repository Manual

**CdSe/CdS Transmission Electron Microscopy (TEM) Image Analysis**

Version 0.2 · Enright et al. 2018 reference dataset

---

## 1. Purpose of this project

This software analyzes **grayscale TEM images** of **CdSe** and **CdS** nanoparticles. Given an image with a visible **scale bar**, it:

1. Finds dark particle blobs on the carbon film background
2. Measures **length** and **width** in **nanometers**
3. Labels each particle (rod, dot, or reject — depending on analysis mode)
4. Saves an **annotated overlay image** and a **CSV spreadsheet**

**Scientific reference:** Enright, M. B. et al. *Mater. Chem. Front.* **2018**, [10.1039/c8qm00056e](https://doi.org/10.1039/c8qm00056e). Supplementary Figure S2 provides validation TEM panels and published rod dimensions.

**What this is not:** A machine-learning black box. It uses classical computer vision (thresholding, shape metrics). It works best on clean TEM exports; screenshots and very dense clusters are harder.

---

## 2. Who this manual is for

| Reader | Start here |
|--------|------------|
| New lab member, no coding | Section 3 → `docs/GETTING_STARTED.md` |
| Occasional user | Section 4 (three ways to run) |
| Someone reading the code | Section 6 (file guide) |
| Developer tuning parameters | Section 7 + `docs/DEVELOPMENT.md` |

---

## 3. Quick start (10 minutes)

### Install once

```bash
git clone https://github.com/fayh601-glitch/TEM-analysis.git
cd TEM-analysis
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

### Analyze with guided script

```bash
bash scripts/analyze_my_image.sh
```

Follow the prompts. Results go to `outputs/user_runs/`.

### Read your results

| Output file | Description |
|-------------|-------------|
| `*_overlay.png` | Image with colored particle outlines and size labels |
| `*_measurements.csv` | One row per detected particle |
| `*_segments_debug.png` | All segmented blobs numbered (troubleshooting) |

---

## 4. Three ways to analyze an image

### A. Guided shell script (easiest)

```bash
bash scripts/analyze_my_image.sh /path/to/image.png
```

Prompts for particle type, scale bar nm, and scale bar pixels.

### B. Web upload (Streamlit)

```bash
pip install -r requirements-optional.txt
streamlit run app/streamlit_app.py
```

Opens a browser window. Upload image, enter scale bar, click Analyze.

### C. Terminal command (full control)

```bash
tem-rods analyze \
  --image data/curated/s2_A_starting_rods.png \
  --preset enright_rods \
  --mode rods \
  --scale-bar-nm 20 \
  --scale-bar-pixels 48 \
  --output-dir outputs/my_run
```

---

## 5. Analysis modes

Use **one codebase** with a **mode flag** — not separate git branches.

| Mode | CLI flag | Meaning |
|------|----------|---------|
| **Rods only** | `--mode rods` | Report elongated particles; round blobs → reject |
| **Dots only** | `--mode dots` | Report round particles; elongated blobs → reject |
| **Both** | `--mode both` | Classify rods and dots (default) |

**Presets** bundle tuned settings:

| Preset | Mode | Best for |
|--------|------|----------|
| `enright_rods` | rods | Enright SI Figure S2 nanorod panels |
| `dots_only` | dots | Spherical quantum dot samples |
| `screenshot` | both | Paper figure screenshots with white margins |
| `sparse_cluster` | both | Large sparse fields (e.g. 200 nm scale bar) |

---

## 6. How the pipeline works

```
  TEM image (pixels)
        │
        ▼
  ┌─────────────┐
  │  io.py      │  Load PNG / TIFF / JPG as grayscale array
  └─────────────┘
        │
        ▼
  ┌─────────────┐
  │ calibrate   │  scale_bar_nm ÷ scale_bar_pixels → nm/pixel
  └─────────────┘
        │
        ▼
  ┌─────────────┐
  │ preprocess  │  Normalize contrast, light blur (reduce grain)
  └─────────────┘
        │
        ▼
  ┌─────────────┐
  │ segment     │  Threshold → find dark blobs → filter noise
  └─────────────┘      mask scale-bar strip, drop faint blobs
        │
        ▼
  ┌─────────────┐
  │ measure     │  Ellipse fit → length, width, eccentricity
  └─────────────┘
        │
        ▼
  ┌─────────────┐
  │ classify    │  rod / dot / reject (+ analysis mode filter)
  └─────────────┘
        │
        ▼
  ┌─────────────┐
  │ pipeline    │  Write CSV + overlay PNG + optional debug image
  └─────────────┘
```

### Measurement definitions

| Quantity | Definition |
|----------|------------|
| **Length** | Major axis of fitted ellipse (longest direction) |
| **Width** | Minor axis of fitted ellipse (shortest direction) |
| **Rod** | High eccentricity AND aspect ratio above threshold |
| **Dot** | Low eccentricity AND roughly round |
| **Reject** | Ambiguous shape, wrong mode, or filtered artifact |

### Overlay legend

| Visual | Meaning |
|--------|---------|
| Solid green contour | Rod boundary |
| Dashed green ellipse | Fitted length/width axes |
| Blue | Dot (only in `both` mode) |
| Orange dashed | Reject — segmented but not reported as rod/dot |
| `R 28.0×3.5 nm` | Class letter + dimensions |

**Blank regions** on the overlay mean the particle was **never segmented** (too faint, merged, or below area threshold). Compare with `*_segments_debug.png`.

---

## 7. Repository layout

```
TEM-analysis/
├── app/
│   └── streamlit_app.py       Web upload UI
├── src/tem_rods/              Main Python package
│   ├── cli.py                 Terminal: tem-rods analyze
│   ├── pipeline.py            End-to-end workflow
│   ├── io.py                  Load images
│   ├── preprocess.py          Contrast + denoise
│   ├── segment.py             Find particles (most critical)
│   ├── measure.py             Length / width in nm
│   ├── classify.py            Rod / dot / reject + mode filter
│   ├── models.py              Settings and data types
│   ├── calibrate.py           Pixels → nanometers
│   ├── scale_bar.py           Auto-detect scale bar
│   └── presets.py             Tuned config bundles
├── scripts/
│   ├── analyze_my_image.sh    Guided analysis (beginners)
│   ├── run_demo.sh            Analyze 3 paper images
│   ├── extract_pdf_images.py  Pull figures from SI PDF
│   ├── prepare_paper_dataset.py
│   └── build_manual_pdf.py    Regenerate this PDF
├── data/
│   ├── curated/               Enright S2 TEM panels
│   ├── calibration.csv        Scale bar per image
│   └── labels/                Manual ground-truth labels
├── outputs/
│   ├── demo/                  Example results (on GitHub)
│   └── user_runs/             Your analyses
├── tests/                     Automatic tests (pytest)
└── docs/
    ├── GETTING_STARTED.md     Beginner walkthrough
    ├── HOW_IT_WORKS.md        Plain-English pipeline guide
    ├── DEVELOPMENT.md         Contributor / tuning guide
    ├── PROJECT_STATUS.md      Roadmap
    └── TEM-analysis-Manual.pdf  This document (PDF)
```

---

## 8. Source code file guide

Read files in this order if you are new to the code:

1. **`pipeline.py`** — Orchestrates everything; best single entry point
2. **`segment.py`** — Where most accuracy is won or lost
3. **`classify.py`** — Shape rules and analysis modes
4. **`measure.py`** — Converts blob geometry to nm
5. **`models.py`** — All tunable parameters in one place
6. **`cli.py`** — How Terminal flags map to settings

Each file has a comment block at the top explaining its job in plain English.

---

## 9. Key parameters (`models.py` / CLI)

| Parameter | Typical value | Purpose |
|-----------|---------------|---------|
| `min_particle_area_px` | 80–150 | Ignore tiny specks |
| `min_eccentricity_rod` | 0.82–0.85 | How elongated = rod |
| `min_aspect_ratio_rod` | 1.38–1.5 | Length/width ratio for rod |
| `mask_bottom_fraction` | 0.10 | Ignore scale-bar strip |
| `split_touching_particles` | off for Enright | Split merged clusters (can fragment rods) |
| `analysis_mode` | rods / dots / both | Sample-type filter |

---

## 10. Troubleshooting

| Problem | Likely cause | What to try |
|---------|--------------|-------------|
| All sizes wrong | Bad scale bar pixels | Re-measure white line in ImageJ |
| Missing rods | Segmentation | Lower `min_area`; try `screenshot` preset; check debug PNG |
| False green rods | Noise / caption text | Crop to TEM field only; raise rod thresholds |
| Many orange rejects | Dense overlaps | Expected for clusters; tune segmentation |
| Blue dots on rod-only sample | Round fragments | Use `--mode rods` or `enright_rods` preset |
| Zero particles | Wrong preset/threshold | Try `screenshot` preset; check image is grayscale TEM |

---

## 11. Validation data

Curated Enright SI panels in `data/curated/`:

| File | Paper mean length × width |
|------|---------------------------|
| `s2_A_starting_rods.png` | 27.6 × 2.9 nm |
| `s2_B_30min.png` | 44.4 × 4.4 nm |
| `s2_D_65min.png` | 99.3 × 5.9 nm |

Run demo: `bash scripts/run_demo.sh`  
Compare: `outputs/validation/paper_comparison.csv`

---

## 12. Testing

```bash
pytest
```

Tests use synthetic images with known rod and dot shapes to catch regressions.

---

## 13. Roadmap (known limitations)

1. Dense touching rods merge — segmentation is the main bottleneck
2. Screenshot images perform worse than raw TEM exports
3. UV–Vis correlation not yet implemented
4. Hand labels in `data/labels/manual_v1.csv` needed for rigorous validation

---

## 14. Commands reference

```bash
# Analyze one image (rods only)
tem-rods analyze -i IMAGE.png --preset enright_rods --mode rods \
  --scale-bar-nm 20 --scale-bar-pixels 45 -o outputs/my_run

# Auto scale bar
tem-rods analyze -i IMAGE.png --auto-scale-bar --scale-bar-nm 20 --preset enright_rods

# Demo all paper panels
bash scripts/run_demo.sh

# Web UI
streamlit run app/streamlit_app.py

# Regenerate manual PDF
python scripts/build_manual_pdf.py
```

---

## 15. License and citation

Academic / research use. TEM reference images derived from Enright et al. 2018 supplementary information.

When publishing results obtained with this tool, cite the Enright paper for validation methodology and document your scale-bar calibration and analysis parameters.

---

*End of manual*

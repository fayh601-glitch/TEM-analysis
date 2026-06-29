# CdSe/CdS TEM Rod & Dot Analyzer

Classical computer-vision pipeline for **transmission electron microscopy (TEM)**
images of **CdSe** and **CdS** nanoparticles. Detects particles, classifies
**rods vs dots**, rejects background noise, and reports **length** and **width**
in nanometers.

**Reference paper:** Enright et al., *Mater. Chem. Front.* **2018** — [DOI: 10.1039/c8qm00056e](https://doi.org/10.1039/c8qm00056e)

> **New here?** Read [How it works](docs/HOW_IT_WORKS.md) for a plain-English walkthrough of every step.

## Quick start

```bash
git clone https://github.com/fayh601-glitch/TEM-analysis.git
cd TEM-analysis
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt && pip install -e .

# Analyze one image
tem-rods analyze \
  --image data/curated/s2_A_starting_rods.png \
  --scale-bar-nm 20 --scale-bar-pixels 48 \
  --min-area 150 --no-watershed

# Or analyze all three paper images at once
bash scripts/run_demo.sh
```

Results land in `outputs/` (or `outputs/demo/` for the demo script).

**View on GitHub:** browse [`outputs/demo/`](outputs/demo/) — overlay PNGs and CSVs are committed for browser preview.

## Repository layout

```
TEM-analysis/
├── src/tem_rods/              ← Main Python package (see src/tem_rods/README.md)
│   ├── cli.py                 ← Terminal command entry point
│   ├── pipeline.py            ← Full analysis workflow
│   ├── segment.py             ← Find particles + filter noise
│   ├── classify.py            ← Rod / dot / reject labels
│   ├── measure.py             ← Length & width in nm
│   └── ...                    ← io, preprocess, calibrate, scale_bar, models
├── scripts/                   ← Batch helpers (see scripts/README.md)
│   ├── run_demo.sh            ← Analyze all 3 paper images
│   ├── extract_pdf_images.py  ← Pull images from SI PDF
│   └── prepare_paper_dataset.py
├── data/                      ← Input images & calibration (see data/README.md)
│   ├── curated/               ← Figure S2 TEM panels
│   └── calibration.csv
├── outputs/                   ← Analysis results (CSVs + overlay PNGs)
│   ├── demo/                  ← Committed demo run (3 paper panels)
│   ├── user_runs/             ← User-uploaded / screenshot analyses
│   └── validation/            ← Comparison to published stats
├── tests/                     ← Automatic tests (`pytest`)
└── docs/
    ├── HOW_IT_WORKS.md        ← Beginner guide (start here)
    ├── DEVELOPMENT.md         ← Contributor guide
    └── PROJECT_STATUS.md      ← What's done / what's next
```

## Commands

| Command | Description |
|---|---|
| `tem-rods analyze --image PATH --auto-scale-bar` | Analyze with automatic scale-bar detection |
| `tem-rods analyze --image PATH --scale-bar-nm 20 --scale-bar-pixels N` | Analyze with manual scale bar |
| `bash scripts/run_demo.sh` | Run all 3 paper images → `outputs/demo/` |
| `python scripts/extract_pdf_images.py SI.pdf -o data/raw` | Extract images from PDF |
| `python scripts/prepare_paper_dataset.py` | Regenerate curated data + validation |
| `pytest` | Run tests |

## Measurement definitions

| Quantity | Definition |
|---|---|
| **Length** | Major axis of fitted ellipse |
| **Width** | Minor axis of fitted ellipse |
| **Rod** | eccentricity ≥ 0.85 and aspect ratio ≥ 1.5 |
| **Dot** | low eccentricity and roughly round |
| **Reject** | ambiguous detection — in CSV only, not on overlay |

## Overlay legend

| Visual | Meaning |
|---|---|
| Solid contour | Actual detected particle boundary |
| Dashed ellipse | Fitted length/width axes |
| Green | Rod |
| Blue | Dot |

## Documentation

- [How it works (beginner)](docs/HOW_IT_WORKS.md)
- [Source code file guide](src/tem_rods/README.md)
- [Scripts folder](scripts/README.md)
- [Development guide](docs/DEVELOPMENT.md)
- [Project status](docs/PROJECT_STATUS.md)
- [Data folder](data/README.md)

## License

Academic / research use. TEM reference images derived from Enright et al. 2018 supplementary information.

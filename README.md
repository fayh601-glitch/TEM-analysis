# CdSe/CdS TEM Rod & Dot Analyzer

Classical computer-vision pipeline for **transmission electron microscopy (TEM)** images of **CdSe** and **CdS** nanoparticles. Detects particles, classifies **rods vs dots**, and reports **length** and **width** in nanometers.

**Reference paper:** Enright et al., *Mater. Chem. Front.* **2018** — [DOI: 10.1039/c8qm00056e](https://doi.org/10.1039/c8qm00056e)

## Quick start

```bash
git clone https://github.com/fayh601-glitch/TEM-analysis.git
cd TEM-analysis
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt && pip install -e .

# Analyze curated Figure S2A starting rods (validated ~28 nm vs paper 27.6 nm)
tem-rods analyze \
  --image data/curated/s2_A_starting_rods.png \
  --scale-bar-nm 20 --scale-bar-pixels 48 \
  --min-area 150 --no-watershed
```

Results: `outputs/s2_A_starting_rods_measurements.csv` and `outputs/s2_A_starting_rods_overlay.png`.

**View results on GitHub:** browse [`outputs/demo/`](outputs/demo/) — overlay PNGs and CSVs are committed so you can preview them in the browser without running locally.

## Repository layout

```
TEM-analysis/
├── src/tem_rods/              # Analysis package
├── scripts/
│   ├── extract_pdf_images.py  # Pull images from SI PDF
│   └── prepare_paper_dataset.py # Build curated set + validation
├── data/
│   ├── curated/               # Figure S2 TEM panels (committed)
│   ├── calibration.csv        # Scale bars + paper reference sizes
│   └── labels/manual_v1.csv   # Rod labels (semi-auto)
├── outputs/validation/        # Comparison to published stats
├── tests/
└── docs/DEVELOPMENT.md        # Contributor guide
```

## Commands

| Command | Description |
|---|---|
| `tem-rods analyze --image PATH --scale-bar-nm 20 --scale-bar-pixels N` | Analyze one TEM image |
| `bash scripts/run_demo.sh` | Run all 3 paper images → `outputs/demo/` (commit to view on GitHub) |
| `python scripts/extract_pdf_images.py SI.pdf -o data/raw` | Extract images from PDF |
| `python scripts/prepare_paper_dataset.py` | Regenerate curated data + validation |
| `pytest` | Run tests |

## Measurement definitions

| Quantity | Definition |
|---|---|
| **Length** | Feret maximum diameter |
| **Width** | Minor axis / Feret min |
| **Rod** | eccentricity ≥ 0.85 and aspect ratio ≥ 1.5 |
| **Dot** | all other detected particles |

## Validation snapshot

| Panel | Paper length | Pipeline length |
|---|---|---|
| S2A starting rods | 27.6 nm | **28.4 nm** |
| S2B 30 min | 44.4 nm | 66.1 nm (needs tuning) |
| S2D 65 min | 99.3 nm | 52.9 nm (needs tuning) |

Full table: `outputs/validation/paper_comparison.csv`

## Documentation

- [Development guide](docs/DEVELOPMENT.md)
- [Project status](docs/PROJECT_STATUS.md)
- [Data folder](data/README.md)

## License

Academic / research use. TEM reference images derived from Enright et al. 2018 supplementary information.

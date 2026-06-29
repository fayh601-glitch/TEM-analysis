# Development guide

## Repo structure

```
TEM-analysis/
├── src/tem_rods/           # Core Python package
│   ├── cli.py              # `tem-rods analyze` entry point
│   ├── pipeline.py         # End-to-end orchestration
│   ├── segment.py          # Particle segmentation
│   ├── classify.py         # Rod vs dot
│   ├── measure.py          # Length / width in nm
│   ├── scale_bar.py        # Auto-detect 20 nm scale bars
│   └── ...
├── scripts/
│   ├── extract_pdf_images.py
│   └── prepare_paper_dataset.py
├── data/                   # See data/README.md
├── outputs/validation/     # Pipeline results vs paper
├── tests/
└── docs/
```

## Local setup

```bash
git clone https://github.com/fayh601-glitch/TEM-analysis.git
cd TEM-analysis
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
pytest
```

Optional: `pip install -r requirements-optional.txt` (Streamlit, Excel export, OCR).

## Run analysis

```bash
# Recommended tuned settings for Enright 2018 SI rods
tem-rods analyze \
  --image data/curated/s2_A_starting_rods.png \
  --scale-bar-nm 20 \
  --scale-bar-pixels 48 \
  --min-area 150 \
  --no-watershed
```

## Validation baseline

See `outputs/validation/paper_comparison.csv`. Panel S2A matches published length within ~3% using tuned parameters.

## Suggested next tasks

1. **Tune segmentation** for S2B and S2D (overlap / merged rods).
2. **Hand-label** 30 rods in Fiji → update `data/labels/manual_v1.csv`.
3. **Streamlit upload UI** (`app/streamlit_app.py`).
4. **Auto scale bar** in CLI via `--auto-scale-bar`.
5. **ML phase** only if classical CV fails on your own TEM images.

## Branch workflow

```bash
git checkout -b feature/your-change
# edit, pytest, commit
git push -u origin feature/your-change
# open PR on GitHub
```

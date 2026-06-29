# Development guide

For a non-technical overview of the pipeline, see [HOW_IT_WORKS.md](HOW_IT_WORKS.md).

## Repo structure

```
TEM-analysis/
├── src/tem_rods/           # Core Python package — see src/tem_rods/README.md
│   ├── cli.py              # `tem-rods analyze` entry point
│   ├── pipeline.py         # End-to-end orchestration + overlay export
│   ├── segment.py          # Particle segmentation + quality filters
│   ├── classify.py         # Rod / dot / reject
│   ├── measure.py          # Length / width in nm
│   ├── scale_bar.py        # Auto-detect 20 nm scale bars
│   └── ...
├── scripts/                # See scripts/README.md
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

## Segmentation quality filters (v0.2)

Default `AnalysisConfig` values in `models.py`:

| Parameter | Default | Purpose |
|---|---|---|
| `mask_bottom_fraction` | 0.10 | Ignore scale-bar strip |
| `min_local_contrast` | 0.025 | Particle must be darker than nearby background |
| `min_solidity` | 0.48 | Reject hollow / irregular blobs |
| `min_extent` | 0.18 | Reject sparse blobs in bounding box |

Rejected particles appear in CSV with `class=reject` but are omitted from overlay PNGs.

## Validation baseline

See `outputs/validation/paper_comparison.csv`. Re-run demo after changes:

```bash
bash scripts/run_demo.sh
```

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

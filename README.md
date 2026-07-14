# CdSe/CdS TEM Rod & Dot Analyzer

Classical computer-vision pipeline for **transmission electron microscopy (TEM)**
images of **CdSe** and **CdS** nanoparticles. Detects particles, measures
**length** and **width** in nanometers, and classifies shape.

**Reference paper:** Enright et al., *Mater. Chem. Front.* **2018** — [DOI: 10.1039/c8qm00056e](https://doi.org/10.1039/c8qm00056e)

> **New here?** Read [Getting started](docs/GETTING_STARTED.md) or open the [PDF manual](docs/TEM-analysis-Manual.pdf).

## Quick start (no coding)

```bash
git clone https://github.com/fayh601-glitch/TEM-analysis.git
cd TEM-analysis
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt && pip install -e .

bash scripts/analyze_my_image.sh
```

Or use the **web upload** (after optional deps):

```bash
pip install -r requirements-optional.txt
streamlit run app/streamlit_app.py
```

After Analyze, click numbered markers on the overlay to **keep (green)** or **discard (red)** particles, then download the approved CSV.

On the website home screen you **pick Rods or Dots** and enter the **scale bar (nm + pixels)** before uploading.

## Public website + QR code

**Live app:** https://tem-analysis-pinc.streamlit.app/

Deploy / update notes: **[docs/DEPLOY_WEBSITE.md](docs/DEPLOY_WEBSITE.md)**

Generate a QR for the public URL:

```bash
pip install 'qrcode[pil]'
python scripts/make_app_qr.py "https://tem-analysis-pinc.streamlit.app"
```

Repo: https://github.com/fayh601-glitch/TEM-analysis

## Quick start (Terminal)

```bash
tem-rods analyze \
  --image data/curated/s2_A_starting_rods.png \
  --preset enright_rods \
  --mode rods \
  --scale-bar-nm 20 --scale-bar-pixels 48 \
  --output-dir outputs/demo
```

## Analysis modes

| Mode | Flag | Use when |
|------|------|----------|
| Nanorods only | `--mode rods` | Enright S2A and other rod-only samples |
| Dots only | `--mode dots` | Spherical QD samples |
| Both | `--mode both` | Mixed or unknown samples |

Presets: `enright_rods`, `dots_only`, `screenshot`, `sparse_cluster`

## Repository layout

```
TEM-analysis/
├── app/streamlit_app.py       ← Web upload UI
├── src/tem_rods/              ← Main Python package
├── scripts/analyze_my_image.sh ← Guided analysis (beginners)
├── data/curated/              ← Enright S2 TEM panels
├── outputs/                   ← CSV + overlay PNG results
├── docs/
│   ├── GETTING_STARTED.md     ← Start here
│   ├── MANUAL.md              ← Full reference
│   └── TEM-analysis-Manual.pdf
└── tests/
```

## Overlay legend

| Visual | Meaning |
|--------|---------|
| Green solid | Rod |
| Blue solid | Dot |
| Orange dashed | Reject (ambiguous / filtered) |
| `*_segments_debug.png` | All blobs before classification |

## Documentation

| Document | Audience |
|----------|----------|
| [Getting started](docs/GETTING_STARTED.md) | First-time users |
| [PDF manual](docs/TEM-analysis-Manual.pdf) | Complete repo reference |
| [How it works](docs/HOW_IT_WORKS.md) | Pipeline in plain English |
| [Source guide](src/tem_rods/README.md) | Which file does what |
| [Development](docs/DEVELOPMENT.md) | Tuning and contributing |
| [Labeling guide](docs/LABELING_GUIDE.md) | Fiji hand measurements for validation |
| [AI workflow](docs/AI_WORKFLOW.md) | Prompting AI for validation sprints |

## Validation status

Pipeline output is compared to **published mean dimensions** from Enright et al.
2018 (see `outputs/validation/paper_comparison.csv`). This is **not** the same
as rigorous per-rod validation.

| Panel | Length error | Width error | Status |
|-------|-------------|-------------|--------|
| S2A starting rods | ~3% | ~62% | Length OK; width overestimated |
| S2B 30 min | ~49% | ~132% | Fails — merged/over-segmented rods |
| S2D 65 min | ~47% | ~15% | Fails — under-counts long rods |

`data/labels/manual_v1.csv` is **semi-automated pipeline output**
(`semi_auto_tuned_pipeline` in the `notes` column), not independent hand labels.
See [data/labels/README.md](data/labels/README.md) and
[docs/LABELING_GUIDE.md](docs/LABELING_GUIDE.md) for creating real Fiji labels.

## License

Academic / research use. TEM reference images derived from Enright et al. 2018 supplementary information.

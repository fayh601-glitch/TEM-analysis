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

## License

Academic / research use. TEM reference images derived from Enright et al. 2018 supplementary information.

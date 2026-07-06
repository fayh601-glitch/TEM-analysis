# Getting started (no coding experience required)

This guide gets you from **zero** to an analyzed TEM image in about 10 minutes.

## What you need

- A Mac with **Python 3.9+** (already on most lab computers)
- A **TEM image** file (PNG, JPG, or TIFF)
- The **scale bar** value from the image (e.g. `20 nm`)

## Step 1 — Download the project

```bash
git clone https://github.com/fayh601-glitch/TEM-analysis.git
cd TEM-analysis
```

## Step 2 — One-time install

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

You only do this once per computer.

## Step 3 — Analyze your image (easiest way)

```bash
bash scripts/analyze_my_image.sh
```

The script will ask you:

1. **Path to your image** (drag the file into Terminal)
2. **Particle type** — choose **nanorods only** for Enright-style rod samples
3. **Scale bar in nm** (usually `20`)
4. **Scale bar in pixels** — measure the white line length in your image

Results appear in `outputs/user_runs/`.

## Step 4 — Open your results

| File | What it is |
|------|------------|
| `*_overlay.png` | Your image with colored outlines and size labels |
| `*_measurements.csv` | Spreadsheet — open in Excel or Google Sheets |
| `*_segments_debug.png` | Every blob found **before** classification (for troubleshooting) |

### Overlay colors

| Color | Meaning |
|-------|---------|
| Green | Nanorod |
| Blue | Round dot (only in "both" mode) |
| Orange dashed | Reject — found but not confident |

## Step 5 — If results look wrong

1. **Wrong sizes?** Re-measure the scale bar in pixels.
2. **Missing rods?** Open `*_segments_debug.png` — if the rod has no number, segmentation missed it.
3. **Too many orange rejects?** Try `--preset screenshot` for phone/screenshot images.
4. **Rods-only sample?** Use `--mode rods` or preset `enright_rods` so round fragments are not called dots.

## Alternative: web upload (no Terminal commands after setup)

```bash
source .venv/bin/activate
pip install -r requirements-optional.txt
streamlit run app/streamlit_app.py
```

Your browser opens an upload page — drag in an image, enter the scale bar, click **Analyze**.

## Alternative: one Terminal command

```bash
tem-rods analyze \
  --image /path/to/my_tem.png \
  --preset enright_rods \
  --mode rods \
  --scale-bar-nm 20 \
  --scale-bar-pixels 45 \
  --output-dir outputs/my_run
```

Replace `45` with your measured scale bar pixels.

## Reading order for the codebase

If you want to understand the code later:

1. [README.md](../README.md) — project overview  
2. [HOW_IT_WORKS.md](HOW_IT_WORKS.md) — pipeline in plain English  
3. [src/tem_rods/README.md](../src/tem_rods/README.md) — which file does what  
4. [MANUAL.md](MANUAL.md) — full reference manual (PDF: `TEM-analysis-Manual.pdf`)  
5. `src/tem_rods/pipeline.py` — start reading code here  

## Analysis modes

| Mode | When to use |
|------|-------------|
| `rods` | Sample contains only nanorods (Enright S2A) |
| `dots` | Sample contains only spherical QDs |
| `both` | Mixed or unknown sample |

Modes live in **one codebase** — you do not need separate git branches.

## Help

- Full manual: [docs/TEM-analysis-Manual.pdf](TEM-analysis-Manual.pdf)  
- Paper reference: Enright et al. 2018 — [DOI: 10.1039/c8qm00056e](https://doi.org/10.1039/c8qm00056e)

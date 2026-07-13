# Scripts folder

Helper programs you run from the terminal. They are **not** imported by the main
Python package — they set up data or run batch analyses.

| Script | What it does |
|---|---|
| [`analyze_my_image.sh`](analyze_my_image.sh) | **Start here** — guided prompts for beginners (no flags to remember) |
| [`run_demo.sh`](run_demo.sh) | Analyze all three curated paper images and save results to `outputs/demo/` |
| [`build_manual_pdf.py`](build_manual_pdf.py) | Regenerate `docs/TEM-analysis-Manual.pdf` from `docs/MANUAL.md` |
| [`extract_pdf_images.py`](extract_pdf_images.py) | Pull embedded TEM images out of a supplementary PDF into `data/raw/` |
| [`prepare_paper_dataset.py`](prepare_paper_dataset.py) | Rebuild curated images, calibration CSV, semi-auto labels, validation comparison |
| [`compare_manual_labels.py`](compare_manual_labels.py) | Compare pipeline counts/lengths/widths to label CSV (default: `manual_v1.csv`) |
| [`make_app_qr.py`](make_app_qr.py) | Make a QR code PNG for the public Streamlit website URL |

## Related docs

| Doc | Purpose |
|---|---|
| [docs/LABELING_GUIDE.md](../docs/LABELING_GUIDE.md) | Fiji hand-measurement workflow |
| [docs/AI_WORKFLOW.md](../docs/AI_WORKFLOW.md) | How to prompt AI for validation sprints |
| [data/labels/README.md](../data/labels/README.md) | What each label file is |

## Typical workflow

```bash
# 1. Extract images from the paper PDF (once)
python scripts/extract_pdf_images.py /path/to/paper_SI.pdf -o data/raw

# 2. Build curated dataset + validation (optional, needs raw images)
python scripts/prepare_paper_dataset.py

# 3. Quick demo on committed curated images
bash scripts/run_demo.sh
```

# Scripts folder

Helper programs you run from the terminal. They are **not** imported by the main
Python package — they set up data or run batch analyses.

| Script | What it does |
|---|---|
| [`run_demo.sh`](run_demo.sh) | Analyze all three curated paper images and save results to `outputs/demo/` |
| [`extract_pdf_images.py`](extract_pdf_images.py) | Pull embedded TEM images out of a supplementary PDF into `data/raw/` |
| [`prepare_paper_dataset.py`](prepare_paper_dataset.py) | Rebuild curated images, calibration CSV, labels, and validation comparison |

## Typical workflow

```bash
# 1. Extract images from the paper PDF (once)
python scripts/extract_pdf_images.py /path/to/paper_SI.pdf -o data/raw

# 2. Build curated dataset + validation (optional, needs raw images)
python scripts/prepare_paper_dataset.py

# 3. Quick demo on committed curated images
bash scripts/run_demo.sh
```

# Data directory

Images, calibration tables, and label files used as **inputs** to the analysis
pipeline. Nothing in this folder is Python code — it is the raw material the
program reads when you run `tem-rods analyze`.

| Path | Purpose |
|---|---|
| `raw/` | Extracted SI images from PDF (local only, gitignored). Run `scripts/extract_pdf_images.py`. |
| `curated/` | Cropped TEM panels ready for analysis (Figure S2 from Enright 2018). |
| `calibration.csv` | Scale bar values and paper reference dimensions per image. |
| `calibration.example.csv` | Template for new images. |
| `labels/manual_v1.csv` | Semi-automated pipeline output (`notes=semi_auto_tuned_pipeline`). **Not** hand labels — see [labels/README.md](labels/README.md). |
| `labels/manual_v1.example.csv` | Legacy column template |
| `labels/manual_v2.template.csv` | Template for Fiji hand measurements |
| `labels/README.md` | What each label file is and validation status |

## Curated panels (Enright 2018 SI Figure S2)

| File | Figure | Paper length × width (nm) |
|---|---|---|
| `s2_A_starting_rods.png` | S2A | 27.6 × 2.9 |
| `s2_B_30min.png` | S2B | 44.4 × 4.4 |
| `s2_D_65min.png` | S2D | 99.3 × 5.9 |
| `enright_s2A_user_screenshot.png` | S2A (user screenshot) | 27.6 × 2.9 |

Regenerate calibration, labels, and validation with:

```bash
python scripts/prepare_paper_dataset.py
```

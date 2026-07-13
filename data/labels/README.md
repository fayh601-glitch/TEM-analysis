# Label files

Particle dimension tables used for validation and comparison. **None of these
files are independent human ground truth** unless `notes` says otherwise.

| File | What it is | Status |
|---|---|---|
| `manual_v1.csv` | Semi-automated pipeline output from `scripts/prepare_paper_dataset.py` using the `enright_rods` preset. Every row has `notes=semi_auto_tuned_pipeline`. | Reference for regression only — **not** hand-measured |
| `manual_v1.example.csv` | Column template with two example rows | Template |
| `manual_v2.template.csv` | Blank template for Fiji hand measurements | **Use this** for real validation labels |

## `manual_v1.csv` — important caveat

This file was historically described as "manual labels" or "ground truth." It is
**not**. It is the output of running the tuned pipeline on Enright 2018 SI
Figure S2 panels and saving each detected rod's ellipse fit.

Do **not** use `manual_v1.csv` to claim the pipeline is validated. Comparing the
pipeline to its own prior output (`compare_manual_labels.py`) only checks
reproducibility, not accuracy.

## Creating real labels (`manual_v2`)

1. Copy `manual_v2.template.csv` to `manual_v2.csv` (do not commit personal drafts
   until reviewed).
2. Follow [docs/LABELING_GUIDE.md](../../docs/LABELING_GUIDE.md) for Fiji steps.
3. Measure 30–50 rods per panel on S2A, S2B, and S2D.
4. Set `measured_by` to your name and leave `notes` blank or describe the tool
   (e.g. `fiji_line_tool`).

## Column reference

| Column | Description |
|---|---|
| `particle_id` | Sequential ID within the image (1, 2, 3, …) |
| `image` | Filename under `data/curated/` |
| `class` | `rod`, `dot`, or `reject` |
| `length_nm` | Major axis in nm |
| `width_nm` | Minor axis in nm |
| `measured_by` | Person who measured (v2 only) |
| `notes` | `semi_auto_tuned_pipeline` for v1; measurement method for v2 |

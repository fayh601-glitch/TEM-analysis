# Fiji / ImageJ hand-labeling guide

Create **independent** rod measurements for validation. The pipeline cannot
validate itself — you need human-measured lengths and widths on the same TEM
panels.

**Target:** 30–50 rods per panel on S2A, S2B, and S2D (`data/curated/`).

**Output:** `data/labels/manual_v2.csv` (copy from `manual_v2.template.csv`).

---

## 1. Install Fiji

Download [Fiji](https://fiji.sc/) (ImageJ bundled with plugins). Open Fiji once
to confirm it launches.

---

## 2. Set scale (calibration)

Use values from `data/calibration.csv`:

| Image | Scale bar (nm) | Scale bar (pixels) |
|---|---|---|
| `s2_A_starting_rods.png` | 20 | 48 |
| `s2_B_30min.png` | 20 | 48 |
| `s2_D_65min.png` | 20 | 52 |

For each image:

1. **File → Open** → select the curated PNG.
2. Draw a line along the white scale bar with the **straight line** tool.
3. **Analyze → Set Scale…**
   - *Distance in pixels:* length of your line (e.g. 48).
   - *Known distance:* 20
   - *Unit of length:* nm
   - Check **Global** if you will measure many rods in this image.
4. Click OK.

Re-check: measuring a line the same length as the scale bar should read ~20 nm.

---

## 3. Measure one rod

1. Select the **straight line** tool.
2. Draw a line along the rod's **long axis** (length).
3. **Analyze → Measure** (or `Ctrl+M` / `Cmd+M`).
4. Note the **Length** value in nm → this is `length_nm`.
5. Draw a second line perpendicular to the first along the **short axis**
   (width) through the rod center.
6. **Analyze → Measure** again → this is `width_nm`.

**Tips:**

- Measure clearly separated rods first; skip heavily overlapping clusters.
- If a rod is tilted, length = longest visible axis; width = shortest.
- Skip ambiguous blobs — do not force a label.

---

## 4. Record in CSV

Open `manual_v2.template.csv` (save as `manual_v2.csv`). One row per rod:

```csv
particle_id,image,class,length_nm,width_nm,measured_by,notes
1,s2_A_starting_rods.png,rod,28.4,3.1,YourName,fiji_line_tool
2,s2_A_starting_rods.png,rod,26.1,2.8,YourName,fiji_line_tool
```

- Increment `particle_id` per image (restart at 1 for each new image).
- Use `class=rod` for elongated particles you would count in the paper.
- Set `measured_by` to your name or initials.

---

## 5. Panel workflow

Repeat sections 2–4 for each panel:

| Panel | File | Paper mean L × W (nm) | Target count |
|---|---|---|---|
| S2A | `s2_A_starting_rods.png` | 27.6 × 2.9 | 30–50 rods |
| S2B | `s2_B_30min.png` | 44.4 × 4.4 | 30–50 rods |
| S2D | `s2_D_65min.png` | 99.3 × 5.9 | 30–50 rods |

Work panel-by-panel: calibrate once per image, then measure rods before switching
files.

---

## 6. Compare to pipeline (after labeling)

```bash
python scripts/compare_manual_labels.py \
  --labels data/labels/manual_v2.csv \
  --preset enright_rods
```

Review mean length and width deltas in `outputs/validation/manual_comparison.csv`.

---

## 7. Quality checks

Before sharing labels:

- [ ] Scale bar calibrated per image (not copied from another panel).
- [ ] 30+ rods per panel with visible, separable outlines.
- [ ] Length > width for every rod row.
- [ ] No duplicate `particle_id` within the same `image`.
- [ ] `notes` documents method (`fiji_line_tool`).

---

## Why this matters

`manual_v1.csv` is **pipeline output**, not hand labels. Only `manual_v2.csv`
(or equivalent Fiji measurements) can support claims that the analyzer matches
published dimensions.

# Enright S2A — user screenshot analysis

**Input:** `data/curated/enright_s2A_user_screenshot.png`  
**Paper reference:** Enright 2018 Figure S2A starting rods — 27.6 × 2.9 nm  
**Scale bar:** 20 nm (auto-detected at 45.0 px → 0.4444 nm/pixel)

## Results summary

| Metric | Pipeline | Paper |
|---|---|---|
| Rod count | 81 | — |
| Mean rod length | **22.8 ± 9.1 nm** | 27.6 nm |
| Mean rod width | **4.7 ± 1.2 nm** | 2.9 nm |
| Dots | 21 | — |
| Rejected | 46 | — |

## How this was generated

```bash
tem-rods analyze \
  --image data/curated/enright_s2A_user_screenshot.png \
  --auto-scale-bar \
  --scale-bar-nm 20 \
  --min-area 150 \
  --output-dir outputs/user_runs/enright_s2A_user_screenshot
```

## Output files

| File | Description |
|---|---|
| [`enright_s2A_user_screenshot_overlay.png`](enright_s2A_user_screenshot_overlay.png) | Annotated TEM — **open this first** |
| [`enright_s2A_user_screenshot_measurements.csv`](enright_s2A_user_screenshot_measurements.csv) | One row per detected particle |

## Overlay legend

| Visual | Meaning |
|---|---|
| Solid green contour | Detected rod boundary |
| Dashed green ellipse | Fitted length/width axes |
| Blue | Dot (round particle) |

Rejected particles appear in the CSV only (not drawn on the overlay).

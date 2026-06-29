# User run outputs

Analysis results for TEM images uploaded or screenshotted outside the paper
curated set. Each subfolder is named after its input image.

| Folder | Input | Description |
|---|---|---|
| [`enright_s2A_user_screenshot/`](enright_s2A_user_screenshot/) | `data/curated/enright_s2A_user_screenshot.png` | User screenshot of Enright 2018 Figure S2A (starting rods) |

## Re-run

```bash
tem-rods analyze \
  --image data/curated/enright_s2A_user_screenshot.png \
  --auto-scale-bar \
  --min-area 150 \
  --output-dir outputs/user_runs/enright_s2A_user_screenshot
```

## Files in each run folder

| File | Description |
|---|---|
| `*_overlay.png` | Annotated image — green contours = rods, blue = dots |
| `*_measurements.csv` | Per-particle length, width, class, and position |

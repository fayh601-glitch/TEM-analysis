# TEM cluster — user screenshot (200 nm scale bar)

**Input:** `data/curated/user_tem_cluster_200nm.png`  
**Scale bar:** 200 nm at 248 px → 0.8065 nm/pixel (auto-detected from bottom-left bar)

## Results summary

| Metric | Value |
|---|---|
| Total particles | 90 |
| Rods | 34 |
| Dots | 24 |
| Rejected | 32 |
| Rod mean length | 34.0 ± 12.4 nm |
| Rod mean width | 10.2 ± 3.7 nm |

## Re-run

```bash
tem-rods analyze \
  --image data/curated/user_tem_cluster_200nm.png \
  --scale-bar-nm 200 \
  --scale-bar-pixels 248 \
  --min-area 150 \
  --output-dir outputs/user_runs/user_tem_cluster_200nm
```

If nm sizes look off, measure the scale bar line in pixels manually and pass `--scale-bar-pixels N`.

## Output files

| File | Description |
|---|---|
| [`user_tem_cluster_200nm_overlay.png`](user_tem_cluster_200nm_overlay.png) | Annotated image — start here |
| [`user_tem_cluster_200nm_measurements.csv`](user_tem_cluster_200nm_measurements.csv) | Per-particle measurements |

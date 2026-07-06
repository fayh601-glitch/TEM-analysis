# How the analysis works

This guide explains the pipeline in plain language — no programming background
required.

## What this project does

You give it a **TEM image** (a grayscale photo of nanoparticles). It:

1. Finds dark blobs that might be particles
2. Measures how long and wide each blob is (in **nanometers**)
3. Labels each one as a **rod**, **dot**, or **reject** (background noise)
4. Saves a **CSV spreadsheet** and an **annotated PNG overlay**

## Step-by-step pipeline

### 1. Load the image (`io.py`)

The program opens your PNG/TIFF/JPG and converts it to a grid of brightness
values (0 = black, 1 = white).

### 2. Calibrate the scale (`calibrate.py`)

TEM images include a scale bar (e.g. "20 nm"). You tell the program how many
**pixels** that bar spans, and it calculates **nm per pixel** so all sizes are
real physical measurements.

### 3. Preprocess (`preprocess.py`)

The image is contrast-normalized and lightly blurred. This reduces film grain
noise so the next step does not treat every speck as a particle.

### 4. Segment — find particles (`segment.py`)

This is the most important step:

- **Threshold:** Everything darker than the average becomes a candidate particle
- **Cleanup:** Tiny specks are removed; nearby blobs may be merged
- **Scale-bar mask:** The bottom strip (where the scale bar lives) is ignored
- **Quality filters:** Blobs are dropped if they are:
  - Too faint compared to nearby background (**local contrast**)
  - Too hollow (**solidity** — area vs convex hull)
  - Too sparse in their bounding box (**extent**)

What survives gets a numbered label.

### 5. Measure (`measure.py`)

For each labeled blob, the program computes:

| Quantity | How it is measured |
|---|---|
| **Length** | Long axis of fitted ellipse |
| **Width** | Short axis of fitted ellipse |
| **Area** | Pixel count × (nm/pixel)² |
| **Eccentricity** | How elongated the shape is (0 = circle, 1 = line) |

### 6. Classify (`classify.py`)

Each blob gets one of three labels (then filtered by **analysis mode**):

| Class | Criteria |
|---|---|
| **Rod** | High eccentricity **and** aspect ratio ≥ threshold |
| **Dot** | Low eccentricity **and** roughly round |
| **Reject** | Ambiguous — or wrong shape for the selected mode |

**Analysis modes** (set with `--mode`):

| Mode | Behavior |
|------|----------|
| `rods` | Round blobs become rejects (for rod-only samples) |
| `dots` | Elongated blobs become rejects (for dot-only samples) |
| `both` | Report both rods and dots |

Rejected blobs appear on the overlay in **orange** when enabled (default on).

### 7. Export (`pipeline.py`)

- **CSV:** One row per particle with size, class, and position
- **Overlay PNG:** Solid contour = actual detected shape; dashed ellipse = fitted
  axes; green = rod, blue = dot

## Reading the overlay

```
  ─────────────  solid green contour  =  actual detected particle boundary
  - - - - - - -  dashed ellipse       =  fitted length/width axes
  R 28.0×3.5 nm  text label           =  class + size
  orange dashed  contour              =  reject (ambiguous)
```

## Folder map (whole repo)

```
TEM-analysis/
├── src/tem_rods/       ← main Python code (see src/tem_rods/README.md)
├── scripts/            ← helper scripts to run batches (see scripts/README.md)
├── data/               ← input images and calibration tables
├── outputs/            ← results (CSVs and overlay PNGs)
├── tests/              ← automatic checks that code still works
└── docs/               ← guides like this one
```

## When results look wrong

Common causes:

1. **Wrong scale bar** → all nm sizes will be off (positions still OK)
2. **Background texture** → may still produce reject entries in CSV
3. **Overlapping rods** → may merge into one blob without watershed splitting

Tune `--min-area` and see [DEVELOPMENT.md](DEVELOPMENT.md) for advanced options.

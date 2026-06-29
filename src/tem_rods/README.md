# Source code — `src/tem_rods/`

This folder is the **main Python package**. Each file has one job in the analysis
chain. Open any file and read the comment block at the top for a plain-English
summary.

## How the files connect

```
cli.py  ──►  pipeline.py  ──►  preprocess.py  ──►  segment.py
                    │                                    │
                    │                                    ▼
                    │                              measure.py
                    │                                    │
                    │                                    ▼
                    │                              classify.py
                    │
                    ├── io.py          (load images)
                    ├── calibrate.py   (pixels → nm)
                    └── scale_bar.py   (auto-find scale bar)
```

## File guide

| File | Role |
|---|---|
| [`cli.py`](cli.py) | Terminal command (`tem-rods analyze`) — start here if you use the CLI |
| [`pipeline.py`](pipeline.py) | Runs the full workflow and saves CSV + overlay PNG |
| [`io.py`](io.py) | Opens image files (PNG, TIFF, JPG) |
| [`preprocess.py`](preprocess.py) | Normalizes brightness and reduces noise |
| [`segment.py`](segment.py) | Finds dark particle blobs; filters background noise |
| [`measure.py`](measure.py) | Computes length, width, and shape numbers |
| [`classify.py`](classify.py) | Labels each blob as **rod**, **dot**, or **reject** |
| [`models.py`](models.py) | Shared settings and result data types |
| [`calibrate.py`](calibrate.py) | Converts scale-bar info to nm/pixel |
| [`scale_bar.py`](scale_bar.py) | Automatically detects the scale bar in an image |
| [`__init__.py`](__init__.py) | Package version marker |

## Particle classes

| Class | Meaning |
|---|---|
| **rod** | Elongated nanoparticle (nanorod) |
| **dot** | Roughly round nanoparticle |
| **reject** | Detected blob that failed shape checks — shown in CSV but not on overlay |

## Further reading

- [How it works (beginner guide)](../docs/HOW_IT_WORKS.md)
- [Development guide](../docs/DEVELOPMENT.md)

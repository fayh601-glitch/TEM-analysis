"""
Freehand TEM tracer — draw directly on the photo.

Custom Streamlit component (HTML/JS). The TEM image is painted onto the same
canvas you draw on, so tracing is on the actual micrograph — not a blank box.
"""

from __future__ import annotations

import base64
from io import BytesIO
from pathlib import Path

import numpy as np
import streamlit.components.v1 as components
from PIL import Image

_COMPONENT_DIR = Path(__file__).resolve().parent / "freehand_frontend"

_freehand = components.declare_component(
    "tem_freehand_trace",
    path=str(_COMPONENT_DIR),
)


def _to_display_jpeg(
    image: np.ndarray | Image.Image,
    *,
    max_width: int = 900,
) -> tuple[str, int, int]:
    if not isinstance(image, Image.Image):
        arr = np.asarray(image)
        if arr.ndim == 2:
            if arr.dtype != np.uint8:
                arr = (
                    (np.clip(arr, 0, 1) * 255).astype(np.uint8)
                    if arr.max() <= 1.5
                    else np.clip(arr, 0, 255).astype(np.uint8)
                )
            image = Image.fromarray(arr, mode="L").convert("RGB")
        else:
            if arr.dtype != np.uint8:
                arr = (
                    (np.clip(arr, 0, 1) * 255).astype(np.uint8)
                    if arr.max() <= 1.5
                    else np.clip(arr, 0, 255).astype(np.uint8)
                )
            image = Image.fromarray(arr).convert("RGB")

    scale = min(1.0, max_width / float(image.width))
    w = max(1, int(round(image.width * scale)))
    h = max(1, int(round(image.height * scale)))
    if image.size != (w, h):
        image = image.resize((w, h), getattr(Image, "Resampling", Image).BILINEAR)

    buf = BytesIO()
    image.save(buf, format="JPEG", quality=85)
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{b64}", w, h


def freehand_trace_on_image(
    image: np.ndarray | Image.Image,
    *,
    stroke_width: int = 3,
    stroke_color: str = "#ff3333",
    clear_token: int = 0,
    key: str | None = None,
    max_width: int = 900,
) -> np.ndarray | None:
    """
    Interactive freehand tracer drawn on top of ``image``.

    Returns RGBA stroke layer at display resolution, or None if nothing drawn yet.
    """
    data_url, w, h = _to_display_jpeg(image, max_width=max_width)
    result = _freehand(
        image_url=data_url,
        width=w,
        height=h,
        stroke_width=int(stroke_width),
        stroke_color=stroke_color,
        clear_token=int(clear_token),
        key=key,
        default=None,
    )
    if not result or not result.get("png"):
        return None
    raw = base64.b64decode(str(result["png"]).split(",")[-1])
    return np.array(Image.open(BytesIO(raw)).convert("RGBA"))

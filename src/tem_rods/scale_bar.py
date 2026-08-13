"""
Scale Bar Detector — automatically find the scale bar in a TEM image
=====================================================================

Published TEM figures usually place a dark or bright horizontal scale bar near
the bottom of the micrograph. This file finds the thin bar line (not the
"200 nm" label text), estimates pixel length, and optionally reads the nm
value from the label or filename.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image
from skimage.measure import label, regionprops

from tem_rods.calibrate import (
    nm_per_pixel_from_cal_text,
    nm_per_pixel_from_scale_bar,
    validate_nm_per_pixel,
)
from tem_rods.preprocess import detect_bottom_info_banner


@dataclass(frozen=True)
class ScaleBarDetection:
    """Result of automatic scale bar detection."""

    bar_pixels: float
    bar_nm: float
    nm_per_pixel: float
    bbox: tuple[int, int, int, int]
    confidence: float = 1.0
    polarity: str = "dark"  # "dark" or "bright"
    in_info_banner: bool = False


def detect_scale_bar(
    image_path: str | Path,
    *,
    scale_bar_nm: float | None = None,
    search_bottom_fraction: float = 0.25,
    search_left_fraction: float = 0.55,
    max_dark_value: float | None = None,
    min_bright_value: float | None = None,
    max_bar_height_px: int = 8,
    min_bar_width_px: int = 18,
    min_bar_aspect: float = 6.0,
    bbox_pad_px: int = 8,
    prefer: str = "auto",
) -> ScaleBarDetection:
    """
    Detect a thin horizontal scale bar in a TEM micrograph.

    Paper figures usually place a dark or bright bar near the bottom-left.
    AMT/Gatan camera exports put the bar in a white info strip at the bottom;
    that strip is searched at full width (including gray anti-aliased lines)
    before particle analysis crops it away. Pass ``scale_bar_nm`` with the
    printed label value (e.g. 50).
    """
    image_path = Path(image_path)
    im = np.array(Image.open(image_path).convert("L"), dtype=float)
    if im.max() > 1.5:
        im = im / 255.0

    h, w = im.shape
    banner_row = detect_bottom_info_banner(im)

    # AMT/Gatan footers put the 50 nm line in the white strip — often gray
    # (not pure black) and not in the bottom-left corner. Measure it there
    # *before* the strip is cropped for particle finding.
    if banner_row is not None and prefer in ("auto", "dark"):
        run_hit = _detect_dark_bar_by_runs(
            im[banner_row:],
            row_offset=banner_row,
            image_width=w,
            min_bar_width_px=min_bar_width_px,
        )
        if run_hit is not None:
            bar_pixels, bbox, polarity = run_hit
            bar_nm = scale_bar_nm
            if bar_nm is None:
                bar_nm = _parse_scale_bar_nm(image_path, im, bbox)
            if bar_nm is None:
                raise ValueError(
                    f"Could not determine scale bar length in nm for {image_path}. "
                    "Pass scale_bar_nm or use a filename like sample_200nm.png."
                )
            nm_per_pixel = validate_nm_per_pixel(
                validate_scale_bar_calibration(
                    bar_pixels,
                    bar_nm,
                    image_width=w,
                    max_bar_fraction=0.60,
                )
            )
            confidence = min(1.0, bar_pixels / min_bar_width_px / 3.0)
            return ScaleBarDetection(
                bar_pixels=bar_pixels,
                bar_nm=bar_nm,
                nm_per_pixel=nm_per_pixel,
                bbox=bbox,
                confidence=confidence,
                polarity=polarity,
                in_info_banner=True,
            )

    if banner_row is not None:
        # Instrument overlays put the scale bar in a full-width info strip.
        row_start = banner_row
        roi = im[row_start:, :]
    else:
        row_start = int(h * (1.0 - search_bottom_fraction))
        col_end = int(w * search_left_fraction)
        roi = im[row_start:, :col_end]

    dark_threshold = max_dark_value if max_dark_value is not None else (0.45 if banner_row is not None else 0.25)
    if max_dark_value is not None and max_dark_value > 1.5:
        dark_threshold = max_dark_value / 255.0
    bright_threshold = min_bright_value if min_bright_value is not None else 0.85
    if min_bright_value is not None and min_bright_value > 1.5:
        bright_threshold = min_bright_value / 255.0

    search_plan: list[tuple[str, np.ndarray]] = []
    if prefer in ("auto", "dark"):
        search_plan.append(("dark", roi < dark_threshold))
    if prefer in ("auto", "bright"):
        search_plan.append(("bright", roi > bright_threshold))
    if prefer == "bright":
        search_plan = [("bright", roi > bright_threshold), ("dark", roi < dark_threshold)]

    max_width_frac = 0.60 if banner_row is not None else 0.4
    candidates: list[tuple[float, object, float, str]] = []
    for polarity, binary in search_plan:
        for score, region, bar_width in _horizontal_bar_candidates(
            binary,
            row_start=row_start,
            image_width=w,
            max_bar_height_px=16 if banner_row is not None else max_bar_height_px,
            min_bar_width_px=min_bar_width_px,
            min_bar_aspect=min_bar_aspect,
            max_width_fraction=max_width_frac,
        ):
            # Prefer longer, thinner bars slightly in the lower part of the ROI.
            candidates.append((score, region, bar_width, polarity))

    if not candidates:
        raise ValueError(f"Could not detect scale bar in {image_path}")

    _, best_region, bar_pixels, polarity = max(candidates, key=lambda item: item[0])
    min_row, min_col, max_row, max_col = best_region.bbox
    bbox = (
        max(0, row_start + min_row - bbox_pad_px),
        max(0, min_col - bbox_pad_px),
        min(h, row_start + max_row + bbox_pad_px),
        min(w, max_col + bbox_pad_px),
    )

    bar_nm = scale_bar_nm
    if bar_nm is None:
        bar_nm = _parse_scale_bar_nm(image_path, im, bbox)
    if bar_nm is None:
        raise ValueError(
            f"Could not determine scale bar length in nm for {image_path}. "
            "Pass scale_bar_nm or use a filename like sample_200nm.png."
        )

    nm_per_pixel = validate_nm_per_pixel(
        validate_scale_bar_calibration(bar_pixels, bar_nm, image_width=w)
    )
    confidence = min(1.0, bar_pixels / min_bar_width_px / 3.0)

    return ScaleBarDetection(
        bar_pixels=bar_pixels,
        bar_nm=bar_nm,
        nm_per_pixel=nm_per_pixel,
        bbox=bbox,
        confidence=confidence,
        polarity=polarity,
        in_info_banner=banner_row is not None,
    )


def _horizontal_bar_candidates(
    binary: np.ndarray,
    *,
    row_start: int,
    image_width: int,
    max_bar_height_px: int,
    min_bar_width_px: int,
    min_bar_aspect: float,
    max_width_fraction: float = 0.4,
) -> list[tuple[float, object, float]]:
    labels = label(binary)
    candidates: list[tuple[float, object, float]] = []

    for region in regionprops(labels):
        min_row, min_col, max_row, max_col = region.bbox
        height = max_row - min_row
        width = max_col - min_col
        if height > max_bar_height_px:
            continue
        if width < min_bar_width_px:
            continue
        if width / max(height, 1) < min_bar_aspect:
            continue
        if width > image_width * max_width_fraction:
            continue

        row_scores = []
        for rel_row in range(min_row, max_row):
            row = binary[rel_row]
            dark = np.where(row)[0]
            if len(dark) < min_bar_width_px:
                continue
            breaks = np.where(np.diff(dark) > 2)[0]
            starts = [0] + (breaks + 1).tolist()
            ends = breaks.tolist() + [len(dark) - 1]
            row_scores.append(max(float(dark[end] - dark[start]) for start, end in zip(starts, ends)))

        bar_width = float(np.median(row_scores)) if row_scores else float(width)
        if bar_width < min_bar_width_px:
            continue

        abs_row = row_start + (min_row + max_row) / 2.0
        score = bar_width + 0.25 * abs_row + 0.02 * (width / max(height, 1))
        candidates.append((score, region, bar_width))

    return candidates


def _longest_true_run(row: np.ndarray) -> tuple[int, int, int]:
    """Return (length, start, end_exclusive) of the longest contiguous True run."""
    if row.size == 0 or not np.any(row):
        return 0, 0, 0
    padded = np.concatenate([[False], np.asarray(row, dtype=bool), [False]])
    diff = np.diff(padded.astype(np.int8))
    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0]
    lengths = ends - starts
    idx = int(np.argmax(lengths))
    return int(lengths[idx]), int(starts[idx]), int(ends[idx])


def _detect_dark_bar_by_runs(
    roi: np.ndarray,
    *,
    row_offset: int,
    image_width: int,
    min_bar_width_px: int = 18,
    max_bar_fraction: float = 0.55,
    max_thickness_px: int = 12,
) -> tuple[float, tuple[int, int, int, int], str] | None:
    """
    Find a thin dark horizontal line in a bright info bar by scanning rows.

    Connected-component detection misses AMT bars that are anti-aliased gray
    or sit next to filename/Cal text. A scale bar is the longest dark run that
    is not a full-width rule and is only a few pixels thick.
    """
    if roi.size == 0:
        return None
    med = float(np.median(roi))
    dark_threshold = max(0.18, min(0.55, med * 0.62))
    binary = roi < dark_threshold
    h, _w = binary.shape
    max_width = int(image_width * max_bar_fraction)

    row_runs = [_longest_true_run(binary[r]) for r in range(h)]
    candidates: list[tuple[float, float, int, int, int, int]] = []
    for row, (length, start, end) in enumerate(row_runs):
        if length < min_bar_width_px or length > max_width:
            continue
        top = row
        bottom = row + 1
        for rr in range(row - 1, -1, -1):
            other_len, other_s, other_e = row_runs[rr]
            if other_len >= 0.82 * length and other_e > start and other_s < end:
                top = rr
            else:
                break
        for rr in range(row + 1, h):
            other_len, other_s, other_e = row_runs[rr]
            if other_len >= 0.82 * length and other_e > start and other_s < end:
                bottom = rr + 1
            else:
                break
        thickness = bottom - top
        if thickness > max_thickness_px:
            continue
        score = float(length) - 1.5 * thickness
        candidates.append((score, float(length), top, bottom, start, end))

    if not candidates:
        return None
    _score, bar_pixels, top, bottom, start, end = max(candidates, key=lambda item: item[0])
    bbox = (
        row_offset + top,
        start,
        row_offset + bottom,
        end,
    )
    return bar_pixels, bbox, "dark"


def detect_scale_bar_pixels(
    image_path: str | Path,
    *,
    scale_bar_nm: float = 20.0,
    search_bottom_fraction: float = 0.22,
    search_left_fraction: float = 0.4,
    max_dark_value: float = 60.0,
) -> tuple[float, float]:
    """
    Backward-compatible helper returning (scale_bar_pixels, nm_per_pixel).
    """
    detection = detect_scale_bar(
        image_path,
        scale_bar_nm=scale_bar_nm,
        search_bottom_fraction=search_bottom_fraction,
        search_left_fraction=search_left_fraction,
        max_dark_value=max_dark_value,
    )
    return detection.bar_pixels, detection.nm_per_pixel


def validate_scale_bar_calibration(
    bar_pixels: float,
    bar_nm: float,
    *,
    image_width: int,
    min_bar_pixels: float = 12.0,
    max_bar_fraction: float = 0.45,
    min_nm_per_pixel: float = 0.02,
    max_nm_per_pixel: float = 8.0,
) -> float:
    """Validate detected scale bar and return nm/pixel."""
    if bar_pixels < min_bar_pixels:
        raise ValueError(f"Scale bar too short ({bar_pixels:.1f} px)")
    if bar_pixels > image_width * max_bar_fraction:
        raise ValueError(f"Scale bar too long ({bar_pixels:.1f} px)")
    nm_per_pixel = nm_per_pixel_from_scale_bar(bar_nm, bar_pixels)
    if nm_per_pixel < min_nm_per_pixel or nm_per_pixel > max_nm_per_pixel:
        raise ValueError(
            f"Implausible calibration {nm_per_pixel:.4f} nm/pixel "
            f"({bar_nm:g} nm / {bar_pixels:.1f} px)"
        )
    return nm_per_pixel


def _parse_scale_bar_nm(
    image_path: Path,
    image: np.ndarray,
    bar_bbox: tuple[int, int, int, int],
) -> float | None:
    """Try OCR, then filename, then label-region heuristics."""
    from_filename = _nm_from_filename(image_path)
    if from_filename is not None:
        return from_filename

    ocr_nm = _ocr_scale_bar_nm(image, bar_bbox)
    if ocr_nm is not None:
        return ocr_nm

    return None


def _nm_from_filename(path: Path) -> float | None:
    match = re.search(r"(\d+(?:\.\d+)?)\s*nm", path.stem, flags=re.IGNORECASE)
    if match:
        return float(match.group(1))
    return None


def _ocr_scale_bar_nm(
    image: np.ndarray,
    bar_bbox: tuple[int, int, int, int],
) -> float | None:
    """Optional OCR via pytesseract when installed."""
    try:
        import pytesseract
    except ImportError:
        return None

    row_min, col_min, row_max, col_max = bar_bbox
    h, w = image.shape
    text_row_min = max(0, row_min - 6)
    text_row_max = min(h, row_max + 20)
    text_col_min = col_min
    text_col_max = min(w, col_max + int(w * 0.18))
    patch = (image[text_row_min:text_row_max, text_col_min:text_col_max] * 255.0).astype(np.uint8)

    text = pytesseract.image_to_string(patch, config="--psm 7")
    match = re.search(r"(\d+(?:\.\d+)?)\s*nm", text, flags=re.IGNORECASE)
    if match:
        return float(match.group(1))
    return None


def parse_embedded_nm_per_pixel(
    source: str | Path | bytes,
    *,
    name: str = "",
    image: np.ndarray | None = None,
) -> float | None:
    """
    Read nm/pixel from AMT-style TIFF tags or burned-in ``Cal: … µm/pix`` text.

    Used when the scale-bar line cannot be measured automatically.
    """
    text_blobs: list[str] = []
    if name:
        text_blobs.append(name)

    data: bytes | None = None
    if isinstance(source, (bytes, bytearray)):
        data = bytes(source)
    else:
        path = Path(source)
        text_blobs.append(path.name)
        if path.exists():
            try:
                data = path.read_bytes()
            except OSError:
                data = None

    if data:
        text_blobs.extend(_tiff_text_blobs(data))

    if image is not None:
        ocr_text = _ocr_full_banner(image)
        if ocr_text:
            text_blobs.append(ocr_text)

    for blob in text_blobs:
        parsed = nm_per_pixel_from_cal_text(blob)
        if parsed is None:
            continue
        try:
            return validate_nm_per_pixel(parsed)
        except ValueError:
            continue
    return None


def _tiff_text_blobs(data: bytes) -> list[str]:
    from io import BytesIO

    blobs: list[str] = []
    try:
        with Image.open(BytesIO(data)) as img:
            for key in ("description", "comment"):
                val = img.info.get(key)
                if isinstance(val, str) and val.strip():
                    blobs.append(val)
            tag = getattr(img, "tag_v2", None)
            if tag is not None:
                for idx in (270, 285, 315):  # ImageDescription, PageName, Artist
                    if idx in tag:
                        blobs.append(str(tag[idx]))
    except Exception:
        return blobs
    return blobs


def _ocr_full_banner(image: np.ndarray) -> str | None:
    try:
        import pytesseract
    except ImportError:
        return None

    img = np.asarray(image, dtype=np.float64)
    if img.max() > 1.5:
        img = img / 255.0
    start = detect_bottom_info_banner(img)
    if start is None:
        start = int(img.shape[0] * 0.75)
    patch = (np.clip(img[start:], 0, 1) * 255.0).astype(np.uint8)
    if patch.size == 0:
        return None
    try:
        return pytesseract.image_to_string(patch, config="--psm 6")
    except Exception:
        return None

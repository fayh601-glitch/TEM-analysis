"""
Particle Segmenter — find dark nanoparticles in a TEM image
==============================================================

This file separates nanoparticles from the lighter carbon-film background using
thresholding and cleanup steps. It ignores the scale-bar strip at the bottom,
drops obvious background noise, and splits large merged blobs that contain two
or more touching rods.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi
from skimage import morphology, segmentation
from skimage.feature import peak_local_max
from skimage.filters import threshold_local, threshold_otsu
from skimage.measure import find_contours, label, regionprops

from tem_rods.models import AnalysisConfig, ThresholdMode


def segment_particles(
    image: np.ndarray,
    *,
    min_particle_area_px: int = 30,
    max_particle_area_px: int | None = None,
    use_watershed: bool = False,
    watershed_min_distance: int = 10,
    exclude_border: bool = True,
    min_solidity: float = 0.55,
    min_extent: float = 0.25,
    min_local_contrast: float = 0.04,
    mask_bottom_fraction: float = 0.12,
    morphology_closing_radius: int = 1,
    fill_holes: bool = True,
    split_touching_particles: bool = True,
    split_min_area_px: int = 500,
    split_max_aspect_ratio: float = 4.5,
    split_min_width_px: float = 22.0,
    split_watershed_min_distance: int = 5,
    threshold_mode: ThresholdMode = ThresholdMode.AUTO,
    percentile_threshold: float = 40.0,
    local_threshold_block_size: int = 35,
    local_threshold_offset: float = 0.01,
    min_darkness: float = 0.08,
    exclude_bbox: tuple[int, int, int, int] | None = None,
) -> np.ndarray:
    """
    Segment dark particles on a lighter TEM background.

    Returns an integer label image (0 = background).
    """
    binary = _binarize_particles(
        image,
        threshold_mode=threshold_mode,
        percentile_threshold=percentile_threshold,
        local_threshold_block_size=local_threshold_block_size,
        local_threshold_offset=local_threshold_offset,
        min_darkness=min_darkness,
    )
    # Remove tiny specks — real nanorods are larger than a few pixels across.
    binary = morphology.remove_small_objects(binary, min_size=min_particle_area_px)
    binary = morphology.binary_opening(binary, morphology.disk(1))
    if morphology_closing_radius > 0:
        binary = morphology.binary_closing(binary, morphology.disk(morphology_closing_radius))

    # Light centers inside rods (TEM diffraction contrast) can split one rod into two blobs.
    if fill_holes:
        binary = ndi.binary_fill_holes(binary)

    # Scale-bar text and the white bar line would otherwise be counted as particles.
    if mask_bottom_fraction > 0:
        binary = _mask_bottom_region(binary, mask_bottom_fraction)

    if exclude_bbox is not None:
        binary = _mask_bbox(binary, exclude_bbox)

    if exclude_border:
        binary = segmentation.clear_border(binary)

    if use_watershed:
        labels = _watershed_labels(binary, watershed_min_distance)
    else:
        labels = label(binary)

    if split_touching_particles:
        # Large merged blobs in dense clusters may contain several touching rods.
        labels = _split_touching_regions(
            labels,
            split_min_area_px=split_min_area_px,
            split_max_aspect_ratio=split_max_aspect_ratio,
            split_min_width_px=split_min_width_px,
            split_watershed_min_distance=split_watershed_min_distance,
            min_piece_area_px=max(min_particle_area_px, 60),
        )

    return _filter_regions(
        labels,
        image,
        min_area=min_particle_area_px,
        max_area=max_particle_area_px,
        min_solidity=min_solidity,
        min_extent=min_extent,
        min_local_contrast=min_local_contrast,
        min_darkness=min_darkness,
    )


def segment_particles_from_config(
    image: np.ndarray,
    config: AnalysisConfig,
    *,
    exclude_bbox: tuple[int, int, int, int] | None = None,
) -> np.ndarray:
    """Convenience wrapper that passes all segmentation settings from AnalysisConfig."""
    bbox = exclude_bbox if config.use_scale_bar_bbox_mask else None
    return segment_particles(
        image,
        min_particle_area_px=config.min_particle_area_px,
        max_particle_area_px=config.max_particle_area_px,
        use_watershed=config.use_watershed,
        watershed_min_distance=config.watershed_min_distance,
        exclude_border=config.exclude_border,
        min_solidity=config.min_solidity,
        min_extent=config.min_extent,
        min_local_contrast=config.min_local_contrast,
        mask_bottom_fraction=config.mask_bottom_fraction,
        morphology_closing_radius=config.morphology_closing_radius,
        fill_holes=config.fill_holes,
        split_touching_particles=config.split_touching_particles,
        split_min_area_px=config.split_min_area_px,
        split_max_aspect_ratio=config.split_max_aspect_ratio,
        split_min_width_px=config.split_min_width_px,
        split_watershed_min_distance=config.split_watershed_min_distance,
        threshold_mode=config.threshold_mode,
        percentile_threshold=config.percentile_threshold,
        local_threshold_block_size=config.local_threshold_block_size,
        local_threshold_offset=config.local_threshold_offset,
        min_darkness=getattr(config, "min_darkness", 0.08),
        exclude_bbox=bbox,
    )


def _watershed_labels(binary: np.ndarray, watershed_min_distance: int) -> np.ndarray:
    distance = ndi.distance_transform_edt(binary)
    coords = peak_local_max(
        distance,
        min_distance=watershed_min_distance,
        labels=binary,
    )
    mask = np.zeros(distance.shape, dtype=bool)
    mask[tuple(coords.T)] = True
    markers = ndi.label(mask)[0]
    return segmentation.watershed(-distance, markers, mask=binary)


def _split_touching_regions(
    labels: np.ndarray,
    *,
    split_min_area_px: int,
    split_max_aspect_ratio: float,
    split_min_width_px: float,
    split_watershed_min_distance: int,
    min_piece_area_px: int,
    bbox_pad: int = 3,
) -> np.ndarray:
    """
    Re-segment only large blobs that likely contain multiple touching rods.

    Small isolated rods are left untouched so length statistics stay stable.
    """
    output = labels.copy()
    next_id = int(output.max()) + 1
    max_split_area = 80_000
    max_split_bbox = 450

    for region in regionprops(labels):
        if not _should_split_region(
            region,
            split_min_area_px=split_min_area_px,
            split_max_aspect_ratio=split_max_aspect_ratio,
            split_min_width_px=split_min_width_px,
        ):
            continue
        if region.area > max_split_area:
            continue
        min_row, min_col, max_row, max_col = region.bbox
        if (max_row - min_row) > max_split_bbox or (max_col - min_col) > max_split_bbox:
            continue

        row_start = max(0, min_row - bbox_pad)
        col_start = max(0, min_col - bbox_pad)
        row_end = min(output.shape[0], max_row + bbox_pad)
        col_end = min(output.shape[1], max_col + bbox_pad)
        submask = labels[row_start:row_end, col_start:col_end] == region.label

        distance = ndi.distance_transform_edt(submask)
        coords = peak_local_max(
            distance,
            min_distance=split_watershed_min_distance,
            labels=submask,
        )
        if len(coords) < 2:
            continue

        marker_mask = np.zeros(distance.shape, dtype=bool)
        marker_mask[tuple(coords.T)] = True
        markers = ndi.label(marker_mask)[0]
        split_labels = segmentation.watershed(-distance, markers, mask=submask)

        output[row_start:row_end, col_start:col_end][submask] = 0
        for piece_id in range(1, split_labels.max() + 1):
            piece = split_labels == piece_id
            if int(piece.sum()) < min_piece_area_px:
                continue
            output[row_start:row_end, col_start:col_end][piece] = next_id
            next_id += 1

    return _relabel_sequential(output)


def _should_split_region(
    region,
    *,
    split_min_area_px: int,
    split_max_aspect_ratio: float,
    split_min_width_px: float,
) -> bool:
    """
    Return True when a blob is large/wide enough to be multiple touching rods.

    Slender single rods (high aspect) are never split. Wide or chunky blobs
    in dense clusters are candidates for watershed separation.
    """
    if region.area <= split_min_area_px:
        return False

    aspect = region.major_axis_length / max(region.minor_axis_length, 1.0)
    if aspect >= split_max_aspect_ratio:
        return False

    # Wide minor axis → likely parallel rods merged side-by-side
    if region.minor_axis_length >= split_min_width_px:
        return True

    # Only split chunky clusters (low aspect, very large area) — avoids fragmenting single rods
    if (
        region.area >= int(split_min_area_px * 1.5)
        and aspect < split_max_aspect_ratio
        and region.minor_axis_length >= split_min_width_px * 0.9
    ):
        return True

    return False


def _relabel_sequential(labels: np.ndarray) -> np.ndarray:
    if int(labels.max()) == 0:
        return labels
    from skimage.segmentation import relabel_sequential as _sk_relabel

    return _sk_relabel(labels)[0]


def _mask_bottom_region(binary: np.ndarray, bottom_fraction: float) -> np.ndarray:
    """Remove the bottom strip where scale bars and figure labels usually appear."""
    masked = binary.copy()
    cutoff = int(binary.shape[0] * (1.0 - bottom_fraction))
    masked[cutoff:, :] = False
    return masked


def _mask_bbox(
    binary: np.ndarray,
    bbox: tuple[int, int, int, int],
) -> np.ndarray:
    """Remove a rectangular region (e.g. detected scale bar + label)."""
    masked = binary.copy()
    row_min, col_min, row_max, col_max = bbox
    masked[row_min:row_max, col_min:col_max] = False
    return masked


def _threshold_dark_particles(image: np.ndarray, *, min_darkness: float) -> np.ndarray:
    """Keep only pixels clearly darker than the carbon-film background.

    Film grain sits a few percent below the median. Nanorods are much darker.
    Statistics are taken from the brighter half of the image so the dark tail
    (the particles) does not inflate the background scale.
    """
    flat = image.reshape(-1)
    median = float(np.median(flat))
    film = flat[flat >= median]
    if film.size < 32:
        film = flat
    film_med = float(np.median(film))
    mad = float(np.median(np.abs(film - film_med)))
    sigma = 1.4826 * mad if mad > 1e-6 else float(max(np.std(film), 1e-3))
    offset = max(float(min_darkness), 2.8 * sigma)
    return image < (median - offset)


def _binarize_particles(
    image: np.ndarray,
    *,
    threshold_mode: ThresholdMode,
    percentile_threshold: float,
    local_threshold_block_size: int,
    local_threshold_offset: float,
    min_darkness: float = 0.08,
) -> np.ndarray:
    mode = threshold_mode
    mode_value = mode.value if isinstance(mode, ThresholdMode) else str(mode)
    if mode == ThresholdMode.AUTO or mode_value == "auto":
        trial = threshold_otsu(image)
        if (image < trial).mean() > 0.75:
            mode = ThresholdMode.PERCENTILE
        else:
            # Never fall back to local thresholding: it labels every film-grain valley.
            mode = ThresholdMode.DARK
            mode_value = "dark"

    if mode == ThresholdMode.PERCENTILE or mode_value == "percentile":
        thresh = float(np.percentile(image, percentile_threshold))
        return image < thresh
    if mode == ThresholdMode.LOCAL or mode_value == "local":
        block = local_threshold_block_size
        if block % 2 == 0:
            block += 1
        local_thresh = threshold_local(
            image,
            block_size=block,
            offset=local_threshold_offset,
        )
        return image < local_thresh
    if mode == ThresholdMode.DARK or mode_value == "dark":
        return _threshold_dark_particles(image, min_darkness=min_darkness)
    thresh = float(threshold_otsu(image))
    return image < thresh


def _local_contrast(image: np.ndarray, particle_mask: np.ndarray, *, padding: int = 5) -> float:
    """How much darker the particle is compared with nearby background (0–1 scale)."""
    dilated = morphology.binary_dilation(particle_mask, morphology.disk(padding))
    ring = dilated & ~particle_mask
    if not ring.any():
        return 0.0
    return float(image[ring].mean() - image[particle_mask].mean())


def _filter_regions(
    labels: np.ndarray,
    image: np.ndarray,
    *,
    min_area: int,
    max_area: int | None,
    min_solidity: float,
    min_extent: float,
    min_local_contrast: float,
    min_darkness: float = 0.08,
) -> np.ndarray:
    kept = np.zeros_like(labels)
    next_id = 1
    padding = 5
    h, w = labels.shape
    background = float(np.median(image))
    for region in regionprops(labels):
        if region.area < min_area:
            continue
        if max_area is not None and region.area > max_area:
            continue
        if region.solidity < min_solidity:
            continue
        if region.extent < min_extent:
            continue

        minr, minc, maxr, maxc = region.bbox
        r0 = max(0, minr - padding)
        c0 = max(0, minc - padding)
        r1 = min(h, maxr + padding)
        c1 = min(w, maxc + padding)
        crop_img = image[r0:r1, c0:c1]
        crop_mask = np.zeros(crop_img.shape, dtype=bool)
        crop_mask[minr - r0 : maxr - r0, minc - c0 : maxc - c0] = region.image
        if not crop_mask.any():
            continue
        # Drop film-grain blobs: they are not much darker than the whole field.
        if float(crop_img[crop_mask].mean()) > background - min_darkness:
            continue
        if _local_contrast(crop_img, crop_mask) < min_local_contrast:
            continue

        kept[minr:maxr, minc:maxc][region.image] = next_id
        next_id += 1
    return kept


def iter_region_contours_xy(region):
    """Yield (xs, ys) contour vertices in full-image coordinates."""
    minr, minc, _maxr, _maxc = region.bbox
    crop = region.image.astype(float)
    if crop.size == 0:
        return
    for contour in find_contours(crop, 0.5):
        yield contour[:, 1] + minc, contour[:, 0] + minr

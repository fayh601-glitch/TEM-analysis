"""
Particle Segmenter — find dark nanoparticles in a TEM image
==============================================================

This file separates nanoparticles from the lighter carbon-film background using
thresholding and cleanup steps. It ignores the scale-bar strip at the bottom and
drops blobs that look like empty background (too faint, too hollow, or too sparse).
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi
from skimage import morphology, segmentation
from skimage.feature import peak_local_max
from skimage.filters import threshold_otsu
from skimage.measure import label, regionprops

from tem_rods.models import AnalysisConfig


def segment_particles(
    image: np.ndarray,
    *,
    min_particle_area_px: int = 30,
    max_particle_area_px: int | None = None,
    use_watershed: bool = True,
    watershed_min_distance: int = 10,
    exclude_border: bool = True,
    min_solidity: float = 0.55,
    min_extent: float = 0.25,
    min_local_contrast: float = 0.04,
    mask_bottom_fraction: float = 0.12,
) -> np.ndarray:
    """
    Segment dark particles on a lighter TEM background.

    Returns an integer label image (0 = background).
    """
    thresh = threshold_otsu(image)
    # Particles are typically darker than the carbon film background.
    binary = image < thresh
    binary = morphology.remove_small_objects(binary, min_size=min_particle_area_px)
    binary = morphology.binary_opening(binary, morphology.disk(1))
    binary = morphology.binary_closing(binary, morphology.disk(2))

    if mask_bottom_fraction > 0:
        binary = _mask_bottom_region(binary, mask_bottom_fraction)

    if exclude_border:
        binary = segmentation.clear_border(binary)

    if use_watershed:
        distance = ndi.distance_transform_edt(binary)
        coords = peak_local_max(
            distance,
            min_distance=watershed_min_distance,
            labels=binary,
        )
        mask = np.zeros(distance.shape, dtype=bool)
        mask[tuple(coords.T)] = True
        markers = ndi.label(mask)[0]
        labels = segmentation.watershed(-distance, markers, mask=binary)
    else:
        labels = label(binary)

    return _filter_regions(
        labels,
        image,
        min_area=min_particle_area_px,
        max_area=max_particle_area_px,
        min_solidity=min_solidity,
        min_extent=min_extent,
        min_local_contrast=min_local_contrast,
    )


def segment_particles_from_config(image: np.ndarray, config: AnalysisConfig) -> np.ndarray:
    """Convenience wrapper that passes all segmentation settings from AnalysisConfig."""
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
    )


def _mask_bottom_region(binary: np.ndarray, bottom_fraction: float) -> np.ndarray:
    """Remove the bottom strip where scale bars and figure labels usually appear."""
    masked = binary.copy()
    cutoff = int(binary.shape[0] * (1.0 - bottom_fraction))
    masked[cutoff:, :] = False
    return masked


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
) -> np.ndarray:
    kept = np.zeros_like(labels)
    next_id = 1
    for region in regionprops(labels):
        if region.area < min_area:
            continue
        if max_area is not None and region.area > max_area:
            continue
        if region.solidity < min_solidity:
            continue
        if region.extent < min_extent:
            continue

        particle_mask = labels == region.label
        if _local_contrast(image, particle_mask) < min_local_contrast:
            continue

        kept[particle_mask] = next_id
        next_id += 1
    return kept

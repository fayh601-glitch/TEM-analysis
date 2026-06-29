from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi
from skimage import morphology, segmentation
from skimage.feature import peak_local_max
from skimage.filters import threshold_otsu
from skimage.measure import label, regionprops


def segment_particles(
    image: np.ndarray,
    *,
    min_particle_area_px: int = 30,
    max_particle_area_px: int | None = None,
    use_watershed: bool = True,
    watershed_min_distance: int = 10,
    exclude_border: bool = True,
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

    labels = _filter_by_area(labels, min_particle_area_px, max_particle_area_px)
    return labels


def _filter_by_area(
    labels: np.ndarray,
    min_area: int,
    max_area: int | None,
) -> np.ndarray:
    kept = np.zeros_like(labels)
    next_id = 1
    for region in regionprops(labels):
        if region.area < min_area:
            continue
        if max_area is not None and region.area > max_area:
            continue
        kept[labels == region.label] = next_id
        next_id += 1
    return kept

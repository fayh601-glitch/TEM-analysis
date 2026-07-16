"""Unit tests for trace-and-match shape similarity."""

from __future__ import annotations

import numpy as np
from skimage.draw import disk, ellipse
from skimage.measure import label

from tem_rods.shape_match import (
    features_from_mask,
    find_similar_in_labels,
    polygon_to_mask,
    shape_distance,
)


def _two_rods_and_a_dot() -> tuple[np.ndarray, np.ndarray]:
    img = np.ones((120, 160), dtype=float) * 0.9
    # Rod A (template-like)
    rr, cc = ellipse(40, 40, 8, 28, rotation=0.2)
    img[rr, cc] = 0.1
    # Similar rod B
    rr, cc = ellipse(40, 110, 7, 26, rotation=-0.3)
    img[rr, cc] = 0.1
    # Round dot (should not match rods tightly)
    rr, cc = disk((90, 80), 10)
    img[rr, cc] = 0.1
    # Labels: invert so dark = True
    binary = img < 0.5
    labels = label(binary)
    return img, labels


def test_polygon_to_mask_and_features():
    mask = polygon_to_mask([(10, 10), (40, 10), (40, 25), (10, 25)], (50, 50))
    assert mask.sum() > 50
    feat = features_from_mask(mask)
    assert feat.aspect_ratio > 1.0
    assert 0 <= feat.circularity <= 1
    assert len(feat.hu_log) == 7


def test_identical_masks_have_low_distance():
    mask = np.zeros((60, 60), dtype=bool)
    rr, cc = ellipse(30, 30, 6, 20)
    mask[rr, cc] = True
    a = features_from_mask(mask)
    assert shape_distance(a, a) < 1e-9


def test_find_similar_prefers_rods_over_dot():
    _img, labels = _two_rods_and_a_dot()
    # Trace around the first rod roughly
    template = polygon_to_mask(
        [(12, 32), (68, 32), (68, 48), (12, 48)],
        labels.shape,
    )
    # Ensure template overlaps a labeled rod
    assert (labels[template] > 0).any()
    _feat, matches = find_similar_in_labels(labels, template, max_score=0.55)
    assert len(matches) >= 1
    # At least one match should be elongated (aspect from features)
    assert any(m.features.aspect_ratio > 1.5 for m in matches)

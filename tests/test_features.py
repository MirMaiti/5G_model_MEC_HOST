"""Feature extraction: shapes, invariances, and the drift guard."""

from __future__ import annotations

import numpy as np
import pytest

from signbridge.features import (
    FEATURE_VERSION,
    FeatureConfig,
    FeatureExtractor,
    extractor_from_describe,
)
from signbridge.landmarks import HANDS, HANDS_POSE, get_layout
from tests.conftest import make_frame


def test_dim_matches_output(extractor, rng):
    landmarks, mask = make_frame(rng)
    assert extractor.transform_frame(landmarks, mask).shape == (extractor.dim,)


def test_hands_layout_is_136_features():
    assert FeatureExtractor(HANDS).dim == 136


def test_pose_layout_adds_a_pose_block():
    assert FeatureExtractor(HANDS_POSE).dim == 136 + 1 + 3 * 33


def test_missing_hand_yields_a_zero_block(extractor, rng):
    landmarks, mask = make_frame(rng, hands=1)  # left hand only
    features = extractor.transform_frame(landmarks, mask)
    assert features[0] == 1.0          # left present
    assert features[66] == 0.0         # right absent
    assert np.all(features[66:132] == 0.0)


def test_translation_invariance(extractor, rng):
    """Moving the signer across the frame must not change the hand shape block."""
    landmarks, mask = make_frame(rng)
    shifted = landmarks + np.array([0.2, -0.1, 0.0], dtype=np.float32)

    base = extractor.transform_frame(landmarks, mask)
    moved = extractor.transform_frame(shifted, mask)

    # Wrist position is deliberately kept, so compare the shape part only.
    assert np.allclose(base[3:66], moved[3:66], atol=1e-5)
    assert not np.allclose(base[1:3], moved[1:3])


def test_scale_invariance(extractor, rng):
    """Stepping closer to the camera must not change the hand shape block."""
    landmarks, mask = make_frame(rng)
    scaled = landmarks * 1.7

    base = extractor.transform_frame(landmarks, mask)
    bigger = extractor.transform_frame(scaled, mask)
    assert np.allclose(base[3:66], bigger[3:66], atol=1e-4)


def test_wrist_position_can_be_switched_off(rng):
    stripped = FeatureExtractor(HANDS, FeatureConfig(include_wrist_position=False))
    assert stripped.dim == 136 - 4
    landmarks, mask = make_frame(rng)
    assert stripped.transform_frame(landmarks, mask).shape == (stripped.dim,)


def test_interhand_block_only_fills_when_both_hands_are_present(extractor, rng):
    one_hand, one_mask = make_frame(rng, hands=1)
    two_hands, two_mask = make_frame(rng, hands=2)

    assert extractor.transform_frame(one_hand, one_mask)[132] == 0.0
    assert extractor.transform_frame(two_hands, two_mask)[132] == 1.0


def test_degenerate_hand_does_not_divide_by_zero(extractor):
    """All landmarks at one point: the scale reference collapses to zero."""
    landmarks = np.zeros((42, 3), dtype=np.float32)
    mask = np.ones(42, dtype=np.float32)
    features = extractor.transform_frame(landmarks, mask)
    assert np.all(np.isfinite(features))


def test_transform_is_deterministic(extractor, rng):
    landmarks, mask = make_frame(rng)
    assert np.array_equal(
        extractor.transform_frame(landmarks, mask),
        extractor.transform_frame(landmarks, mask),
    )


def test_wrong_landmark_count_is_rejected(extractor):
    with pytest.raises(ValueError, match="expects landmarks shaped"):
        extractor.transform_frame(np.zeros((75, 3)), np.ones(75))


def test_sequence_transform_shape(extractor, rng):
    from tests.conftest import make_clip

    landmarks, mask = make_clip(rng, frames=30)
    assert extractor.transform_sequence(landmarks, mask).shape == (30, extractor.dim)


def test_describe_round_trip(extractor):
    rebuilt = extractor_from_describe(extractor.describe())
    assert rebuilt.dim == extractor.dim
    assert rebuilt.layout.name == extractor.layout.name


def test_stale_feature_version_is_refused(extractor):
    """A checkpoint from older feature maths must not be served silently."""
    described = extractor.describe()
    described["config"]["feature_version"] = FEATURE_VERSION - 1
    with pytest.raises(ValueError, match="feature version"):
        extractor_from_describe(described)


def test_unknown_layout_name():
    with pytest.raises(KeyError, match="Unknown landmark layout"):
        get_layout("elbows")

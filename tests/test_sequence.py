"""Length normalisation."""

from __future__ import annotations

import numpy as np
import pytest

from signbridge.sequence import resample_sequence, time_shift


@pytest.mark.parametrize("frames,target", [(10, 45), (90, 45), (45, 45), (1, 30)])
def test_resample_hits_the_target_length(frames, target):
    sequence = np.random.default_rng(0).random((frames, 8)).astype(np.float32)
    assert resample_sequence(sequence, target).shape == (target, 8)


def test_resample_preserves_endpoints():
    sequence = np.linspace(0, 1, 10, dtype=np.float32)[:, None]
    resampled = resample_sequence(sequence, 45)
    assert resampled[0, 0] == pytest.approx(0.0)
    assert resampled[-1, 0] == pytest.approx(1.0)


def test_resample_is_linear_on_a_ramp():
    sequence = np.linspace(0, 1, 11, dtype=np.float32)[:, None]
    resampled = resample_sequence(sequence, 21)
    assert np.allclose(resampled[:, 0], np.linspace(0, 1, 21), atol=1e-5)


def test_single_frame_is_repeated():
    sequence = np.array([[1.0, 2.0]], dtype=np.float32)
    assert np.array_equal(resample_sequence(sequence, 4), np.tile(sequence, (4, 1)))


def test_empty_sequence_is_rejected():
    with pytest.raises(ValueError, match="empty"):
        resample_sequence(np.zeros((0, 5)), 10)


def test_time_shift_clamps_rather_than_wrapping():
    """Wrapping would put the end of a sign before its beginning."""
    sequence = np.arange(5, dtype=np.float32)[:, None]
    shifted = time_shift(sequence, 2)
    assert shifted[0, 0] == 0.0 and shifted[1, 0] == 0.0
    assert shifted[-1, 0] == 2.0


def test_zero_shift_is_identity():
    sequence = np.arange(5, dtype=np.float32)[:, None]
    assert np.array_equal(time_shift(sequence, 0), sequence)

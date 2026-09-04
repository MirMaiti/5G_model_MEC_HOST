"""Length normalisation for landmark clips.

The models take a fixed number of frames, but real clips vary: a recorded
sample runs as long as you held the sign, and the server's rolling buffer is
short until it fills. Both are put on the same footing here, by the same
function, so a partially-filled buffer at inference time looks like the clips
the model saw during training.

Linear interpolation along the time axis is used rather than dropping or
repeating frames, because it keeps the trajectory of a moving sign smooth.
"""

from __future__ import annotations

import numpy as np


def resample_sequence(sequence: np.ndarray, target_length: int) -> np.ndarray:
    """Resample a ``(T, D)`` sequence to exactly ``target_length`` frames.

    Args:
        sequence: Frames by features. Must hold at least one frame.
        target_length: Frames wanted out.

    Returns:
        A ``(target_length, D)`` float32 array. A clip already at the target
        length is returned unchanged apart from dtype.

    Raises:
        ValueError: If the sequence is empty or not 2-D, or the target is < 1.
    """
    sequence = np.asarray(sequence, dtype=np.float32)
    if sequence.ndim != 2:
        raise ValueError(f"Expected a sequence shaped (T, D); got {tuple(sequence.shape)}.")
    frames, dim = sequence.shape
    if frames == 0:
        raise ValueError("Cannot resample an empty sequence.")
    if target_length < 1:
        raise ValueError(f"target_length must be at least 1; got {target_length}.")
    if frames == target_length:
        return sequence

    if frames == 1:
        return np.repeat(sequence, target_length, axis=0)

    source = np.linspace(0.0, frames - 1, num=frames, dtype=np.float32)
    wanted = np.linspace(0.0, frames - 1, num=target_length, dtype=np.float32)
    out = np.empty((target_length, dim), dtype=np.float32)
    for channel in range(dim):
        out[:, channel] = np.interp(wanted, source, sequence[:, channel])
    return out


def time_shift(sequence: np.ndarray, shift: int) -> np.ndarray:
    """Roll a clip along time, holding the end frames rather than wrapping.

    Wrapping would teach the model that the end of a sign precedes its
    beginning, so the edges are clamped instead.
    """
    sequence = np.asarray(sequence, dtype=np.float32)
    if shift == 0:
        return sequence
    frames = sequence.shape[0]
    index = np.clip(np.arange(frames) - shift, 0, frames - 1)
    return sequence[index]

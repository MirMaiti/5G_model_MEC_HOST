"""The capture-side interface: something that produces landmark frames.

Keeping this abstract is what lets the same client loop run against a live
webcam, a recorded clip or a synthetic generator. The TCP client depends only on
:class:`LandmarkSource`, so nothing in the network path knows whether a camera
is involved.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Iterator, Optional

import numpy as np


@dataclass
class LandmarkFrame:
    """One frame of landmarks, plus the image it came from when there is one.

    Args:
        landmarks: ``(L, 3)`` normalised coordinates in the configured layout.
        mask: ``(L,)`` presence flags; 0 marks a landmark that was not detected.
        timestamp_ms: Capture time in milliseconds since the epoch.
        image: The BGR frame, kept only for the preview window. It is never
            sent over the network.
    """

    landmarks: np.ndarray
    mask: np.ndarray
    timestamp_ms: float
    image: Optional[np.ndarray] = None

    @property
    def hands_detected(self) -> int:
        """How many of the two hand slots were filled."""
        from ..landmarks import NUM_HAND_LANDMARKS

        left = self.mask[:NUM_HAND_LANDMARKS].max() > 0
        right = self.mask[NUM_HAND_LANDMARKS : 2 * NUM_HAND_LANDMARKS].max() > 0
        return int(left) + int(right)


class LandmarkSource(ABC):
    """Yields :class:`LandmarkFrame` objects until exhausted or stopped."""

    @abstractmethod
    def frames(self) -> Iterator[LandmarkFrame]:
        """Iterate frames. Infinite for a live camera, finite for a recording."""

    def close(self) -> None:
        """Release the camera, file handle or model."""

    def __enter__(self) -> "LandmarkSource":
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

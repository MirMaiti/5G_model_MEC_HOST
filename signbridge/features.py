"""Turning raw landmarks into the vector the model actually sees.

This module is the single source of truth for normalisation. The collector, the
trainer and the server all call the same :class:`FeatureExtractor`, and the
config that built it is written into the checkpoint - so a model can never be
served with features that differ from the ones it was trained on.

Why normalise at all: raw MediaPipe coordinates move when the signer steps
closer to the camera or stands off-centre, but the *sign* does not. Each hand is
therefore re-expressed relative to its own wrist and divided by its own size,
which removes camera distance and position. The wrist's position in the frame is
kept as two separate features, because for some signs *where* the hand is does
carry meaning.

Layout of one hand block (66 values)::

    [0]      presence flag, 1.0 when the hand was detected
    [1:3]    wrist x, y in normalised frame coordinates
    [3:66]   21 landmarks x (x, y, z), wrist-centred and scale-normalised
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np

from .landmarks import (
    HAND_MIDDLE_MCP,
    HAND_WRIST,
    NUM_HAND_LANDMARKS,
    POSE_LEFT_SHOULDER,
    POSE_RIGHT_SHOULDER,
    LandmarkLayout,
)

#: Values per hand block: presence + wrist xy + 21 landmarks x 3.
HAND_BLOCK_DIM = 1 + 2 + NUM_HAND_LANDMARKS * 3
#: Values in the inter-hand block: presence + the wrist-to-wrist vector.
INTERHAND_BLOCK_DIM = 1 + 3

#: Below this, a scale reference is treated as degenerate rather than divided by.
_EPS = 1e-6
#: A hand counts as detected when at least this fraction of its points are present.
_PRESENCE_THRESHOLD = 0.5

#: Bumped whenever the feature maths changes, so old checkpoints fail loudly
#: instead of being served with silently different inputs.
FEATURE_VERSION = 1


@dataclass(frozen=True)
class FeatureConfig:
    """Switches controlling which feature blocks are produced.

    Args:
        include_wrist_position: Keep each wrist's position in the frame. Turn
            off to make the model fully translation-invariant.
        include_interhand: Add the vector between the two wrists, which
            separates signs that differ only in how the hands are spaced.
    """

    include_wrist_position: bool = True
    include_interhand: bool = True

    def describe(self) -> Dict[str, object]:
        """Serialisable form, stored in the checkpoint."""
        return {
            "include_wrist_position": self.include_wrist_position,
            "include_interhand": self.include_interhand,
            "feature_version": FEATURE_VERSION,
        }


class FeatureExtractor:
    """Converts ``(L, 3)`` landmark frames into fixed-length feature vectors.

    The transform is pure: no fitted statistics, no state carried between
    frames. That is what makes it safe to run the identical code on the capture
    device, in the training loop and on the server.

    Args:
        layout: The landmark layout the incoming arrays follow.
        config: Which optional feature blocks to include.
    """

    def __init__(self, layout: LandmarkLayout, config: Optional[FeatureConfig] = None) -> None:
        self.layout = layout
        self.config = config or FeatureConfig()

    @property
    def dim(self) -> int:
        """Length of the feature vector produced for one frame."""
        total = 2 * HAND_BLOCK_DIM
        if not self.config.include_wrist_position:
            total -= 2 * 2
        if self.config.include_interhand:
            total += INTERHAND_BLOCK_DIM
        if self.layout.include_pose:
            pose_count = self.layout.pose.stop - self.layout.pose.start
            total += 1 + 3 * pose_count
        return total

    def describe(self) -> Dict[str, object]:
        """Everything needed to rebuild this extractor from a checkpoint."""
        return {
            "layout": self.layout.describe(),
            "config": self.config.describe(),
            "dim": self.dim,
        }

    def _hand_block(self, coords: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Normalise one hand to a wrist-centred, scale-free block."""
        present = float(mask.mean()) >= _PRESENCE_THRESHOLD
        width = HAND_BLOCK_DIM if self.config.include_wrist_position else HAND_BLOCK_DIM - 2
        block = np.zeros(width, dtype=np.float32)
        if not present:
            return block

        wrist = coords[HAND_WRIST]
        relative = coords - wrist

        # Wrist-to-knuckle distance is stable across hand poses, so it makes a
        # better size reference than the bounding box, which collapses when the
        # fingers curl.
        scale = float(np.linalg.norm(relative[HAND_MIDDLE_MCP]))
        if scale < _EPS:
            scale = float(np.linalg.norm(relative, axis=-1).max())
        if scale < _EPS:
            scale = 1.0
        relative = relative / scale

        block[0] = 1.0
        if self.config.include_wrist_position:
            block[1:3] = wrist[:2]
            block[3:] = relative.reshape(-1)
        else:
            block[1:] = relative.reshape(-1)
        return block

    def _pose_block(self, coords: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Normalise the pose against shoulder centre and shoulder width."""
        count = coords.shape[0]
        block = np.zeros(1 + 3 * count, dtype=np.float32)
        if float(mask.mean()) < _PRESENCE_THRESHOLD:
            return block

        left, right = coords[POSE_LEFT_SHOULDER], coords[POSE_RIGHT_SHOULDER]
        centre = (left + right) / 2.0
        scale = float(np.linalg.norm(left - right))
        if scale < _EPS:
            scale = 1.0
        block[0] = 1.0
        block[1:] = ((coords - centre) / scale).reshape(-1)
        return block

    def transform_frame(self, landmarks: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Build the feature vector for one frame.

        Args:
            landmarks: ``(L, 3)`` coordinates in MediaPipe's normalised space.
            mask: ``(L,)`` presence flags; 0 marks a landmark that was never
                detected, which is the only way to tell it apart from one
                genuinely measured at the origin.

        Returns:
            A ``(dim,)`` float32 vector.

        Raises:
            ValueError: If the arrays do not match the configured layout.
        """
        expected = self.layout.num_landmarks
        landmarks = np.asarray(landmarks, dtype=np.float32)
        mask = np.asarray(mask, dtype=np.float32)
        if landmarks.shape != (expected, 3):
            raise ValueError(
                f"Layout {self.layout.name!r} expects landmarks shaped "
                f"({expected}, 3); got {tuple(landmarks.shape)}."
            )
        if mask.shape != (expected,):
            raise ValueError(
                f"Layout {self.layout.name!r} expects a mask shaped ({expected},); "
                f"got {tuple(mask.shape)}."
            )

        left = self.layout.left_hand
        right = self.layout.right_hand
        blocks = [
            self._hand_block(landmarks[left], mask[left]),
            self._hand_block(landmarks[right], mask[right]),
        ]

        if self.config.include_interhand:
            interhand = np.zeros(INTERHAND_BLOCK_DIM, dtype=np.float32)
            both = blocks[0][0] > 0.0 and blocks[1][0] > 0.0
            if both:
                interhand[0] = 1.0
                interhand[1:] = landmarks[right][HAND_WRIST] - landmarks[left][HAND_WRIST]
            blocks.append(interhand)

        if self.layout.include_pose:
            pose = self.layout.pose
            blocks.append(self._pose_block(landmarks[pose], mask[pose]))

        return np.concatenate(blocks).astype(np.float32, copy=False)

    def transform_sequence(self, landmarks: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Apply :meth:`transform_frame` across a ``(T, L, 3)`` clip.

        Returns:
            A ``(T, dim)`` float32 array.
        """
        landmarks = np.asarray(landmarks, dtype=np.float32)
        mask = np.asarray(mask, dtype=np.float32)
        if landmarks.ndim != 3:
            raise ValueError(
                f"Expected a sequence shaped (T, L, 3); got {tuple(landmarks.shape)}."
            )
        if mask.shape != landmarks.shape[:2]:
            raise ValueError(
                f"Mask shaped {tuple(mask.shape)} does not match landmarks "
                f"{tuple(landmarks.shape[:2])}."
            )
        out = np.empty((landmarks.shape[0], self.dim), dtype=np.float32)
        for index in range(landmarks.shape[0]):
            out[index] = self.transform_frame(landmarks[index], mask[index])
        return out


def extractor_from_describe(described: Dict[str, object]) -> FeatureExtractor:
    """Rebuild an extractor from the dict a checkpoint stored.

    Raises:
        ValueError: If the checkpoint was written by an incompatible version of
            the feature code.
    """
    from .landmarks import get_layout

    layout_info = dict(described["layout"])  # type: ignore[arg-type]
    config_info = dict(described["config"])  # type: ignore[arg-type]

    stored_version = int(config_info.get("feature_version", 0))
    if stored_version != FEATURE_VERSION:
        raise ValueError(
            f"This checkpoint was built with feature version {stored_version}, but "
            f"the installed code is version {FEATURE_VERSION}. Retrain, or check "
            "out the matching revision - serving it would feed the model inputs "
            "it was never trained on."
        )

    layout = get_layout(str(layout_info["name"]))
    config = FeatureConfig(
        include_wrist_position=bool(config_info["include_wrist_position"]),
        include_interhand=bool(config_info["include_interhand"]),
    )
    return FeatureExtractor(layout=layout, config=config)

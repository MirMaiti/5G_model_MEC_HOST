"""Canonical landmark layout shared by capture, training and inference.

Every stage of the pipeline agrees on one ordering for the landmark tensor::

    [  0 ..  20]  left hand   (21 landmarks)
    [ 21 ..  41]  right hand  (21 landmarks)
    [ 42 ..  74]  pose        (33 landmarks, only when pose is enabled)

Keeping the layout in a single module means the collector, the trainer and the
server can never disagree about which slice is which. A model trained on the
``hands`` layout will refuse to load against ``hands_pose`` because the feature
dimension changes, which is the failure we want: loud, and at startup.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

#: Landmarks MediaPipe returns for one hand.
NUM_HAND_LANDMARKS = 21
#: Landmarks MediaPipe returns for the body pose (BlazePose ordering).
NUM_POSE_LANDMARKS = 33

#: Index of the wrist within a single hand's 21 landmarks.
HAND_WRIST = 0
#: Index of the middle-finger MCP joint, used as the hand's scale reference.
HAND_MIDDLE_MCP = 9

#: Indices within the 33 pose landmarks.
POSE_LEFT_SHOULDER = 11
POSE_RIGHT_SHOULDER = 12


@dataclass(frozen=True)
class LandmarkLayout:
    """Slice boundaries for one landmark configuration.

    Args:
        name: Layout identifier, stored in checkpoints and sent in the TCP
            handshake so a mismatched client fails immediately.
        include_pose: Whether the tensor carries the 33 pose landmarks.
    """

    name: str
    include_pose: bool

    @property
    def num_landmarks(self) -> int:
        """Total landmarks per frame."""
        pose = NUM_POSE_LANDMARKS if self.include_pose else 0
        return 2 * NUM_HAND_LANDMARKS + pose

    @property
    def left_hand(self) -> slice:
        """Slice covering the left hand."""
        return slice(0, NUM_HAND_LANDMARKS)

    @property
    def right_hand(self) -> slice:
        """Slice covering the right hand."""
        return slice(NUM_HAND_LANDMARKS, 2 * NUM_HAND_LANDMARKS)

    @property
    def pose(self) -> slice:
        """Slice covering the pose landmarks, empty when pose is disabled."""
        start = 2 * NUM_HAND_LANDMARKS
        end = start + (NUM_POSE_LANDMARKS if self.include_pose else 0)
        return slice(start, end)

    def describe(self) -> Dict[str, object]:
        """Serialisable form, embedded in checkpoints and the TCP handshake."""
        return {
            "name": self.name,
            "include_pose": self.include_pose,
            "num_landmarks": self.num_landmarks,
        }


#: Hands only - 42 landmarks. The default, and what "send hand landmarks" means.
HANDS = LandmarkLayout(name="hands", include_pose=False)
#: Hands plus body pose - 75 landmarks, for signs that depend on body location.
HANDS_POSE = LandmarkLayout(name="hands_pose", include_pose=True)

_LAYOUTS: Dict[str, LandmarkLayout] = {layout.name: layout for layout in (HANDS, HANDS_POSE)}


def get_layout(name: str) -> LandmarkLayout:
    """Look up a layout by name.

    Raises:
        KeyError: If the name is not a registered layout.
    """
    try:
        return _LAYOUTS[name]
    except KeyError:
        known = ", ".join(sorted(_LAYOUTS))
        raise KeyError(f"Unknown landmark layout {name!r}. Known layouts: {known}.") from None


def layout_names() -> Tuple[str, ...]:
    """Names of every registered layout."""
    return tuple(sorted(_LAYOUTS))

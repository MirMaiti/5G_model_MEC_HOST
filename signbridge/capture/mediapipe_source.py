"""Landmark extraction from a live camera, using the MediaPipe Tasks API.

This is the only module on the host that touches a camera, and the only one
that imports MediaPipe or OpenCV - both are imported lazily so the server side
never pulls them in.

MediaPipe 1.0 removed the old ``mp.solutions`` API, so this uses Tasks, which
needs a downloaded ``.task`` bundle::

    python -m signbridge.cli.fetch_models

A note on handedness. MediaPipe labels each detected hand ``Left`` or ``Right``,
and that label decides which half of the landmark tensor the hand lands in.
Whether it matches the signer's own left and right depends on whether the
camera image is mirrored - but it does not need to. What matters is that the
label is applied identically when recording data and when predicting, which it
is, because both paths run this class. The preview is mirrored for the signer's
comfort; the frame fed to MediaPipe is not, so the convention stays fixed.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any, Iterator, Optional, Tuple

import numpy as np

from ..landmarks import NUM_HAND_LANDMARKS, NUM_POSE_LANDMARKS, LandmarkLayout
from .base import LandmarkFrame, LandmarkSource

logger = logging.getLogger(__name__)


class CameraError(RuntimeError):
    """Raised when the camera cannot be opened or read."""


class MediaPipeSource(LandmarkSource):
    """Reads a camera and emits landmark frames in the configured layout.

    Args:
        layout: Which landmarks to produce.
        hand_model_path: Path to ``hand_landmarker.task``.
        pose_model_path: Path to ``pose_landmarker_lite.task``; only needed when
            the layout includes pose.
        camera_index: Index passed to OpenCV.
        width: Requested capture width.
        height: Requested capture height.
        min_hand_detection_confidence: MediaPipe detection threshold.
        min_hand_presence_confidence: MediaPipe presence threshold.
        min_tracking_confidence: MediaPipe tracking threshold.
        mirror_preview: Mirror the preview image only, not the analysed frame.

    Raises:
        FileNotFoundError: If a required ``.task`` bundle is missing.
    """

    def __init__(
        self,
        layout: LandmarkLayout,
        hand_model_path: str = "models/hand_landmarker.task",
        pose_model_path: str = "models/pose_landmarker_lite.task",
        camera_index: int = 0,
        width: int = 640,
        height: int = 480,
        min_hand_detection_confidence: float = 0.5,
        min_hand_presence_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        mirror_preview: bool = True,
    ) -> None:
        self.layout = layout
        self.camera_index = camera_index
        self.width = width
        self.height = height
        self.mirror_preview = mirror_preview

        self._hand_model_path = _require_model(hand_model_path, "hand_landmarker.task")
        self._pose_model_path = (
            _require_model(pose_model_path, "pose_landmarker_lite.task")
            if layout.include_pose
            else None
        )
        self._hand_options = (
            min_hand_detection_confidence,
            min_hand_presence_confidence,
            min_tracking_confidence,
        )
        self._capture: Optional[Any] = None
        self._hands: Optional[Any] = None
        self._pose: Optional[Any] = None

    def _open(self) -> None:
        """Open the camera and build the MediaPipe landmarkers."""
        import cv2
        from mediapipe.tasks.python import BaseOptions
        from mediapipe.tasks.python import vision

        capture = cv2.VideoCapture(self.camera_index)
        if not capture.isOpened():
            raise CameraError(
                f"Could not open camera {self.camera_index}. On macOS the app hosting "
                "this shell needs camera permission (System Settings > Privacy & "
                "Security > Camera); an embedded terminal panel often cannot get it, "
                "so try running from Terminal.app."
            )
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self._capture = capture

        detection, presence, tracking = self._hand_options
        self._hands = vision.HandLandmarker.create_from_options(
            vision.HandLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=str(self._hand_model_path)),
                running_mode=vision.RunningMode.VIDEO,
                num_hands=2,
                min_hand_detection_confidence=detection,
                min_hand_presence_confidence=presence,
                min_tracking_confidence=tracking,
            )
        )
        if self._pose_model_path is not None:
            self._pose = vision.PoseLandmarker.create_from_options(
                vision.PoseLandmarkerOptions(
                    base_options=BaseOptions(model_asset_path=str(self._pose_model_path)),
                    running_mode=vision.RunningMode.VIDEO,
                    num_poses=1,
                )
            )

    def frames(self) -> Iterator[LandmarkFrame]:
        """Yield landmark frames from the camera until it is closed.

        Raises:
            CameraError: If the camera cannot be opened or stops delivering.
        """
        import cv2
        import mediapipe as mp

        if self._capture is None:
            self._open()
        assert self._capture is not None and self._hands is not None

        # MediaPipe's VIDEO mode requires strictly increasing timestamps, and
        # rejects the stream if one repeats - so this counter drives it rather
        # than the wall clock, which can stall or step backwards.
        frame_index = 0
        while True:
            ok, image = self._capture.read()
            if not ok:
                raise CameraError(f"Camera {self.camera_index} stopped delivering frames.")

            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            timestamp_ms = int(frame_index * 1000 / 30) + 1
            frame_index += 1

            hand_result = self._hands.detect_for_video(mp_image, timestamp_ms)
            pose_result = (
                self._pose.detect_for_video(mp_image, timestamp_ms)
                if self._pose is not None
                else None
            )
            landmarks, mask = self._assemble(hand_result, pose_result)

            preview = cv2.flip(image, 1) if self.mirror_preview else image
            yield LandmarkFrame(
                landmarks=landmarks,
                mask=mask,
                timestamp_ms=time.time() * 1000.0,
                image=preview,
            )

    def _assemble(self, hand_result: Any, pose_result: Any) -> Tuple[np.ndarray, np.ndarray]:
        """Place detections into the canonical tensor, zero-filling what is missing."""
        total = self.layout.num_landmarks
        landmarks = np.zeros((total, 3), dtype=np.float32)
        mask = np.zeros(total, dtype=np.float32)

        hand_landmarks = getattr(hand_result, "hand_landmarks", None) or []
        handedness = getattr(hand_result, "handedness", None) or []
        for points, categories in zip(hand_landmarks, handedness):
            if not categories:
                continue
            label = categories[0].category_name
            target = self.layout.left_hand if label == "Left" else self.layout.right_hand
            landmarks[target] = _points_to_array(points, NUM_HAND_LANDMARKS)
            mask[target] = 1.0

        if self.layout.include_pose and pose_result is not None:
            pose_landmarks = getattr(pose_result, "pose_landmarks", None) or []
            if pose_landmarks:
                target = self.layout.pose
                landmarks[target] = _points_to_array(pose_landmarks[0], NUM_POSE_LANDMARKS)
                mask[target] = 1.0

        return landmarks, mask

    def close(self) -> None:
        """Release the camera and the landmarker models."""
        for resource in (self._hands, self._pose):
            if resource is not None:
                try:
                    resource.close()
                except Exception:  # pragma: no cover
                    pass
        if self._capture is not None:
            self._capture.release()
        self._capture = self._hands = self._pose = None


def _points_to_array(points: Any, expected: int) -> np.ndarray:
    """Convert MediaPipe normalized landmarks to an ``(expected, 3)`` array."""
    array = np.zeros((expected, 3), dtype=np.float32)
    for index, point in enumerate(points[:expected]):
        array[index] = (point.x, point.y, point.z)
    return array


def _require_model(path: str, name: str) -> Path:
    """Check a ``.task`` bundle exists, with a fixable message when it does not.

    Raises:
        FileNotFoundError: If the bundle is missing.
    """
    resolved = Path(path)
    if not resolved.exists():
        raise FileNotFoundError(
            f"MediaPipe model bundle {name} not found at {resolved}.\n"
            "Download it with:\n"
            "    python -m signbridge.cli.fetch_models"
        )
    return resolved


def draw_landmarks(image: np.ndarray, landmarks: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Draw the hand skeleton onto a preview image.

    The preview is mirrored while the analysed frame is not, so the drawing is
    mirrored here too, otherwise the dots would not sit on the hands.

    Returns:
        The same image, annotated in place.
    """
    import cv2

    height, width = image.shape[:2]
    connections = [
        (0, 1), (1, 2), (2, 3), (3, 4),           # thumb
        (0, 5), (5, 6), (6, 7), (7, 8),           # index
        (5, 9), (9, 10), (10, 11), (11, 12),      # middle
        (9, 13), (13, 14), (14, 15), (15, 16),    # ring
        (13, 17), (17, 18), (18, 19), (19, 20),   # little
        (0, 17),
    ]
    for hand_index in range(2):
        start = hand_index * NUM_HAND_LANDMARKS
        block = slice(start, start + NUM_HAND_LANDMARKS)
        if mask[block].max() <= 0:
            continue
        points = landmarks[block]
        colour = (0, 200, 255) if hand_index == 0 else (255, 180, 0)
        pixels = [(int((1.0 - p[0]) * width), int(p[1] * height)) for p in points]
        for a, b in connections:
            cv2.line(image, pixels[a], pixels[b], colour, 2)
        for pixel in pixels:
            cv2.circle(image, pixel, 3, (255, 255, 255), -1)
    return image

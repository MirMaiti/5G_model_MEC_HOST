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

The camera can be a device index, a device path, or a network stream such as an
IP camera's ``rtsp://`` address. A stream is read on its own thread that keeps
only the newest frame: OpenCV otherwise queues every frame the camera sends, and
on a device slower than the camera the delay grows without bound.
"""

from __future__ import annotations

import logging
import os
import threading
import time
from pathlib import Path
from typing import Any, Iterator, Optional, Tuple, Union
from urllib.parse import urlsplit, urlunsplit

import numpy as np

from ..landmarks import NUM_HAND_LANDMARKS, NUM_POSE_LANDMARKS, LandmarkLayout
from .base import LandmarkFrame, LandmarkSource

logger = logging.getLogger(__name__)

#: Environment variable read when ``--camera`` is not given, so a stream URL
#: carrying a password never has to be written into ``config.yaml``.
CAMERA_ENV = "SIGNBRIDGE_CAMERA"

Camera = Union[int, str]


class CameraError(RuntimeError):
    """Raised when the camera cannot be opened or read."""


def parse_camera(value: str) -> Camera:
    """Turn ``"0"`` into a device index; leave paths and URLs as strings."""
    value = value.strip()
    return int(value) if value.isdigit() else value


def resolve_camera(cli_value: Optional[str], config_index: int) -> Camera:
    """Pick the camera: the flag, then ``$SIGNBRIDGE_CAMERA``, then the config."""
    if cli_value is not None:
        return parse_camera(cli_value)
    from_env = os.environ.get(CAMERA_ENV, "").strip()
    if from_env:
        return parse_camera(from_env)
    return config_index


def is_stream(camera: Camera) -> bool:
    """Whether ``camera`` is a network address rather than a local device."""
    return isinstance(camera, str) and "://" in camera


def describe_camera(camera: Camera) -> str:
    """Name the camera for messages, with any password in a URL masked."""
    if not is_stream(camera):
        return str(camera)
    parts = urlsplit(str(camera))
    if parts.password is None:
        return str(camera)
    host = parts.hostname or ""
    if parts.port is not None:
        host = f"{host}:{parts.port}"
    return urlunsplit(parts._replace(netloc=f"{parts.username}:***@{host}"))


class LatestFrameReader:
    """Reads a capture on a background thread, handing out only the newest frame.

    Frames that arrive while the caller is busy are overwritten, never queued,
    so the frame returned is at most one camera interval old however slowly the
    caller runs. It has the ``read``/``release`` shape of ``cv2.VideoCapture``.

    Args:
        capture: An opened ``cv2.VideoCapture`` or anything with ``read``/``release``.
        timeout: Seconds :meth:`read` waits for a new frame before failing.
    """

    def __init__(self, capture: Any, timeout: float = 10.0) -> None:
        self._capture = capture
        self._timeout = timeout
        self._condition = threading.Condition()
        self._frame: Any = None
        self._fresh = False
        self._ended = False
        self._stopping = False
        self._thread = threading.Thread(target=self._run, name="camera-reader", daemon=True)
        self._thread.start()

    def _run(self) -> None:
        while not self._stopping:
            ok, image = self._capture.read()
            with self._condition:
                if not ok:
                    self._ended = True
                    self._condition.notify_all()
                    return
                self._frame = image
                self._fresh = True
                self._condition.notify_all()

    def read(self) -> Tuple[bool, Any]:
        """Return the newest frame not yet handed out, as ``(ok, image)``."""
        with self._condition:
            self._condition.wait_for(lambda: self._fresh or self._ended, timeout=self._timeout)
            if not self._fresh:
                return False, None
            self._fresh = False
            return True, self._frame

    def release(self) -> None:
        """Stop the thread and release the underlying capture."""
        self._stopping = True
        self._thread.join(timeout=2.0)
        self._capture.release()


class MediaPipeSource(LandmarkSource):
    """Reads a camera and emits landmark frames in the configured layout.

    Args:
        layout: Which landmarks to produce.
        hand_model_path: Path to ``hand_landmarker.task``.
        pose_model_path: Path to ``pose_landmarker_lite.task``; only needed when
            the layout includes pose.
        camera_index: Device index, device path, or stream URL such as
            ``rtsp://user:pass@192.168.1.64:554/stream``.
        width: Requested capture width. A stream's resolution is set on the
            camera, so its frames are instead scaled down to at most this width.
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
        camera_index: Camera = 0,
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
        self._live_clock = False
        self._capture: Optional[Any] = None
        self._hands: Optional[Any] = None
        self._pose: Optional[Any] = None

    def _open(self) -> None:
        """Open the camera and build the MediaPipe landmarkers."""
        import cv2
        from mediapipe.tasks.python import BaseOptions
        from mediapipe.tasks.python import vision

        if is_stream(self.camera_index):
            # UDP, FFmpeg's default for RTSP, loses packets on a busy network
            # and smears the decoded image; TCP delivers whole frames or none.
            os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;tcp")
            capture = cv2.VideoCapture(str(self.camera_index), cv2.CAP_FFMPEG)
            if not capture.isOpened():
                raise CameraError(
                    f"Could not open stream {describe_camera(self.camera_index)}. Check the "
                    "address, path and credentials, and that this device can reach the camera."
                )
            self._capture = LatestFrameReader(capture)
            # Frames are dropped, so frame counting would understate elapsed
            # time; MediaPipe's tracker gets the real clock instead.
            self._live_clock = True
        else:
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

        # Force CPU: the default delegate selection tries GPU on some
        # machines (notably some Macs) where MediaPipe's GPU service isn't
        # reliably available from a plain Python venv, crashing with "Check
        # failed: service Service is unavailable" instead of falling back -
        # CPU is fast enough for landmarks anyway.
        detection, presence, tracking = self._hand_options
        self._hands = vision.HandLandmarker.create_from_options(
            vision.HandLandmarkerOptions(
                base_options=BaseOptions(
                    model_asset_path=str(self._hand_model_path), delegate=BaseOptions.Delegate.CPU
                ),
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
                    base_options=BaseOptions(
                        model_asset_path=str(self._pose_model_path), delegate=BaseOptions.Delegate.CPU
                    ),
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
        # rejects the stream if one repeats - so a frame counter drives it
        # rather than the wall clock, which can stall or step backwards. A
        # stream drops frames, so it uses the monotonic clock, forced upward.
        frame_index = 0
        started = time.monotonic()
        last_timestamp_ms = 0
        while True:
            ok, image = self._capture.read()
            if not ok:
                raise CameraError(
                    f"Camera {describe_camera(self.camera_index)} stopped delivering frames."
                )

            if self._live_clock and image.shape[1] > self.width:
                # Landmarks are normalised, so scaling changes nothing but cost.
                scale = self.width / image.shape[1]
                image = cv2.resize(
                    image, (self.width, round(image.shape[0] * scale)), interpolation=cv2.INTER_AREA
                )

            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            if self._live_clock:
                timestamp_ms = max(int((time.monotonic() - started) * 1000), last_timestamp_ms + 1)
            else:
                timestamp_ms = int(frame_index * 1000 / 30) + 1
            last_timestamp_ms = timestamp_ms
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

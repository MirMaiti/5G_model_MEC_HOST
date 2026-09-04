"""A landmark source backed by recorded ``.npz`` clips.

This is how the network path, the server and the speech output get tested
without a camera in the loop - and how a fixed clip can be replayed to compare
two models on identical input.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Iterator, List, Optional, Sequence, Union

import numpy as np

from .base import LandmarkFrame, LandmarkSource

PathLike = Union[str, Path]


class ReplaySource(LandmarkSource):
    """Replays landmark clips at a chosen frame rate.

    Args:
        paths: ``.npz`` clips written by the collector.
        fps: Replay rate. ``0`` replays as fast as possible, for tests.
        loop: Restart from the first clip when the last one ends.

    Raises:
        FileNotFoundError: If a path does not exist.
        ValueError: If no paths are given.
    """

    def __init__(
        self, paths: Sequence[PathLike], fps: float = 30.0, loop: bool = False
    ) -> None:
        if not paths:
            raise ValueError("ReplaySource needs at least one clip.")
        self.paths: List[Path] = []
        for path in paths:
            resolved = Path(path)
            if not resolved.exists():
                raise FileNotFoundError(f"No clip at {resolved}")
            self.paths.append(resolved)
        self.fps = float(fps)
        self.loop = bool(loop)

    @classmethod
    def from_directory(
        cls, directory: PathLike, fps: float = 30.0, loop: bool = False, pattern: str = "*.npz"
    ) -> "ReplaySource":
        """Replay every clip in a directory tree, in sorted order.

        Raises:
            FileNotFoundError: If the directory holds no matching clips.
        """
        root = Path(directory)
        found = sorted(root.rglob(pattern))
        if not found:
            raise FileNotFoundError(f"No {pattern} clips under {root}")
        return cls(found, fps=fps, loop=loop)

    def frames(self) -> Iterator[LandmarkFrame]:
        """Yield every frame of every clip, paced to ``fps``."""
        interval = 1.0 / self.fps if self.fps > 0 else 0.0
        while True:
            for path in self.paths:
                with np.load(path, allow_pickle=False) as clip:
                    landmarks = clip["landmarks"].astype(np.float32)
                    mask = clip["mask"].astype(np.float32)
                for index in range(landmarks.shape[0]):
                    started = time.perf_counter()
                    yield LandmarkFrame(
                        landmarks=landmarks[index],
                        mask=mask[index],
                        timestamp_ms=time.time() * 1000.0,
                    )
                    if interval:
                        remaining = interval - (time.perf_counter() - started)
                        if remaining > 0:
                            time.sleep(remaining)
            if not self.loop:
                return


class SyntheticSource(LandmarkSource):
    """Generates arbitrary landmark motion, for load-testing the link.

    The output is meaningless as signing. It exists to measure throughput and
    latency without a camera or a signer.

    Args:
        num_landmarks: Landmarks per frame, matching the server's layout.
        count: Frames to emit; ``0`` runs until stopped.
        fps: Emission rate.
        seed: Fixes the sequence so runs are comparable.
    """

    def __init__(
        self, num_landmarks: int, count: int = 0, fps: float = 30.0, seed: int = 0
    ) -> None:
        self.num_landmarks = int(num_landmarks)
        self.count = int(count)
        self.fps = float(fps)
        self._rng = np.random.default_rng(seed)

    def frames(self) -> Iterator[LandmarkFrame]:
        """Yield synthetic frames until ``count`` is reached."""
        interval = 1.0 / self.fps if self.fps > 0 else 0.0
        emitted = 0
        phase = 0.0
        while self.count == 0 or emitted < self.count:
            started = time.perf_counter()
            phase += 0.1
            base = np.tile(
                np.array([0.5 + 0.1 * np.sin(phase), 0.5 + 0.1 * np.cos(phase), 0.0], dtype=np.float32),
                (self.num_landmarks, 1),
            )
            jitter = self._rng.normal(0.0, 0.02, size=(self.num_landmarks, 3)).astype(np.float32)
            yield LandmarkFrame(
                landmarks=base + jitter,
                mask=np.ones(self.num_landmarks, dtype=np.float32),
                timestamp_ms=time.time() * 1000.0,
            )
            emitted += 1
            if interval:
                remaining = interval - (time.perf_counter() - started)
                if remaining > 0:
                    time.sleep(remaining)

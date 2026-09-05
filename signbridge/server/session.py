"""Per-connection state: the rolling buffer and the smoothing that sits on top.

Each TCP connection gets its own :class:`InferenceSession`, so two capture
devices streaming at once never contaminate each other's buffers.

Frames arrive one at a time. Running the model on every one would be wasteful
and jittery, so the session accumulates a window, re-runs the model every few
frames, and majority-votes over recent results before reporting anything. A
prediction is only announced as ``changed`` when the smoothed label actually
moves, which is the signal the host uses to decide when to speak.
"""

from __future__ import annotations

import collections
import time
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, List, Optional

import numpy as np

from ..features import FeatureExtractor
from ..sequence import resample_sequence
from .predictor import Predictor

#: Label reported when nothing is confident enough to name.
SILENCE = ""


@dataclass
class SessionConfig:
    """Buffering and smoothing settings.

    Args:
        window_size: Frames kept in the rolling buffer.
        inference_interval: Run the model every N frames.
        min_buffer: Frames required before the first prediction.
        vote_window: Predictions to majority-vote over; 1 disables smoothing.
        min_confidence: Below this, the prediction is reported as silence.
    """

    window_size: int = 45
    inference_interval: int = 5
    min_buffer: int = 20
    vote_window: int = 5
    min_confidence: float = 0.6


class InferenceSession:
    """Accumulates frames for one connection and predicts on a schedule.

    Args:
        predictor: The shared model wrapper.
        config: Buffering and smoothing settings.
    """

    def __init__(self, predictor: Predictor, config: Optional[SessionConfig] = None) -> None:
        self.predictor = predictor
        self.config = config or SessionConfig()
        self._buffer: Deque[np.ndarray] = collections.deque(
            maxlen=max(1, self.config.window_size)
        )
        self._votes: Deque[str] = collections.deque(maxlen=max(1, self.config.vote_window))
        self.frames_seen = 0
        self.current_label = SILENCE
        #: Consecutive frames added with no hand detected at all. Used to skip
        #: the model - rather than trust it to learn the pattern - when the
        #: buffer holds nothing but empty frames.
        self._frames_without_hands = 0

    @property
    def extractor(self) -> FeatureExtractor:
        """The feature extractor the model was trained with."""
        return self.predictor.extractor

    @property
    def buffered(self) -> int:
        """Frames currently in the rolling buffer."""
        return len(self._buffer)

    def reset(self) -> None:
        """Clear the buffer and vote history, as at the start of a new sign."""
        self._buffer.clear()
        self._votes.clear()
        self.frames_seen = 0
        self.current_label = SILENCE
        self._frames_without_hands = 0

    def add_landmarks(self, landmarks: np.ndarray, mask: np.ndarray) -> Optional[Dict[str, Any]]:
        """Add one frame of raw landmarks and predict when one is due.

        Feature extraction happens here rather than on the capture device: the
        host stays a thin sensor, and changing the normalisation only means
        redeploying the server.

        Returns:
            A response dict on inference frames, otherwise ``None`` - the host
            should simply keep sending.
        """
        return self.add_features(self.extractor.transform_frame(landmarks, mask))

    def add_features(self, features: np.ndarray) -> Optional[Dict[str, Any]]:
        """Add one already-extracted feature vector.

        Returns:
            A response dict on inference frames, otherwise ``None``.
        """
        features = np.asarray(features, dtype=np.float32)
        self._buffer.append(features)
        self.frames_seen += 1
        self._frames_without_hands = (
            0 if self.extractor.hands_present(features) else self._frames_without_hands + 1
        )

        if len(self._buffer) < self.config.min_buffer:
            return None
        if self.frames_seen % max(1, self.config.inference_interval) != 0:
            return None

        started = time.perf_counter()
        # Nothing currently buffered had a hand in it at all - this is certain,
        # not a judgement call, so it is handled as a rule instead of asking a
        # model trained on a finite number of clips to recognise an all-zero
        # input. Also skips a wasted forward pass.
        no_hands = self._frames_without_hands >= len(self._buffer)
        if no_hands:
            raw_label, confidence, top = SILENCE, 0.0, {}
        else:
            # The buffer is shorter than the training window until it fills;
            # resample so a half-full buffer still looks like a whole clip.
            window = resample_sequence(np.stack(self._buffer), self.predictor.window)
            prediction = self.predictor.predict(window)
            raw_label, confidence, top = prediction.label, prediction.confidence, prediction.top
        elapsed_ms = (time.perf_counter() - started) * 1000.0

        candidate = raw_label if (not no_hands and confidence >= self.config.min_confidence) else SILENCE
        self._votes.append(candidate)
        stable = collections.Counter(self._votes).most_common(1)[0][0]

        changed = stable != self.current_label
        self.current_label = stable

        return {
            "label": stable,
            "raw_label": raw_label,
            "confidence": round(float(confidence), 4),
            "top": {k: round(float(v), 4) for k, v in top.items()},
            "changed": changed,
            "buffered": len(self._buffer),
            "frames_seen": self.frames_seen,
            "inference_ms": round(elapsed_ms, 2),
            "untrained": self.predictor.untrained,
        }


def session_config_from(config: Any) -> SessionConfig:
    """Build a :class:`SessionConfig` from a :class:`~signbridge.config.ServerConfig`."""
    return SessionConfig(
        window_size=config.window_size,
        inference_interval=config.inference_interval,
        min_buffer=config.min_buffer,
        vote_window=config.vote_window,
        min_confidence=config.min_confidence,
    )

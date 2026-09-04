"""Buffering, the inference schedule, and vote smoothing."""

from __future__ import annotations

import numpy as np
import pytest

from signbridge.features import FeatureExtractor
from signbridge.landmarks import HANDS
from signbridge.server.predictor import Predictor, Prediction, UntrainedPredictor
from signbridge.server.session import InferenceSession, SessionConfig
from tests.conftest import make_frame


class ScriptedPredictor(Predictor):
    """Returns a caller-supplied sequence of labels, for testing smoothing."""

    def __init__(self, extractor, labels, script):
        self._extractor = extractor
        self._labels = list(labels)
        self._script = list(script)
        self.calls = 0

    @property
    def labels(self):
        return self._labels

    @property
    def window(self):
        return 45

    @property
    def extractor(self):
        return self._extractor

    def predict(self, features):
        label, confidence = self._script[min(self.calls, len(self._script) - 1)]
        self.calls += 1
        return Prediction(label=label, confidence=confidence, top={label: confidence})


@pytest.fixture
def extractor():
    return FeatureExtractor(HANDS)


def feed(session, rng, count):
    """Push ``count`` frames through the session, collecting the replies."""
    results = []
    for _ in range(count):
        landmarks, mask = make_frame(rng)
        reply = session.add_landmarks(landmarks, mask)
        if reply is not None:
            results.append(reply)
    return results


def test_no_prediction_before_min_buffer(extractor, rng):
    session = InferenceSession(
        UntrainedPredictor(extractor, ["a", "b"], 45),
        SessionConfig(min_buffer=20, inference_interval=1, min_confidence=0.0),
    )
    assert feed(session, rng, 19) == []


def test_predicts_on_the_configured_interval(extractor, rng):
    session = InferenceSession(
        UntrainedPredictor(extractor, ["a", "b"], 45),
        SessionConfig(min_buffer=10, inference_interval=5, min_confidence=0.0),
    )
    # Frames 10, 15, 20, 25, 30 are the multiples of 5 at or past the minimum.
    assert len(feed(session, rng, 30)) == 5


def test_buffer_is_capped_at_window_size(extractor, rng):
    session = InferenceSession(
        UntrainedPredictor(extractor, ["a", "b"], 45),
        SessionConfig(window_size=20, min_buffer=5, inference_interval=5, min_confidence=0.0),
    )
    feed(session, rng, 60)
    assert session.buffered == 20


def test_low_confidence_reports_silence(extractor, rng):
    session = InferenceSession(
        ScriptedPredictor(extractor, ["a", "b"], [("a", 0.2)] * 10),
        SessionConfig(min_buffer=5, inference_interval=5, vote_window=1, min_confidence=0.6),
    )
    replies = feed(session, rng, 20)
    assert replies and all(reply["label"] == "" for reply in replies)
    assert replies[0]["raw_label"] == "a"


def test_vote_window_suppresses_a_single_flicker(extractor, rng):
    script = [("a", 0.9), ("a", 0.9), ("b", 0.9), ("a", 0.9), ("a", 0.9)]
    session = InferenceSession(
        ScriptedPredictor(extractor, ["a", "b"], script),
        SessionConfig(min_buffer=5, inference_interval=5, vote_window=3, min_confidence=0.5),
    )
    labels = [reply["label"] for reply in feed(session, rng, 25)]
    assert labels == ["a", "a", "a", "a", "a"]


def test_changed_flag_marks_only_transitions(extractor, rng):
    script = [("a", 0.9)] * 3 + [("b", 0.9)] * 3
    session = InferenceSession(
        ScriptedPredictor(extractor, ["a", "b"], script),
        SessionConfig(min_buffer=5, inference_interval=5, vote_window=1, min_confidence=0.5),
    )
    replies = feed(session, rng, 30)
    changes = [reply["label"] for reply in replies if reply["changed"]]
    assert changes == ["a", "b"]


def test_reset_clears_buffer_and_history(extractor, rng):
    session = InferenceSession(
        UntrainedPredictor(extractor, ["a", "b"], 45),
        SessionConfig(min_buffer=5, inference_interval=5, min_confidence=0.0),
    )
    feed(session, rng, 20)
    session.reset()
    assert session.buffered == 0
    assert session.frames_seen == 0
    assert session.current_label == ""


def test_partial_buffer_is_resampled_to_the_model_window(extractor, rng):
    """A half-full buffer must still reach the model as a full window."""
    seen = {}

    class Recording(UntrainedPredictor):
        def predict(self, features):
            seen["shape"] = features.shape
            return super().predict(features)

    session = InferenceSession(
        Recording(extractor, ["a", "b"], 45),
        SessionConfig(window_size=45, min_buffer=10, inference_interval=10, min_confidence=0.0),
    )
    feed(session, rng, 10)
    assert seen["shape"] == (45, extractor.dim)


def test_untrained_flag_propagates_to_the_reply(extractor, rng):
    session = InferenceSession(
        UntrainedPredictor(extractor, ["a", "b"], 45),
        SessionConfig(min_buffer=5, inference_interval=5, min_confidence=0.0),
    )
    assert feed(session, rng, 10)[0]["untrained"] is True

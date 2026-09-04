"""Speech gating: the rules that stop the host stuttering."""

from __future__ import annotations

import pytest

from signbridge.host.tts import NullBackend, Speaker, backend_names, build_backend


class RecordingBackend(NullBackend):
    """Captures phrases instead of speaking them."""

    def __init__(self):
        self.spoken = []

    def say(self, text):
        self.spoken.append(text)


@pytest.fixture
def speaker():
    backend = RecordingBackend()
    speaker = Speaker(backend, repeat_cooldown=2.0, stability_seconds=0.5)
    yield speaker, backend
    speaker.close()


def test_a_label_must_hold_before_it_is_spoken(speaker):
    sp, _ = speaker
    assert sp.offer("hello", now=100.0) is False   # first sighting
    assert sp.offer("hello", now=100.3) is False   # not held long enough
    assert sp.offer("hello", now=100.6) is True    # held 0.6s > 0.5s


def test_flicker_resets_the_timer(speaker):
    sp, _ = speaker
    sp.offer("hello", now=100.0)
    sp.offer("yes", now=100.2)      # a different label restarts the clock
    assert sp.offer("hello", now=100.4) is False
    assert sp.offer("hello", now=101.0) is True


def test_same_phrase_does_not_repeat_within_the_cooldown(speaker):
    sp, _ = speaker
    sp.offer("hello", now=100.0)
    assert sp.offer("hello", now=100.6) is True
    assert sp.offer("hello", now=101.5) is False   # inside the 2s cooldown
    assert sp.offer("hello", now=103.0) is True    # past it


def test_silence_clears_the_candidate(speaker):
    sp, _ = speaker
    sp.offer("hello", now=100.0)
    assert sp.offer("", now=100.2) is False
    assert sp.offer("hello", now=100.4) is False   # candidate had to restart


def test_a_different_label_speaks_without_waiting_for_the_cooldown(speaker):
    sp, _ = speaker
    sp.offer("hello", now=100.0)
    assert sp.offer("hello", now=100.6) is True
    sp.offer("thanks", now=100.7)
    assert sp.offer("thanks", now=101.3) is True


def test_phrases_reach_the_backend(speaker):
    import time

    sp, backend = speaker
    sp.offer("hello", now=100.0)
    sp.offer("hello", now=100.6)
    time.sleep(0.3)
    assert backend.spoken == ["hello"]


def test_null_backend_is_always_available():
    assert "null" in backend_names()
    assert build_backend("null") is not None


def test_unknown_backend_names_the_alternatives():
    with pytest.raises(KeyError, match="Available"):
        build_backend("robot_voice")

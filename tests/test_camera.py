"""Camera selection and the newest-frame reader used for network streams."""

from __future__ import annotations

import threading
import time

from signbridge.capture.mediapipe_source import (
    CAMERA_ENV,
    LatestFrameReader,
    describe_camera,
    is_stream,
    parse_camera,
    resolve_camera,
)


def test_digits_are_a_device_index():
    assert parse_camera("0") == 0
    assert parse_camera(" 2 ") == 2


def test_paths_and_urls_stay_strings():
    assert parse_camera("/dev/video0") == "/dev/video0"
    assert parse_camera("rtsp://cam/stream") == "rtsp://cam/stream"


def test_flag_beats_environment_beats_config(monkeypatch):
    monkeypatch.setenv(CAMERA_ENV, "rtsp://env/stream")
    assert resolve_camera("1", 0) == 1
    assert resolve_camera(None, 0) == "rtsp://env/stream"
    monkeypatch.delenv(CAMERA_ENV)
    assert resolve_camera(None, 3) == 3


def test_blank_environment_falls_back_to_config(monkeypatch):
    monkeypatch.setenv(CAMERA_ENV, "  ")
    assert resolve_camera(None, 0) == 0


def test_only_urls_are_streams():
    assert is_stream("rtsp://192.168.1.64/stream")
    assert not is_stream("/dev/video0")
    assert not is_stream(0)


def test_password_is_masked_in_messages():
    described = describe_camera("rtsp://admin:secret@192.168.1.64:554/Streaming/Channels/102")
    assert "secret" not in described
    assert described == "rtsp://admin:***@192.168.1.64:554/Streaming/Channels/102"


def test_urls_without_a_password_are_unchanged():
    assert describe_camera("rtsp://192.168.1.64/stream") == "rtsp://192.168.1.64/stream"
    assert describe_camera(0) == "0"


class FakeCapture:
    """Delivers numbered frames as fast as asked, then ends."""

    def __init__(self, count):
        self.count = count
        self.sent = 0
        self.released = False

    def read(self):
        if self.sent >= self.count:
            return False, None
        self.sent += 1
        time.sleep(0.001)
        return True, self.sent

    def release(self):
        self.released = True


def test_slow_reader_skips_to_the_newest_frame():
    capture = FakeCapture(count=200)
    reader = LatestFrameReader(capture, timeout=2.0)
    ok, first = reader.read()
    time.sleep(0.5)
    ok, later = reader.read()
    assert ok
    assert later - first > 50, "frames that arrived while the caller was busy should be skipped"
    reader.release()


def test_each_frame_is_handed_out_once_and_end_is_reported():
    capture = FakeCapture(count=3)
    reader = LatestFrameReader(capture, timeout=1.0)
    seen = []
    while True:
        ok, frame = reader.read()
        if not ok:
            break
        seen.append(frame)
    assert seen == sorted(set(seen))
    assert seen[-1] == 3
    reader.release()
    assert capture.released


def test_read_times_out_when_the_stream_stalls():
    class Stalled:
        def __init__(self):
            self.gate = threading.Event()

        def read(self):
            self.gate.wait()
            return False, None

        def release(self):
            pass

    capture = Stalled()
    reader = LatestFrameReader(capture, timeout=0.1)
    assert reader.read() == (False, None)
    capture.gate.set()
    reader.release()

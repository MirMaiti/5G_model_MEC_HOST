"""The capture host's MJPEG camera preview."""

from __future__ import annotations

import http.client
import threading
import time

import numpy as np
import pytest

from signbridge.host.video_server import BOUNDARY, VideoServer


@pytest.fixture
def video():
    server = VideoServer(("127.0.0.1", 0)).start()
    yield server
    server.stop()


def wait_for(condition, timeout=5.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if condition():
            return True
        time.sleep(0.02)
    return False


def test_nothing_is_encoded_while_nobody_watches(video):
    image = np.zeros((360, 640, 3), dtype=np.uint8)
    called = []
    assert video.offer(image, annotate=lambda frame: called.append(frame) or frame) is False
    assert called == []
    assert video.latest() is None


def test_announce_tells_the_mec_where_the_stream_is(video):
    assert video.announce() == {"port": video.port, "mirrored": True}


def test_a_watching_browser_receives_jpeg_frames(video):
    connection = http.client.HTTPConnection("127.0.0.1", video.port, timeout=5)
    connection.request("GET", "/video.mjpg")
    response = connection.getresponse()
    assert response.getheader("Content-Type") == f"multipart/x-mixed-replace; boundary={BOUNDARY}"
    assert wait_for(lambda: video.viewers == 1)

    video.publish_jpeg(b"\xff\xd8first\xff\xd9")
    assert response.fp.readline().strip() == f"--{BOUNDARY}".encode()
    headers = {}
    while True:
        line = response.fp.readline().strip()
        if not line:
            break
        key, _, value = line.decode().partition(":")
        headers[key.lower()] = value.strip()
    assert headers["content-type"] == "image/jpeg"
    assert response.fp.read(int(headers["content-length"])) == b"\xff\xd8first\xff\xd9"

    response.close()
    connection.close()

    def gone():
        video.publish_jpeg(b"\xff\xd8next\xff\xd9")
        return video.viewers == 0

    assert wait_for(gone), "a closed browser is noticed at the next frame, and encoding stops"



def test_frames_are_scaled_annotated_and_encoded_for_viewers(video):
    pytest.importorskip("cv2")
    video._viewer_joined()
    try:
        image = np.full((720, 1280, 3), 90, dtype=np.uint8)
        seen = []
        assert video.offer(image, annotate=lambda frame: seen.append(frame.shape) or frame) is True
        assert seen == [(360, 640, 3)]
        assert video.latest()[:2] == b"\xff\xd8"
        assert image.max() == 90, "the caller's frame must not be drawn on"
    finally:
        video._viewer_left()


def test_snapshot_before_any_frame_explains_itself(video):
    connection = http.client.HTTPConnection("127.0.0.1", video.port, timeout=5)
    connection.request("GET", "/snapshot.jpg")
    assert connection.getresponse().status == 503

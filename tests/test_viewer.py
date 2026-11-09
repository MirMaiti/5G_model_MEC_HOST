"""The live web viewer: event fan-out, and predictions reaching a browser."""

from __future__ import annotations

import http.client
import json
import threading
import time

import pytest

from signbridge.capture.replay_source import SyntheticSource
from signbridge.features import FeatureExtractor
from signbridge.host.client import HostClient, run_session
from signbridge.landmarks import HANDS
from signbridge.server.predictor import UntrainedPredictor
from signbridge.server.session import SessionConfig
from signbridge.server.tcp_server import SignBridgeServer
from signbridge.server.viewer import SUBSCRIBER_QUEUE, FrameRate, PredictionHub, ViewerServer


def test_every_watcher_gets_every_event():
    hub = PredictionHub()
    first, second = hub.subscribe(), hub.subscribe()
    hub.publish({"type": "prediction", "label": "hello"})
    assert first.get_nowait()["label"] == "hello"
    assert second.get_nowait()["label"] == "hello"


def test_a_stalled_watcher_loses_old_events_instead_of_blocking():
    hub = PredictionHub()
    stalled = hub.subscribe()
    for index in range(SUBSCRIBER_QUEUE + 50):
        hub.publish({"n": index})
    received = [stalled.get_nowait()["n"] for _ in range(stalled.qsize())]
    assert len(received) == SUBSCRIBER_QUEUE
    assert received[-1] == SUBSCRIBER_QUEUE + 49


def test_unsubscribed_watchers_stop_receiving():
    hub = PredictionHub()
    watcher = hub.subscribe()
    hub.unsubscribe(watcher)
    hub.publish({"type": "prediction"})
    assert watcher.empty()
    assert hub.watchers == 0


def test_snapshot_lists_connected_hosts():
    hub = PredictionHub()
    hub.host_connected("10.0.0.5:4000")
    assert [host["peer"] for host in hub.snapshot()["hosts"]] == ["10.0.0.5:4000"]
    hub.host_disconnected("10.0.0.5:4000", frames=12)
    assert hub.snapshot()["hosts"] == []


def test_frame_rate_over_a_sliding_window():
    rate = FrameRate(span_seconds=5.0)
    value = 0.0
    for index in range(11):
        value = rate.tick(now=index * 0.4)
    assert value == pytest.approx(2.5)
    assert rate.tick(now=100.0) == 0.0


@pytest.fixture
def live():
    """A TCP server publishing into a viewer, both on ephemeral ports."""
    hub = PredictionHub()
    predictor = UntrainedPredictor(FeatureExtractor(HANDS), ["hello", "thanks", "yes"], window=45)
    hub.info = predictor.info()
    server = SignBridgeServer(
        ("127.0.0.1", 0),
        predictor,
        SessionConfig(window_size=45, min_buffer=10, inference_interval=5, min_confidence=0.0),
        hub=hub,
    )
    threading.Thread(target=server.serve_forever, daemon=True).start()
    viewer = ViewerServer(("127.0.0.1", 0), hub).start()
    yield server, viewer
    viewer.stop()
    server.shutdown()
    server.server_close()


def read_events(response, wanted, timeout=10.0):
    """Collect SSE events until one of type ``wanted`` arrives."""
    events = []
    deadline = time.time() + timeout
    while time.time() < deadline:
        line = response.fp.readline().decode().strip()
        if not line.startswith("data: "):
            continue
        event = json.loads(line[len("data: "):])
        events.append(event)
        if event["type"] == wanted:
            return events
    raise AssertionError(f"no {wanted!r} event; got {[event['type'] for event in events]}")


def test_page_and_status_are_served(live):
    _, viewer = live
    connection = http.client.HTTPConnection("127.0.0.1", viewer.port, timeout=5)
    connection.request("GET", "/")
    page = connection.getresponse()
    assert page.status == 200
    body = page.read().decode()
    assert "<title>SignBridge Live</title>" in body
    assert "https://" not in body, "the page must work on a network with no internet"

    connection.request("GET", "/status")
    status = json.loads(connection.getresponse().read())
    assert status["info"]["labels"] == ["hello", "thanks", "yes"]

    connection.request("GET", "/nope")
    assert connection.getresponse().status == 404


def test_predictions_from_a_capture_host_reach_the_browser(live):
    server, viewer = live
    stream = http.client.HTTPConnection("127.0.0.1", viewer.port, timeout=10)
    stream.request("GET", "/events")
    response = stream.getresponse()
    assert response.getheader("Content-Type") == "text/event-stream"
    assert read_events(response, "status")[0]["hosts"] == []

    client = HostClient("127.0.0.1", server.port, connect_timeout=5.0)
    client.connect()
    try:
        run_session(client, SyntheticSource(num_landmarks=42, count=30, fps=0), linger_seconds=1.0)
    finally:
        client.close()

    events = read_events(response, "prediction")
    assert events[0]["type"] == "host" and events[0]["connected"] is True
    prediction = events[-1]
    assert prediction["peer"] == events[0]["peer"]
    assert {"label", "confidence", "fps", "frames_seen", "inference_ms"} <= prediction.keys()

    events = read_events(response, "host")
    assert events[-1]["connected"] is False
    stream.close()

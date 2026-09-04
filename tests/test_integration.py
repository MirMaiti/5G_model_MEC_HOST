"""End-to-end: host and server over a real TCP socket.

These are the tests that would have caught the two bugs that actually bit
during development - a client assuming request/response pairing, and a finite
source exiting before the server's last replies arrived.
"""

from __future__ import annotations

import threading
import time

import numpy as np
import pytest

from signbridge.capture.replay_source import SyntheticSource
from signbridge.config import load_config
from signbridge.features import FeatureExtractor
from signbridge.host.client import ConnectionFailed, HostClient, run_session
from signbridge.landmarks import HANDS
from signbridge.server.predictor import UntrainedPredictor
from signbridge.server.session import SessionConfig
from signbridge.server.tcp_server import SignBridgeServer


@pytest.fixture
def server():
    """A server on an ephemeral port, torn down after the test."""
    extractor = FeatureExtractor(HANDS)
    predictor = UntrainedPredictor(extractor, ["hello", "thanks", "yes"], window=45)
    instance = SignBridgeServer(
        ("127.0.0.1", 0),
        predictor,
        SessionConfig(window_size=45, min_buffer=10, inference_interval=5, min_confidence=0.0),
        max_connections=2,
    )
    thread = threading.Thread(target=instance.serve_forever, daemon=True)
    thread.start()
    yield instance
    instance.shutdown()
    instance.server_close()


def connect(server, layout="hands"):
    """Open a client against the fixture server."""
    client = HostClient("127.0.0.1", server.port, layout=layout, connect_timeout=5.0)
    client.connect()
    return client


def test_handshake_reports_the_model(server):
    client = connect(server)
    try:
        assert client.info["labels"] == ["hello", "thanks", "yes"]
        assert client.info["feature_dim"] == 136
        assert client.info["layout"]["num_landmarks"] == 42
        assert client.untrained is True
    finally:
        client.close()


def test_streaming_produces_predictions(server, rng):
    client = connect(server)
    try:
        source = SyntheticSource(num_landmarks=42, count=60, fps=0)
        stats = run_session(client, source, linger_seconds=2.0)
        assert stats.frames_sent == 60
        assert stats.predictions > 0
    finally:
        client.close()


def test_predictions_echo_the_frame_sequence(server):
    """The echoed seq is what lets the host measure end-to-end latency."""
    client = connect(server)
    try:
        received = []
        run_session(
            client,
            SyntheticSource(num_landmarks=42, count=40, fps=0),
            on_prediction=received.append,
            linger_seconds=2.0,
        )
        assert received
        assert all("seq" in item and "capture_t_ms" in item for item in received)
        assert received[0]["seq"] < received[-1]["seq"]
    finally:
        client.close()


def test_trailing_predictions_are_not_dropped(server):
    """A finite source finishes sending well before the last replies arrive."""
    client = connect(server)
    try:
        with_linger = run_session(
            client, SyntheticSource(num_landmarks=42, count=200, fps=0), linger_seconds=2.0
        )
    finally:
        client.close()

    client = connect(server)
    try:
        without_linger = run_session(
            client, SyntheticSource(num_landmarks=42, count=200, fps=0), linger_seconds=0.0
        )
    finally:
        client.close()

    assert with_linger.predictions > without_linger.predictions


def test_layout_mismatch_is_refused_at_the_handshake(server):
    with pytest.raises(ConnectionFailed, match="layout"):
        connect(server, layout="hands_pose")


def test_connection_limit_is_enforced(server):
    first = connect(server)
    second = connect(server)
    try:
        with pytest.raises(ConnectionFailed, match="limit"):
            connect(server)
    finally:
        first.close()
        second.close()


def test_wrong_landmark_count_does_not_kill_the_session(server):
    """A bad frame must produce an error, not drop the connection."""
    from signbridge.capture.base import LandmarkFrame

    client = connect(server)
    try:
        bad = LandmarkFrame(
            landmarks=np.zeros((75, 3), dtype=np.float32),
            mask=np.ones(75, dtype=np.float32),
            timestamp_ms=time.time() * 1000.0,
        )
        client.send_frame(bad)
        time.sleep(0.3)

        good = SyntheticSource(num_landmarks=42, count=40, fps=0)
        stats = run_session(client, good, linger_seconds=2.0)
        assert stats.predictions > 0
    finally:
        client.close()


def test_reset_clears_the_server_buffer(server):
    client = connect(server)
    try:
        source = SyntheticSource(num_landmarks=42, count=30, fps=0)
        for frame in source.frames():
            client.send_frame(frame)
        time.sleep(0.3)
        client.reset()
        time.sleep(0.3)

        buffered = []
        run_session(
            client,
            SyntheticSource(num_landmarks=42, count=15, fps=0),
            on_prediction=lambda r: buffered.append(r["buffered"]),
            linger_seconds=2.0,
        )
        # After a reset the buffer refills from empty, so it cannot already be full.
        assert buffered and min(buffered) <= 15
    finally:
        client.close()


def test_unreachable_server_explains_how_to_start_one():
    client = HostClient("127.0.0.1", 1, connect_timeout=1.0)
    with pytest.raises(ConnectionFailed, match="signbridge.cli.serve"):
        client.connect()


def test_train_then_serve_round_trip(tmp_path, rng):
    """The whole pipeline: record -> train -> serve -> predict over TCP."""
    from signbridge.model.checkpoint import load_checkpoint
    from signbridge.server.predictor import TorchPredictor
    from signbridge.training.dataset import ClipDataset, build_vocabulary, load_clips
    from signbridge.training.trainer import train

    # Two well-separated synthetic signs. This exercises the code path; it is
    # not a claim about accuracy on real signing.
    data_root = tmp_path / "data"
    for offset, label in ((0.2, "alpha"), (0.8, "beta")):
        for index in range(8):
            clip = np.full((45, 42, 3), offset, dtype=np.float32)
            clip += rng.normal(0, 0.01, clip.shape).astype(np.float32)
            directory = data_root / label
            directory.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                directory / f"{index}.npz",
                landmarks=clip,
                mask=np.ones((45, 42), dtype=np.float32),
                label=np.array(label),
            )

    config = load_config()
    config.training.epochs = 4
    config.training.device = "cpu"
    config.model.architecture = "mlp"
    config.model.hidden_dim = 32
    extractor = config.feature_extractor()

    clips = load_clips(data_root, extractor)
    labels = build_vocabulary(clips)
    dataset = ClipDataset(clips, labels, window=config.model.window)
    result = train(
        config=config,
        extractor=extractor,
        labels=labels,
        train_dataset=dataset,
        val_dataset=dataset,
        checkpoint_dir=tmp_path / "models",
    )
    assert result.checkpoint_path is not None

    loaded = load_checkpoint(result.checkpoint_path, device="cpu")
    assert loaded.labels == labels
    assert loaded.extractor.dim == extractor.dim

    predictor = TorchPredictor(result.checkpoint_path, device="cpu")
    assert predictor.untrained is False

    instance = SignBridgeServer(
        ("127.0.0.1", 0),
        predictor,
        SessionConfig(window_size=45, min_buffer=10, inference_interval=5, min_confidence=0.0),
    )
    thread = threading.Thread(target=instance.serve_forever, daemon=True)
    thread.start()
    try:
        client = HostClient("127.0.0.1", instance.port, connect_timeout=5.0)
        client.connect()
        assert client.untrained is False
        received = []
        try:
            run_session(
                client,
                SyntheticSource(num_landmarks=42, count=40, fps=0),
                on_prediction=received.append,
                linger_seconds=2.0,
            )
        finally:
            client.close()
        assert received
        assert received[0]["label"] in labels
        assert received[0]["untrained"] is False
    finally:
        instance.shutdown()
        instance.server_close()

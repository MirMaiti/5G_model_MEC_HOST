"""The MEC-side TCP server.

Landmarks in, predictions out. One thread per connection, each with its own
:class:`~signbridge.server.session.InferenceSession`; the model itself is shared
and guarded by a lock inside the predictor.

Nothing in this module imports MediaPipe or OpenCV. The server never sees a
pixel - that is the whole architecture in one sentence, and it is what lets the
MEC node stay small and run on hardware with no camera stack at all.
"""

from __future__ import annotations

import logging
import socket
import socketserver
import threading
import time
from typing import Any, Dict, Optional

import numpy as np

from ..protocol import (
    Connection,
    MessageType,
    ProtocolError,
    PROTOCOL_VERSION,
    decode_frame,
    decode_json,
)
from .predictor import Predictor
from .session import InferenceSession, SessionConfig

logger = logging.getLogger(__name__)

#: How long to wait for the opening HELLO before dropping the connection.
HANDSHAKE_TIMEOUT = 10.0


class _Handler(socketserver.BaseRequestHandler):
    """Serves one connection for its lifetime."""

    server: "SignBridgeServer"

    def handle(self) -> None:
        """Run the handshake, then the frame loop, until the peer goes away."""
        connection = Connection(self.request)
        peer = connection.peer

        if not self.server.acquire_slot():
            logger.warning("Refusing %s: %d connections already active", peer, self.server.max_connections)
            connection.send_json(
                MessageType.ERROR,
                {"error": f"Server is at its {self.server.max_connections}-connection limit."},
            )
            connection.close()
            return

        session = InferenceSession(self.server.predictor, self.server.session_config)
        logger.info("Host connected: %s", peer)
        try:
            if not self._handshake(connection, session):
                return
            self._serve(connection, session)
        except ProtocolError as exc:
            logger.warning("Protocol error from %s: %s", peer, exc)
            self._try_error(connection, str(exc))
        except (ConnectionResetError, BrokenPipeError, TimeoutError, socket.timeout):
            logger.info("Host %s disconnected abruptly", peer)
        except Exception:  # pragma: no cover - unexpected, but must not kill the server
            logger.exception("Unhandled error serving %s", peer)
        finally:
            self.server.release_slot()
            connection.close()
            logger.info("Host disconnected: %s (%d frames)", peer, session.frames_seen)

    def _handshake(self, connection: Connection, session: InferenceSession) -> bool:
        """Exchange HELLO/WELCOME. False when the client should be dropped."""
        self.request.settimeout(HANDSHAKE_TIMEOUT)
        message = connection.read_message()
        if message is None:
            return False
        message_type, payload = message
        if message_type is not MessageType.HELLO:
            self._try_error(connection, f"Expected HELLO first; got {message_type.name}.")
            return False

        hello = decode_json(payload)
        expected = session.extractor.layout
        claimed = hello.get("layout")
        if claimed is not None and claimed != expected.name:
            self._try_error(
                connection,
                f"This server serves the {expected.name!r} layout "
                f"({expected.num_landmarks} landmarks); the host offered {claimed!r}.",
            )
            return False

        # No read timeout once streaming: a signer pausing between signs is not
        # an error, and the OS will tell us if the link actually dies.
        self.request.settimeout(None)
        info = self.server.predictor.info()
        info.update({"protocol_version": PROTOCOL_VERSION, "server_time_ms": time.time() * 1000.0})
        connection.send_json(MessageType.WELCOME, info)
        return True

    def _serve(self, connection: Connection, session: InferenceSession) -> None:
        """Handle messages until the peer closes or says BYE."""
        expected_landmarks = session.extractor.layout.num_landmarks

        while True:
            message = connection.read_message()
            if message is None:
                return
            message_type, payload = message

            if message_type is MessageType.FRAME:
                self._on_frame(connection, session, payload, expected_landmarks)
            elif message_type is MessageType.RESET:
                session.reset()
                connection.send_json(MessageType.RESET, {"reset": True})
            elif message_type is MessageType.PING:
                connection.send_json(MessageType.PONG, decode_json(payload) if payload else {})
            elif message_type is MessageType.BYE:
                return
            else:
                self._try_error(connection, f"Unexpected message type {message_type.name}.")

    def _on_frame(
        self,
        connection: Connection,
        session: InferenceSession,
        payload: bytes,
        expected_landmarks: int,
    ) -> None:
        """Decode one frame, buffer it, and reply if a prediction came due."""
        seq, t_ms, landmarks, mask = decode_frame(payload)
        if landmarks.shape[0] != expected_landmarks:
            self._try_error(
                connection,
                f"This model expects {expected_landmarks} landmarks per frame; "
                f"got {landmarks.shape[0]}.",
            )
            return

        try:
            result = session.add_landmarks(landmarks, mask)
        except ValueError as exc:
            self._try_error(connection, str(exc))
            return

        if result is None:
            return

        result.update({"seq": seq, "capture_t_ms": t_ms, "server_t_ms": time.time() * 1000.0})
        connection.send_json(MessageType.PREDICTION, result)
        self.server.note_prediction()

    def _try_error(self, connection: Connection, message: str) -> None:
        """Send an ERROR, ignoring a peer that has already gone."""
        try:
            connection.send_json(MessageType.ERROR, {"error": message})
        except OSError:  # pragma: no cover - peer vanished mid-error
            pass


class SignBridgeServer(socketserver.ThreadingTCPServer):
    """Threaded TCP server wrapping a :class:`~signbridge.server.predictor.Predictor`.

    Args:
        address: ``(host, port)`` to bind.
        predictor: The shared model wrapper.
        session_config: Buffering and smoothing settings for each connection.
        max_connections: Refuse further hosts beyond this many.
    """

    allow_reuse_address = True
    daemon_threads = True

    def __init__(
        self,
        address: tuple,
        predictor: Predictor,
        session_config: Optional[SessionConfig] = None,
        max_connections: int = 8,
    ) -> None:
        self.predictor = predictor
        self.session_config = session_config or SessionConfig()
        self.max_connections = max(1, max_connections)
        self._active = 0
        self._predictions = 0
        self._counter_lock = threading.Lock()
        self._started = time.time()
        super().__init__(address, _Handler)

    @property
    def port(self) -> int:
        """The bound port, useful when binding to port 0 in tests."""
        return int(self.server_address[1])

    def acquire_slot(self) -> bool:
        """Take a connection slot, or return False when full."""
        with self._counter_lock:
            if self._active >= self.max_connections:
                return False
            self._active += 1
            return True

    def release_slot(self) -> None:
        """Give a connection slot back."""
        with self._counter_lock:
            self._active = max(0, self._active - 1)

    def note_prediction(self) -> None:
        """Count one prediction sent, for the stats line."""
        with self._counter_lock:
            self._predictions += 1

    def stats(self) -> Dict[str, Any]:
        """Live counters, for logging."""
        with self._counter_lock:
            return {
                "active_connections": self._active,
                "predictions_served": self._predictions,
                "uptime_seconds": round(time.time() - self._started, 1),
            }

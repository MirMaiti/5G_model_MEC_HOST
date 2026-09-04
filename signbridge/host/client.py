"""The capture-side client: landmarks out over TCP, predictions back, speech.

Predictions do not arrive one-per-frame. The server buffers and only answers
every few frames, so this is not a request/response protocol - the client must
keep sending while replies arrive whenever they are ready. A reader thread owns
the receive side and drops results into a queue; the capture loop never blocks
on the network, and the camera never stalls waiting for a prediction.
"""

from __future__ import annotations

import logging
import queue
import socket
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterator, List, Optional

from ..capture.base import LandmarkFrame, LandmarkSource
from ..protocol import Connection, MessageType, ProtocolError, decode_json

logger = logging.getLogger(__name__)


class ConnectionFailed(RuntimeError):
    """Raised when the server cannot be reached or refuses the handshake."""


@dataclass
class SessionStats:
    """Counters for the status line."""

    frames_sent: int = 0
    predictions: int = 0
    latencies_ms: List[float] = field(default_factory=list)

    def note_latency(self, value: float) -> None:
        """Record one end-to-end latency, keeping only a recent window."""
        self.latencies_ms.append(value)
        if len(self.latencies_ms) > 200:
            del self.latencies_ms[:100]

    @property
    def mean_latency_ms(self) -> float:
        """Mean of the recent latencies, or 0 when there are none."""
        return sum(self.latencies_ms) / len(self.latencies_ms) if self.latencies_ms else 0.0


class HostClient:
    """Owns the TCP connection to the MEC server.

    Args:
        host: Server address.
        port: Server port.
        layout: Layout name announced in the handshake, so a mismatch is caught
            immediately rather than producing quiet nonsense.
        connect_timeout: Seconds to wait for the connection and handshake.
    """

    def __init__(
        self,
        host: str,
        port: int,
        layout: str = "hands",
        connect_timeout: float = 5.0,
    ) -> None:
        self.host = host
        self.port = port
        self.layout = layout
        self.connect_timeout = connect_timeout
        self.info: Dict[str, Any] = {}
        self._connection: Optional[Connection] = None
        self._queue: "queue.Queue[Dict[str, Any]]" = queue.Queue()
        self._reader: Optional[threading.Thread] = None
        self._closing = threading.Event()
        self._seq = 0

    @property
    def connected(self) -> bool:
        """Whether the connection is currently open."""
        return self._connection is not None

    @property
    def untrained(self) -> bool:
        """True when the server admits its weights are untrained."""
        return bool(self.info.get("untrained", False))

    def connect(self) -> Dict[str, Any]:
        """Open the connection and complete the handshake.

        Returns:
            The server's WELCOME payload: labels, window, feature dim, device.

        Raises:
            ConnectionFailed: If the server is unreachable or rejects the host.
        """
        try:
            sock = socket.create_connection((self.host, self.port), timeout=self.connect_timeout)
        except OSError as exc:
            raise ConnectionFailed(
                f"Could not reach the MEC server at {self.host}:{self.port} - {exc}\n"
                "Start it with:\n"
                "    python -m signbridge.cli.serve --config config.yaml"
            ) from exc

        connection = Connection(sock)
        connection.send_json(
            MessageType.HELLO,
            {"layout": self.layout, "client": "signbridge-host", "t_ms": time.time() * 1000.0},
        )
        try:
            message = connection.read_message()
        except ProtocolError as exc:
            connection.close()
            raise ConnectionFailed(f"Bad handshake from {self.host}:{self.port}: {exc}") from exc
        if message is None:
            connection.close()
            raise ConnectionFailed("Server closed the connection during the handshake.")

        message_type, payload = message
        body = decode_json(payload)
        if message_type is MessageType.ERROR:
            connection.close()
            raise ConnectionFailed(f"Server refused the connection: {body.get('error')}")
        if message_type is not MessageType.WELCOME:
            connection.close()
            raise ConnectionFailed(f"Expected WELCOME; got {message_type.name}.")

        sock.settimeout(None)
        self.info = body
        self._connection = connection
        self._closing.clear()
        self._reader = threading.Thread(target=self._read_loop, name="signbridge-reader", daemon=True)
        self._reader.start()
        logger.info(
            "Connected to %s:%s - %d labels, window %s, %s",
            self.host, self.port, len(body.get("labels", [])), body.get("window"),
            body.get("device", "unknown device"),
        )
        return body

    def _read_loop(self) -> None:
        """Receive predictions and errors until the link closes."""
        connection = self._connection
        if connection is None:
            return
        try:
            while not self._closing.is_set():
                message = connection.read_message()
                if message is None:
                    break
                message_type, payload = message
                if message_type is MessageType.PREDICTION:
                    self._queue.put(decode_json(payload))
                elif message_type is MessageType.ERROR:
                    logger.warning("Server error: %s", decode_json(payload).get("error"))
                elif message_type in (MessageType.PONG, MessageType.RESET):
                    continue
        except (OSError, ProtocolError) as exc:
            if not self._closing.is_set():
                logger.warning("Connection lost: %s", exc)
        finally:
            self._queue.put({"__disconnected__": True})

    def send_frame(self, frame: LandmarkFrame) -> None:
        """Send one landmark frame.

        Raises:
            ConnectionFailed: If the link is not open or the write fails.
        """
        if self._connection is None:
            raise ConnectionFailed("Not connected.")
        try:
            self._connection.send_frame(
                self._seq, frame.timestamp_ms, frame.landmarks, frame.mask
            )
        except OSError as exc:
            raise ConnectionFailed(f"Send failed: {exc}") from exc
        self._seq += 1

    def drain(self) -> List[Dict[str, Any]]:
        """Take every prediction that has arrived since the last call.

        Returns:
            Predictions in arrival order. A ``__disconnected__`` marker is
            included when the link dropped.
        """
        results: List[Dict[str, Any]] = []
        while True:
            try:
                results.append(self._queue.get_nowait())
            except queue.Empty:
                return results

    def reset(self) -> None:
        """Ask the server to clear its rolling buffer."""
        if self._connection is not None:
            self._connection.send_json(MessageType.RESET, {})

    def close(self) -> None:
        """Say BYE and shut the connection down."""
        self._closing.set()
        if self._connection is not None:
            try:
                self._connection.send_json(MessageType.BYE, {})
            except OSError:
                pass
            self._connection.close()
            self._connection = None
        if self._reader is not None:
            self._reader.join(timeout=2.0)
            self._reader = None

    def __enter__(self) -> "HostClient":
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()


def run_session(
    client: HostClient,
    source: LandmarkSource,
    on_prediction: Optional[Callable[[Dict[str, Any]], None]] = None,
    on_frame: Optional[Callable[[LandmarkFrame, SessionStats], bool]] = None,
    stats: Optional[SessionStats] = None,
    linger_seconds: float = 1.0,
) -> SessionStats:
    """Stream a source to the server, dispatching predictions as they arrive.

    Args:
        client: An already-connected client.
        source: Where frames come from.
        on_prediction: Called with each prediction dict - this is where speech
            and console output are wired in.
        on_frame: Called with each captured frame; return False to stop. This is
            where a preview window is drawn.
        stats: Counters to accumulate into; a fresh set is made when omitted.
        linger_seconds: After the source runs dry, keep collecting replies for
            this long. A finite source - a replayed clip - can finish sending
            long before the server has answered for the last frames, and
            without this those predictions would be dropped on the floor.

    Returns:
        The accumulated stats.

    Raises:
        ConnectionFailed: If the link drops mid-session.
    """
    counters = stats or SessionStats()

    for frame in source.frames():
        client.send_frame(frame)
        counters.frames_sent += 1

        for result in client.drain():
            if result.get("__disconnected__"):
                raise ConnectionFailed("The MEC server closed the connection.")
            counters.predictions += 1
            capture_t = result.get("capture_t_ms")
            if capture_t:
                counters.note_latency(time.time() * 1000.0 - float(capture_t))
            if on_prediction is not None:
                on_prediction(result)

        if on_frame is not None and not on_frame(frame, counters):
            break

    deadline = time.monotonic() + max(0.0, linger_seconds)
    while time.monotonic() < deadline:
        pending = client.drain()
        if not pending:
            time.sleep(0.02)
            continue
        for result in pending:
            if result.get("__disconnected__"):
                return counters
            counters.predictions += 1
            capture_t = result.get("capture_t_ms")
            if capture_t:
                counters.note_latency(time.time() * 1000.0 - float(capture_t))
            if on_prediction is not None:
                on_prediction(result)

    return counters

"""A read-only web page showing what the TCP server is recognising, live.

The capture host (for example the UNO Q reading a CCTV camera) talks to the
TCP server; this module lets anyone on the same network watch the result in a
browser - current sign, confidence, a running caption, frame rate - without
touching the capture path. It is purely a spectator: it never sends anything
back to the model.

Standard library only (``http.server`` plus Server-Sent Events), and the page
loads nothing from the internet, so it works on an isolated lab network.

    GET /          the viewer page
    GET /events    text/event-stream of JSON events
    GET /status    JSON snapshot: model info and connected hosts
"""

from __future__ import annotations

import json
import logging
import queue
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

PAGE_PATH = Path(__file__).with_name("viewer.html")

#: Seconds between keep-alive comments on an idle event stream, so proxies and
#: browsers do not decide the connection is dead while nobody is signing.
KEEPALIVE_SECONDS = 15.0

#: Events buffered per browser before the oldest are dropped. A stalled tab
#: must never slow the model down.
SUBSCRIBER_QUEUE = 256


class PredictionHub:
    """Fans server events out to any number of watching browsers.

    ``publish`` never blocks: each subscriber has a bounded queue, and a
    subscriber that falls behind loses its oldest events rather than holding
    up the TCP handler that produced them.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._subscribers: List["queue.Queue[Dict[str, Any]]"] = []
        self._hosts: Dict[str, Dict[str, Any]] = {}
        self.info: Dict[str, Any] = {}

    def subscribe(self) -> "queue.Queue[Dict[str, Any]]":
        """Register a new watcher and return its event queue."""
        subscriber: "queue.Queue[Dict[str, Any]]" = queue.Queue(maxsize=SUBSCRIBER_QUEUE)
        with self._lock:
            self._subscribers.append(subscriber)
        return subscriber

    def unsubscribe(self, subscriber: "queue.Queue[Dict[str, Any]]") -> None:
        """Forget a watcher whose browser went away."""
        with self._lock:
            if subscriber in self._subscribers:
                self._subscribers.remove(subscriber)

    @property
    def watchers(self) -> int:
        """Browsers currently connected."""
        with self._lock:
            return len(self._subscribers)

    def snapshot(self) -> Dict[str, Any]:
        """Model info and currently connected capture hosts."""
        with self._lock:
            hosts = [dict(host) for host in self._hosts.values()]
        return {"type": "status", "info": self.info, "hosts": hosts, "time_ms": time.time() * 1000.0}

    def host_connected(self, peer: str) -> None:
        """A capture host opened a session."""
        with self._lock:
            self._hosts[peer] = {"peer": peer, "since_ms": time.time() * 1000.0}
        self.publish({"type": "host", "peer": peer, "connected": True})

    def host_disconnected(self, peer: str, frames: int) -> None:
        """A capture host went away."""
        with self._lock:
            self._hosts.pop(peer, None)
        self.publish({"type": "host", "peer": peer, "connected": False, "frames": frames})

    def publish(self, event: Dict[str, Any]) -> None:
        """Send one event to every watcher, dropping old events for slow ones."""
        event = dict(event)
        event.setdefault("time_ms", time.time() * 1000.0)
        with self._lock:
            subscribers = list(self._subscribers)
        for subscriber in subscribers:
            while True:
                try:
                    subscriber.put_nowait(event)
                    break
                except queue.Full:
                    try:
                        subscriber.get_nowait()
                    except queue.Empty:
                        pass


class FrameRate:
    """Frames per second over a sliding window of recent arrivals."""

    def __init__(self, span_seconds: float = 5.0) -> None:
        self.span = span_seconds
        self._times: List[float] = []

    def tick(self, now: Optional[float] = None) -> float:
        """Record one frame and return the current rate."""
        now = time.monotonic() if now is None else now
        self._times.append(now)
        cutoff = now - self.span
        while self._times and self._times[0] < cutoff:
            self._times.pop(0)
        if len(self._times) < 2:
            return 0.0
        elapsed = self._times[-1] - self._times[0]
        return (len(self._times) - 1) / elapsed if elapsed > 0 else 0.0


class _ViewerHandler(BaseHTTPRequestHandler):
    server: "ViewerServer"
    protocol_version = "HTTP/1.1"

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002 - stdlib signature
        logger.debug("viewer %s: " + format, self.address_string(), *args)

    def do_GET(self) -> None:  # noqa: N802 - stdlib naming
        path = self.path.split("?", 1)[0]
        if path in ("/", "/index.html"):
            self._send(200, "text/html; charset=utf-8", self.server.page())
        elif path == "/status":
            self._send(200, "application/json", json.dumps(self.server.hub.snapshot()).encode())
        elif path == "/events":
            self._stream()
        else:
            self._send(404, "text/plain; charset=utf-8", b"Not found\n")

    def _send(self, status: int, content_type: str, body: bytes) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _stream(self) -> None:
        """Hold the connection open and write each event as it happens."""
        hub = self.server.hub
        subscriber = hub.subscribe()
        logger.info("Viewer connected: %s (%d watching)", self.address_string(), hub.watchers)
        try:
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Connection", "keep-alive")
            self.end_headers()
            self._write_event(hub.snapshot())
            while not self.server.stopping.is_set():
                try:
                    event = subscriber.get(timeout=KEEPALIVE_SECONDS)
                except queue.Empty:
                    self.wfile.write(b": keep-alive\n\n")
                    self.wfile.flush()
                    continue
                self._write_event(event)
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError, TimeoutError):
            pass
        finally:
            hub.unsubscribe(subscriber)
            self.close_connection = True
            logger.info("Viewer disconnected: %s", self.address_string())

    def _write_event(self, event: Dict[str, Any]) -> None:
        self.wfile.write(b"data: " + json.dumps(event).encode() + b"\n\n")
        self.wfile.flush()


class ViewerServer(ThreadingHTTPServer):
    """Serves the viewer page and its event stream.

    Args:
        address: ``(host, port)`` to bind.
        hub: Where the TCP server publishes its predictions.
    """

    allow_reuse_address = True
    daemon_threads = True

    def __init__(self, address: tuple, hub: PredictionHub) -> None:
        self.hub = hub
        self.stopping = threading.Event()
        self._thread: Optional[threading.Thread] = None
        super().__init__(address, _ViewerHandler)

    @property
    def port(self) -> int:
        """The bound port, useful when binding to port 0 in tests."""
        return int(self.server_address[1])

    @staticmethod
    def page() -> bytes:
        """The viewer page, read fresh so edits show up on reload."""
        return PAGE_PATH.read_bytes()

    def start(self) -> "ViewerServer":
        """Serve in a background thread."""
        self._thread = threading.Thread(target=self.serve_forever, name="viewer", daemon=True)
        self._thread.start()
        return self

    def stop(self) -> None:
        """Stop serving and let open event streams finish."""
        self.stopping.set()
        self.shutdown()
        self.server_close()

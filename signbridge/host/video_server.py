"""An optional MJPEG preview of the capture camera, for the live web viewer.

The capture device normally shows its camera in a desktop window. A headless
board has no screen, so this serves the same annotated preview over HTTP
instead, as ``multipart/x-mixed-replace`` JPEGs that any browser plays in a
plain ``<img>`` tag.

The video goes straight from this device to the watching browser. It never
passes through the MEC server, which still only ever receives landmarks.

Encoding costs CPU on a small board, so frames are only encoded while at least
one browser is actually watching; with nobody connected :meth:`offer` returns
immediately.

    GET /video.mjpg     the live stream
    GET /snapshot.jpg   the most recent frame
"""

from __future__ import annotations

import logging
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Optional

import numpy as np

logger = logging.getLogger(__name__)

BOUNDARY = "signbridgeframe"

#: Seconds a stream waits for a new frame before repeating the last one, so a
#: paused camera does not look like a dead connection.
REPEAT_SECONDS = 5.0


class VideoServer(ThreadingHTTPServer):
    """Serves the most recent annotated camera frame as MJPEG.

    Args:
        address: ``(host, port)`` to bind.
        max_width: Frames wider than this are scaled down before encoding.
        quality: JPEG quality, 1-100.
        mirrored: Whether the frames handed to :meth:`offer` are mirrored.
            Only reported to viewers, so a skeleton drawn elsewhere lines up.
    """

    allow_reuse_address = True
    daemon_threads = True

    def __init__(
        self,
        address: tuple,
        max_width: int = 640,
        quality: int = 70,
        mirrored: bool = True,
    ) -> None:
        self.max_width = max_width
        self.quality = quality
        self.mirrored = mirrored
        self._condition = threading.Condition()
        self._jpeg: Optional[bytes] = None
        self._sequence = 0
        self._viewers = 0
        self.stopping = threading.Event()
        super().__init__(address, _VideoHandler)

    @property
    def port(self) -> int:
        """The bound port, useful when binding to port 0 in tests."""
        return int(self.server_address[1])

    @property
    def viewers(self) -> int:
        """Browsers currently streaming."""
        with self._condition:
            return self._viewers

    def announce(self) -> dict:
        """What the capture host tells the MEC, so the viewer can find the stream."""
        return {"port": self.port, "mirrored": self.mirrored}

    def start(self) -> "VideoServer":
        """Serve in a background thread."""
        threading.Thread(target=self.serve_forever, name="video", daemon=True).start()
        return self

    def stop(self) -> None:
        """Stop serving and release any waiting streams."""
        self.stopping.set()
        with self._condition:
            self._condition.notify_all()
        self.shutdown()
        self.server_close()

    def offer(self, image: Optional[np.ndarray], annotate: Any = None) -> bool:
        """Publish one frame, if anyone is watching.

        Args:
            image: The BGR preview frame.
            annotate: Optional ``fn(image) -> image`` that draws on a copy of
                the frame - only called when the frame will actually be sent.

        Returns:
            True when the frame was encoded and published.
        """
        if image is None or self.viewers == 0:
            return False
        import cv2

        frame = image
        if frame.shape[1] > self.max_width:
            scale = self.max_width / frame.shape[1]
            frame = cv2.resize(frame, (self.max_width, round(frame.shape[0] * scale)), interpolation=cv2.INTER_AREA)
        else:
            frame = frame.copy()
        if annotate is not None:
            frame = annotate(frame)
        ok, encoded = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, self.quality])
        if not ok:
            return False
        self.publish_jpeg(encoded.tobytes())
        return True

    def publish_jpeg(self, jpeg: bytes) -> None:
        """Make an already-encoded frame the current one."""
        with self._condition:
            self._jpeg = jpeg
            self._sequence += 1
            self._condition.notify_all()

    def latest(self) -> Optional[bytes]:
        """The most recent frame, or None before the first."""
        with self._condition:
            return self._jpeg

    def wait_for_frame(self, after: int, timeout: float) -> tuple:
        """Block until a frame newer than ``after`` exists, or ``timeout`` passes.

        Returns:
            ``(sequence, jpeg)``; the sequence is unchanged on a timeout.
        """
        deadline = time.monotonic() + timeout
        with self._condition:
            while self._sequence == after and not self.stopping.is_set():
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                self._condition.wait(remaining)
            return self._sequence, self._jpeg

    def _viewer_joined(self) -> None:
        with self._condition:
            self._viewers += 1

    def _viewer_left(self) -> None:
        with self._condition:
            self._viewers = max(0, self._viewers - 1)


class _VideoHandler(BaseHTTPRequestHandler):
    server: VideoServer
    protocol_version = "HTTP/1.1"

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002 - stdlib signature
        logger.debug("video %s: " + format, self.address_string(), *args)

    def do_GET(self) -> None:  # noqa: N802 - stdlib naming
        path = self.path.split("?", 1)[0]
        if path == "/video.mjpg":
            self._stream()
        elif path == "/snapshot.jpg":
            jpeg = self.server.latest()
            if jpeg is None:
                self._plain(503, b"No frame yet - open /video.mjpg to start the camera preview.\n")
            else:
                self._send(200, "image/jpeg", jpeg)
        else:
            self._plain(404, b"Not found. The stream is at /video.mjpg\n")

    def _plain(self, status: int, body: bytes) -> None:
        self._send(status, "text/plain; charset=utf-8", body)

    def _send(self, status: int, content_type: str, body: bytes) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def _stream(self) -> None:
        video = self.server
        video._viewer_joined()
        logger.info("Video viewer connected: %s", self.address_string())
        try:
            self.send_response(200)
            self.send_header("Content-Type", f"multipart/x-mixed-replace; boundary={BOUNDARY}")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Connection", "close")
            self.end_headers()
            sequence = -1
            while not video.stopping.is_set():
                sequence, jpeg = video.wait_for_frame(sequence, REPEAT_SECONDS)
                if jpeg is None:
                    continue
                self.wfile.write(
                    f"--{BOUNDARY}\r\nContent-Type: image/jpeg\r\nContent-Length: {len(jpeg)}\r\n\r\n".encode()
                    + jpeg
                    + b"\r\n"
                )
                self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError, TimeoutError):
            pass
        finally:
            video._viewer_left()
            self.close_connection = True
            logger.info("Video viewer disconnected: %s", self.address_string())

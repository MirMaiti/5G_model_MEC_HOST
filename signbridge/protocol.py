"""The wire format spoken between the capture host and the MEC server.

A raw TCP framing layer, deliberately small. Landmarks are the bulk of the
traffic and travel as packed binary; control messages and predictions are JSON,
because they are rare and being able to read them in a packet capture is worth
more than the bytes.

Every message is::

    magic   2 bytes   b"SB"
    version 1 byte
    type    1 byte    MessageType
    length  4 bytes   uint32, big-endian, payload length
    payload length bytes

A FRAME payload is packed rather than JSON-encoded::

    seq           4 bytes   uint32, big-endian
    t_ms          8 bytes   float64, big-endian, capture time (ms since epoch)
    n_landmarks   2 bytes   uint16, big-endian
    coords        4 * 3L    float32, big-endian, x/y/z per landmark
    mask          L bytes   uint8, 1 = detected

For the 42-landmark hands layout that is 14 + 504 + 42 = 560 bytes of payload,
568 on the wire with the header - about 17 kB/s at 30 fps. The same frame as a JSON array of floats runs roughly
five times larger, and a 640x480 JPEG - what the image-based pipeline would have
sent - is 20 to 40 kB, so around 50x more. That ratio is the whole point of
moving landmark extraction onto the host.
"""

from __future__ import annotations

import enum
import json
import socket
import struct
from typing import Any, Dict, Optional, Tuple

import numpy as np

#: Identifies our framing immediately, so a client pointed at the wrong port
#: fails on the first message instead of blocking on a bad length prefix.
MAGIC = b"SB"
#: Bumped on any incompatible change to the framing or payload layouts.
PROTOCOL_VERSION = 1

_HEADER = struct.Struct(">2sBBI")
HEADER_SIZE = _HEADER.size

_FRAME_HEADER = struct.Struct(">IdH")

#: Refuse anything larger rather than allocating it. One frame is well under 1 kB.
MAX_PAYLOAD = 8 * 1024 * 1024

_COORD_DTYPE = np.dtype(">f4")


class MessageType(enum.IntEnum):
    """Message kinds carried by the framing layer."""

    HELLO = 1       # host -> server, opens the session
    WELCOME = 2     # server -> host, model metadata
    FRAME = 3       # host -> server, one frame of landmarks (binary)
    PREDICTION = 4  # server -> host, a prediction
    RESET = 5       # host -> server, clear the rolling buffer
    ERROR = 6       # server -> host, a message was rejected
    PING = 7        # either way, round-trip timing
    PONG = 8
    BYE = 9         # either way, orderly shutdown


class ProtocolError(Exception):
    """Raised when bytes on the wire cannot be interpreted."""


def encode_message(message_type: MessageType, payload: bytes = b"") -> bytes:
    """Wrap a payload in the framing header."""
    if len(payload) > MAX_PAYLOAD:
        raise ProtocolError(f"Payload of {len(payload)} bytes exceeds the {MAX_PAYLOAD}-byte limit.")
    return _HEADER.pack(MAGIC, PROTOCOL_VERSION, int(message_type), len(payload)) + payload


def encode_json(message_type: MessageType, body: Dict[str, Any]) -> bytes:
    """Encode a control message whose payload is JSON."""
    return encode_message(message_type, json.dumps(body).encode("utf-8"))


def decode_json(payload: bytes) -> Dict[str, Any]:
    """Decode a JSON payload.

    Raises:
        ProtocolError: If the payload is not a JSON object.
    """
    try:
        body = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ProtocolError(f"Payload is not valid JSON: {exc}") from exc
    if not isinstance(body, dict):
        raise ProtocolError(f"Expected a JSON object; got {type(body).__name__}.")
    return body


def encode_frame(seq: int, t_ms: float, landmarks: np.ndarray, mask: np.ndarray) -> bytes:
    """Pack one frame of landmarks into a FRAME message.

    Args:
        seq: Monotonic frame counter, echoed back with the prediction so the
            host can measure end-to-end latency.
        t_ms: Capture timestamp in milliseconds since the epoch.
        landmarks: ``(L, 3)`` float array of normalised coordinates.
        mask: ``(L,)`` presence flags.

    Raises:
        ProtocolError: If the shapes disagree.
    """
    landmarks = np.asarray(landmarks, dtype=np.float32)
    mask = np.asarray(mask)
    if landmarks.ndim != 2 or landmarks.shape[1] != 3:
        raise ProtocolError(f"Expected landmarks shaped (L, 3); got {tuple(landmarks.shape)}.")
    count = landmarks.shape[0]
    if mask.shape != (count,):
        raise ProtocolError(
            f"Mask shaped {tuple(mask.shape)} does not match {count} landmarks."
        )
    payload = (
        _FRAME_HEADER.pack(int(seq) & 0xFFFFFFFF, float(t_ms), count)
        + landmarks.astype(_COORD_DTYPE, copy=False).tobytes()
        + (mask > 0).astype(np.uint8).tobytes()
    )
    return encode_message(MessageType.FRAME, payload)


def decode_frame(payload: bytes) -> Tuple[int, float, np.ndarray, np.ndarray]:
    """Unpack a FRAME payload.

    Returns:
        ``(seq, t_ms, landmarks (L, 3) float32, mask (L,) float32)``.

    Raises:
        ProtocolError: If the payload is truncated or self-inconsistent.
    """
    if len(payload) < _FRAME_HEADER.size:
        raise ProtocolError(f"FRAME payload is {len(payload)} bytes, too short for its header.")
    seq, t_ms, count = _FRAME_HEADER.unpack_from(payload, 0)

    coords_start = _FRAME_HEADER.size
    coords_end = coords_start + count * 3 * _COORD_DTYPE.itemsize
    expected = coords_end + count
    if len(payload) != expected:
        raise ProtocolError(
            f"FRAME claims {count} landmarks, which needs {expected} bytes; got {len(payload)}."
        )

    coords = np.frombuffer(payload, dtype=_COORD_DTYPE, count=count * 3, offset=coords_start)
    mask = np.frombuffer(payload, dtype=np.uint8, count=count, offset=coords_end)
    return (
        int(seq),
        float(t_ms),
        coords.astype(np.float32).reshape(count, 3),
        mask.astype(np.float32),
    )


class Connection:
    """A framed message channel over one TCP socket.

    Wraps the socket rather than subclassing it, so the transport can be swapped
    (TLS, a Unix socket, an in-memory pair for tests) without touching callers.

    Args:
        sock: A connected stream socket.
    """

    def __init__(self, sock: socket.socket) -> None:
        self.sock = sock
        self._buffer = bytearray()
        # Landmark frames are small and latency-sensitive; Nagle's algorithm
        # would hold them back waiting for more data to coalesce, which is
        # exactly wrong when each frame should leave the moment it exists.
        try:
            sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        except OSError:  # pragma: no cover - not every transport supports it
            pass

    @property
    def peer(self) -> str:
        """Human-readable address of the far end, for logs."""
        try:
            host, port = self.sock.getpeername()[:2]
            return f"{host}:{port}"
        except OSError:  # pragma: no cover - socket already closed
            return "<disconnected>"

    def send(self, data: bytes) -> None:
        """Send an already-encoded message."""
        self.sock.sendall(data)

    def send_json(self, message_type: MessageType, body: Dict[str, Any]) -> None:
        """Encode and send a JSON control message."""
        self.sock.sendall(encode_json(message_type, body))

    def send_frame(self, seq: int, t_ms: float, landmarks: np.ndarray, mask: np.ndarray) -> None:
        """Encode and send one landmark frame."""
        self.sock.sendall(encode_frame(seq, t_ms, landmarks, mask))

    def _fill(self, count: int) -> bool:
        """Read until the buffer holds ``count`` bytes. False on clean EOF."""
        while len(self._buffer) < count:
            chunk = self.sock.recv(65536)
            if not chunk:
                return False
            self._buffer.extend(chunk)
        return True

    def read_message(self) -> Optional[Tuple[MessageType, bytes]]:
        """Read one whole message, reassembling across TCP segment boundaries.

        Returns:
            ``(type, payload)``, or ``None`` when the peer closed cleanly.

        Raises:
            ProtocolError: On a bad magic, an unsupported version, an unknown
                message type, or an oversized length.
        """
        if not self._fill(HEADER_SIZE):
            return None
        magic, version, raw_type, length = _HEADER.unpack_from(self._buffer, 0)
        if magic != MAGIC:
            raise ProtocolError(
                f"Bad magic {magic!r}; expected {MAGIC!r}. Is this a SignBridge server?"
            )
        if version != PROTOCOL_VERSION:
            raise ProtocolError(
                f"Peer speaks protocol version {version}, this build speaks "
                f"{PROTOCOL_VERSION}. Upgrade both ends."
            )
        if length > MAX_PAYLOAD:
            raise ProtocolError(f"Message claims {length} bytes, over the {MAX_PAYLOAD} limit.")
        try:
            message_type = MessageType(raw_type)
        except ValueError:
            raise ProtocolError(f"Unknown message type {raw_type}.") from None

        if not self._fill(HEADER_SIZE + length):
            raise ProtocolError(
                f"Connection closed mid-message: wanted {length} payload bytes, "
                f"got {len(self._buffer) - HEADER_SIZE}."
            )
        payload = bytes(self._buffer[HEADER_SIZE : HEADER_SIZE + length])
        del self._buffer[: HEADER_SIZE + length]
        return message_type, payload

    def close(self) -> None:
        """Close the underlying socket, ignoring an already-closed one."""
        try:
            self.sock.close()
        except OSError:  # pragma: no cover
            pass

    def __enter__(self) -> "Connection":
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

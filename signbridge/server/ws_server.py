"""The MEC-side WebSocket server, for browsers instead of the Python capture host.

The TCP server (:mod:`signbridge.server.tcp_server`) speaks a compact binary
protocol to the Python capture host. A browser can't open a raw TCP socket at
all, so this is a second, independent front door onto the exact same model:
same :class:`~signbridge.server.predictor.Predictor`, same
:class:`~signbridge.server.session.InferenceSession`, same config - only the
transport and wire format differ, because JSON over WebSocket is what a
browser can actually speak, and landmark payloads are small enough that the
TCP protocol's binary packing buys nothing here.

Message shapes, all JSON text frames:

    -> {"type": "hello", "layout": "hands"}
    <- {"type": "welcome", "labels": [...], "window": 45, "feature_dim": 136,
        "layout": {...}, "untrained": false}
    -> {"type": "frame", "landmarks": [[x, y, z], ...], "mask": [0.0/1.0, ...]}
    <- {"type": "prediction", "label": "hello", "raw_label": "hello",
        "confidence": 0.93, "top": {...}, "changed": true, ...}
    -> {"type": "reset"}
    <- {"type": "error", "error": "..."}

One :class:`~signbridge.server.session.InferenceSession` per connection, same
as the TCP server - two browser tabs never share a rolling buffer.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import Any, Dict, Optional

import numpy as np

from .predictor import Predictor
from .session import InferenceSession, SessionConfig

logger = logging.getLogger(__name__)

try:
    import websockets
    from websockets.server import WebSocketServerProtocol
except ImportError:  # pragma: no cover - optional dependency, checked at call time
    websockets = None  # type: ignore[assignment]
    WebSocketServerProtocol = Any  # type: ignore[assignment,misc]


class WebSocketBridge:
    """Serves the model over WebSocket, mirroring the TCP server's behaviour.

    Args:
        predictor: The shared model wrapper - the exact same object a TCP
            server would use, so both front doors can run against one model.
        session_config: Buffering and smoothing settings for each connection.
    """

    def __init__(self, predictor: Predictor, session_config: Optional[SessionConfig] = None) -> None:
        if websockets is None:
            raise ImportError(
                "The 'websockets' package is required for the WebSocket bridge. "
                "Install it with: pip install websockets"
            )
        self.predictor = predictor
        self.session_config = session_config or SessionConfig()
        self._active = 0
        self._predictions = 0
        self._started = time.time()

    def stats(self) -> Dict[str, Any]:
        """Live counters, for logging - mirrors the TCP server's stats()."""
        return {
            "active_connections": self._active,
            "predictions_served": self._predictions,
            "uptime_seconds": round(time.time() - self._started, 1),
        }

    async def _handle(self, websocket: "WebSocketServerProtocol") -> None:
        """Serve one browser connection for its lifetime."""
        peer = getattr(websocket, "remote_address", "unknown")
        session = InferenceSession(self.predictor, self.session_config)
        self._active += 1
        logger.info("Browser connected: %s", peer)
        try:
            async for raw in websocket:
                try:
                    message = json.loads(raw)
                except (json.JSONDecodeError, TypeError):
                    await self._send_error(websocket, "Message was not valid JSON.")
                    continue
                await self._on_message(websocket, session, message)
        except Exception:  # pragma: no cover - a browser dropping mid-frame is routine
            logger.info("Browser %s disconnected", peer)
        finally:
            self._active = max(0, self._active - 1)
            logger.info("Browser disconnected: %s (%d frames)", peer, session.frames_seen)

    async def _on_message(
        self, websocket: "WebSocketServerProtocol", session: InferenceSession, message: Dict[str, Any]
    ) -> None:
        """Dispatch one decoded message by its ``type``."""
        kind = message.get("type")

        if kind == "hello":
            expected = session.extractor.layout
            claimed = message.get("layout")
            if claimed is not None and claimed != expected.name:
                await self._send_error(
                    websocket,
                    f"This server serves the {expected.name!r} layout "
                    f"({expected.num_landmarks} landmarks); the browser offered {claimed!r}.",
                )
                return
            info = dict(self.predictor.info())
            info["type"] = "welcome"
            await websocket.send(json.dumps(info))
            return

        if kind == "frame":
            await self._on_frame(websocket, session, message)
            return

        if kind == "reset":
            session.reset()
            await websocket.send(json.dumps({"type": "reset", "reset": True}))
            return

        await self._send_error(websocket, f"Unexpected message type {kind!r}.")

    async def _on_frame(
        self, websocket: "WebSocketServerProtocol", session: InferenceSession, message: Dict[str, Any]
    ) -> None:
        """Decode one browser-sent frame, buffer it, and reply if a prediction came due."""
        expected = session.extractor.layout.num_landmarks
        try:
            landmarks = np.asarray(message["landmarks"], dtype=np.float32)
            mask = np.asarray(message["mask"], dtype=np.float32)
        except (KeyError, ValueError, TypeError):
            await self._send_error(websocket, "A 'frame' message needs 'landmarks' (L,3) and 'mask' (L,) arrays.")
            return

        if landmarks.shape != (expected, 3) or mask.shape != (expected,):
            await self._send_error(
                websocket,
                f"This model expects {expected} landmarks per frame; "
                f"got landmarks {tuple(landmarks.shape)}, mask {tuple(mask.shape)}.",
            )
            return

        try:
            result = session.add_landmarks(landmarks, mask)
        except ValueError as exc:
            await self._send_error(websocket, str(exc))
            return

        if result is None:
            return

        result["type"] = "prediction"
        result["server_t_ms"] = time.time() * 1000.0
        await websocket.send(json.dumps(result))
        self._predictions += 1

    @staticmethod
    async def _send_error(websocket: "WebSocketServerProtocol", text: str) -> None:
        """Send an ERROR, ignoring a peer that has already gone."""
        try:
            await websocket.send(json.dumps({"type": "error", "error": text}))
        except Exception:  # pragma: no cover - peer vanished mid-error
            pass

    async def serve(self, host: str, port: int) -> None:
        """Bind and serve until cancelled."""
        async with websockets.serve(self._handle, host, port, max_size=2**20):
            await asyncio.Future()  # run forever


def run_forever(predictor: Predictor, session_config: SessionConfig, host: str, port: int) -> None:
    """Blocking entry point for the CLI: run the bridge until Ctrl-C."""
    bridge = WebSocketBridge(predictor, session_config)
    try:
        asyncio.run(bridge.serve(host, port))
    except KeyboardInterrupt:
        pass

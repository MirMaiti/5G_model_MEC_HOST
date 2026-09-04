"""Run the capture host: camera -> landmarks -> TCP -> prediction -> speech.

    python -m signbridge.cli.run_host --config config.yaml

This is the piece that runs on the device with the camera. MediaPipe runs here;
only landmark coordinates leave the machine. Predictions come back from the MEC
server and are spoken locally.

Without a camera, ``--source replay`` streams recorded clips through the exact
same path, which is the way to test the server and the speech output.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..capture.base import LandmarkSource
from ..capture.replay_source import ReplaySource, SyntheticSource
from ..config import load_config
from ..host.client import ConnectionFailed, HostClient, SessionStats, run_session
from ..host.tts import Speaker, build_backend
from ..landmarks import get_layout

logger = logging.getLogger(__name__)


def build_source(args: argparse.Namespace, config: Any) -> LandmarkSource:
    """Construct the landmark source the flags asked for.

    Raises:
        FileNotFoundError: If a replay path holds no clips.
    """
    layout = get_layout(config.layout)

    if args.source == "replay":
        if not args.clips:
            raise FileNotFoundError(
                "--source replay needs --clips pointing at a .npz file or a directory."
            )
        path = Path(args.clips)
        if path.is_dir():
            return ReplaySource.from_directory(path, fps=args.fps, loop=args.loop)
        return ReplaySource([path], fps=args.fps, loop=args.loop)

    if args.source == "synthetic":
        return SyntheticSource(layout.num_landmarks, count=args.count, fps=args.fps)

    from ..capture.mediapipe_source import MediaPipeSource

    return MediaPipeSource(
        layout=layout,
        hand_model_path=config.capture.hand_model_path,
        pose_model_path=config.capture.pose_model_path,
        camera_index=args.camera if args.camera is not None else config.capture.camera_index,
        width=config.capture.width,
        height=config.capture.height,
        min_hand_detection_confidence=config.capture.min_hand_detection_confidence,
        min_hand_presence_confidence=config.capture.min_hand_presence_confidence,
        min_tracking_confidence=config.capture.min_tracking_confidence,
        mirror_preview=config.capture.mirror_preview,
    )


def main(argv: Optional[List[str]] = None) -> int:
    """Connect, stream, speak."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="config.yaml", help="Config file (default: config.yaml)")
    parser.add_argument("--server", default=None, help="Server address as host:port")
    parser.add_argument("--source", choices=("camera", "replay", "synthetic"), default="camera",
                        help="Where landmarks come from (default: camera)")
    parser.add_argument("--clips", default=None, help="Clip file or directory for --source replay")
    parser.add_argument("--fps", type=float, default=30.0, help="Replay/synthetic frame rate")
    parser.add_argument("--loop", action="store_true", help="Loop replayed clips")
    parser.add_argument("--count", type=int, default=300, help="Frames for --source synthetic")
    parser.add_argument("--camera", type=int, default=None, help="Camera index override")
    parser.add_argument("--no-preview", action="store_true", help="Do not open a preview window")
    parser.add_argument("--no-speech", action="store_true", help="Print predictions instead of speaking")
    parser.add_argument("--tts-backend", default=None, help="auto, macos_say, pyttsx3 or null")
    parser.add_argument("--voice", default=None, help="Voice name for the TTS backend")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    overrides: Dict[str, Any] = {"tts": {"backend": args.tts_backend, "voice": args.voice}}
    if args.server:
        host, _, port = args.server.partition(":")
        overrides["host"] = {"server_host": host, "server_port": int(port) if port else None}
    config = load_config(args.config if Path(args.config).exists() else None, overrides=overrides)

    client = HostClient(
        host=config.host.server_host,
        port=config.host.server_port,
        layout=config.layout,
        connect_timeout=config.host.connect_timeout,
    )
    try:
        welcome = client.connect()
    except ConnectionFailed as exc:
        print(f"\n{exc}\n", file=sys.stderr)
        return 1

    print(f"Connected to {config.host.server_host}:{config.host.server_port}")
    print(f"  labels: {', '.join(welcome.get('labels', []))}")
    print(f"  window {welcome.get('window')} frames, {welcome.get('feature_dim')} features/frame")
    if welcome.get("device"):
        print(f"  server device: {welcome['device']}")

    speak = not args.no_speech
    if client.untrained:
        print("\n*** The server is running an UNTRAINED model. Its predictions are")
        print("*** meaningless, so speech is disabled. Train a model first:")
        print("***     python -m signbridge.cli.train --config config.yaml\n")
        speak = False

    speaker: Optional[Speaker] = None
    if speak:
        speaker = Speaker(
            backend=build_backend(config.tts.backend, voice=config.tts.voice, rate=config.tts.rate),
            repeat_cooldown=config.tts.repeat_cooldown,
            stability_seconds=config.tts.stability_seconds,
        )

    show_preview = config.host.show_preview and not args.no_preview and args.source == "camera"
    cv2 = None
    if show_preview:
        import cv2 as _cv2

        cv2 = _cv2

    last_label = ""

    def on_prediction(result: Dict[str, Any]) -> None:
        """Print each change, and speak it when the model is trained."""
        nonlocal last_label
        label = result.get("label", "")
        if result.get("changed") and label:
            spoken = speaker.offer(label) if speaker is not None else False
            marker = "spoken" if spoken else ("untrained" if result.get("untrained") else "")
            print(
                f"  {label:<16} {result.get('confidence', 0):5.1%}  "
                f"{result.get('inference_ms', 0):5.1f} ms  {marker}"
            )
        elif speaker is not None:
            speaker.offer(label)
        last_label = label

    def on_frame(frame: Any, stats: SessionStats) -> bool:
        """Draw the preview; return False to stop the session."""
        if cv2 is None or frame.image is None:
            return True
        from ..capture.mediapipe_source import draw_landmarks

        image = draw_landmarks(frame.image, frame.landmarks, frame.mask)
        cv2.putText(image, last_label or "...", (12, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 255, 0), 2)
        cv2.putText(
            image,
            f"{stats.frames_sent} frames  {stats.mean_latency_ms:.0f} ms round trip  hands: {frame.hands_detected}",
            (12, 72), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (230, 230, 230), 1,
        )
        cv2.imshow("signbridge - host", image)
        return (cv2.waitKey(1) & 0xFF) != ord("q")

    source = None
    stats = SessionStats()
    try:
        source = build_source(args, config)
        print("\nStreaming landmarks. Ctrl-C to stop.\n")
        with source:
            run_session(client, source, on_prediction=on_prediction, on_frame=on_frame, stats=stats)
    except (ConnectionFailed, FileNotFoundError) as exc:
        print(f"\n{exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        pass
    finally:
        if cv2 is not None:
            cv2.destroyAllWindows()
        if speaker is not None:
            speaker.close()
        client.close()

    print(
        f"\n{stats.frames_sent} frames sent, {stats.predictions} predictions, "
        f"mean round trip {stats.mean_latency_ms:.1f} ms"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

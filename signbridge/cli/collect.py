"""Record labelled landmark clips from the webcam.

    python -m signbridge.cli.collect --label hello --samples 30

Records one clip per sample: a countdown, then ``data.clip_frames`` frames of
landmarks written to ``data/<split>/<label>/``. Only landmarks are saved - no
video ever touches the disk, which is worth knowing before you record yourself
signing.

Aim for at least 20-30 clips per sign, and vary something between them:
distance from the camera, which way you are facing, lighting, speed. A model
trained on 30 identical clips learns the room, not the sign.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
import uuid
from pathlib import Path
from typing import List, Optional

import numpy as np

from ..capture.base import LandmarkFrame
from ..capture.mediapipe_source import CameraError, MediaPipeSource, draw_landmarks
from ..config import load_config
from ..landmarks import get_layout

logger = logging.getLogger(__name__)


def write_clip(
    destination: Path, label: str, frames: List[LandmarkFrame], layout_name: str
) -> Path:
    """Write one clip to a ``.npz``.

    Returns:
        The path written.
    """
    destination.mkdir(parents=True, exist_ok=True)
    path = destination / f"{label}-{uuid.uuid4().hex[:12]}.npz"
    np.savez_compressed(
        path,
        landmarks=np.stack([f.landmarks for f in frames]).astype(np.float32),
        mask=np.stack([f.mask for f in frames]).astype(np.float32),
        label=np.array(label),
        layout=np.array(layout_name),
    )
    return path


def main(argv: Optional[List[str]] = None) -> int:
    """Run the collector."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="config.yaml", help="Config file (default: config.yaml)")
    parser.add_argument("--label", required=True, help="The sign being recorded, e.g. hello")
    parser.add_argument("--samples", type=int, default=30, help="Clips to record (default: 30)")
    parser.add_argument("--split", default=None, help="Subdirectory of data root (default: the config's train split)")
    parser.add_argument("--frames", type=int, default=None, help="Frames per clip (default: from config)")
    parser.add_argument("--camera", type=int, default=None, help="Camera index override")
    parser.add_argument("--no-preview", action="store_true", help="Record without a preview window")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    config = load_config(args.config if Path(args.config).exists() else None)
    layout = get_layout(config.layout)

    split = args.split or config.data.train_split
    clip_frames = args.frames or config.data.clip_frames
    destination = Path(config.data.root) / split / args.label

    source = MediaPipeSource(
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

    show_preview = not args.no_preview
    cv2 = None
    if show_preview:
        import cv2 as _cv2

        cv2 = _cv2

    print(f"Recording '{args.label}': {args.samples} clips of {clip_frames} frames -> {destination}")
    print("SPACE starts a clip, q quits.\n")

    recorded = 0
    state = "waiting"
    countdown_until = 0.0
    buffer: List[LandmarkFrame] = []

    try:
        with source:
            for frame in source.frames():
                now = time.monotonic()

                if state == "counting" and now >= countdown_until:
                    state, buffer = "recording", []
                if state == "recording":
                    buffer.append(frame)
                    if len(buffer) >= clip_frames:
                        detected = sum(1 for f in buffer if f.hands_detected > 0)
                        if detected < clip_frames // 2:
                            print(
                                f"  discarded: hands visible in only {detected}/{clip_frames} "
                                "frames. Keep both hands in shot."
                            )
                        else:
                            path = write_clip(destination, args.label, buffer, layout.name)
                            recorded += 1
                            print(f"  [{recorded}/{args.samples}] {path.name}  ({detected}/{clip_frames} frames with hands)")
                        state, buffer = "waiting", []
                        if recorded >= args.samples:
                            print(f"\nDone: {recorded} clips in {destination}")
                            break

                if show_preview and frame.image is not None and cv2 is not None:
                    image = draw_landmarks(frame.image, frame.landmarks, frame.mask)
                    if state == "counting":
                        remaining = max(0.0, countdown_until - now)
                        banner, colour = f"Get ready... {remaining:0.1f}", (0, 200, 255)
                    elif state == "recording":
                        banner, colour = f"RECORDING {len(buffer)}/{clip_frames}", (0, 0, 255)
                    else:
                        banner, colour = f"SPACE to record  [{recorded}/{args.samples}]", (0, 255, 0)
                    cv2.putText(image, banner, (12, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.8, colour, 2)
                    cv2.putText(image, f"label: {args.label}   hands: {frame.hands_detected}",
                                (12, 64), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (230, 230, 230), 1)
                    cv2.imshow("signbridge - collect", image)

                    key = cv2.waitKey(1) & 0xFF
                    if key == ord("q"):
                        print(f"\nStopped: {recorded} clips recorded.")
                        break
                    if key == ord(" ") and state == "waiting":
                        state = "counting"
                        countdown_until = now + config.data.countdown_seconds
                elif state == "waiting":
                    # Headless: no keyboard, so pace the clips automatically.
                    state = "counting"
                    countdown_until = now + config.data.countdown_seconds
    except (CameraError, FileNotFoundError) as exc:
        print(f"\n{exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print(f"\nInterrupted: {recorded} clips recorded.")
    finally:
        if cv2 is not None:
            cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

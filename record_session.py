"""Record one continuous live take for tune_session.py to evaluate.

Unlike contribute_data.py, this does not chop the recording into fixed-length
labelled clips - it captures every frame from start to stop as one continuous
sequence, so tune_session.py can replay your actual, natural mix of signs,
pauses and transitions through several server parameter combos and let you
judge the results yourself.

Usage:

    python record_session.py

Each run is saved as its own timestamped file under sessions/, so recording
several takes never overwrites an earlier one - pass --out to name it
yourself instead. Sign naturally for however long you like - a few real
signs, pause, drift into another, repeat. Press 'q' (or Ctrl+C) to stop and
save. Only landmark coordinates and per-frame timestamps are written - no
video.

Needs: pip install mediapipe opencv-python numpy
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import numpy as np

from contribute_data import NUM_HAND_LANDMARKS, assemble, ensure_model

SESSIONS_DIR = "sessions"


def default_output_path() -> Path:
    """A timestamped path under sessions/, so takes accumulate instead of overwriting."""
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return Path(SESSIONS_DIR) / f"session_{stamp}.npz"


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--out", default=None,
        help=f"Output file (default: {SESSIONS_DIR}/session_<timestamp>.npz)",
    )
    parser.add_argument("--model", default="models/hand_landmarker.task", help="Path to the model bundle")
    parser.add_argument("--camera", type=int, default=0, help="Camera index")
    parser.add_argument("--no-preview", action="store_true", help="Record without a preview window")
    args = parser.parse_args(argv)

    try:
        import cv2
        import mediapipe as mp
        from mediapipe.tasks.python import BaseOptions, vision
    except ImportError as exc:
        print(f"Missing dependency: {exc}\nInstall with: pip install mediapipe opencv-python numpy", file=sys.stderr)
        return 1

    model_path = ensure_model(Path(args.model))
    capture = cv2.VideoCapture(args.camera)
    if not capture.isOpened():
        print(f"Could not open camera {args.camera}.", file=sys.stderr)
        return 1

    landmarker = vision.HandLandmarker.create_from_options(
        vision.HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(model_path)),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=2,
        )
    )

    show_preview = not args.no_preview
    print("Recording. Sign naturally - real signs, pauses, transitions between them.")
    print("Press 'q' in the preview window (or Ctrl+C here) to stop and save.\n")

    landmarks_log: List[np.ndarray] = []
    masks_log: List[np.ndarray] = []
    timestamps_log: List[float] = []
    frame_index = 0
    started = time.monotonic()

    try:
        while True:
            ok, image = capture.read()
            if not ok:
                print("Camera stopped delivering frames.", file=sys.stderr)
                break

            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            timestamp_ms = int(frame_index * 1000 / 30) + 1
            frame_index += 1
            result = landmarker.detect_for_video(mp_image, timestamp_ms)
            landmarks, mask = assemble(result)

            elapsed = time.monotonic() - started
            landmarks_log.append(landmarks)
            masks_log.append(mask)
            timestamps_log.append(elapsed)

            if show_preview:
                preview = cv2.flip(image, 1)
                hands_detected = int(mask[:NUM_HAND_LANDMARKS].max() > 0) + int(mask[NUM_HAND_LANDMARKS:].max() > 0)
                cv2.putText(preview, f"RECORDING  {elapsed:0.1f}s  {len(landmarks_log)} frames",
                            (12, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                cv2.putText(preview, f"hands: {hands_detected}   q to stop",
                            (12, 64), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (230, 230, 230), 1)
                cv2.imshow("record_session", preview)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        capture.release()
        landmarker.close()
        if show_preview:
            cv2.destroyAllWindows()

    if not landmarks_log:
        print("No frames recorded.", file=sys.stderr)
        return 1

    out_path = Path(args.out) if args.out else default_output_path()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        landmarks=np.stack(landmarks_log).astype(np.float32),
        mask=np.stack(masks_log).astype(np.float32),
        timestamps=np.array(timestamps_log, dtype=np.float64),
        layout=np.array("hands"),
    )
    print(f"\nSaved {len(landmarks_log)} frames ({timestamps_log[-1]:.1f}s) to {out_path}")
    print("Copy this file to wherever the trained model/tune_session.py runs, then:")
    print(f"    python tune_session.py --checkpoint models/best.pt --live-session {out_path.as_posix()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

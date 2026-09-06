"""Standalone data-collection script for SignBridge contributors.

Records labelled hand-landmark clips compatible with the main project's
training pipeline, without needing the full ``signbridge`` package, a venv
built from its requirements files, or even a clone of the repo. Only needs:

    pip install mediapipe opencv-python numpy

Usage:

    python contribute_data.py --label hello --samples 30

Each clip is written as a ``.npz`` under ``data/train/<label>/``, in the same
shape and landmark ordering the main project's trainer reads - so the
resulting folder can be zipped up and merged straight into the project's
``data/train/``, or committed directly if you have a fork with write access.

SPACE starts a clip, q quits. Nothing but coordinates is ever written to
disk - no video, no images.
"""

from __future__ import annotations

import argparse
import sys
import time
import urllib.request
import uuid
from pathlib import Path
from typing import List, Optional

import numpy as np

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/1/hand_landmarker.task"
)
NUM_HAND_LANDMARKS = 21
NUM_LANDMARKS = 2 * NUM_HAND_LANDMARKS  # 42: left hand (0-20), right hand (21-41)


def ensure_model(path: Path) -> Path:
    """Download the hand-landmarker bundle if it isn't already there."""
    if path.exists():
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading hand landmark model to {path} ...")
    urllib.request.urlretrieve(MODEL_URL, path)
    print("  done.")
    return path


def assemble(hand_result) -> tuple[np.ndarray, np.ndarray]:
    """Turn one MediaPipe detection result into the canonical (42, 3) tensor."""
    landmarks = np.zeros((NUM_LANDMARKS, 3), dtype=np.float32)
    mask = np.zeros(NUM_LANDMARKS, dtype=np.float32)

    hand_landmarks = getattr(hand_result, "hand_landmarks", None) or []
    handedness = getattr(hand_result, "handedness", None) or []
    for points, categories in zip(hand_landmarks, handedness):
        if not categories:
            continue
        is_left = categories[0].category_name == "Left"
        start = 0 if is_left else NUM_HAND_LANDMARKS
        block = slice(start, start + NUM_HAND_LANDMARKS)
        for index, point in enumerate(points[:NUM_HAND_LANDMARKS]):
            landmarks[start + index] = (point.x, point.y, point.z)
        mask[block] = 1.0

    return landmarks, mask


_INVALID_LABEL_CHARS = set('<>:"/\\|?*')
_RESERVED_WINDOWS_NAMES = {
    "con", "prn", "aux", "nul",
    *(f"com{i}" for i in range(1, 10)),
    *(f"lpt{i}" for i in range(1, 10)),
}


def normalize_label(raw: str) -> str:
    """Make a label safe to use as a directory/file name on every OS.

    Different contributors on different platforms recording the same sign
    with different casing or stray characters is how a merged dataset ends
    up with duplicate classes (silent) or unrecoverable git checkout errors
    on a case-insensitive filesystem like Windows' (loud) - so labels are
    forced to a single canonical form before anything is written to disk.

    Raises:
        SystemExit: If the label contains a path separator or a character
            that is invalid in a Windows filename, or is empty after
            normalizing - re-recording with a different label is the only
            fix, so this fails fast rather than writing something that would
            corrupt someone else's merge later.
    """
    label = raw.strip().lower()
    if not label:
        raise SystemExit("--label cannot be empty (after trimming whitespace).")
    if any(char in _INVALID_LABEL_CHARS for char in label):
        raise SystemExit(
            f"--label {raw!r} contains a character that is invalid in a Windows "
            f"filename ({''.join(sorted(_INVALID_LABEL_CHARS))}). Pick a plain "
            "word, e.g. 'thank_you' instead of 'thank/you'."
        )
    if label in _RESERVED_WINDOWS_NAMES:
        raise SystemExit(f"--label {raw!r} is a reserved Windows device name. Pick a different label.")
    if label != raw:
        print(f"Note: using label '{label}' (normalized from '{raw}') so it matches across contributors.")
    return label


def write_clip(destination: Path, label: str, landmarks: List[np.ndarray], masks: List[np.ndarray]) -> Path:
    """Write one recorded clip as a ``.npz``, matching the main project's schema."""
    destination.mkdir(parents=True, exist_ok=True)
    path = destination / f"{label}-{uuid.uuid4().hex[:12]}.npz"
    np.savez_compressed(
        path,
        landmarks=np.stack(landmarks).astype(np.float32),
        mask=np.stack(masks).astype(np.float32),
        label=np.array(label),
        layout=np.array("hands"),
    )
    return path


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--label", required=True, help="The sign being recorded, e.g. hello")
    parser.add_argument("--samples", type=int, default=30, help="Clips to record (default: 30)")
    parser.add_argument("--frames", type=int, default=45, help="Frames per clip (default: 45, ~1.5s at 30fps)")
    parser.add_argument("--countdown", type=float, default=2.0, help="Seconds of countdown before each clip")
    parser.add_argument("--out", default="data/train", help="Output root (default: data/train)")
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

    args.label = normalize_label(args.label)
    model_path = ensure_model(Path(args.model))
    destination = Path(args.out) / args.label

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
    print(f"Recording '{args.label}': {args.samples} clips of {args.frames} frames -> {destination}")
    print("SPACE starts a clip, q quits.\n")

    recorded = 0
    state = "waiting"
    countdown_until = 0.0
    landmarks_buf: List[np.ndarray] = []
    masks_buf: List[np.ndarray] = []
    frame_index = 0

    try:
        while recorded < args.samples:
            ok, image = capture.read()
            if not ok:
                print("Camera stopped delivering frames.", file=sys.stderr)
                return 1

            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            timestamp_ms = int(frame_index * 1000 / 30) + 1
            frame_index += 1
            result = landmarker.detect_for_video(mp_image, timestamp_ms)
            landmarks, mask = assemble(result)

            now = time.monotonic()
            if state == "counting" and now >= countdown_until:
                state, landmarks_buf, masks_buf = "recording", [], []
            if state == "recording":
                landmarks_buf.append(landmarks)
                masks_buf.append(mask)
                if len(landmarks_buf) >= args.frames:
                    detected = sum(1 for m in masks_buf if m.max() > 0)
                    if detected < args.frames // 2:
                        print(f"  discarded: hands visible in only {detected}/{args.frames} frames.")
                    else:
                        path = write_clip(destination, args.label, landmarks_buf, masks_buf)
                        recorded += 1
                        print(f"  [{recorded}/{args.samples}] {path.name}  ({detected}/{args.frames} frames with hands)")
                    state, landmarks_buf, masks_buf = "waiting", [], []

            if show_preview:
                preview = cv2.flip(image, 1)
                if state == "counting":
                    remaining = max(0.0, countdown_until - now)
                    banner, colour = f"Get ready... {remaining:0.1f}", (0, 200, 255)
                elif state == "recording":
                    banner, colour = f"RECORDING {len(landmarks_buf)}/{args.frames}", (0, 0, 255)
                else:
                    banner, colour = f"SPACE to record  [{recorded}/{args.samples}]", (0, 255, 0)
                cv2.putText(preview, banner, (12, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.8, colour, 2)
                hands_detected = int(mask[:NUM_HAND_LANDMARKS].max() > 0) + int(mask[NUM_HAND_LANDMARKS:].max() > 0)
                cv2.putText(preview, f"label: {args.label}   hands: {hands_detected}",
                            (12, 64), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (230, 230, 230), 1)
                cv2.imshow("contribute_data - collect", preview)

                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    print(f"\nStopped: {recorded} clips recorded.")
                    break
                if key == ord(" ") and state == "waiting":
                    state = "counting"
                    countdown_until = now + args.countdown
            elif state == "waiting":
                state = "counting"
                countdown_until = now + args.countdown
    except KeyboardInterrupt:
        print(f"\nInterrupted: {recorded} clips recorded.")
    finally:
        capture.release()
        landmarker.close()
        if show_preview:
            cv2.destroyAllWindows()

    print(f"\nDone: {recorded} clips in {destination}")
    print("Zip that folder (or your whole 'data' folder) and send it back, or")
    print("if you have a fork with write access:")
    print(f"  git add {destination} && git commit -m 'Add {args.label} clips' && git push")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

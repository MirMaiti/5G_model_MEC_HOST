"""Download the MediaPipe Tasks bundles the capture host needs.

    python -m signbridge.cli.fetch_models

These are Google's published hand and pose landmarker models. Only the capture
host needs them; the MEC server never runs MediaPipe.
"""

from __future__ import annotations

import argparse
import shutil
import sys
import urllib.request
from pathlib import Path
from typing import Dict

BASE = "https://storage.googleapis.com/mediapipe-models"

MODELS: Dict[str, str] = {
    "hand_landmarker.task": f"{BASE}/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
    "pose_landmarker_lite.task": f"{BASE}/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task",
}


def main(argv: list = None) -> int:
    """Fetch the bundles into the destination directory."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dest", default="models", help="Destination directory (default: models)")
    parser.add_argument(
        "--only", choices=sorted(MODELS), help="Fetch just one bundle instead of all"
    )
    parser.add_argument(
        "--from-dir",
        help="Copy from a local directory that already holds the bundles instead of downloading",
    )
    parser.add_argument("--force", action="store_true", help="Re-fetch even if present")
    args = parser.parse_args(argv)

    destination = Path(args.dest)
    destination.mkdir(parents=True, exist_ok=True)
    wanted = [args.only] if args.only else list(MODELS)

    for name in wanted:
        target = destination / name
        if target.exists() and not args.force:
            print(f"  {name}: already present ({target.stat().st_size / 1e6:.1f} MB)")
            continue
        try:
            if args.from_dir:
                source = Path(args.from_dir) / name
                if not source.exists():
                    print(f"  {name}: not found in {args.from_dir}", file=sys.stderr)
                    return 1
                shutil.copy2(source, target)
                print(f"  {name}: copied from {source} ({target.stat().st_size / 1e6:.1f} MB)")
            else:
                print(f"  {name}: downloading from {MODELS[name]}")
                urllib.request.urlretrieve(MODELS[name], target)
                print(f"  {name}: saved to {target} ({target.stat().st_size / 1e6:.1f} MB)")
        except (OSError, urllib.error.URLError) as exc:
            print(f"  {name}: failed - {exc}", file=sys.stderr)
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

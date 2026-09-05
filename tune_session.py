"""Sweep server smoothing/threshold settings offline, without a camera or a
live server restart.

The idea: stitch your own recorded clips end-to-end into one long stream -
real signs interleaved with your recorded ``idle`` clips (including the
"mid-transition" ones, if you recorded that sub-type) - so the stream itself
contains exactly the kind of ambiguous, hands-visible transitions you're
worried about. Replay that stream through ``InferenceSession`` in-process,
once per candidate (inference_interval, vote_window, min_confidence) combo,
and count how many times each combo reports a label that doesn't match what
was actually being signed at that moment. Lower is better. No TCP, no camera,
no server restart - every combo runs in a fraction of a second.

Usage:

    python tune_session.py --checkpoint models/best.pt --data data/train

Run this wherever the checkpoint and PyTorch live (the MEC/WSL), not the
capture host.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from signbridge.server.predictor import TorchPredictor
from signbridge.server.session import SILENCE, InferenceSession, SessionConfig
from signbridge.training.dataset import Clip, DatasetError, load_clips


def build_stream(
    clips: Sequence[Clip],
    real_labels: Sequence[str],
    gap_label: str,
    segments_per_label: int,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, List[Tuple[str, int]]]:
    """Concatenate clips into one stream, tracking the true label per segment.

    A clip labelled ``gap_label`` (your ``idle`` clips, ideally including
    the mid-transition ones) is inserted between every pair of real signs,
    so the stream contains the exact kind of ambiguous stretch a live demo
    produces between two signs.

    Returns:
        ``(features, segments)`` - the concatenated ``(T, D)`` feature array,
        and a list of ``(true_label, frame_count)`` in stream order.
    """
    by_label: Dict[str, List[Clip]] = {}
    for clip in clips:
        by_label.setdefault(clip.label, []).append(clip)

    order: List[Clip] = []
    for label in real_labels:
        pool = by_label.get(label, [])
        if not pool:
            print(f"warning: no clips for label {label!r}, skipping it in the stream", file=sys.stderr)
            continue
        count = min(segments_per_label, len(pool))
        chosen = rng.choice(len(pool), size=count, replace=False)
        order.extend(pool[i] for i in chosen)
    rng.shuffle(order)

    gap_pool = by_label.get(gap_label, [])
    if not gap_pool:
        print(f"warning: no {gap_label!r} clips found; stream will have no transition gaps", file=sys.stderr)

    pieces: List[np.ndarray] = []
    segments: List[Tuple[str, int]] = []
    for clip in order:
        pieces.append(clip.features)
        segments.append((clip.label, clip.features.shape[0]))
        if gap_pool:
            gap = gap_pool[int(rng.integers(0, len(gap_pool)))]
            pieces.append(gap.features)
            segments.append((gap_label, gap.features.shape[0]))

    return np.concatenate(pieces, axis=0), segments


def evaluate(
    predictor: TorchPredictor,
    features: np.ndarray,
    segments: Sequence[Tuple[str, int]],
    config: SessionConfig,
) -> Dict[str, float]:
    """Replay ``features`` through one session config and score the result.

    ``wrong_events`` counts every reported label change that doesn't match
    what was actually being signed at that frame - a flicker mid-sign and a
    real-sign label leaking into a transition gap both count here, since
    both are exactly what a live demo audience would see as a mistake.
    ``mean_latency`` is frames-from-segment-start to the first frame the
    session's current label agrees with the truth, averaged over real-sign
    segments only (lower is more responsive) - reported so you can see the
    responsiveness you're trading away for stability.
    """
    session = InferenceSession(predictor, config)
    truth = np.empty(features.shape[0], dtype=object)
    offset = 0
    for label, count in segments:
        truth[offset : offset + count] = label
        offset += count

    effective = SILENCE
    wrong_events = 0
    total_changes = 0
    latencies: List[int] = []
    segment_start = 0
    for label, count in segments:
        found_at: Optional[int] = None
        for i in range(count):
            frame_index = segment_start + i
            reply = session.add_features(features[frame_index])
            if reply is not None:
                effective = reply["label"]
                if reply["changed"]:
                    total_changes += 1
                    if effective != truth[frame_index]:
                        wrong_events += 1
            if label != SILENCE and label != "idle" and found_at is None and effective == label:
                found_at = i
        if label != SILENCE and label != "idle":
            latencies.append(found_at if found_at is not None else count)
        segment_start += count

    return {
        "wrong_events": wrong_events,
        "total_changes": total_changes,
        "mean_latency_frames": float(np.mean(latencies)) if latencies else float("nan"),
    }


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", default="models/best.pt")
    parser.add_argument("--data", default="data/train")
    parser.add_argument("--gap-label", default="idle", help="Label used as the transition/idle filler (default: idle)")
    parser.add_argument("--segments-per-label", type=int, default=4, help="Real-sign clips per label in the stream")
    parser.add_argument("--intervals", default="3,5,8", help="Comma-separated inference_interval candidates")
    parser.add_argument("--votes", default="3,5,7", help="Comma-separated vote_window candidates")
    parser.add_argument("--confidences", default="0.5,0.65,0.8", help="Comma-separated min_confidence candidates")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default=None)
    args = parser.parse_args(argv)

    intervals = [int(x) for x in args.intervals.split(",")]
    votes = [int(x) for x in args.votes.split(",")]
    confidences = [float(x) for x in args.confidences.split(",")]

    predictor = TorchPredictor(args.checkpoint, device=args.device)
    real_labels = [label for label in predictor.labels if label != args.gap_label]

    try:
        clips = load_clips(args.data, predictor.extractor)
    except DatasetError as exc:
        print(f"\n{exc}", file=sys.stderr)
        return 1

    rng = np.random.default_rng(args.seed)
    features, segments = build_stream(clips, real_labels, args.gap_label, args.segments_per_label, rng)
    print(f"Stream: {len(segments)} segments, {features.shape[0]} frames total\n")

    results = []
    for interval in intervals:
        for vote_window in votes:
            for min_confidence in confidences:
                config = SessionConfig(
                    inference_interval=interval, vote_window=vote_window, min_confidence=min_confidence,
                )
                metrics = evaluate(predictor, features, segments, config)
                results.append((interval, vote_window, min_confidence, metrics))

    results.sort(key=lambda r: (r[3]["wrong_events"], r[3]["mean_latency_frames"]))

    print(f"{'interval':>8} {'vote':>5} {'min_conf':>9}  {'wrong':>6} {'changes':>8} {'latency(frames)':>16}")
    for interval, vote_window, min_confidence, metrics in results:
        print(
            f"{interval:>8} {vote_window:>5} {min_confidence:>9.2f}  "
            f"{metrics['wrong_events']:>6} {metrics['total_changes']:>8} "
            f"{metrics['mean_latency_frames']:>16.1f}"
        )

    best = results[0]
    print(f"\nBest: inference_interval={best[0]}, vote_window={best[1]}, min_confidence={best[2]}")
    print("Copy these into config.yaml's `server:` block, then restart `signbridge.cli.serve`.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

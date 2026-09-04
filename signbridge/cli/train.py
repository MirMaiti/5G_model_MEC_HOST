"""Train a classifier on the clips you recorded.

    python -m signbridge.cli.train --config config.yaml

Reads ``data/<train split>``, holds out a stratified validation set, trains the
architecture named in the config, and writes ``models/best.pt`` together with a
``metrics.json`` report. The checkpoint carries the label list and the exact
feature description, so the server cannot serve it with mismatched inputs.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import List, Optional

from ..config import load_config
from ..training.dataset import (
    ClipDataset,
    DatasetError,
    build_vocabulary,
    label_counts,
    load_clips,
    stratified_split,
)
from ..training.trainer import format_confusion, train

logger = logging.getLogger(__name__)


def main(argv: Optional[List[str]] = None) -> int:
    """Run training."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="config.yaml", help="Config file (default: config.yaml)")
    parser.add_argument("--data", default=None, help="Dataset directory override")
    parser.add_argument("--val-data", default=None, help="Explicit validation directory; otherwise the training clips are split")
    parser.add_argument("--architecture", default=None, help="Override the architecture (mlp, gru, transformer)")
    parser.add_argument("--epochs", type=int, default=None, help="Override the epoch count")
    parser.add_argument("--batch-size", type=int, default=None, help="Override the batch size")
    parser.add_argument("--window", type=int, default=None, help="Override the frames per window")
    parser.add_argument("--out", default=None, help="Checkpoint directory override")
    parser.add_argument("--device", default=None, help="Force a device, e.g. cpu, mps, cuda")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    overrides = {
        "model": {"architecture": args.architecture, "window": args.window},
        "training": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "device": args.device,
            "checkpoint_dir": args.out,
        },
    }
    config = load_config(args.config if Path(args.config).exists() else None, overrides=overrides)
    extractor = config.feature_extractor()

    train_root = Path(args.data) if args.data else Path(config.data.root) / config.data.train_split

    try:
        clips = load_clips(train_root, extractor)

        if args.val_data:
            val_clips = load_clips(Path(args.val_data), extractor)
            train_clips = clips
        else:
            explicit_val = Path(config.data.root) / config.data.val_split
            if explicit_val.exists() and any(explicit_val.rglob("*.npz")):
                train_clips = clips
                val_clips = load_clips(explicit_val, extractor)
            else:
                train_clips, val_clips = stratified_split(
                    clips, config.training.val_fraction, config.training.seed
                )

        labels = build_vocabulary(train_clips + val_clips)
    except DatasetError as exc:
        print(f"\n{exc}", file=sys.stderr)
        return 1

    print(f"\nDataset: {train_root}")
    print(f"  {len(train_clips)} training clips, {len(val_clips)} validation clips")
    print(f"  {len(labels)} labels: {', '.join(labels)}")
    for label, count in label_counts(train_clips).items():
        print(f"    {label:<20} {count} train / {label_counts(val_clips).get(label, 0)} val")
    print(f"  {extractor.dim} features per frame, window {config.model.window} frames")
    print(f"  architecture: {config.model.architecture}\n")

    train_dataset = ClipDataset(
        train_clips, labels, window=config.model.window, augment=True,
        noise=config.training.augment_noise, max_shift=config.training.augment_time_shift,
        seed=config.training.seed,
    )
    val_dataset = ClipDataset(
        val_clips, labels, window=config.model.window, augment=False, seed=config.training.seed
    )

    result = train(
        config=config,
        extractor=extractor,
        labels=labels,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        checkpoint_dir=Path(args.out) if args.out else None,
        progress=lambda epoch: print(epoch.format()),
    )

    print(f"\nBest validation accuracy {result.best_accuracy:.2%} at epoch {result.best_epoch}")
    if result.confusion is not None:
        print("\nConfusion (rows = truth, columns = prediction):")
        print(format_confusion(result.confusion, labels))
    print(f"\nCheckpoint: {result.checkpoint_path}")
    print("\nServe it with:")
    print(f"    python -m signbridge.cli.serve --checkpoint {result.checkpoint_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Per-class precision/recall/F1 from a trained checkpoint's confusion matrix.

signbridge.cli.train reports accuracy and a confusion matrix in
models/metrics.json, but not precision/recall/F1 - this derives them from
that same confusion matrix rather than re-running anything.

Usage:

    python precision_recall.py --metrics models/metrics.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Optional


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--metrics", default="models/metrics.json", help="Path to metrics.json (default: models/metrics.json)")
    args = parser.parse_args(argv)

    data = json.loads(Path(args.metrics).read_text())
    labels = data["labels"]
    confusion = data["confusion"]  # rows = truth, columns = prediction

    n = len(labels)
    print(f"From {args.metrics} (accuracy {data['best_accuracy']:.2%}, epoch {data['best_epoch']})\n")
    print(f"{'label':<14} {'precision':>10} {'recall':>10} {'f1':>10} {'support':>8}")

    macro_p = macro_r = macro_f1 = 0.0
    for i in range(n):
        tp = confusion[i][i]
        fp = sum(confusion[r][i] for r in range(n)) - tp   # predicted i, actually something else
        fn = sum(confusion[i][c] for c in range(n)) - tp   # actually i, predicted something else
        support = sum(confusion[i])

        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        macro_p += precision; macro_r += recall; macro_f1 += f1

        print(f"{labels[i]:<14} {precision:>10.2%} {recall:>10.2%} {f1:>10.2%} {support:>8}")

    print(f"\n{'macro avg':<14} {macro_p / n:>10.2%} {macro_r / n:>10.2%} {macro_f1 / n:>10.2%}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

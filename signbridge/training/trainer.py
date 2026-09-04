"""The training loop.

Deliberately plain: AdamW, cross-entropy, early stopping on validation
accuracy, best-checkpoint saving. The interesting decisions in this project are
in the features and the transport, not here, and a training loop you can read in
one sitting is easier to adapt than a framework.

Nothing in this module is run automatically. It trains only on clips you have
recorded, when you run ``python -m signbridge.cli.train``.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn

from ..config import Config
from ..features import FeatureExtractor
from ..model.architectures import build_model, model_hyperparameters
from ..model.checkpoint import default_device, describe_device, save_checkpoint
from .dataset import ClipDataset

logger = logging.getLogger(__name__)


@dataclass
class EpochResult:
    """Metrics for one epoch."""

    epoch: int
    train_loss: float
    train_accuracy: float
    val_loss: float
    val_accuracy: float
    seconds: float

    def format(self) -> str:
        """One-line summary for the console."""
        return (
            f"epoch {self.epoch:3d}  "
            f"train loss {self.train_loss:.4f} acc {self.train_accuracy:6.2%}  |  "
            f"val loss {self.val_loss:.4f} acc {self.val_accuracy:6.2%}  "
            f"({self.seconds:.1f}s)"
        )


@dataclass
class TrainingResult:
    """Everything a finished run produced."""

    best_accuracy: float
    best_epoch: int
    epochs: List[EpochResult] = field(default_factory=list)
    checkpoint_path: Optional[Path] = None
    confusion: Optional[np.ndarray] = None
    labels: List[str] = field(default_factory=list)


def _run_epoch(
    model: nn.Module,
    dataset: ClipDataset,
    batch_size: int,
    device: torch.device,
    criterion: nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
) -> Tuple[float, float, np.ndarray]:
    """Run one pass. With no optimizer this is an evaluation pass.

    Returns:
        ``(mean_loss, accuracy, confusion)``.
    """
    training = optimizer is not None
    model.train(training)

    total_loss = 0.0
    total_correct = 0
    total_seen = 0
    classes = len(dataset.labels)
    confusion = np.zeros((classes, classes), dtype=np.int64)

    for features, targets in dataset.batches(batch_size, shuffle=training):
        inputs = torch.from_numpy(features).to(device)
        labels = torch.from_numpy(targets).to(device)

        with torch.set_grad_enabled(training):
            logits = model(inputs)
            loss = criterion(logits, labels)

        if training:
            assert optimizer is not None
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            # Recurrent models can produce very large gradients on short, noisy
            # clips; clipping keeps a single bad batch from wrecking the run.
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

        predicted = logits.argmax(dim=-1)
        total_loss += float(loss.item()) * labels.shape[0]
        total_correct += int((predicted == labels).sum().item())
        total_seen += int(labels.shape[0])
        for target, guess in zip(labels.cpu().numpy(), predicted.cpu().numpy()):
            confusion[int(target), int(guess)] += 1

    if total_seen == 0:
        return 0.0, 0.0, confusion
    return total_loss / total_seen, total_correct / total_seen, confusion


def train(
    config: Config,
    extractor: FeatureExtractor,
    labels: Sequence[str],
    train_dataset: ClipDataset,
    val_dataset: ClipDataset,
    checkpoint_dir: Optional[Path] = None,
    progress: Optional[Any] = None,
) -> TrainingResult:
    """Train a classifier on recorded clips.

    Args:
        config: The full configuration.
        extractor: The feature extractor used to build both datasets; it is
            stored in the checkpoint so serving cannot drift from training.
        labels: The vocabulary, in model output order.
        train_dataset: Windows to learn from.
        val_dataset: Windows to score against.
        checkpoint_dir: Where to write ``best.pt``; defaults to the config.
        progress: Optional callable invoked with each :class:`EpochResult`.

    Returns:
        The run's metrics and the path of the best checkpoint.
    """
    settings = config.training
    torch.manual_seed(settings.seed)
    np.random.seed(settings.seed)

    device = torch.device(settings.device) if settings.device else default_device()
    model = build_model(
        architecture=config.model.architecture,
        input_dim=extractor.dim,
        num_classes=len(labels),
        window=config.model.window,
        **model_hyperparameters(config.model),
    ).to(device)

    parameters = sum(p.numel() for p in model.parameters())
    logger.info(
        "Training %s on %s: %d parameters, %d features/frame, %d classes",
        config.model.architecture, describe_device(device), parameters, extractor.dim, len(labels),
    )

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=settings.learning_rate, weight_decay=settings.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=max(2, settings.patience // 3)
    )
    criterion = nn.CrossEntropyLoss()

    destination = Path(checkpoint_dir or settings.checkpoint_dir)
    destination.mkdir(parents=True, exist_ok=True)

    result = TrainingResult(best_accuracy=-1.0, best_epoch=-1, labels=list(labels))
    epochs_without_improvement = 0

    for epoch in range(1, settings.epochs + 1):
        started = time.time()
        train_loss, train_accuracy, _ = _run_epoch(
            model, train_dataset, settings.batch_size, device, criterion, optimizer
        )
        val_loss, val_accuracy, confusion = _run_epoch(
            model, val_dataset, settings.batch_size, device, criterion, None
        )
        scheduler.step(val_accuracy)

        epoch_result = EpochResult(
            epoch=epoch,
            train_loss=train_loss,
            train_accuracy=train_accuracy,
            val_loss=val_loss,
            val_accuracy=val_accuracy,
            seconds=time.time() - started,
        )
        result.epochs.append(epoch_result)
        if progress is not None:
            progress(epoch_result)
        else:
            logger.info("%s", epoch_result.format())

        if val_accuracy > result.best_accuracy:
            result.best_accuracy = val_accuracy
            result.best_epoch = epoch
            result.confusion = confusion
            epochs_without_improvement = 0
            result.checkpoint_path = save_checkpoint(
                path=destination / "best.pt",
                model=model,
                extractor=extractor,
                labels=list(labels),
                architecture=config.model.architecture,
                hyperparameters=model_hyperparameters(config.model),
                config=config.to_dict(),
                metrics={
                    "val_accuracy": val_accuracy,
                    "val_loss": val_loss,
                    "train_accuracy": train_accuracy,
                    "epoch": epoch,
                },
            )
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= settings.patience:
                logger.info(
                    "Stopping early: no improvement for %d epochs (best %.2f%% at epoch %d)",
                    settings.patience, result.best_accuracy * 100, result.best_epoch,
                )
                break

    _write_report(destination, result, config)
    return result


def _write_report(destination: Path, result: TrainingResult, config: Config) -> None:
    """Save metrics next to the checkpoint, so a run can be compared later."""
    report: Dict[str, Any] = {
        "best_accuracy": result.best_accuracy,
        "best_epoch": result.best_epoch,
        "labels": result.labels,
        "architecture": config.model.architecture,
        "epochs": [
            {
                "epoch": e.epoch,
                "train_loss": e.train_loss,
                "train_accuracy": e.train_accuracy,
                "val_loss": e.val_loss,
                "val_accuracy": e.val_accuracy,
            }
            for e in result.epochs
        ],
    }
    if result.confusion is not None:
        report["confusion"] = result.confusion.tolist()
    (destination / "metrics.json").write_text(json.dumps(report, indent=2))


def format_confusion(confusion: np.ndarray, labels: Sequence[str], width: int = 10) -> str:
    """Render a confusion matrix as text, rows = truth, columns = prediction."""
    header = " " * (width + 1) + " ".join(label[:width].rjust(width) for label in labels)
    lines = [header]
    for index, label in enumerate(labels):
        cells = " ".join(str(int(value)).rjust(width) for value in confusion[index])
        lines.append(f"{label[:width].ljust(width)} {cells}")
    return "\n".join(lines)

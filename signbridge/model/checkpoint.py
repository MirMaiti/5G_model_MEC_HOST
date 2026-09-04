"""Saving and loading checkpoints that carry everything needed to serve them.

A bare ``state_dict`` is not enough. The server has to rebuild the *identical*
feature extractor, or it will feed the model vectors it never saw in training
and produce confident nonsense. So the feature description, the label list and
the architecture settings travel with the weights, and loading rebuilds all
three together.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import torch

from ..features import FeatureExtractor, extractor_from_describe
from .architectures import SequenceClassifier, build_model

PathLike = Union[str, Path]

#: Bumped when the checkpoint dict layout changes.
CHECKPOINT_VERSION = 1


@dataclass
class LoadedModel:
    """A checkpoint reconstituted into working parts."""

    model: SequenceClassifier
    extractor: FeatureExtractor
    labels: List[str]
    window: int
    device: torch.device
    metadata: Dict[str, Any]

    @property
    def num_classes(self) -> int:
        """Number of labels this model predicts."""
        return len(self.labels)


def save_checkpoint(
    path: PathLike,
    model: SequenceClassifier,
    extractor: FeatureExtractor,
    labels: List[str],
    architecture: str,
    hyperparameters: Dict[str, Any],
    config: Optional[Dict[str, Any]] = None,
    metrics: Optional[Dict[str, Any]] = None,
) -> Path:
    """Write a checkpoint holding weights, features, labels and architecture.

    Returns:
        The path written.
    """
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "checkpoint_version": CHECKPOINT_VERSION,
        "state_dict": model.state_dict(),
        "labels": list(labels),
        "features": extractor.describe(),
        "architecture": architecture,
        "hyperparameters": dict(hyperparameters),
        "window": model.window,
        "input_dim": model.input_dim,
        "config": config or {},
        "metrics": metrics or {},
    }
    torch.save(payload, destination)
    return destination


def load_checkpoint(
    path: PathLike, device: Optional[Union[str, torch.device]] = None
) -> LoadedModel:
    """Rebuild a model, its feature extractor and its labels from disk.

    Args:
        path: The checkpoint file.
        device: Where to place the model. Defaults to the best available.

    Raises:
        FileNotFoundError: If the checkpoint does not exist.
        ValueError: If it was written by an incompatible version, or its stored
            feature dimension disagrees with the extractor it describes.
    """
    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(
            f"No checkpoint at {source}. Train one first:\n"
            f"    python -m signbridge.cli.train --config config.yaml"
        )

    resolved = torch.device(device) if device is not None else default_device()
    # These files are produced by our own training run, so the full pickle is
    # expected; weights_only would reject the metadata we deliberately store.
    payload = torch.load(source, map_location=resolved, weights_only=False)

    version = int(payload.get("checkpoint_version", 0))
    if version != CHECKPOINT_VERSION:
        raise ValueError(
            f"{source} is a version {version} checkpoint; this build reads version "
            f"{CHECKPOINT_VERSION}. Retrain it."
        )

    extractor = extractor_from_describe(payload["features"])
    labels = list(payload["labels"])
    stored_dim = int(payload["input_dim"])
    if stored_dim != extractor.dim:
        raise ValueError(
            f"{source} was trained on {stored_dim} features per frame, but its "
            f"feature description rebuilds to {extractor.dim}. The checkpoint is "
            "inconsistent; retrain it."
        )

    model = build_model(
        architecture=str(payload["architecture"]),
        input_dim=stored_dim,
        num_classes=len(labels),
        window=int(payload["window"]),
        **dict(payload.get("hyperparameters", {})),
    )
    model.load_state_dict(payload["state_dict"])
    model.to(resolved)
    model.eval()

    return LoadedModel(
        model=model,
        extractor=extractor,
        labels=labels,
        window=int(payload["window"]),
        device=resolved,
        metadata={
            "path": str(source),
            "architecture": payload["architecture"],
            "metrics": payload.get("metrics", {}),
            "config": payload.get("config", {}),
        },
    )


def default_device() -> torch.device:
    """Pick the best device available: CUDA, then Apple Metal, then CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def describe_device(device: torch.device) -> str:
    """Readable device name for logs and the TCP handshake."""
    if device.type == "cuda":  # pragma: no cover - needs a GPU
        return f"cuda ({torch.cuda.get_device_name(device)})"
    if device.type == "mps":
        return "mps (Apple GPU)"
    return "cpu"

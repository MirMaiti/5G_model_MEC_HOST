"""The model-facing layer of the server.

:class:`Predictor` is the seam between the transport and the model. The TCP
server only ever calls :meth:`Predictor.predict`, so swapping in a different
backend - ONNX Runtime, a quantised build, a remote accelerator - means writing
one class and changing one line, not touching the networking.
"""

from __future__ import annotations

import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np

from ..features import FeatureExtractor
from ..model.checkpoint import LoadedModel, describe_device, load_checkpoint


@dataclass
class Prediction:
    """One model output."""

    label: str
    confidence: float
    #: The few highest-scoring labels, for debugging a shaky prediction.
    top: Dict[str, float]

    def to_dict(self) -> Dict[str, Any]:
        """JSON-serialisable form."""
        return {
            "label": self.label,
            "confidence": round(float(self.confidence), 4),
            "top": {k: round(float(v), 4) for k, v in self.top.items()},
        }


class Predictor(ABC):
    """Turns a window of features into a :class:`Prediction`."""

    #: True when the weights are not trained, so callers can refuse to speak them.
    untrained: bool = False

    @property
    @abstractmethod
    def labels(self) -> List[str]:
        """The label set, in model output order."""

    @property
    @abstractmethod
    def window(self) -> int:
        """Frames the model expects per window."""

    @property
    @abstractmethod
    def extractor(self) -> FeatureExtractor:
        """The feature extractor this model was trained with."""

    @abstractmethod
    def predict(self, features: np.ndarray) -> Prediction:
        """Classify one ``(T, D)`` window of features."""

    def info(self) -> Dict[str, Any]:
        """Metadata sent to the host in the WELCOME message."""
        return {
            "labels": list(self.labels),
            "window": self.window,
            "feature_dim": self.extractor.dim,
            "layout": self.extractor.layout.describe(),
            "features": self.extractor.describe(),
            "untrained": self.untrained,
        }


class TorchPredictor(Predictor):
    """Runs a trained checkpoint under PyTorch.

    Args:
        checkpoint: Path to a file written by ``signbridge.cli.train``.
        device: Optional device override.
        top_k: How many labels to report alongside the winner.
    """

    def __init__(
        self,
        checkpoint: Union[str, Path],
        device: Optional[str] = None,
        top_k: int = 3,
    ) -> None:
        self._loaded: LoadedModel = load_checkpoint(checkpoint, device=device)
        self._top_k = max(1, top_k)
        # One shared model across connection threads; a forward pass is not
        # guaranteed re-entrant, and the server is I/O-bound anyway.
        self._lock = threading.Lock()

    @property
    def labels(self) -> List[str]:
        """The label set, in model output order."""
        return self._loaded.labels

    @property
    def window(self) -> int:
        """Frames the model expects per window."""
        return self._loaded.window

    @property
    def extractor(self) -> FeatureExtractor:
        """The feature extractor this model was trained with."""
        return self._loaded.extractor

    @property
    def device_name(self) -> str:
        """Readable device description."""
        return describe_device(self._loaded.device)

    def predict(self, features: np.ndarray) -> Prediction:
        """Classify one ``(T, D)`` window of features."""
        import torch

        batch = torch.from_numpy(np.ascontiguousarray(features, dtype=np.float32))[None, ...]
        with self._lock:
            batch = batch.to(self._loaded.device)
            with torch.no_grad():
                logits = self._loaded.model(batch)
                probabilities = torch.softmax(logits, dim=-1)[0].cpu().numpy()
        return _prediction_from_scores(probabilities, self.labels, self._top_k)

    def info(self) -> Dict[str, Any]:
        """Metadata sent to the host in the WELCOME message."""
        base = super().info()
        base.update(
            {
                "device": self.device_name,
                "architecture": self._loaded.metadata.get("architecture"),
                "checkpoint": self._loaded.metadata.get("path"),
            }
        )
        return base


class UntrainedPredictor(Predictor):
    """A randomly initialised model, for exercising the link before training.

    Every prediction it makes is meaningless, and it says so: ``untrained`` is
    True here, in the WELCOME handshake and on every prediction, and the host
    refuses to speak its output. It exists so the TCP path, the buffering and
    the preview can be verified end to end before any data is collected.
    """

    untrained = True

    def __init__(self, extractor: FeatureExtractor, labels: List[str], window: int) -> None:
        self._extractor = extractor
        self._labels = list(labels)
        self._window = int(window)

    @property
    def labels(self) -> List[str]:
        """The placeholder label set."""
        return self._labels

    @property
    def window(self) -> int:
        """Frames per window."""
        return self._window

    @property
    def extractor(self) -> FeatureExtractor:
        """The configured feature extractor."""
        return self._extractor

    def predict(self, features: np.ndarray) -> Prediction:
        """Return a deterministic but meaningless score vector."""
        features = np.asarray(features, dtype=np.float32)
        if features.ndim != 2 or features.shape[1] != self._extractor.dim:
            raise ValueError(
                f"Expected a window shaped (T, {self._extractor.dim}); "
                f"got {tuple(features.shape)}."
            )
        # Derived from the input so the same window gives the same answer, which
        # makes transport tests reproducible.
        seed = int(abs(float(features.sum())) * 1000) % (2**32)
        scores = np.random.default_rng(seed).random(len(self._labels)).astype(np.float32)
        scores /= scores.sum()
        return _prediction_from_scores(scores, self._labels, top_k=3)


def _prediction_from_scores(
    scores: np.ndarray, labels: List[str], top_k: int
) -> Prediction:
    """Build a :class:`Prediction` from a probability vector."""
    if scores.shape[0] != len(labels):
        raise ValueError(
            f"Model produced {scores.shape[0]} scores for {len(labels)} labels."
        )
    order = np.argsort(scores)[::-1]
    best = int(order[0])
    top = {labels[int(i)]: float(scores[int(i)]) for i in order[: min(top_k, len(labels))]}
    return Prediction(label=labels[best], confidence=float(scores[best]), top=top)

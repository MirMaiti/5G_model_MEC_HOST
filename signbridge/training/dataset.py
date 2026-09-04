"""Loading recorded clips and turning them into training tensors.

Clips are ``.npz`` files holding ``landmarks`` ``(T, L, 3)`` and ``mask``
``(T, L)``. A clip's label is resolved in this order:

1. a ``label`` array inside the ``.npz`` (what the collector writes),
2. the name of the directory holding it - ``data/train/hello/clip_003.npz``,
3. the part of the filename before the first ``-`` - ``hello-a1b2c3.npz``.

That covers the collector's own output and the two layouts people most often
arrive with, so an existing dataset usually needs no reorganising.

Features are extracted once at load rather than per epoch. Landmark clips are
small - a 45-frame hands clip is about 23 kB - so even a few thousand fit in
memory comfortably, and every epoch after the first costs nothing.
"""

from __future__ import annotations

import logging
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Sequence, Tuple, Union

import numpy as np

from ..features import FeatureExtractor
from ..sequence import resample_sequence, time_shift

logger = logging.getLogger(__name__)

PathLike = Union[str, Path]


class DatasetError(RuntimeError):
    """Raised when the dataset on disk cannot be used for training."""


@dataclass
class Clip:
    """One recorded sample, already converted to features."""

    features: np.ndarray  # (T, D)
    label: str
    path: Path


def resolve_label(path: Path, stored: Optional[object], root: Path) -> str:
    """Work out a clip's label from the file, its directory, or its name.

    Raises:
        DatasetError: If no label can be determined.
    """
    if stored is not None:
        text = str(np.asarray(stored).item() if np.asarray(stored).ndim == 0 else stored)
        if text:
            return text

    parent = path.parent
    if parent != root and parent.name:
        return parent.name

    stem = path.stem
    if "-" in stem:
        return stem.split("-", 1)[0]

    raise DatasetError(
        f"Cannot determine a label for {path}. Store one in the .npz as 'label', "
        "put the clip in a directory named after its sign, or name the file "
        "'<label>-<id>.npz'."
    )


def load_clips(
    root: PathLike,
    extractor: FeatureExtractor,
    pattern: str = "*.npz",
) -> List[Clip]:
    """Load and featurise every clip under ``root``.

    Raises:
        DatasetError: If the directory is missing, empty, or holds a clip whose
            landmark count does not match the configured layout.
    """
    directory = Path(root)
    if not directory.exists():
        raise DatasetError(
            f"No dataset directory at {directory}. Record clips first:\n"
            f"    python -m signbridge.cli.collect --label hello --samples 30"
        )

    paths = sorted(directory.rglob(pattern))
    if not paths:
        raise DatasetError(
            f"No {pattern} clips under {directory}. Record some with "
            "'python -m signbridge.cli.collect'."
        )

    expected = extractor.layout.num_landmarks
    clips: List[Clip] = []
    for path in paths:
        try:
            with np.load(path, allow_pickle=False) as data:
                if "landmarks" not in data:
                    raise DatasetError(f"{path} has no 'landmarks' array.")
                landmarks = data["landmarks"].astype(np.float32)
                mask = (
                    data["mask"].astype(np.float32)
                    if "mask" in data
                    # Absent landmarks are written as exact zeros, so this
                    # reconstructs the mask for clips recorded without one.
                    else (np.abs(landmarks).sum(axis=-1) > 0).astype(np.float32)
                )
                stored_label = data["label"] if "label" in data else None
        except (OSError, ValueError) as exc:
            raise DatasetError(f"Could not read {path}: {exc}") from exc

        if landmarks.ndim != 3 or landmarks.shape[2] != 3:
            raise DatasetError(
                f"{path} holds landmarks shaped {tuple(landmarks.shape)}; "
                "expected (frames, landmarks, 3)."
            )
        if landmarks.shape[1] != expected:
            raise DatasetError(
                f"{path} has {landmarks.shape[1]} landmarks per frame, but the "
                f"configured layout '{extractor.layout.name}' expects {expected}. "
                "Re-record, or switch the layout in config.yaml."
            )

        label = resolve_label(path, stored_label, directory)
        clips.append(
            Clip(features=extractor.transform_sequence(landmarks, mask), label=label, path=path)
        )

    logger.info("Loaded %d clips from %s", len(clips), directory)
    return clips


def build_vocabulary(clips: Sequence[Clip]) -> List[str]:
    """The sorted label set, which fixes the model's output order.

    Raises:
        DatasetError: If there are fewer than two labels to tell apart.
    """
    labels = sorted({clip.label for clip in clips})
    if len(labels) < 2:
        raise DatasetError(
            f"Training needs at least two labels; the data holds {labels or 'none'}. "
            "Record clips for more signs."
        )
    return labels


def label_counts(clips: Sequence[Clip]) -> Dict[str, int]:
    """Clips per label, for the pre-training summary."""
    return dict(sorted(Counter(clip.label for clip in clips).items()))


def stratified_split(
    clips: Sequence[Clip], val_fraction: float, seed: int
) -> Tuple[List[Clip], List[Clip]]:
    """Split clips per label, so every sign appears in both halves.

    A plain random split can leave a rare sign entirely out of validation, which
    makes the reported accuracy quietly meaningless.

    Raises:
        DatasetError: If a label has too few clips to appear on both sides.
    """
    if not 0.0 < val_fraction < 1.0:
        raise DatasetError(f"val_fraction must be between 0 and 1; got {val_fraction}.")

    rng = np.random.default_rng(seed)
    by_label: Dict[str, List[Clip]] = {}
    for clip in clips:
        by_label.setdefault(clip.label, []).append(clip)

    train: List[Clip] = []
    validation: List[Clip] = []
    for label, group in sorted(by_label.items()):
        if len(group) < 2:
            raise DatasetError(
                f"Label '{label}' has only {len(group)} clip. Every sign needs at "
                "least 2 so it can appear in both training and validation - "
                "record more."
            )
        order = rng.permutation(len(group))
        held_out = max(1, int(round(len(group) * val_fraction)))
        held_out = min(held_out, len(group) - 1)  # never leave training empty
        validation.extend(group[int(i)] for i in order[:held_out])
        train.extend(group[int(i)] for i in order[held_out:])
    return train, validation


class ClipDataset:
    """Fixed-length feature windows and their label indices.

    Args:
        clips: The clips to serve.
        labels: The vocabulary, fixing label order.
        window: Frames per window; clips are resampled to this length.
        augment: Apply jitter and time shift. Training split only - augmenting
            validation would make the score depend on the random seed.
        noise: Standard deviation of the Gaussian jitter.
        max_shift: Largest time shift in frames.
        seed: Fixes the augmentation stream.
    """

    def __init__(
        self,
        clips: Sequence[Clip],
        labels: Sequence[str],
        window: int,
        augment: bool = False,
        noise: float = 0.0,
        max_shift: int = 0,
        seed: int = 0,
    ) -> None:
        self.clips = list(clips)
        self.labels = list(labels)
        self.window = int(window)
        self.augment = bool(augment)
        self.noise = float(noise)
        self.max_shift = int(max_shift)
        self._index = {label: position for position, label in enumerate(self.labels)}
        self._rng = np.random.default_rng(seed)

        unknown = {clip.label for clip in self.clips} - set(self._index)
        if unknown:
            raise DatasetError(
                f"Clips carry labels absent from the vocabulary: {', '.join(sorted(unknown))}."
            )

    def __len__(self) -> int:
        """Number of clips."""
        return len(self.clips)

    def __getitem__(self, position: int) -> Tuple[np.ndarray, int]:
        """Return one ``(window, dim)`` window and its label index."""
        clip = self.clips[position]
        features = resample_sequence(clip.features, self.window)

        if self.augment:
            if self.max_shift > 0:
                shift = int(self._rng.integers(-self.max_shift, self.max_shift + 1))
                features = time_shift(features, shift)
            if self.noise > 0.0:
                features = features + self._rng.normal(
                    0.0, self.noise, size=features.shape
                ).astype(np.float32)

        return features.astype(np.float32, copy=False), self._index[clip.label]

    def batches(self, batch_size: int, shuffle: bool) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """Yield ``(features (B, T, D), labels (B,))`` numpy batches."""
        order = self._rng.permutation(len(self)) if shuffle else np.arange(len(self))
        for start in range(0, len(order), batch_size):
            chunk = order[start : start + batch_size]
            features, targets = zip(*(self[int(i)] for i in chunk))
            yield np.stack(features), np.asarray(targets, dtype=np.int64)

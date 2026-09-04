"""Shared fixtures."""

from __future__ import annotations

import numpy as np
import pytest

from signbridge.config import load_config
from signbridge.features import FeatureExtractor
from signbridge.landmarks import HANDS


@pytest.fixture
def extractor() -> FeatureExtractor:
    """A hands-layout feature extractor with default settings."""
    return FeatureExtractor(HANDS)


@pytest.fixture
def config():
    """Default configuration."""
    return load_config()


@pytest.fixture
def rng() -> np.random.Generator:
    """A seeded generator, so failures reproduce."""
    return np.random.default_rng(20260904)


def make_frame(rng: np.random.Generator, num_landmarks: int = 42, hands: int = 2):
    """Build one plausible landmark frame and its mask."""
    landmarks = rng.random((num_landmarks, 3)).astype(np.float32)
    mask = np.zeros(num_landmarks, dtype=np.float32)
    if hands >= 1:
        mask[:21] = 1.0
    if hands >= 2:
        mask[21:42] = 1.0
    landmarks[mask == 0] = 0.0
    return landmarks, mask


def make_clip(rng: np.random.Generator, frames: int = 45, num_landmarks: int = 42):
    """Build a clip of landmarks and masks."""
    pairs = [make_frame(rng, num_landmarks) for _ in range(frames)]
    return (
        np.stack([p[0] for p in pairs]),
        np.stack([p[1] for p in pairs]),
    )

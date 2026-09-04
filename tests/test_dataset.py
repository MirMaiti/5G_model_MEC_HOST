"""Loading recorded clips: label resolution, splitting, and clear failures."""

from __future__ import annotations

import numpy as np
import pytest

from signbridge.training.dataset import (
    ClipDataset,
    DatasetError,
    build_vocabulary,
    label_counts,
    load_clips,
    stratified_split,
)
from tests.conftest import make_clip


def write_clip(path, rng, label=None, frames=45, num_landmarks=42):
    """Write one .npz clip, optionally embedding the label."""
    path.parent.mkdir(parents=True, exist_ok=True)
    landmarks, mask = make_clip(rng, frames=frames, num_landmarks=num_landmarks)
    arrays = {"landmarks": landmarks, "mask": mask}
    if label is not None:
        arrays["label"] = np.array(label)
    np.savez_compressed(path, **arrays)


def test_label_comes_from_the_npz_field(tmp_path, extractor, rng):
    write_clip(tmp_path / "anything.npz", rng, label="hello")
    assert load_clips(tmp_path, extractor)[0].label == "hello"


def test_label_falls_back_to_the_directory_name(tmp_path, extractor, rng):
    write_clip(tmp_path / "thanks" / "clip_001.npz", rng)
    assert load_clips(tmp_path, extractor)[0].label == "thanks"


def test_label_falls_back_to_the_filename_prefix(tmp_path, extractor, rng):
    write_clip(tmp_path / "yes-a1b2c3.npz", rng)
    assert load_clips(tmp_path, extractor)[0].label == "yes"


def test_mask_is_inferred_when_absent(tmp_path, extractor, rng):
    """Clips recorded without a mask must still load."""
    landmarks, _ = make_clip(rng, frames=10)
    landmarks[:, 21:, :] = 0.0  # right hand never detected
    np.savez_compressed(tmp_path / "hello-1.npz", landmarks=landmarks)
    clip = load_clips(tmp_path, extractor)[0]
    assert clip.features.shape == (10, extractor.dim)


def test_missing_directory_says_how_to_record(tmp_path, extractor):
    with pytest.raises(DatasetError, match="collect"):
        load_clips(tmp_path / "nope", extractor)


def test_empty_directory_is_an_error(tmp_path, extractor):
    with pytest.raises(DatasetError, match="No .* clips"):
        load_clips(tmp_path, extractor)


def test_wrong_landmark_count_names_the_layout(tmp_path, extractor, rng):
    write_clip(tmp_path / "hello-1.npz", rng, label="hello", num_landmarks=75)
    with pytest.raises(DatasetError, match="expects 42"):
        load_clips(tmp_path, extractor)


def test_vocabulary_needs_two_labels(tmp_path, extractor, rng):
    write_clip(tmp_path / "hello-1.npz", rng, label="hello")
    clips = load_clips(tmp_path, extractor)
    with pytest.raises(DatasetError, match="at least two labels"):
        build_vocabulary(clips)


def test_stratified_split_puts_every_label_on_both_sides(tmp_path, extractor, rng):
    for label in ("hello", "thanks", "yes"):
        for index in range(6):
            write_clip(tmp_path / label / f"{index}.npz", rng)
    clips = load_clips(tmp_path, extractor)
    train, validation = stratified_split(clips, val_fraction=0.25, seed=1)

    assert {c.label for c in train} == {"hello", "thanks", "yes"}
    assert {c.label for c in validation} == {"hello", "thanks", "yes"}
    assert len(train) + len(validation) == len(clips)


def test_split_refuses_a_label_with_a_single_clip(tmp_path, extractor, rng):
    write_clip(tmp_path / "hello" / "1.npz", rng)
    write_clip(tmp_path / "hello" / "2.npz", rng)
    write_clip(tmp_path / "rare" / "1.npz", rng)
    clips = load_clips(tmp_path, extractor)
    with pytest.raises(DatasetError, match="only 1 clip"):
        stratified_split(clips, val_fraction=0.2, seed=1)


def test_split_never_empties_the_training_side(tmp_path, extractor, rng):
    for index in range(2):
        write_clip(tmp_path / "hello" / f"{index}.npz", rng)
        write_clip(tmp_path / "yes" / f"{index}.npz", rng)
    clips = load_clips(tmp_path, extractor)
    train, validation = stratified_split(clips, val_fraction=0.9, seed=1)
    assert len(train) == 2 and len(validation) == 2


def test_dataset_resamples_every_clip_to_the_window(tmp_path, extractor, rng):
    write_clip(tmp_path / "hello" / "1.npz", rng, frames=12)
    write_clip(tmp_path / "yes" / "1.npz", rng, frames=88)
    clips = load_clips(tmp_path, extractor)
    dataset = ClipDataset(clips, ["hello", "yes"], window=45)

    for index in range(len(dataset)):
        features, target = dataset[index]
        assert features.shape == (45, extractor.dim)
        assert target in (0, 1)


def test_augmentation_perturbs_only_when_enabled(tmp_path, extractor, rng):
    write_clip(tmp_path / "hello" / "1.npz", rng)
    write_clip(tmp_path / "yes" / "1.npz", rng)
    clips = load_clips(tmp_path, extractor)

    plain = ClipDataset(clips, ["hello", "yes"], window=45, augment=False)
    assert np.array_equal(plain[0][0], plain[0][0])

    noisy = ClipDataset(clips, ["hello", "yes"], window=45, augment=True, noise=0.05, max_shift=2)
    assert not np.array_equal(noisy[0][0], noisy[0][0])


def test_batches_cover_every_clip(tmp_path, extractor, rng):
    for label in ("hello", "yes"):
        for index in range(5):
            write_clip(tmp_path / label / f"{index}.npz", rng)
    clips = load_clips(tmp_path, extractor)
    dataset = ClipDataset(clips, ["hello", "yes"], window=45)

    seen = sum(features.shape[0] for features, _ in dataset.batches(4, shuffle=True))
    assert seen == len(dataset)


def test_label_counts(tmp_path, extractor, rng):
    write_clip(tmp_path / "hello" / "1.npz", rng)
    write_clip(tmp_path / "hello" / "2.npz", rng)
    write_clip(tmp_path / "yes" / "1.npz", rng)
    assert label_counts(load_clips(tmp_path, extractor)) == {"hello": 2, "yes": 1}

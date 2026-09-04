"""Configuration: one YAML file, plain dataclasses, no framework.

Every stage reads the same file. Sections map one-to-one onto dataclasses, so a
typo in the YAML raises a clear ``TypeError`` naming the bad key instead of
being silently ignored.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import yaml

PathLike = Union[str, Path]


@dataclass
class CaptureConfig:
    """Camera and landmark extraction on the host."""

    camera_index: int = 0
    width: int = 640
    height: int = 480
    target_fps: int = 30
    #: MediaPipe Tasks bundle. Fetch with ``python -m signbridge.cli.fetch_models``.
    hand_model_path: str = "models/hand_landmarker.task"
    pose_model_path: str = "models/pose_landmarker_lite.task"
    min_hand_detection_confidence: float = 0.5
    min_hand_presence_confidence: float = 0.5
    min_tracking_confidence: float = 0.5
    #: Mirror the preview so it reads like a mirror, as signers expect.
    mirror_preview: bool = True


@dataclass
class DataConfig:
    """Where recorded clips live and how long they are."""

    root: str = "data"
    train_split: str = "train"
    val_split: str = "val"
    #: Frames recorded per sample by the collector.
    clip_frames: int = 45
    #: Countdown before recording starts, so you can get your hands up.
    countdown_seconds: float = 2.0


@dataclass
class ModelConfig:
    """Architecture selection and its hyperparameters."""

    #: One of the names in ``signbridge.model.registry``.
    architecture: str = "gru"
    hidden_dim: int = 192
    num_layers: int = 2
    dropout: float = 0.3
    #: Frames the sequence models see at once. Ignored by ``mlp``.
    window: int = 45


@dataclass
class TrainingConfig:
    """Optimisation settings for ``signbridge.cli.train``."""

    epochs: int = 60
    batch_size: int = 32
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    #: Fraction of training clips held out when no explicit val split exists.
    val_fraction: float = 0.2
    #: Stop when validation accuracy has not improved for this many epochs.
    patience: int = 15
    seed: int = 1234
    device: Optional[str] = None
    checkpoint_dir: str = "models"
    #: Small jitter applied to training clips only. Zero disables it.
    augment_noise: float = 0.01
    augment_time_shift: int = 3


@dataclass
class ServerConfig:
    """The MEC-side TCP server."""

    host: str = "0.0.0.0"
    port: int = 9009
    checkpoint: str = "models/best.pt"
    device: Optional[str] = None
    #: Frames kept in the rolling buffer per connection.
    window_size: int = 45
    #: Run the model every N frames rather than on every frame.
    inference_interval: int = 5
    #: Frames required before the first prediction.
    min_buffer: int = 20
    #: Majority-vote over this many recent predictions; 1 disables smoothing.
    vote_window: int = 5
    #: Predictions below this confidence are reported as empty.
    min_confidence: float = 0.6
    max_connections: int = 8


@dataclass
class TTSConfig:
    """Speech synthesis on the host."""

    #: ``auto``, ``macos_say``, ``pyttsx3`` or ``null``.
    backend: str = "auto"
    voice: Optional[str] = None
    rate: Optional[int] = None
    #: Never repeat the same phrase within this many seconds.
    repeat_cooldown: float = 2.5
    #: A prediction must hold steady this long before it is spoken.
    stability_seconds: float = 0.4


@dataclass
class HostConfig:
    """The capture-side client."""

    server_host: str = "127.0.0.1"
    server_port: int = 9009
    #: Seconds to wait on connect before giving up.
    connect_timeout: float = 5.0
    #: Show an OpenCV preview window with the landmarks drawn on it.
    show_preview: bool = True
    #: Reconnect automatically when the link drops.
    auto_reconnect: bool = True
    reconnect_delay: float = 2.0


@dataclass
class Config:
    """The whole configuration tree."""

    layout: str = "hands"
    include_wrist_position: bool = True
    include_interhand: bool = True
    capture: CaptureConfig = field(default_factory=CaptureConfig)
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    server: ServerConfig = field(default_factory=ServerConfig)
    tts: TTSConfig = field(default_factory=TTSConfig)
    host: HostConfig = field(default_factory=HostConfig)

    def to_dict(self) -> Dict[str, Any]:
        """Plain-dict form, for writing into a checkpoint."""
        return dataclasses.asdict(self)

    def feature_extractor(self):
        """Build the :class:`~signbridge.features.FeatureExtractor` this config describes."""
        from .features import FeatureConfig, FeatureExtractor
        from .landmarks import get_layout

        return FeatureExtractor(
            layout=get_layout(self.layout),
            config=FeatureConfig(
                include_wrist_position=self.include_wrist_position,
                include_interhand=self.include_interhand,
            ),
        )


_SECTIONS = {
    "capture": CaptureConfig,
    "data": DataConfig,
    "model": ModelConfig,
    "training": TrainingConfig,
    "server": ServerConfig,
    "tts": TTSConfig,
    "host": HostConfig,
}


def _build_section(name: str, values: Any):
    """Instantiate one section dataclass, naming any unknown key."""
    cls = _SECTIONS[name]
    if values is None:
        return cls()
    if not isinstance(values, dict):
        raise TypeError(f"Config section '{name}' must be a mapping; got {type(values).__name__}.")
    known = {f.name for f in dataclasses.fields(cls)}
    unknown = set(values) - known
    if unknown:
        raise TypeError(
            f"Unknown key(s) in config section '{name}': {', '.join(sorted(unknown))}. "
            f"Valid keys: {', '.join(sorted(known))}."
        )
    return cls(**values)


def load_config(path: Optional[PathLike] = None, overrides: Optional[Dict[str, Any]] = None) -> Config:
    """Load a config from YAML, falling back to defaults.

    Args:
        path: The YAML file. ``None`` returns pure defaults.
        overrides: Nested dict merged over the file, for CLI flags.

    Raises:
        FileNotFoundError: If ``path`` is given but missing.
        TypeError: If the file contains an unknown key.
    """
    raw: Dict[str, Any] = {}
    if path is not None:
        config_path = Path(path)
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        loaded = yaml.safe_load(config_path.read_text()) or {}
        if not isinstance(loaded, dict):
            raise TypeError(f"{config_path} must contain a YAML mapping at the top level.")
        raw = loaded

    if overrides:
        raw = _deep_merge(raw, overrides)

    top_level = {k: v for k, v in raw.items() if k not in _SECTIONS}
    known_top = {"layout", "include_wrist_position", "include_interhand"}
    unknown = set(top_level) - known_top
    if unknown:
        raise TypeError(
            f"Unknown top-level config key(s): {', '.join(sorted(unknown))}. "
            f"Valid: {', '.join(sorted(known_top | set(_SECTIONS)))}."
        )

    sections = {name: _build_section(name, raw.get(name)) for name in _SECTIONS}
    return Config(**top_level, **sections)


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Merge ``override`` into ``base`` without mutating either."""
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        elif value is not None:
            merged[key] = value
    return merged

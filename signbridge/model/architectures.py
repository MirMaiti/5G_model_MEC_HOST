"""The classifiers themselves.

All three take the same input - a batch of fixed-length feature windows shaped
``(B, T, D)`` - and return logits shaped ``(B, C)``. That uniform signature is
what lets the architecture be a string in ``config.yaml`` rather than a branch
scattered through the training and serving code.

Which to use:

``mlp``
    Pools the window and classifies. Fastest, and enough for static handshapes
    where the pose alone identifies the sign.
``gru``
    Bidirectional recurrent encoder. The default: signs like THANKYOU are a
    movement, not a pose, and order matters.
``transformer``
    Self-attention encoder. More capacity than the GRU and worth trying once
    the vocabulary grows, but it wants more data.
"""

from __future__ import annotations

import math
from typing import Any, Dict

import torch
import torch.nn as nn

from .registry import register_architecture


class SequenceClassifier(nn.Module):
    """Base class fixing the input and output contract.

    Args:
        input_dim: Features per frame, from the feature extractor.
        num_classes: Size of the label set.
        window: Frames per input window.
    """

    def __init__(self, input_dim: int, num_classes: int, window: int, **_: Any) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.num_classes = int(num_classes)
        self.window = int(window)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # pragma: no cover - abstract
        """Map ``(B, T, D)`` features to ``(B, C)`` logits."""
        raise NotImplementedError

    def _check(self, x: torch.Tensor) -> None:
        """Fail with a useful message rather than a shape error deep in a layer."""
        if x.ndim != 3:
            raise ValueError(f"Expected input shaped (B, T, D); got {tuple(x.shape)}.")
        if x.shape[-1] != self.input_dim:
            raise ValueError(
                f"This model was built for {self.input_dim} features per frame; "
                f"got {x.shape[-1]}. The feature config and the checkpoint disagree."
            )


@register_architecture("mlp")
class PooledMLP(SequenceClassifier):
    """Mean-and-max pooling over time, then a small MLP.

    Pooling discards frame order, which is the point: for a static handshape it
    removes a nuisance variable, and it makes the model indifferent to how the
    window happens to be aligned with the sign.
    """

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        window: int,
        hidden_dim: int = 192,
        num_layers: int = 2,
        dropout: float = 0.3,
        **_: Any,
    ) -> None:
        super().__init__(input_dim, num_classes, window)
        layers: list[nn.Module] = []
        width = input_dim * 2  # mean and max concatenated
        for _index in range(max(1, num_layers)):
            layers += [nn.Linear(width, hidden_dim), nn.LayerNorm(hidden_dim), nn.ReLU(), nn.Dropout(dropout)]
            width = hidden_dim
        self.encoder = nn.Sequential(*layers)
        self.head = nn.Linear(width, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Map ``(B, T, D)`` features to ``(B, C)`` logits."""
        self._check(x)
        pooled = torch.cat([x.mean(dim=1), x.max(dim=1).values], dim=-1)
        return self.head(self.encoder(pooled))


@register_architecture("gru")
class GRUClassifier(SequenceClassifier):
    """Bidirectional GRU over the window, classified from the final states."""

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        window: int,
        hidden_dim: int = 192,
        num_layers: int = 2,
        dropout: float = 0.3,
        **_: Any,
    ) -> None:
        super().__init__(input_dim, num_classes, window)
        self.input_norm = nn.LayerNorm(input_dim)
        self.rnn = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=max(1, num_layers),
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Map ``(B, T, D)`` features to ``(B, C)`` logits."""
        self._check(x)
        output, _hidden = self.rnn(self.input_norm(x))
        # Last step of the forward pass and first of the backward pass: together
        # they have seen the whole window from both ends.
        forward_last = output[:, -1, : output.shape[-1] // 2]
        backward_first = output[:, 0, output.shape[-1] // 2 :]
        return self.head(self.dropout(torch.cat([forward_last, backward_first], dim=-1)))


class _PositionalEncoding(nn.Module):
    """Fixed sinusoidal positions; attention is otherwise order-blind."""

    def __init__(self, dim: int, max_len: int = 512) -> None:
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1).float()
        divisor = torch.exp(torch.arange(0, dim, 2).float() * (-math.log(10000.0) / dim))
        encoding = torch.zeros(max_len, dim)
        encoding[:, 0::2] = torch.sin(position * divisor)
        encoding[:, 1::2] = torch.cos(position * divisor[: encoding[:, 1::2].shape[-1]])
        self.register_buffer("encoding", encoding.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encodings to ``(B, T, D)``."""
        return x + self.encoding[:, : x.shape[1]]


@register_architecture("transformer")
class TransformerClassifier(SequenceClassifier):
    """Transformer encoder with mean pooling over time."""

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        window: int,
        hidden_dim: int = 192,
        num_layers: int = 2,
        dropout: float = 0.3,
        num_heads: int = 4,
        **_: Any,
    ) -> None:
        super().__init__(input_dim, num_classes, window)
        # nn.MultiheadAttention requires the model width to divide evenly by the
        # head count; round up rather than failing on an awkward hidden_dim.
        if hidden_dim % num_heads != 0:
            hidden_dim = num_heads * math.ceil(hidden_dim / num_heads)
        self.project = nn.Linear(input_dim, hidden_dim)
        self.positions = _PositionalEncoding(hidden_dim, max_len=max(window, 512))
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=max(1, num_layers))
        self.norm = nn.LayerNorm(hidden_dim)
        self.head = nn.Linear(hidden_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Map ``(B, T, D)`` features to ``(B, C)`` logits."""
        self._check(x)
        encoded = self.encoder(self.positions(self.project(x)))
        return self.head(self.norm(encoded.mean(dim=1)))


def build_model(
    architecture: str, input_dim: int, num_classes: int, window: int, **kwargs: Any
) -> SequenceClassifier:
    """Instantiate a registered architecture by name.

    Raises:
        KeyError: If the architecture is not registered.
    """
    from .registry import get_architecture

    cls = get_architecture(architecture)
    return cls(input_dim=input_dim, num_classes=num_classes, window=window, **kwargs)


def model_hyperparameters(config: Any) -> Dict[str, Any]:
    """Pull the architecture kwargs out of a :class:`~signbridge.config.ModelConfig`."""
    return {
        "hidden_dim": config.hidden_dim,
        "num_layers": config.num_layers,
        "dropout": config.dropout,
    }

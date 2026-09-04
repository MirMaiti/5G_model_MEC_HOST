"""A name-to-architecture registry.

Adding an architecture means writing the module and decorating it - nothing
else in the codebase needs to change, and the name becomes valid in
``config.yaml`` immediately.

    @register_architecture("my_net")
    class MyNet(SequenceClassifier):
        ...
"""

from __future__ import annotations

from typing import Callable, Dict, Tuple, Type

_REGISTRY: Dict[str, Type] = {}


def register_architecture(name: str) -> Callable[[Type], Type]:
    """Class decorator registering an architecture under ``name``.

    Raises:
        ValueError: If the name is already taken.
    """

    def decorator(cls: Type) -> Type:
        if name in _REGISTRY:
            raise ValueError(f"Architecture {name!r} is already registered.")
        _REGISTRY[name] = cls
        return cls

    return decorator


def get_architecture(name: str) -> Type:
    """Look up a registered architecture.

    Raises:
        KeyError: If no architecture goes by that name.
    """
    try:
        return _REGISTRY[name]
    except KeyError:
        known = ", ".join(sorted(_REGISTRY)) or "<none registered>"
        raise KeyError(f"Unknown architecture {name!r}. Available: {known}.") from None


def architecture_names() -> Tuple[str, ...]:
    """Every registered architecture name."""
    return tuple(sorted(_REGISTRY))

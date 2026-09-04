"""Speech synthesis on the capture host.

Two problems separate a working demo from an unusable one, and both are handled
here rather than in the client loop:

*Blocking.* Synthesis takes far longer than a frame interval. Speaking inline
would stall capture and the video would visibly hitch, so a worker thread owns
the backend and the capture loop only ever enqueues.

*Chatter.* The server emits a prediction several times a second. Speaking each
one would produce a stutter of repeated words, so a label must hold steady for
``stability_seconds`` before it is spoken, and the same phrase will not repeat
within ``repeat_cooldown`` seconds.

Backends are looked up by name, so adding one - a cloud voice, a different
engine - means writing a class and registering it.
"""

from __future__ import annotations

import logging
import platform
import queue
import shutil
import subprocess
import threading
import time
from abc import ABC, abstractmethod
from typing import Callable, Dict, Optional

logger = logging.getLogger(__name__)

_BACKENDS: Dict[str, Callable[..., "TTSBackend"]] = {}


def register_backend(name: str) -> Callable[[type], type]:
    """Class decorator registering a TTS backend under ``name``."""

    def decorator(cls: type) -> type:
        _BACKENDS[name] = cls  # type: ignore[assignment]
        return cls

    return decorator


class TTSBackend(ABC):
    """Speaks a phrase, blocking until it finishes."""

    @abstractmethod
    def say(self, text: str) -> None:
        """Speak ``text``. Called only from the worker thread."""

    def close(self) -> None:
        """Release any resources the backend holds."""

    @staticmethod
    def available() -> bool:
        """Whether this backend can run on this machine."""
        return True


@register_backend("null")
class NullBackend(TTSBackend):
    """Logs instead of speaking. Used on machines with no working engine."""

    def say(self, text: str) -> None:
        """Log the phrase that would have been spoken."""
        logger.info("[tts:null] %s", text)


@register_backend("macos_say")
class MacSayBackend(TTSBackend):
    """macOS ``say``. No extra dependency and it is always present.

    Args:
        voice: A voice name from ``say -v '?'``. ``None`` uses the system voice.
        rate: Words per minute. ``None`` uses the system rate.
    """

    def __init__(self, voice: Optional[str] = None, rate: Optional[int] = None) -> None:
        self._command = ["say"]
        if voice:
            self._command += ["-v", voice]
        if rate:
            self._command += ["-r", str(int(rate))]

    @staticmethod
    def available() -> bool:
        """True on macOS with ``say`` on PATH."""
        return platform.system() == "Darwin" and shutil.which("say") is not None

    def say(self, text: str) -> None:
        """Speak via the ``say`` binary."""
        try:
            subprocess.run(self._command + [text], check=True, capture_output=True, timeout=30)
        except (subprocess.SubprocessError, OSError) as exc:
            logger.warning("say failed for %r: %s", text, exc)


@register_backend("pyttsx3")
class Pyttsx3Backend(TTSBackend):
    """Cross-platform offline synthesis through pyttsx3.

    Args:
        voice: Substring matched against installed voice names.
        rate: Words per minute.

    Raises:
        ImportError: If pyttsx3 is not installed.
    """

    def __init__(self, voice: Optional[str] = None, rate: Optional[int] = None) -> None:
        import pyttsx3  # imported lazily: it is an optional dependency

        self._engine = pyttsx3.init()
        if rate:
            self._engine.setProperty("rate", int(rate))
        if voice:
            for candidate in self._engine.getProperty("voices"):
                if voice.lower() in candidate.name.lower():
                    self._engine.setProperty("voice", candidate.id)
                    break
            else:
                logger.warning("No pyttsx3 voice matching %r; using the default.", voice)

    @staticmethod
    def available() -> bool:
        """True when pyttsx3 imports."""
        try:
            import pyttsx3  # noqa: F401
        except Exception:
            return False
        return True

    def say(self, text: str) -> None:
        """Speak through the pyttsx3 engine."""
        self._engine.say(text)
        self._engine.runAndWait()

    def close(self) -> None:
        """Stop the engine."""
        try:
            self._engine.stop()
        except Exception:  # pragma: no cover - engine already torn down
            pass


def build_backend(
    name: str = "auto", voice: Optional[str] = None, rate: Optional[int] = None
) -> TTSBackend:
    """Construct a backend by name.

    Args:
        name: A registered name, or ``auto`` to pick the best available.

    Raises:
        KeyError: If the name is not registered.
    """
    if name == "auto":
        for candidate in ("macos_say", "pyttsx3"):
            cls = _BACKENDS[candidate]
            if cls.available():  # type: ignore[attr-defined]
                logger.info("TTS backend: %s", candidate)
                return cls(voice=voice, rate=rate)  # type: ignore[call-arg]
        logger.warning("No speech engine available; predictions will only be printed.")
        return NullBackend()

    if name not in _BACKENDS:
        known = ", ".join(sorted(_BACKENDS))
        raise KeyError(f"Unknown TTS backend {name!r}. Available: {known}.")
    cls = _BACKENDS[name]
    if name == "null":
        return cls()  # type: ignore[call-arg]
    return cls(voice=voice, rate=rate)  # type: ignore[call-arg]


def backend_names() -> tuple:
    """Every registered backend name."""
    return tuple(sorted(_BACKENDS))


class Speaker:
    """Decides *whether* to speak, and speaks off the capture thread.

    Args:
        backend: The engine to speak through.
        repeat_cooldown: Seconds before the same phrase may repeat.
        stability_seconds: How long a label must hold before it is spoken.
        queue_size: Pending phrases to hold; older ones are dropped when full,
            because stale predictions are not worth saying late.
    """

    def __init__(
        self,
        backend: TTSBackend,
        repeat_cooldown: float = 2.5,
        stability_seconds: float = 0.4,
        queue_size: int = 4,
    ) -> None:
        self.backend = backend
        self.repeat_cooldown = float(repeat_cooldown)
        self.stability_seconds = float(stability_seconds)
        self._queue: "queue.Queue[Optional[str]]" = queue.Queue(maxsize=max(1, queue_size))
        self._candidate: Optional[str] = None
        self._candidate_since = 0.0
        self._last_spoken: Optional[str] = None
        self._last_spoken_at = 0.0
        self._closed = False
        self._worker = threading.Thread(target=self._run, name="signbridge-tts", daemon=True)
        self._worker.start()

    def _run(self) -> None:
        """Drain the queue, speaking each phrase in turn."""
        while True:
            text = self._queue.get()
            if text is None:
                return
            try:
                self.backend.say(text)
            except Exception:  # pragma: no cover - a bad engine must not kill the host
                logger.exception("TTS backend failed on %r", text)

    def offer(self, label: str, now: Optional[float] = None) -> bool:
        """Offer the current stable label; speak it if it has earned it.

        Args:
            label: The smoothed label from the server. Empty means silence.
            now: Injectable clock, for tests.

        Returns:
            True when the phrase was queued for speech.
        """
        moment = time.monotonic() if now is None else now

        if not label:
            self._candidate = None
            return False

        if label != self._candidate:
            self._candidate = label
            self._candidate_since = moment
            return False

        if moment - self._candidate_since < self.stability_seconds:
            return False

        if label == self._last_spoken and moment - self._last_spoken_at < self.repeat_cooldown:
            return False

        self._last_spoken = label
        self._last_spoken_at = moment
        try:
            self._queue.put_nowait(label)
        except queue.Full:
            # Speech is already backed up; saying this one late would be worse
            # than not saying it.
            logger.debug("TTS queue full, dropping %r", label)
            return False
        return True

    def close(self) -> None:
        """Stop the worker and release the backend."""
        if self._closed:
            return
        self._closed = True
        try:
            self._queue.put_nowait(None)
        except queue.Full:  # pragma: no cover
            pass
        self._worker.join(timeout=2.0)
        self.backend.close()

    def __enter__(self) -> "Speaker":
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

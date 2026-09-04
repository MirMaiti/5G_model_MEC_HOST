"""The MEC server must never need a camera stack, and vice versa.

This is the architectural claim the whole project rests on: because the host
sends landmarks rather than pixels, the server needs no MediaPipe and no
OpenCV, and the container can stay small and architecture-independent. It is
easy to break with one careless top-level import, so it is asserted here.

Each check runs in a fresh interpreter - once a module is imported in this
process it would stay in ``sys.modules`` and the test would pass vacuously.
"""

from __future__ import annotations

import subprocess
import sys

CHECK = (
    "import sys, {module};"
    "loaded=[m for m in {watch} if m in sys.modules];"
    "print(','.join(loaded))"
)


def _modules_loaded_by(module: str, watch: tuple) -> set:
    """Import ``module`` in a clean interpreter and report which of ``watch`` it pulled in."""
    result = subprocess.run(
        [sys.executable, "-c", CHECK.format(module=module, watch=watch)],
        capture_output=True,
        text=True,
        check=True,
    )
    return {name for name in result.stdout.strip().split(",") if name}


def test_server_does_not_import_a_camera_stack():
    assert _modules_loaded_by("signbridge.cli.serve", ("mediapipe", "cv2")) == set()


def test_host_does_not_import_torch():
    """The capture device runs no model, so it should not need PyTorch."""
    assert _modules_loaded_by("signbridge.cli.run_host", ("torch",)) == set()


def test_core_modules_are_numpy_only():
    """Protocol and features are shared by both ends; keep them light."""
    for module in ("signbridge.protocol", "signbridge.features", "signbridge.sequence"):
        assert _modules_loaded_by(module, ("torch", "mediapipe", "cv2")) == set()

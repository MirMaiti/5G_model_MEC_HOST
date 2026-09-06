"""SignBridge: landmarks on the capture host, inference on the MEC server.

This module patches a stdlib gap before anything else runs. See the
docstring on ``_patch_int_str_digits_limit`` below for why.
"""

from __future__ import annotations


def _patch_int_str_digits_limit() -> None:
    """Work around a broken system Python 3.11 install.

    Ubuntu 22.04's ``python3.11`` package is frozen at the ``3.11.0~rc1``
    pre-release build (jammy-apps-security never shipped a newer one), and
    that build predates ``sys.set_int_max_str_digits`` /
    ``sys.get_int_max_str_digits`` (PEP added late in the 3.11 cycle).
    torch >= 2.x unconditionally references
    ``sys.get_int_max_str_digits`` in ``torch._dynamo.polyfills.sys`` on
    any interpreter reporting ``version_info >= (3, 11)``, so constructing
    an optimizer (which lazily imports ``torch._dynamo``) crashes with::

        AttributeError: module 'sys' has no attribute 'get_int_max_str_digits'

    The real fix is a complete Python 3.11+ (e.g. the system's
    python3.10, or python3.11/3.12 from deadsnakes); this shim just fills
    in the two functions with stdlib-equivalent behaviour so torch stops
    tripping over the gap. It only installs if the attribute is actually
    missing, so it is a no-op on any correct interpreter.
    """
    import sys

    if hasattr(sys, "get_int_max_str_digits"):
        return

    state = {"limit": 4300}  # CPython's default (sys.int_info.default_max_str_digits)

    def get_int_max_str_digits() -> int:
        return state["limit"]

    def set_int_max_str_digits(maxdigits: int) -> None:
        if maxdigits != 0 and maxdigits < 640:
            raise ValueError("maxdigits must be 0 or >= 640")
        state["limit"] = maxdigits

    sys.get_int_max_str_digits = get_int_max_str_digits
    sys.set_int_max_str_digits = set_int_max_str_digits


_patch_int_str_digits_limit()

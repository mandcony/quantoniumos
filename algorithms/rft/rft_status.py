#!/usr/bin/env python3
# SPDX-License-Identifier: LicenseRef-QuantoniumOS-Claims-NC
"""Centralized reporting of Unitary RFT availability.

This module probes the native Φ-RFT kernel once and exposes a shared
status dictionary to every subsystem (engine, synth, GUI). The DAW can now
truthfully signal when it is running the real unitary kernel versus a FFT
fallback so reviewers know exactly what they are hearing.
"""

from __future__ import annotations

import importlib
import os
import threading
from typing import Any, Dict, Optional

__all__ = [
    "get_status",
    "refresh_status",
    "is_unitary_available",
]

_VARIANT_NAMES = {
    0: "standard",
    1: "harmonic",
    2: "fibonacci",
    3: "chaotic",
    4: "geometric",
    5: "hybrid",
    6: "adaptive",
}

_STATUS_CACHE: Optional[Dict[str, Any]] = None
_STATUS_LOCK = threading.Lock()


def _probe_status() -> Dict[str, Any]:
    """Detect whether the native Φ-RFT kernel is active."""
    status: Dict[str, Any] = {
        "unitary": False,
        "variant": None,
        "kernel_version": None,
        "library_path": None,
        "is_mock": True,
        "error": None,
    }

    try:
        bindings = importlib.import_module(
            "algorithms.rft.kernels.python_bindings.unitary_rft"
        )
    except Exception as exc:  # noqa: BLE001 - we want the exact import error text
        status["error"] = f"import-error: {exc}"
        return status

    status["kernel_version"] = getattr(
        bindings, "__version__", os.getenv("RFT_KERNEL_VERSION")
    )

    variant_guess = getattr(bindings, "RFT_VARIANT_HARMONIC", None)
    status["variant"] = _VARIANT_NAMES.get(variant_guess)

    try:
        test_engine = bindings.UnitaryRFT(
            size=64,
            flags=getattr(bindings, "RFT_FLAG_UNITARY", 0),
            variant=variant_guess or getattr(bindings, "RFT_VARIANT_STANDARD", 0),
        )
    except Exception as exc:  # noqa: BLE001 - need precise failure reason
        status["error"] = f"init-error: {exc}"
        return status

    is_mock = getattr(test_engine, "_is_mock", True)
    status["is_mock"] = is_mock
    status["unitary"] = not is_mock

    variant_code = getattr(test_engine, "variant", None)
    if variant_code is not None:
        status["variant"] = _VARIANT_NAMES.get(variant_code, status["variant"])

    lib_handle = getattr(test_engine, "lib", None)
    lib_name = getattr(lib_handle, "_name", None) if lib_handle else None
    status["library_path"] = lib_name

    # Drop the temporary engine so its ctypes resources get cleaned up.
    del test_engine

    return status


def refresh_status() -> Dict[str, Any]:
    """Force a re-probe of the kernel."""
    global _STATUS_CACHE
    with _STATUS_LOCK:
        _STATUS_CACHE = _probe_status()
        return dict(_STATUS_CACHE)


def get_status(use_cache: bool = True) -> Dict[str, Any]:
    """Return the cached Φ-RFT status dictionary."""
    global _STATUS_CACHE
    with _STATUS_LOCK:
        if not use_cache or _STATUS_CACHE is None:
            _STATUS_CACHE = _probe_status()
        return dict(_STATUS_CACHE)


def is_unitary_available() -> bool:
    """Convenience helper for callers that only need a boolean."""
    return get_status().get("unitary", False)

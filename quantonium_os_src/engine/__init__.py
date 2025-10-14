"""Compatibility layer exposing engine modules under a stable namespace."""
from __future__ import annotations

import sys
from importlib import import_module
from types import ModuleType
from typing import Dict

__all__ = [
    "vertex_assembly",
    "open_quantum_systems",
]


_definitions: Dict[str, ModuleType] = {}


def _ensure_submodule(name: str) -> ModuleType:
    """Import a submodule from the historical engine layout."""
    if name not in _definitions:
        module = import_module(f"{__name__}.engine.{name}")
        sys.modules[f"{__name__}.{name}"] = module
        _definitions[name] = module
    return _definitions[name]


# Eagerly expose the primary modules required by the public tests.
vertex_assembly = _ensure_submodule("vertex_assembly")
open_quantum_systems = _ensure_submodule("open_quantum_systems")


def __getattr__(name: str) -> ModuleType:
    if name in __all__:
        return _ensure_submodule(name)
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))

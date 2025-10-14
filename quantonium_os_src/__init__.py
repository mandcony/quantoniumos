"""Top-level package for QuantoniumOS source modules.

This package primarily mirrors the internal project layout while
providing a stable import surface for external integrations and tests.
"""
from importlib import import_module as _import_module
from types import ModuleType as _ModuleType

__all__ = ["engine", "apps", "frontend", "safety"]


def _load_subpackage(name: str) -> _ModuleType:
    """Import a first-level subpackage on demand."""
    return _import_module(f"{__name__}.{name}")


def __getattr__(name: str) -> _ModuleType:
    if name in __all__:
        return _load_subpackage(name)
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))

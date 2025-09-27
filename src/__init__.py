"""QuantoniumOS source root package.# Package marker for src


Ensures `import src.*` works in tests and tools. Windows-friendly.
"""

from importlib import import_module as _import_module  # re-export convenience if desired

__all__ = [
    "_import_module",
]

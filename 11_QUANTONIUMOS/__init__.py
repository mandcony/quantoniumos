"""
QuantoniumOS Core Package

The unified QuantoniumOS core system providing quantum-enhanced
operating system capabilities, file systems, and core services.

This is the main QuantoniumOS package containing the core OS
functionality and unified interfaces.
"""

try:
    from .quantonium_os_unified import QuantoniumOS

    __all__ = ["QuantoniumOS"]
except ImportError:
    __all__ = []

__version__ = "1.0.0"

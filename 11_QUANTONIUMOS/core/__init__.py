#!/usr/bin/env python3
"""
QuantoniumOS - Complete Quantum Operating System
================================================

The world's first quantum operating system featuring:
- 1000-qubit quantum vertex network
- True Recursive Frequency Transform (RFT) engine
- Enhanced quantum cryptography
- Unified desktop and web interface

Quick Start:
    >>> from core.quantoniumos import QuantoniumOSUnified
    >>> os = QuantoniumOSUnified()
    >>> os.run()

Modules:
    - quantonium_os_unified: Main unified interface
    - start_quantoniumos: Launcher utilities
    - quantum_vertex_kernel: 1000-qubit quantum kernel
    - true_rft: Canonical RFT implementation
    - enhanced_crypto: RFT-based cryptography

Repository: https://github.com/mandcony/quantoniumos
"""

__version__ = "1.0.0"
__author__ = "QuantoniumOS Development Team"
__email__ = "dev@quantoniumos.com"
__license__ = "Proprietary"
__copyright__ = "Copyright 2024-2025 QuantoniumOS"

# Core imports
try:
    from .quantonium_os_unified import QuantoniumOSUnified
    from .start_quantoniumos import core.main as mainas launch
except ImportError:
    # Fallback for development environment
    import os
import sys

    sys.path.insert(0, os.path.dirname(__file__))
    try:
        from start_quantoniumos import core.main as mainas launch

        from core.quantonium_os_unified import QuantoniumOSUnified
    except ImportError:
        QuantoniumOSUnified = None
        launch = None

# Package metadata
__all__ = [
    "QuantoniumOSUnified",
    "launch",
    "__version__",
    "__author__",
    "__email__",
]


def version():
    """Return the current QuantoniumOS version."""
    return __version__


def info():
    """Display QuantoniumOS package information."""
    info_text = f"""
QuantoniumOS v{__version__}
===========================
Complete Quantum Operating System

Features:
• 1000-qubit quantum vertex network
• True RFT engine with patent protection
• Enhanced quantum cryptography  
• Unified desktop/web interface
• Real-time quantum simulation

Repository: https://github.com/mandcony/quantoniumos
Copyright: {__copyright__}
License: {__license__}
    """
    print(info_text)
    return info_text


def quick_start():
    """Launch QuantoniumOS with default configuration."""
    if QuantoniumOSUnified is not None:
        print("🚀 Starting QuantoniumOS...")
        os_instance = QuantoniumOSUnified()
        os_instance.run()
    else:
        print("❌ QuantoniumOS core modules not available")
        print("Please ensure the package is properly installed")


# Package initialization
if __name__ == "__main__":
    info()

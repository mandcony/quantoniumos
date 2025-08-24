#!/usr/bin/env python3
"""
PyQt5 QuantoniumOS Launcher
============================
Launches the modern PyQt5-based QuantoniumOS desktop interface.
"""

import os
import sys

# Add the current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def main():
    """Launch the PyQt5 QuantoniumOS interface."""
    print("=" * 60)
    print("    QuantoniumOS - PyQt5 Modern Interface")
    print("=" * 60)
    print("🚀 Initializing quantum desktop environment...")

    try:
        from quantonium_os_pyqt5 import QuantoniumOSPyQt5

        print("✅ PyQt5 interface loaded successfully")
        print("🔗 Connecting to quantum kernel...")

        app = QuantoniumOSPyQt5()
        print("🌟 Launching QuantoniumOS Desktop...")
        app.run()

    except ImportError as e:
        print(f"❌ Error: {e}")
        print("\n📋 Missing dependencies. Please install:")
        print("   pip install PyQt5")
        print("   or")
        print("   pip install -r requirements.txt")

    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()

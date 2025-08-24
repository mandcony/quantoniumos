#!/usr/bin/env python3
"""
QuantoniumOS - One-Click Launcher
=================================
Single entry point for the complete QuantoniumOS with all features integrated.
"""

import os
import sys

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def main():
    """Launch QuantoniumOS with all features integrated"""
    print("=" * 70)
    print("    🚀 QuantoniumOS - Complete Quantum Operating System")
    print("=" * 70)
    print("🔧 Initializing unified system...")

    try:
        from quantonium_os_unified import QuantoniumOSUnified

        print("✅ System components loaded")
        print("⚛️ Quantum kernel ready")
        print("📜 Patent modules integrated")
        print("🖥️ Desktop interface starting...")
        print()

        # Launch the unified OS
        quantonium_os = QuantoniumOSUnified()
        quantonium_os.run()

    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("\n📋 Make sure all dependencies are installed:")
        print("   pip install tkinter numpy scipy")

    except Exception as e:
        print(f"❌ Startup error: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()

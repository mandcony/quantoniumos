#!/usr/bin/env python3
"""
QuantoniumOS Main Launcher
=========================
Simple launcher for the organized QuantoniumOS project.
"""
import sys
from pathlib import Path

# Add the main OS directory to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / "11_QUANTONIUMOS"))


def main():
    """Launch QuantoniumOS"""
    print("🌟 QuantoniumOS - Launching Main System")
    print("=" * 50)

    try:
        # Import and run the main unified OS
        from core.quantonium_os_unified import core.main as mainas quantonium_main

        print("✅ QuantoniumOS imported successfully")
        return quantonium_main()

    except ImportError as e:
        print(f"❌ Import failed: {e}")
        print("\n🔧 Manual launch options:")
        print("   python 11_QUANTONIUMOS/quantonium_os_unified.py")
        print("   python 15_DEPLOYMENT/production/app.py")
        print("   python 03_RUNNING_SYSTEMS/app.py")
        return 1


if __name__ == "__main__":
    sys.exit(main())

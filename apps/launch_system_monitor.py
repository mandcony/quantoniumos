#!/usr/bin/env python3
"""
System Monitor App Launcher with Unified Design
"""
import os
import sys

# Add the app directory to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from quantonium_app_wrapper import launch_app
    from system_monitor import SystemMonitor

    def main():
        return launch_app(SystemMonitor, "System Monitor", "System Monitor")

    if __name__ == "__main__":
        sys.exit(main())

except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Please ensure PyQt5 is installed: pip install PyQt5")

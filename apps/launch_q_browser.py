#!/usr/bin/env python3
"""
Q-Browser App Launcher with Unified Design
"""
import os
import sys

# Add the app directory to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from q_browser import QuantumBrowser
    from quantonium_app_wrapper import launch_app

    def main():
        return launch_app(QuantumBrowser, "Q-Browser", "Quantum Browser")

    if __name__ == "__main__":
        sys.exit(main())

except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Please ensure PyQt5 and required packages are installed")

    # Fallback main function for validation
    def main():
        print("Q-Browser launcher (fallback mode)")
        return 0

#!/usr/bin/env python3
"""
QuantoniumOS GUI Launcher
Simple launcher for the GUI system
"""

import os
import sys
from pathlib import Path


def main():
    # Get project directories
    current_dir = Path(__file__).parent
    project_root = current_dir
    gui_dir = project_root / "11_QUANTONIUMOS"

    # Ensure design system is available
    sys.path.insert(0, str(project_root))

    # Create a copy of the design system in the GUI directory if needed
    design_system_file = project_root / "quantonium_design_system.py"
    gui_design_system_file = gui_dir / "quantonium_design_system.py"

    if not gui_design_system_file.exists() and design_system_file.exists():
        print("Creating design system file in GUI directory...")
        with open(design_system_file, "r") as src:
            with open(gui_design_system_file, "w") as dst:
                dst.write(src.read())

    print("Launching QuantoniumOS GUI...")

    # Run the unified GUI directly using subprocess
    import subprocess

    os.chdir(str(gui_dir))
    subprocess.run([sys.executable, "quantonium_os_unified.py"])


if __name__ == "__main__":
    main()

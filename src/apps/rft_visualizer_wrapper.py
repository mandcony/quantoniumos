#!/usr/bin/env python3
"""
RFT Visualizer Wrapper - Completely isolated launcher
This wrapper ensures the RFT visualizer runs independently from the main OS launcher
"""

import sys
import os
import subprocess
import time

def main():
    # Get the directory where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Change to the parent directory (quantoniumos root)
    root_dir = os.path.dirname(script_dir)
    os.chdir(root_dir)
    
    # Now rft_visualizer.py path is relative to the root directory
    rft_visualizer_path = "apps/rft_visualizer.py"
    
    # Launch the RFT visualizer with complete process isolation
    try:
        # Use CREATE_NEW_PROCESS_GROUP for Windows to ensure complete isolation
        if sys.platform == "win32":
            # Windows: Use shell=True and detach the process
            subprocess.Popen(
                [sys.executable, rft_visualizer_path],
                shell=False,
                creationflags=subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.DETACHED_PROCESS,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                stdin=subprocess.DEVNULL
            )
        else:
            # Unix-like systems
            subprocess.Popen(
                [sys.executable, rft_visualizer_path],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                stdin=subprocess.DEVNULL
            )
        
        print("🌊 RFT Visualizer launched successfully!")
        
    except Exception as e:
        print(f"❌ Failed to launch RFT Visualizer: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())

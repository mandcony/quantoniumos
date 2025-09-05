#!/usr/bin/env python3
"""
RFT Visualizer - QuantoniumOS
=============================
This is a bridge module that loads the actual implementation from launch_rft_visualizer.py
for compatibility with the rest of the system.
"""

import os
import sys

# Import the actual implementation from the launcher file
try:
    from launch_rft_visualizer import RFTVisualizer, main
    
    # Re-export the main classes
    __all__ = ['RFTVisualizer']
    
    if __name__ == "__main__":
        # If this file is run directly, start the launcher
        main()
except ImportError as e:
    print(f"Error importing RFT Visualizer: {e}")
    sys.exit(1)

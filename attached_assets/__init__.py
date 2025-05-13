"""
QuantoniumOS Desktop Applications Package

This package contains all the desktop applications for QuantoniumOS.
Desktop apps are launched from the OS interface and run in separate windows.
"""

import os
import sys

# Add parent directory to path to ensure imports work correctly
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)
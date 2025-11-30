#!/usr/bin/env python3
"""
QuantSoundDesign DAW Launcher

Professional Sound Design Studio with Φ-RFT Native Engine.
"""

import sys
import os

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, project_root)

# Now import and run
from src.apps.quantsounddesign.gui import main

if __name__ == "__main__":
    main()

"""
QuantoniumOS Test Suite

Unified test suite for QuantoniumOS with proper test discovery
and standardized test execution.

Test Structure:
- unit/: Unit tests for individual components  
- integration/: Integration tests for system interactions
- benchmarks/: Performance and benchmark tests

Usage:
    python -m pytest tests/
    python tests/run_all_tests.py
"""

import os
import sys
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

__version__ = "1.0.0"
__all__ = []

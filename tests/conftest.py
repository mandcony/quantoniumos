"""Pytest configuration: ensure project root is on sys.path for `import src`.

Windows terminals sometimes ignore temporary PYTHONPATH modifications in the same session.
This file makes the tests robust without requiring environment setup.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

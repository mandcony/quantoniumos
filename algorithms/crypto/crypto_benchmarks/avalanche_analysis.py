#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025 Luis M. Minier / quantoniumos
"""Legacy wrapper kept for backwards compatibility.

Delegates to `tools.crypto.avalanche_analyzer` so existing test pipelines that
invoke this script continue to work without modification.
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.crypto.avalanche_analyzer import main  # type: ignore


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Luis M. Minier / quantoniumos
"""
Fetch a small subset of Sleep-EDF for research testing (sleep EEG).

⚠️  RESEARCH USE ONLY — NOT FOR CLINICAL OR DIAGNOSTIC USE  ⚠️

This data and any software using it is provided strictly for research,
educational, and experimental purposes. Do NOT use for medical diagnosis,
treatment decisions, or clinical patient care.

Requirements:
- PhysioNet account; accept Sleep-EDF terms.
- USE_REAL_DATA=1 environment variable to enable.

Downloads a minimal set (e.g., two PSG/ Hypnogram pairs) to data/physionet/sleepedf_subset/.
"""

import os
import sys
import pathlib
import subprocess

BASE_URL = "https://physionet.org/files/sleep-edfx/1.0.0/sleep-cassette"
RECORDS = [
    "SC4001E0-PSG.edf",
    "SC4001EC-Hypnogram.edf",
    "SC4002E0-PSG.edf",
    "SC4002EC-Hypnogram.edf",
]
OUTPUT_DIR = pathlib.Path(__file__).resolve().parent / "physionet" / "sleepedf"


def ensure_env():
    if os.environ.get("USE_REAL_DATA") != "1":
        print("USE_REAL_DATA not set to 1; aborting fetch to avoid accidental downloads.")
        sys.exit(1)


def fetch_file(fname: str):
    dest = OUTPUT_DIR / fname
    if dest.exists():
        print(f"✓ Exists {dest}")
        return
    url = f"{BASE_URL}/{fname}"
    print(f"↓ Fetching {url}")
    res = subprocess.run(["curl", "-fsSL", url, "-o", str(dest)], capture_output=True, text=True)
    if res.returncode != 0:
        print(f"  Error: {res.stderr}")
        raise RuntimeError(f"Failed to fetch {url}")


def main():
    ensure_env()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for fname in RECORDS:
        fetch_file(fname)
    print("Done. Stored under", OUTPUT_DIR)


if __name__ == "__main__":
    main()

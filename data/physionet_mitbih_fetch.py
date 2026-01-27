#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Luis M. Minier / quantoniumos
"""
Fetch a small subset of MIT-BIH Arrhythmia Database for research testing.

⚠️  RESEARCH USE ONLY — NOT FOR CLINICAL OR DIAGNOSTIC USE  ⚠️

This data and any software using it is provided strictly for research,
educational, and experimental purposes. Do NOT use for medical diagnosis,
treatment decisions, or clinical patient care.

Requirements:
- PhysioNet account and credentialed cookies (~.netrc or environment)
- Data Use Agreement acceptance on PhysioNet

Behavior:
- Downloads a tiny subset (e.g., records 100 and 101 first 10 minutes) to avoid large footprints.
- Writes under data/physionet/mitbih_subset/
- Skips if files already present.

Usage:
    USE_REAL_DATA=1 python data/physionet_mitbih_fetch.py

Warning: MIT-BIH is research-only; verify downstream use complies with PhysioNet terms.
"""

import os
import sys
import pathlib
import subprocess

BASE_URL = "https://physionet.org/files/mitdb/1.0.0"
# Extended subset with different arrhythmia types:
# 100, 101: Normal sinus rhythm (some PVCs)
# 200: PVCs, VT episodes
# 207: Left bundle branch block, VT
# 208: Many arrhythmia types (PVCs, APCs, VT)
# 217: Ventricular bigeminy/trigeminy
# 219: Atrial fibrillation episodes
# 221: Atrial flutter
RECORDS = ["100", "101", "200", "207", "208", "217", "219", "221"]
OUTPUT_DIR = pathlib.Path(__file__).resolve().parent / "physionet" / "mitbih"


def ensure_env():
    if os.environ.get("USE_REAL_DATA") != "1":
        print("USE_REAL_DATA not set to 1; aborting fetch to avoid accidental downloads.")
        sys.exit(1)


def fetch_record(record: str):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    files = [f"{record}.dat", f"{record}.hea", f"{record}.atr"]
    for fname in files:
        url = f"{BASE_URL}/{fname}"
        dest = OUTPUT_DIR / fname
        if dest.exists():
            print(f"✓ Exists {dest}")
            continue
        print(f"↓ Fetching {url}")
        res = subprocess.run(["curl", "-fsSL", url, "-o", str(dest)], capture_output=True, text=True)
        if res.returncode != 0:
            # .atr may not exist for all records, skip it
            if fname.endswith(".atr"):
                print(f"  (annotation file not found, skipping)")
                continue
            print(res.stderr)
            raise RuntimeError(f"Failed to fetch {url}")


def main():
    ensure_env()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for rec in RECORDS:
        fetch_record(rec)
    print("Done. Stored under", OUTPUT_DIR)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
ULTRA FAST Cleanup - No Hanging Guaranteed
==========================================

This script will ACTUALLY finish and clean up your duplicates.
"""

import os
from datetime import datetime
from pathlib import Path


def ultra_fast_cleanup():
    root = Path("C:\\quantoniumos-1")
    attic = root / "09_LEGACY_BACKUPS" / f"_attic_{datetime.now().strftime('%Y%m%d')}"

    print(f"[START] Ultra fast cleanup")
    print(f"[ROOT] {root}")

    # Create attic
    attic.mkdir(parents=True, exist_ok=True)
    print(f"[ATTIC] Created {attic}")

    moved_count = 0

    # Simple patterns to move
    patterns_to_move = [
        "*.backup",
        "*_backup.*",
        "*_old.*",
        "*_fixed.*",
        "*_copy.*",
        "*_temp.*",
        "*.tmp",
    ]

    print("\n=== MOVING BACKUP FILES ===")
    for pattern in patterns_to_move:
        for file_path in root.glob(pattern):
            if file_path.is_file() and "09_LEGACY_BACKUPS" not in str(file_path):
                try:
                    target = attic / file_path.name
                    file_path.rename(target)
                    print(f"[MOVED] {file_path.name}")
                    moved_count += 1
                except Exception as e:
                    print(f"[ERROR] {file_path.name}: {e}")

    print("\n=== MOVING EMPTY FILES ===")
    empty_count = 0
    for file_path in root.iterdir():
        if file_path.is_file():
            try:
                if file_path.stat().st_size == 0:
                    target = attic / f"empty_{file_path.name}"
                    file_path.rename(target)
                    print(f"[MOVED] Empty file: {file_path.name}")
                    empty_count += 1
                    moved_count += 1
            except Exception as e:
                print(f"[ERROR] {file_path.name}: {e}")

            # Limit to prevent hanging
            if empty_count > 50:
                print("[LIMIT] Stopped at 50 empty files")
                break

    print(f"\n=== CLEANUP COMPLETE ===")
    print(f"Total files moved: {moved_count}")
    print(f"Moved to: {attic}")

    return moved_count


if __name__ == "__main__":
    ultra_fast_cleanup()

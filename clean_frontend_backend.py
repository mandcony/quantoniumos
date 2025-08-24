#!/usr/bin/env python3
"""
Frontend and Backend Cleanup Script for QuantoniumOS

This script specifically targets cleanup in the frontend and backend areas of the codebase:
- Removes duplicate or obsolete components
- Cleans up unnecessary UI artifacts
- Removes temporary or debug code
- Consolidates similar functionality

Usage:
    python clean_frontend_backend.py [--dry-run] [--verbose]

Options:
    --dry-run   Show what would be deleted without actually deleting
    --verbose   Show detailed information about each file processed
"""

import argparse
import hashlib
import json
import os
import re
import shutil
import sys
from pathlib import Path

# Define frontend and backend directories
FRONTEND_DIRS = [
    "frontend",
    "wave_ui",
    "web",
    "11_QUANTONIUMOS/gui",
    "apps",
]

BACKEND_DIRS = [
    "core",
    "src",
    "phase3",
    "phase4",
    "03_RUNNING_SYSTEMS",
    "04_RFT_ALGORITHMS",
    "05_QUANTUM_ENGINES",
    "06_CRYPTOGRAPHY",
]

# Files that should be kept despite matching removal patterns
ESSENTIAL_FILES = [
    # Add any critical files here that should never be removed
    "apps/__init__.py",
    "core/__init__.py",
    "frontend/__init__.py",
]


def compute_file_hash(filepath):
    """Compute hash of file contents"""
    hasher = hashlib.md5()
    with open(filepath, "rb") as f:
        buf = f.read()
        hasher.update(buf)
    return hasher.hexdigest()


def find_duplicate_files(directory):
    """Find duplicate files based on content hash"""
    files_by_hash = {}
    duplicates = []

    for root, _, files in os.walk(directory):
        for filename in files:
            filepath = os.path.join(root, filename)
            if os.path.isfile(filepath):
                try:
                    file_hash = compute_file_hash(filepath)
                    if file_hash in files_by_hash:
                        # Prefer to keep files in standard locations vs. backup/temp locations
                        existing_file = files_by_hash[file_hash]
                        if any(
                            pattern in filepath.lower()
                            for pattern in ["backup", "temp", "old", "legacy"]
                        ):
                            duplicates.append(filepath)
                        elif any(
                            pattern in existing_file.lower()
                            for pattern in ["backup", "temp", "old", "legacy"]
                        ):
                            duplicates.append(existing_file)
                            files_by_hash[file_hash] = filepath
                        else:
                            # If neither has obvious backup markers, keep the shorter path
                            if len(filepath) > len(existing_file):
                                duplicates.append(filepath)
                            else:
                                duplicates.append(existing_file)
                                files_by_hash[file_hash] = filepath
                    else:
                        files_by_hash[file_hash] = filepath
                except (IOError, OSError):
                    continue

    return duplicates


def find_debug_files(directory):
    """Find files that appear to be debugging or testing related"""
    debug_patterns = [
        r"debug",
        r"test_",
        r"_test",
        r"mock",
        r"stub",
        r"dummy",
        r"temp",
        r"tmp",
        r"experimental",
        r"playground",
        r"sandbox",
        r"scratch",
        r"wip",
        r"draft",
    ]
    debug_files = []

    for root, _, files in os.walk(directory):
        for filename in files:
            filepath = os.path.join(root, filename)
            lower_path = filepath.lower()

            # Skip essential files
            if any(essential in filepath for essential in ESSENTIAL_FILES):
                continue

            # Look for debug patterns in filename
            if any(re.search(pattern, filename.lower()) for pattern in debug_patterns):
                debug_files.append(filepath)
                continue

            # Look for debug code inside Python files
            if filename.endswith(".py"):
                try:
                    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
                        content = f.read().lower()
                        # Check for obvious debug markers
                        if (
                            "# debug" in content
                            or 'print("debug' in content
                            or 'console.log("debug' in content
                            or "debug only" in content
                        ):
                            debug_files.append(filepath)
                except (IOError, OSError):
                    continue

    return debug_files


def find_obsolete_files(directory):
    """Find files that appear to be obsolete or deprecated"""
    obsolete_patterns = [
        r"deprecated",
        r"obsolete",
        r"old",
        r"legacy",
        r"archive",
        r"backup",
        r"superseded",
        r"replaced",
        r"unused",
        r"_v\d+",
        r"\.old\.",
    ]
    obsolete_files = []

    for root, _, files in os.walk(directory):
        for filename in files:
            filepath = os.path.join(root, filename)

            # Skip essential files
            if any(essential in filepath for essential in ESSENTIAL_FILES):
                continue

            # Look for obsolete patterns in filename or path
            if any(
                re.search(pattern, filepath.lower()) for pattern in obsolete_patterns
            ):
                obsolete_files.append(filepath)

    return obsolete_files


def is_essential(filepath):
    """Check if a file is essential and should not be removed"""
    for essential in ESSENTIAL_FILES:
        if essential in filepath:
            return True
    return False


def main():
    parser = argparse.ArgumentParser(
        description="Clean frontend and backend code in QuantoniumOS"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be deleted without actually deleting",
    )
    parser.add_argument(
        "--verbose", action="store_true", help="Show detailed information"
    )
    args = parser.parse_args()

    root_dir = Path(os.path.dirname(os.path.abspath(__file__)))

    # Stats for reporting
    stats = {
        "duplicate_files": 0,
        "debug_files": 0,
        "obsolete_files": 0,
        "bytes_freed": 0,
    }

    # Process frontend directories
    print("Analyzing frontend directories...")
    frontend_paths = [
        root_dir / dir_name
        for dir_name in FRONTEND_DIRS
        if (root_dir / dir_name).exists()
    ]

    # Process backend directories
    print("Analyzing backend directories...")
    backend_paths = [
        root_dir / dir_name
        for dir_name in BACKEND_DIRS
        if (root_dir / dir_name).exists()
    ]

    all_paths = frontend_paths + backend_paths

    # Find duplicate files
    print("Finding duplicate files...")
    all_duplicates = []
    for path in all_paths:
        if args.verbose:
            print(f"Checking for duplicates in {path}")
        duplicates = find_duplicate_files(path)
        all_duplicates.extend(duplicates)

    # Find debug files
    print("Finding debug files...")
    all_debug_files = []
    for path in all_paths:
        if args.verbose:
            print(f"Checking for debug files in {path}")
        debug_files = find_debug_files(path)
        all_debug_files.extend(debug_files)

    # Find obsolete files
    print("Finding obsolete files...")
    all_obsolete_files = []
    for path in all_paths:
        if args.verbose:
            print(f"Checking for obsolete files in {path}")
        obsolete_files = find_obsolete_files(path)
        all_obsolete_files.extend(obsolete_files)

    # Remove duplicate files
    if all_duplicates:
        print(f"\nFound {len(all_duplicates)} duplicate files:")
        for file_path in all_duplicates:
            if is_essential(file_path):
                if args.verbose:
                    print(f"Skipping essential file: {file_path}")
                continue

            if args.dry_run:
                print(f"Would remove duplicate: {file_path}")
            else:
                if args.verbose:
                    print(f"Removing duplicate: {file_path}")
                try:
                    file_size = os.path.getsize(file_path)
                    stats["bytes_freed"] += file_size
                    os.remove(file_path)
                    stats["duplicate_files"] += 1
                except Exception as e:
                    print(f"Error removing {file_path}: {e}")

    # Remove debug files
    if all_debug_files:
        print(f"\nFound {len(all_debug_files)} debug files:")
        for file_path in all_debug_files:
            if is_essential(file_path):
                if args.verbose:
                    print(f"Skipping essential file: {file_path}")
                continue

            if args.dry_run:
                print(f"Would remove debug file: {file_path}")
            else:
                if args.verbose:
                    print(f"Removing debug file: {file_path}")
                try:
                    file_size = os.path.getsize(file_path)
                    stats["bytes_freed"] += file_size
                    os.remove(file_path)
                    stats["debug_files"] += 1
                except Exception as e:
                    print(f"Error removing {file_path}: {e}")

    # Remove obsolete files
    if all_obsolete_files:
        print(f"\nFound {len(all_obsolete_files)} obsolete files:")
        for file_path in all_obsolete_files:
            if is_essential(file_path):
                if args.verbose:
                    print(f"Skipping essential file: {file_path}")
                continue

            if args.dry_run:
                print(f"Would remove obsolete file: {file_path}")
            else:
                if args.verbose:
                    print(f"Removing obsolete file: {file_path}")
                try:
                    file_size = os.path.getsize(file_path)
                    stats["bytes_freed"] += file_size
                    os.remove(file_path)
                    stats["obsolete_files"] += 1
                except Exception as e:
                    print(f"Error removing {file_path}: {e}")

    # Print summary
    print("\nCleanup Summary:")
    if args.dry_run:
        print("DRY RUN - No files were actually deleted")
    print(f"Duplicate files: {stats['duplicate_files']}")
    print(f"Debug files: {stats['debug_files']}")
    print(f"Obsolete files: {stats['obsolete_files']}")
    print(
        f"Total files that would be removed: {stats['duplicate_files'] + stats['debug_files'] + stats['obsolete_files']}"
    )
    print(
        f"Total space that would be freed: {stats['bytes_freed'] / (1024*1024):.2f} MB"
    )

    # Create report of files to remove
    report = {
        "duplicate_files": all_duplicates,
        "debug_files": all_debug_files,
        "obsolete_files": all_obsolete_files,
        "stats": {
            "duplicate_count": len(all_duplicates),
            "debug_count": len(all_debug_files),
            "obsolete_count": len(all_obsolete_files),
            "total_count": len(all_duplicates)
            + len(all_debug_files)
            + len(all_obsolete_files),
        },
    }

    # Write report to file
    with open("frontend_backend_cleanup_report.json", "w") as f:
        json.dump(report, f, indent=2)

    print("\nDetailed report saved to frontend_backend_cleanup_report.json")


if __name__ == "__main__":
    main()

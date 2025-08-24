#!/usr/bin/env python3
"""
Clean Unnecessary Files Script for QuantoniumOS

This script identifies and removes unnecessary files like:
- Python bytecode files (.pyc)
- Temporary and backup files
- Cache directories
- Old test artifacts
- Legacy backups
- Empty directories

Usage:
    python clean_unnecessary_files.py [--dry-run] [--verbose]

Options:
    --dry-run   Show what would be deleted without actually deleting
    --verbose   Show detailed information about each file processed
"""

import argparse
import os
import re
import shutil
import sys
from pathlib import Path

# Files and directories to keep (even if they match removal patterns)
ESSENTIAL_KEEP = [
    # Core system files
    r"core[\\/]python[\\/]engines[\\/]__init__.py",
    r"src[\\/]quantoniumos[\\/]__init__.py",
]

# File patterns to remove
REMOVE_PATTERNS = [
    r"\.pyc$",  # Python bytecode
    r"\.pyo$",  # Optimized bytecode
    r"\.pyd$",  # Python DLL files
    r"\.o$",  # Object files
    r"\.obj$",  # Object files
    r"\.so$",  # Shared object files
    r"\.bak$",  # Backup files
    r"\.backup$",  # Backup files
    r"\.tmp$",  # Temporary files
    r"\.temp$",  # Temporary files
    r"\.old$",  # Old files
    r".*_OLD.*",  # Files with OLD in name
    r".*_BACKUP.*",  # Files with BACKUP in name
    r".*\.py\.legacy_backup$",  # Legacy Python backups
]

# Directory patterns to remove
REMOVE_DIR_PATTERNS = [
    r"__pycache__$",  # Python cache
    r"\.cache$",  # Generic cache
    r"cache$",  # Cache dirs
    r"temp$",  # Temp dirs
    r"tmp$",  # Tmp dirs
]

# Directories to clean completely
CLEAN_DIRS = [
    "09_LEGACY_BACKUPS",
    "17_BUILD_ARTIFACTS/cache",
    "12_TEST_RESULTS",
]


def is_essential(path):
    """Check if a file is in the essential keep list"""
    path_str = str(path)
    for pattern in ESSENTIAL_KEEP:
        if re.search(pattern, path_str, re.IGNORECASE):
            return True
    return False


def should_remove_file(path):
    """Check if a file should be removed"""
    if is_essential(path):
        return False

    path_str = str(path)
    for pattern in REMOVE_PATTERNS:
        if re.search(pattern, path_str, re.IGNORECASE):
            return True
    return False


def should_remove_dir(path):
    """Check if a directory should be removed"""
    path_str = str(path)
    for pattern in REMOVE_DIR_PATTERNS:
        if re.search(pattern, path_str, re.IGNORECASE):
            return True
    return False


def clean_directory(directory, dry_run=False, verbose=False):
    """Completely clean a directory but keep the directory itself"""
    if not os.path.exists(directory):
        if verbose:
            print(f"Directory does not exist: {directory}")
        return

    for item in os.listdir(directory):
        item_path = os.path.join(directory, item)
        if os.path.isdir(item_path):
            if dry_run:
                print(f"Would remove directory: {item_path}")
            else:
                if verbose:
                    print(f"Removing directory: {item_path}")
                shutil.rmtree(item_path, ignore_errors=True)
        else:
            if dry_run:
                print(f"Would remove file: {item_path}")
            else:
                if verbose:
                    print(f"Removing file: {item_path}")
                os.remove(item_path)


def main():
    parser = argparse.ArgumentParser(
        description="Clean unnecessary files from QuantoniumOS"
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
        "files_removed": 0,
        "dirs_removed": 0,
        "bytes_freed": 0,
    }

    # First, completely clean specific directories
    for clean_dir in CLEAN_DIRS:
        dir_path = root_dir / clean_dir
        if args.verbose:
            print(f"Cleaning directory: {dir_path}")
        clean_directory(dir_path, args.dry_run, args.verbose)

    # Then remove specific file and directory patterns
    for root, dirs, files in os.walk(root_dir, topdown=True):
        # Filter out directories to skip (don't process their contents)
        dirs_to_remove = []
        for i, d in enumerate(dirs):
            dir_path = os.path.join(root, d)

            # Skip .git, .venv, and third_party directories
            if d in [".git", ".venv", ".venv-1"] or root.endswith("third_party"):
                dirs_to_remove.append(i)
                continue

            if should_remove_dir(dir_path):
                if args.dry_run:
                    print(f"Would remove directory: {dir_path}")
                else:
                    if args.verbose:
                        print(f"Removing directory: {dir_path}")
                    try:
                        dir_size = sum(
                            os.path.getsize(os.path.join(dirpath, filename))
                            for dirpath, _, filenames in os.walk(dir_path)
                            for filename in filenames
                        )
                        stats["bytes_freed"] += dir_size
                        shutil.rmtree(dir_path, ignore_errors=True)
                        stats["dirs_removed"] += 1
                    except Exception as e:
                        print(f"Error removing {dir_path}: {e}")

                dirs_to_remove.append(i)

        # Remove directories in reverse order to avoid index issues
        for i in sorted(dirs_to_remove, reverse=True):
            del dirs[i]

        # Process files
        for file in files:
            file_path = os.path.join(root, file)
            if should_remove_file(file_path):
                if args.dry_run:
                    print(f"Would remove file: {file_path}")
                else:
                    if args.verbose:
                        print(f"Removing file: {file_path}")
                    try:
                        file_size = os.path.getsize(file_path)
                        stats["bytes_freed"] += file_size
                        os.remove(file_path)
                        stats["files_removed"] += 1
                    except Exception as e:
                        print(f"Error removing {file_path}: {e}")

    # Print summary
    print("\nCleanup Summary:")
    if args.dry_run:
        print("DRY RUN - No files were actually deleted")
    print(f"Files that would be removed: {stats['files_removed']}")
    print(f"Directories that would be removed: {stats['dirs_removed']}")
    print(
        f"Total space that would be freed: {stats['bytes_freed'] / (1024*1024):.2f} MB"
    )


if __name__ == "__main__":
    main()

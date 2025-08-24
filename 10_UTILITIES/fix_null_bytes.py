#!/usr/bin/env python3
"""
Fix files with null bytes in the 02_CORE_VALIDATORS directory
"""

import os
import sys
from pathlib import Path


def fix_null_bytes(file_path):
    """Remove null bytes from a file"""
    try:
        # Read the file in binary mode
        with open(file_path, "rb") as f:
            content = f.read()

        # Remove null bytes
        fixed_content = content.replace(b"\x00", b"")

        # Write the fixed content back
        with open(file_path, "wb") as f:
            f.write(fixed_content)

        print(f"Fixed: {file_path}")
        return True
    except Exception as e:
        print(f"Error fixing {file_path}: {e}")
        return False


def main():
    """Fix all Python files in 02_CORE_VALIDATORS"""
    validators_dir = Path(__file__).parent / "02_CORE_VALIDATORS"

    if not validators_dir.exists():
        print(f"Error: Directory {validators_dir} does not exist")
        return False

    fixed_count = 0
    error_count = 0

    for file_path in validators_dir.glob("*.py"):
        if fix_null_bytes(file_path):
            fixed_count += 1
        else:
            error_count += 1

    print(f"\nFixed {fixed_count} files")
    print(f"Errors: {error_count}")

    return error_count == 0


if __name__ == "__main__":
    sys.exit(0 if main() else 1)

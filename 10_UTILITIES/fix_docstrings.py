#!/usr/bin/env python3
"""
Fix broken docstrings in Python files.
This script finds and fixes docstrings that are malformed with """text"""
"""

import os
import re
import sys
from pathlib import Path


def fix_broken_docstrings(file_path):
    """Fix broken docstrings in a Python file."""
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read()
    
    # Find all instances of broken docstrings
    pattern = r'""""""([^"]*?)""""""'
    fixed_content = re.sub(pattern, r'"""\1"""', content)
    
    if fixed_content != content:
        print(f"Fixing docstrings in {file_path}")
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(fixed_content)
        return True
    return False

def process_directory(directory):
    """Process all Python files in a directory recursively."""
    fixed_count = 0
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.py'):
                file_path = os.path.join(root, file)
                if fix_broken_docstrings(file_path):
                    fixed_count += 1
    
    return fixed_count

if __name__ == "__main__":
    directory = sys.argv[1] if len(sys.argv) > 1 else '.'
    fixed_count = process_directory(directory)
    print(f"Fixed docstrings in {fixed_count} files.")

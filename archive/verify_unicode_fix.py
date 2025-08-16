#!/usr/bin/env python3
"""
Unicode Corruption Verification Script
======================================

This script verifies that all unicode corruption has been fixed across the workspace.
"""

import os
import re
from pathlib import Path

def check_corruption_patterns(content):
    """Check for various corruption patterns."""
    patterns = [
        r'\|\|\|\|',  # The main corruption pattern
        r'[^\x00-\x7F]{4,}',  # Long sequences of non-ASCII characters
    ]
    
    issues = []
    for pattern in patterns:
        matches = re.findall(pattern, content)
        if matches:
            issues.append(f"Found pattern '{pattern}': {len(matches)} occurrences")
    
    return issues

def verify_file(file_path):
    """Verify a single file for corruption."""
    try:
        with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
            content = f.read()
        
        issues = check_corruption_patterns(content)
        return issues
    
    except Exception as e:
        return [f"Error reading file: {e}"]

def verify_workspace(workspace_dir):
    """Verify all files in the workspace."""
    workspace_path = Path(workspace_dir)
    
    # File extensions to check
    extensions = {'.py', '.md', '.txt', '.json', '.yaml', '.yml', '.js', '.ts', '.cpp', '.h', '.hpp', '.c'}
    
    corrupted_files = []
    total_files = 0
    
    print("Verifying workspace for unicode corruption...")
    print("=" * 60)
    
    for file_path in workspace_path.rglob('*'):
        if file_path.is_file() and file_path.suffix in extensions:
            total_files += 1
            
            issues = verify_file(file_path)
            
            if issues:
                corrupted_files.append((str(file_path), issues))
                print(f"⚠️  Issues in: {file_path.name}")
                for issue in issues:
                    print(f"    - {issue}")
    
    print("=" * 60)
    print(f"Verification complete!")
    print(f"Total files checked: {total_files}")
    print(f"Files with issues: {len(corrupted_files)}")
    
    if not corrupted_files:
        print("✅ No unicode corruption found! All files are clean.")
    else:
        print("\n⚠️  Files still with issues:")
        for file_path, issues in corrupted_files:
            print(f"  - {file_path}")
            for issue in issues:
                print(f"    {issue}")

def main():
    """Main function."""
    workspace_dir = Path(__file__).parent
    verify_workspace(workspace_dir)

if __name__ == "__main__":
    main()

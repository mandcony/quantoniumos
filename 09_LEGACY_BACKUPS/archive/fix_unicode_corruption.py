#!/usr/bin/env python3
"""
Unicode Corruption Fix Script === This script systematically fixes unicode corruption across all files in the workspace. The corruption pattern appears to be characters separated by .
"""

import os
import re from pathlib
import Path
import chardet
import codecs
def detect_encoding(file_path): """
        Detect the encoding of a file.
"""

        try: with open(file_path, 'rb') as f: raw_data = f.read() result = chardet.detect(raw_data)
        return result['encoding']
        if result['encoding'] else 'utf-8'
        except:
        return 'utf-8'
def fix_unicode_corruption(content): """
        Fix unicode corruption patterns in content.
"""

        # Pattern 1: Characters separated by content = re.sub(r'\|\|\|\|(.)\|\|\|\|', r'\1', content)

        # Pattern 2: Multiple consecutive content = re.sub(r'\|\|\|\|+', '', content)

        # Pattern 3: Leading/trailing content = re.sub(r'^\|\|\|\|', '', content, flags=re.MULTILINE) content = re.sub(r'\|\|\|\|$', '', content, flags=re.MULTILINE)

        # Pattern 4: Fix specific corruption patterns content = content.replace('', '')

        # Pattern 5: Fix broken string patterns content = re.sub(r'(\w)\|\|\|\|(\w)', r'\1\2', content)
        return content
def is_corrupted(content): """
        Check
        if content appears to be corrupted.
"""

        return '' in content
def fix_file(file_path): """
        Fix unicode corruption in a single file.
"""

        try:

        # Detect encoding encoding = detect_encoding(file_path)

        # Read file content with open(file_path, 'r', encoding=encoding, errors='replace') as f: content = f.read()

        # Check
        if file is corrupted
        if not is_corrupted(content):
        return False, "No corruption detected"

        # Fix corruption fixed_content = fix_unicode_corruption(content)

        # Write back fixed content with open(file_path, 'w', encoding='utf-8') as f: f.write(fixed_content)
        return True, "Fixed successfully" except Exception as e:
        return False, f"Error: {e}"
def scan_and_fix_workspace(workspace_dir): """
        Scan and fix all files in the workspace.
"""
        workspace_path = Path(workspace_dir)

        # File extensions to check extensions = {'.py', '.md', '.txt', '.json', '.yaml', '.yml', '.js', '.ts', '.cpp', '.h', '.hpp', '.c'} fixed_files = [] error_files = [] total_files = 0
        print("Scanning workspace for unicode corruption...")
        print("=" * 60)
        for file_path in workspace_path.rglob('*'):
        if file_path.is_file() and file_path.suffix in extensions: total_files += 1 success, message = fix_file(file_path)
        if success: fixed_files.append(str(file_path))
        print(f"✓ Fixed: {file_path.name}")
        elif "Error:" in message: error_files.append((str(file_path), message))
        print(f"✗ Error: {file_path.name} - {message}")
        else:

        # No corruption detected - silent pass
        print("=" * 60)
        print(f"Scan complete!")
        print(f"Total files scanned: {total_files}")
        print(f"Files fixed: {len(fixed_files)}")
        print(f"Files with errors: {len(error_files)}")
        if fixed_files:
        print("\nFixed files:")
        for file_path in fixed_files:
        print(f" - {file_path}")
        if error_files:
        print("\nFiles with errors:") for file_path, error in error_files:
        print(f" - {file_path}: {error}")
def main(): """
        Main function.
"""
        workspace_dir = Path(__file__).parent scan_and_fix_workspace(workspace_dir)

if __name__ == "__main__": main()
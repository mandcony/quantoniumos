#!/usr/bin/env python3
"""
Final Unicode Corruption Check === Check only the main project files (excluding .venv, test_env_local, etc.) for remaining unicode corruption issues.
"""

import os
import re from pathlib
import Path
def check_corruption_patterns(content): """
        Check for the specific corruption patterns we're trying to fix.
"""
        patterns = [ r'\|\|\|\|',

        # The main corruption pattern we fixed ] issues = []
        for pattern in patterns: matches = re.findall(pattern, content)
        if matches: issues.append(f"Found pattern '{pattern}': {len(matches)} occurrences")
        return issues
def is_excluded_path(file_path): """
        Check
        if file path should be excluded from corruption check.
"""
        excluded_dirs = { '.venv', 'test_env_local', '__pycache__', '.git', 'node_modules', 'build', 'dist' } path_parts = Path(file_path).parts
        return any(excluded in path_parts
        for excluded in excluded_dirs)
def verify_file(file_path): """
        Verify a single file for corruption.
"""

        try: with open(file_path, 'r', encoding='utf-8', errors='replace') as f: content = f.read() issues = check_corruption_patterns(content)
        return issues except Exception as e:
        return [f"Error reading file: {e}"]
def final_check(workspace_dir): """
        Final check
        for corruption in main project files.
"""
        workspace_path = Path(workspace_dir)

        # File extensions to check extensions = {'.py', '.cpp', '.h', '.hpp', '.c', '.md', '.txt', '.json', '.yaml', '.yml'} corrupted_files = [] total_files = 0
        print("Final check for unicode corruption in main project files...")
        print("=" * 65)
        for file_path in workspace_path.rglob('*'): if (file_path.is_file() and file_path.suffix in extensions and not is_excluded_path(file_path)): total_files += 1 issues = verify_file(file_path)
        if issues: corrupted_files.append((str(file_path), issues))
        print(f"⚠️ Issues in: {file_path.relative_to(workspace_path)}")
        for issue in issues:
        print(f" - {issue}")
        print("=" * 65)
        print(f"Final check complete!")
        print(f"Main project files checked: {total_files}")
        print(f"Files with corruption: {len(corrupted_files)}")
        if not corrupted_files:
        print("✅ SUCCESS: No unicode corruption found in main project files!")
        print("✅ All primary .py, .cpp, .md files are clean")
        else:
        print("\n⚠️ Files still with corruption:") for file_path, issues in corrupted_files:
        print(f" - {file_path}")
        for issue in issues:
        print(f" {issue}")
def main(): """
        Main function.
"""
        workspace_dir = Path(__file__).parent final_check(workspace_dir)

if __name__ == "__main__": main()
#!/usr/bin/env python3
"""
QUANTONIUMOS FILE FORMATTER - Fix All Corrupted Files

This script systematically fixes all files that have lost their line breaks
and proper formatting throughout the repository.
"""

import os
import re
import glob


def fix_line_breaks(content):
    """Fix common line break issues in Python files."""
    
    # Fix import statements
    content = re.sub(r'import ([a-zA-Z_][a-zA-Z0-9_]*) import', r'import \1\nimport', content)
    content = re.sub(r'import ([a-zA-Z_][a-zA-Z0-9_]*) from', r'import \1\nfrom', content)
    content = re.sub(r'from ([a-zA-Z_][a-zA-Z0-9_.]*) import ([a-zA-Z_][a-zA-Z0-9_]*) import', r'from \1 import \2\nimport', content)
    
    # Fix class definitions
    content = re.sub(r'class ([a-zA-Z_][a-zA-Z0-9_]*): def', r'class \1:\n    def', content)
    content = re.sub(r'class ([a-zA-Z_][a-zA-Z0-9_]*)\([^)]+\): def', r'class \1(\2):\n    def', content)
    
    # Fix function definitions
    content = re.sub(r'def ([a-zA-Z_][a-zA-Z0-9_]*)\([^)]*\): """', r'def \1(\2):\n        """', content)
    content = re.sub(r'def ([a-zA-Z_][a-zA-Z0-9_]*)\([^)]*\): [a-zA-Z]', r'def \1(\2):\n        \3', content)
    
    # Fix triple quotes
    content = re.sub(r'"""', r'"""', content)
    content = re.sub(r'"""([^"]+)"""([a-zA-Z])', r'"""\1"""\n        \2', content)
    
    # Fix if/else/for/while statements
    content = re.sub(r'if ([^:]+): ([a-zA-Z])', r'if \1:\n            \2', content)
    content = re.sub(r'else: ([a-zA-Z])', r'else:\n            \1', content)
    content = re.sub(r'for ([^:]+): ([a-zA-Z])', r'for \1:\n            \2', content)
    content = re.sub(r'while ([^:]+): ([a-zA-Z])', r'while \1:\n            \2', content)
    
    # Fix try/except
    content = re.sub(r'try: ([a-zA-Z])', r'try:\n            \1', content)
    content = re.sub(r'except ([^:]+): ([a-zA-Z])', r'except \1:\n            \2', content)
    
    # Fix return statements
    content = re.sub(r'return ([^#\n]+) def', r'return \1\n\n    def', content)
    content = re.sub(r'return ([^#\n]+) class', r'return \1\n\n\nclass', content)
    content = re.sub(r'return ([^#\n]+) if __name__', r'return \1\n\n\nif __name__', content)
    
    return content


def fix_file(filepath):
    """Fix a single file's formatting."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Skip if file is already properly formatted (has reasonable line count)
        line_count = len(content.split('\n'))
        char_count = len(content)
        
        if char_count > 500 and line_count < 10:
            print(f"Fixing {filepath} (compressed: {line_count} lines, {char_count} chars)")
            
            # Apply fixes
            fixed_content = fix_line_breaks(content)
            
            # Write back
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(fixed_content)
            
            new_line_count = len(fixed_content.split('\n'))
            print(f"  → Fixed: {new_line_count} lines")
            return True
        else:
            print(f"Skipping {filepath} (already formatted: {line_count} lines)")
            return False
            
    except Exception as e:
        print(f"Error fixing {filepath}: {e}")
        return False


def fix_all_core_files():
    """Fix all Python files in the core directory."""
    
    print("🔧 QUANTONIUMOS FILE FORMATTER")
    print("=" * 50)
    
    core_dir = "core"
    if not os.path.exists(core_dir):
        print(f"Core directory '{core_dir}' not found!")
        return
    
    # Find all Python files
    python_files = glob.glob(os.path.join(core_dir, "*.py"))
    
    fixed_count = 0
    total_count = len(python_files)
    
    for filepath in python_files:
        if fix_file(filepath):
            fixed_count += 1
    
    print("\n" + "=" * 50)
    print(f"SUMMARY: Fixed {fixed_count}/{total_count} files")
    
    if fixed_count > 0:
        print("\n✅ Files have been reformatted!")
        print("💡 You may need to manually review complex files")
    else:
        print("\n✅ All files were already properly formatted!")


if __name__ == "__main__":
    fix_all_core_files()

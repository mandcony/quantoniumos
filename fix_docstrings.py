#!/usr/bin/env python3
"""
Fix malformed docstrings with 6 quotes instead of 3
"""

import os
import re
import glob

def fix_docstrings_in_file(filepath):
    """Fix malformed docstrings in a single file."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Track changes
        original_content = content
        
        # Fix opening docstrings: """" -> """
        content = re.sub(r'"""', '"""', content)
        
        if content != original_content:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(content)
            print(f"✅ Fixed docstrings in: {filepath}")
            return True
        else:
            return False
            
    except Exception as e:
        print(f"❌ Error processing {filepath}: {e}")
        return False

def main():
    """Fix all Python files with malformed docstrings."""
    
    # Find all Python files
    python_files = []
    for root, dirs, files in os.walk('.'):
        # Skip certain directories
        if any(skip in root for skip in ['__pycache__', '.git', 'node_modules', 'venv', 'env']):
            continue
            
        for file in files:
            if file.endswith('.py'):
                python_files.append(os.path.join(root, file))
    
    print(f"🔍 Found {len(python_files)} Python files to check...")
    
    fixed_count = 0
    for filepath in python_files:
        if fix_docstrings_in_file(filepath):
            fixed_count += 1
    
    print(f"\n✅ Fixed docstrings in {fixed_count} files")

if __name__ == "__main__":
    main()

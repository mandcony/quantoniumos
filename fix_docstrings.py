#!/usr/bin/env python3
"""
Quick script to fix malformed docstrings with 6+ quotes
"""

import re
import sys

def fix_docstrings(file_path):
    """Fix malformed docstrings in a Python file"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Fix docstrings with 6+ quotes at start and end
        # Pattern: """"""text""""""  ->  """text"""
        pattern = r'"{6,}([^"]*?)"{6,}'
        fixed_content = re.sub(pattern, r'"""\1"""', content)
        
        # Fix docstrings with 6+ quotes at start only
        # Pattern: """"""text"""  ->  """text"""
        pattern2 = r'"{6,}([^"]*?)"{3}'
        fixed_content = re.sub(pattern2, r'"""\1"""', fixed_content)
        
        # Fix docstrings with 6+ quotes at end only
        # Pattern: """text""""""  ->  """text"""
        pattern3 = r'"{3}([^"]*?)"{6,}'
        fixed_content = re.sub(pattern3, r'"""\1"""', fixed_content)
        
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(fixed_content)
        
        print(f"✅ Fixed docstrings in {file_path}")
        return True
        
    except Exception as e:
        print(f"❌ Error fixing {file_path}: {e}")
        return False

if __name__ == "__main__":
    file_path = sys.argv[1] if len(sys.argv) > 1 else "core/encryption/geometric_waveform_hash.py"
    fix_docstrings(file_path)

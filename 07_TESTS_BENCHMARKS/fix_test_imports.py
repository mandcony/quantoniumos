#!/usr/bin/env python3
"""
Automated script to fix import errors in test files

This script scans Python files in the specified directory and fixes common import errors:
1. Invalid imports like 'import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), '05_QUANTUM_ENGINES'))
from xyz import abc'
2. Missing module imports
3. Other common import patterns that need fixing
"""

import os
import re
import sys
from pathlib import Path


def fix_invalid_decimal_import(file_path):
    """Fix imports like 'import importlib.util
import os

# Load the xyz module
spec = importlib.util.spec_from_file_location(
    "xyz", 
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                "05_QUANTUM_ENGINES/xyz.py")
)
xyz = importlib.util.module_from_spec(spec)
spec.loader.exec_module(xyz)

# Import specific functions/classes
abc = xyz.abc'"""
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read()
    
    # Look for import patterns like 'import importlib.util
import os

# Load the xyz module
spec = importlib.util.spec_from_file_location(
    "xyz", 
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                "05_QUANTUM_ENGINES/xyz.py")
)
xyz = importlib.util.module_from_spec(spec)
spec.loader.exec_module(xyz)

# Import specific functions/classes
abc = xyz.abc'
    pattern = r'from\s+(\d+)_([A-Z_]+)\.([a-zA-Z0-9_]+)\s+import\s+([a-zA-Z0-9_,\s]+)'
    matches = re.findall(pattern, content)
    
    if matches:
        print(f"Found {len(matches)} invalid imports in {file_path}")
        
        for match in matches:
            folder_num, folder_name, module_name, imports = match
            old_import = f"from {folder_num}_{folder_name}.{module_name}
import {imports}"
            
            # Create new import using importlib
            new_import = f"""import importlib.util
import os

# Load the {module_name} module
spec = importlib.util.spec_from_file_location(
    "{module_name}", 
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                "{folder_num}_{folder_name}/{module_name}.py")
)
{module_name} = importlib.util.module_from_spec(spec)
spec.loader.exec_module({module_name})

# Import specific functions/classes
{', '.join(imp.strip() for imp in imports.split(','))} = {', '.join(f'{module_name}.{imp.strip()}' for imp in imports.split(','))}"""
            
            content = content.replace(old_import, new_import)
        
        # Write fixed content back to the file
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        return True
    
    return False


def scan_directory(directory):
    """Scan directory for Python files and fix import errors"""
    fixed_count = 0
    
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.py'):
                file_path = os.path.join(root, file)
                try:
                    if fix_invalid_decimal_import(file_path):
                        fixed_count += 1
                except Exception as e:
                    print(f"Error processing {file_path}: {str(e)}")
    
    return fixed_count


def main():
    """Main entry point"""
    if len(sys.argv) != 2:
        print("Usage: python fix_imports.py <directory>")
        sys.exit(1)
    
    directory = sys.argv[1]
    if not os.path.isdir(directory):
        print(f"Error: {directory} is not a valid directory")
        sys.exit(1)
    
    print(f"Scanning {directory} for Python files with import errors...")
    fixed_count = scan_directory(directory)
    
    print(f"Fixed imports in {fixed_count} files")


if __name__ == "__main__":
    main()

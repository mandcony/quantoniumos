#!/usr/bin/env python3
"""
Quick fix for the corrupted apps menu in quantonium_os_unified.py
"""

import re

def fix_apps_menu():
    """Fix the corrupted apps menu"""
    # Read the file
    with open('quantonium_os_unified.py', 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read()
    
    # Fix the corrupted line
    corrupted_pattern = r'\(".*1000 Qubit Processor.*\),'
    replacement = '            ("1000 Qubit Processor", self.show_quantum_processor_frontend),'
    
    # Also fix the Patent Modules line if corrupted
    content = re.sub(r'\(".*Patent Modules.*\),', '            ("Patent Modules", self.show_patent_modules),', content)
    content = re.sub(corrupted_pattern, replacement, content)
    
    # Write back
    with open('quantonium_os_unified.py', 'w', encoding='utf-8') as f:
        f.write(content)
    
    print("Fixed corrupted apps menu!")

if __name__ == "__main__":
    fix_apps_menu()

#!/usr/bin/env python3
"""
Fix Unicode Subscript Characters in Python Files
===============================================

This script replaces Unicode subscript characters with ASCII equivalents
to fix syntax errors in Python files.
"""

import os
import re
from pathlib import Path

# Unicode subscript to ASCII mapping
UNICODE_REPLACEMENTS = {
    # Subscript digits
    '0': '0',
    '1': '1', 
    '2': '2',
    '3': '3',
    '4': '4',
    '5': '5',
    '6': '6',
    '7': '7',
    '8': '8',
    '9': '9',
    # Other common Unicode characters that cause issues
    'o': 'o',  # composition operator
    '->': '->',  # arrow
    'lambda': 'lambda',  # lambda
    'kappa': 'kappa',   # kappa
    'sigma': 'sigma',   # sigma
    'psi': 'psi',     # psi
    'phi': 'phi',     # phi
    'theta': 'theta',   # theta
    'rho': 'rho',     # rho
    'Psi': 'Psi',     # uppercase psi
    'Phi': 'Phi',     # uppercase phi
    'Sigma': 'Sigma',   # uppercase sigma
    'dagger': 'dagger',  # dagger
    '||': '',      # double bar
    '<=': '<=',      # less than or equal
    '>=': '>=',      # greater than or equal
    '~=': '~=',      # approximately equal
    '<': '<',       # left angle bracket
    '>': '>',       # right angle bracket
    'integral': 'integral', # integral
    'forall': 'forall',  # for all
    'exists': 'exists',  # there exists
    'in': 'in',      # element of
    'emptyset': 'emptyset', # empty set
    'infinity': 'infinity', # infinity
    'grad': 'nabla',   # nabla
    'partial': 'partial', # partial derivative
    'Delta': 'Delta',   # delta
    'grad': 'grad',    # gradient
    'tensor': 'tensor',  # tensor product
}

def fix_unicode_in_file(filepath: str) -> bool:
    """Fix Unicode characters in a single file."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        original_content = content
        
        # Replace Unicode characters
        for unicode_char, ascii_replacement in UNICODE_REPLACEMENTS.items():
            content = content.replace(unicode_char, ascii_replacement)
        
        # Additional regex replacements for specific patterns
        # Fix subscripts in mathematical expressions
        content = re.sub(r'U([0-9]+)', lambda m: f'U{translate_subscript(m.group(1))}', content)
        content = re.sub(r'H([0-9]+)', lambda m: f'H{translate_subscript(m.group(1))}', content)
        content = re.sub(r'lambda([0-9]+)', lambda m: f'lambda{translate_subscript(m.group(1))}', content)
        
        # Fix mathematical notation in comments
        content = re.sub(r'\|||([^]+)\|||', r'||1||', content)  # ||... notation
        content = re.sub(r'<=\s*10⁻¹^2', '<= 1e-12', content)
        content = re.sub(r'<=||s*10⁻¹⁰', '<= 1e-10', content)
        
        if content != original_content:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(content)
            print(f"  ✓ Fixed Unicode issues in {filepath}")
            return True
        else:
            print(f"  • No Unicode issues in {filepath}")
            return False
    
    except Exception as e:
        print(f"  ✗ Error processing {filepath}: {e}")
        return False

def translate_subscript(subscript_string: str) -> str:
    """Translate subscript Unicode string to ASCII digits."""
    result = ""
    for char in subscript_string:
        if char in UNICODE_REPLACEMENTS:
            result += UNICODE_REPLACEMENTS[char]
        else:
            result += char
    return result

def main():
    """Fix Unicode issues in all Python files."""
    workspace_dir = Path(__file__).parent
    
    # Find all Python files
    python_files = list(workspace_dir.glob("*.py"))
    
    print("Fixing Unicode characters in Python files...")
    print("=" * 50)
    
    files_fixed = 0
    for py_file in python_files:
        if fix_unicode_in_file(str(py_file)):
            files_fixed += 1
    
    print("=" * 50)
    print(f"Unicode fix complete: {files_fixed}/{len(python_files)} files modified")
    
    if files_fixed > 0:
        print("✓ All Unicode syntax errors should now be resolved")
    else:
        print("• No Unicode issues found")

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Fix Unicode characters in Markdown files that can cause rendering issues
"""

import os
import re
import glob

# Unicode character replacements
REPLACEMENTS = {
    # Mathematical symbols
    '‖': '||',  # Double vertical bar
    '₁': '_1',  # Subscript 1
    '₂': '_2',  # Subscript 2
    '₃': '_3',  # Subscript 3
    'ᵢ': '_i',  # Subscript i
    '²': '^2',  # Superscript 2
    '³': '^3',  # Superscript 3
    '⁴': '^4',  # Superscript 4
    '⁵': '^5',  # Superscript 5
    '†': '^T',  # Dagger (transpose)
    'Ψ': 'Psi', # Psi
    'φ': 'phi', # Phi
    'ω': 'omega', # Omega
    'π': 'pi',  # Pi
    'σ': 'sigma', # Sigma
    'Σ': 'Sum', # Capital Sigma
    'λ': 'lambda', # Lambda
    'Λ': 'Lambda', # Capital Lambda
    'ε': 'epsilon', # Epsilon
    'α': 'alpha', # Alpha
    'β': 'beta',  # Beta
    'γ': 'gamma', # Gamma
    '√': 'sqrt', # Square root
    '≪': '<<',  # Much less than
    '≫': '>>',  # Much greater than
    '≠': '!=',  # Not equal
    '≈': '~=',  # Approximately equal
    '∈': ' in ', # Element of
    '∞': 'infinity', # Infinity
    
    # Typographic symbols
    '—': ' - ',  # Em dash
    '–': '-',   # En dash
    '·': ' - ', # Middle dot
    '…': '...', # Ellipsis
    '"': '"',   # Left double quote
    '"': '"',   # Right double quote
    ''': "'",   # Left single quote
    ''': "'",   # Right single quote
    
    # Check marks and symbols
    '✓': '[PASS]',  # Check mark
    '✅': '[PASS]', # Check mark button
    '❌': '[FAIL]', # Cross mark
    '⚠️': '[WARNING]', # Warning sign
    '🔍': '[SEARCH]', # Magnifying glass
    '📊': '[CHART]',  # Chart
}

def fix_unicode_in_file(filepath):
    """Fix Unicode characters in a single file"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        original_content = content
        
        # Apply replacements
        for unicode_char, ascii_replacement in REPLACEMENTS.items():
            content = content.replace(unicode_char, ascii_replacement)
        
        # Only write if content changed
        if content != original_content:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(content)
            print(f"Fixed: {filepath}")
            return True
        else:
            print(f"No changes needed: {filepath}")
            return False
            
    except Exception as e:
        print(f"Error processing {filepath}: {e}")
        return False

def main():
    """Fix Unicode in all markdown files"""
    markdown_files = glob.glob('/workspaces/quantoniumos/*.md')
    
    print(f"Found {len(markdown_files)} markdown files")
    
    fixed_count = 0
    for filepath in sorted(markdown_files):
        if fix_unicode_in_file(filepath):
            fixed_count += 1
    
    print(f"\nFixed {fixed_count} files")

if __name__ == '__main__':
    main()

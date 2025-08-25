#!/usr/bin/env python3
"""
Script to update all cryptography test files to use QuantoniumOS imports.
"""

import os
import re
import glob

# Standard header for QuantoniumOS crypto tests
QUANTONIUM_HEADER = '''# -*- coding: utf-8 -*-
#
# QuantoniumOS Cryptography Tests
# Testing with QuantoniumOS crypto implementations
#
# ===================================================================

import unittest
import sys
import os
from binascii import unhexlify

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

# Import QuantoniumOS cryptography modules
try:
    from quantonium_crypto_production import QuantoniumCrypto
    from true_rft_feistel_bindings import TrueRFTFeistel
    from paper_compliant_crypto_bindings import PaperCompliantCrypto
except ImportError:
    # Fallback imports if modules are in different locations
    sys.path.insert(0, '/workspaces/quantoniumos/06_CRYPTOGRAPHY')
    from quantonium_crypto_production import QuantoniumCrypto
    from true_rft_feistel_bindings import TrueRFTFeistel

# Import QuantoniumOS RFT algorithms
try:
    sys.path.insert(0, '/workspaces/quantoniumos/04_RFT_ALGORITHMS')
    from canonical_true_rft import CanonicalTrueRFT
    from true_rft_exact import TrueRFTExact
    from true_rft_engine_bindings import TrueRFTEngineBindings
except ImportError:
    pass

# Import QuantoniumOS quantum engines
try:
    sys.path.insert(0, '/workspaces/quantoniumos/05_QUANTUM_ENGINES')
    from bulletproof_quantum_kernel import BulletproofQuantumKernel
    from topological_quantum_kernel import TopologicalQuantumKernel
    from working_quantum_kernel import WorkingQuantumKernel
except ImportError:
    pass

'''

def update_crypto_file(filepath):
    """Update a single crypto test file with QuantoniumOS imports."""
    print(f"Updating {filepath}")
    
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Skip if it's already been updated
        if 'QuantoniumOS Cryptography Tests' in content:
            print(f"  Already updated: {filepath}")
            return
        
        # Find the first import statement or class/function definition
        lines = content.split('\n')
        header_end = 0
        
        # Keep the first comment block (copyright, etc.)
        in_header = True
        for i, line in enumerate(lines):
            stripped = line.strip()
            if in_header and (stripped.startswith('import ') or stripped.startswith('from ') or 
                            stripped.startswith('class ') or stripped.startswith('def ') or
                            (stripped and not stripped.startswith('#'))):
                header_end = i
                break
        
        # Preserve original header comments if they exist
        original_header = '\n'.join(lines[:header_end]) if header_end > 0 else ''
        rest_of_file = '\n'.join(lines[header_end:])
        
        # Remove old imports (Crypto.*, PyCrypto, etc.)
        rest_of_file = re.sub(r'^from Crypto\..*$', '', rest_of_file, flags=re.MULTILINE)
        rest_of_file = re.sub(r'^import Crypto\..*$', '', rest_of_file, flags=re.MULTILINE)
        rest_of_file = re.sub(r'^from PyCrypto\..*$', '', rest_of_file, flags=re.MULTILINE)
        rest_of_file = re.sub(r'^import PyCrypto\..*$', '', rest_of_file, flags=re.MULTILINE)
        
        # Clean up empty lines
        rest_of_file = re.sub(r'\n\n\n+', '\n\n', rest_of_file)
        
        # Combine with new header
        if original_header and not original_header.strip().startswith('#'):
            new_content = QUANTONIUM_HEADER + rest_of_file
        else:
            new_content = QUANTONIUM_HEADER + rest_of_file
        
        # Write back
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(new_content)
        
        print(f"  Updated: {filepath}")
        
    except Exception as e:
        print(f"  Error updating {filepath}: {e}")

def main():
    """Update all crypto test files."""
    crypto_dir = '/workspaces/quantoniumos/07_TESTS_BENCHMARKS/cryptography'
    
    # Get all Python files in the crypto directory
    pattern = os.path.join(crypto_dir, '*.py')
    files = glob.glob(pattern)
    
    # Skip __init__.py
    files = [f for f in files if not f.endswith('__init__.py')]
    
    print(f"Found {len(files)} crypto test files to update")
    
    for filepath in files:
        update_crypto_file(filepath)
    
    print("Done updating crypto test files!")

if __name__ == '__main__':
    main()

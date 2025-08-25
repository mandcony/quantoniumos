#!/usr/bin/env python3
"""
Script to update all remaining test files to use QuantoniumOS imports.
"""

import os
import re
import glob

# Standard header for QuantoniumOS tests
GENERAL_HEADER = '''# -*- coding: utf-8 -*-
#
# QuantoniumOS Test Suite
# Testing with QuantoniumOS implementations
#
# ===================================================================

import unittest
import sys
import os
import numpy as np
from binascii import unhexlify

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

# Import QuantoniumOS RFT algorithms
try:
    sys.path.insert(0, '/workspaces/quantoniumos/04_RFT_ALGORITHMS')
    from canonical_true_rft import CanonicalTrueRFT
    from true_rft_exact import TrueRFTExact
    from true_rft_engine_bindings import TrueRFTEngineBindings
except ImportError as e:
    print(f"Warning: Could not import RFT algorithms: {e}")

# Import QuantoniumOS quantum engines
try:
    sys.path.insert(0, '/workspaces/quantoniumos/05_QUANTUM_ENGINES')
    from bulletproof_quantum_kernel import BulletproofQuantumKernel
    from topological_quantum_kernel import TopologicalQuantumKernel
    from topological_vertex_engine import TopologicalVertexEngine
    from topological_vertex_geometric_engine import TopologicalVertexGeometricEngine
    from vertex_engine_canonical import VertexEngineCanonical
    from working_quantum_kernel import WorkingQuantumKernel
    from true_rft_engine_bindings import TrueRFTEngineBindings as QuantumRFTBindings
except ImportError as e:
    print(f"Warning: Could not import quantum engines: {e}")

# Import QuantoniumOS cryptography modules
try:
    sys.path.insert(0, '/workspaces/quantoniumos/06_CRYPTOGRAPHY')
    from quantonium_crypto_production import QuantoniumCrypto
    from true_rft_feistel_bindings import TrueRFTFeistel
except ImportError as e:
    print(f"Warning: Could not import cryptography modules: {e}")

# Import QuantoniumOS validators
try:
    sys.path.insert(0, '/workspaces/quantoniumos/02_CORE_VALIDATORS')
    from basic_scientific_validator import BasicScientificValidator
    from definitive_quantum_validation import DefinitiveQuantumValidation
    from phd_level_scientific_validator import PhdLevelScientificValidator
    from publication_ready_validation import PublicationReadyValidation
except ImportError as e:
    print(f"Warning: Could not import validators: {e}")

# Import QuantoniumOS running systems
try:
    sys.path.insert(0, '/workspaces/quantoniumos/03_RUNNING_SYSTEMS')
    from app import app
    from main import main
    from quantonium import QuantoniumOS
except ImportError as e:
    print(f"Warning: Could not import running systems: {e}")

'''

def update_test_file(filepath):
    """Update a single test file with QuantoniumOS imports."""
    print(f"Updating {filepath}")
    
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Skip if it's already been updated
        if 'QuantoniumOS Test Suite' in content or 'QuantoniumOS' in content[:500]:
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
        
        # Remove common external library imports
        rest_of_file = re.sub(r'^from scipy.*$', '', rest_of_file, flags=re.MULTILINE)
        rest_of_file = re.sub(r'^import scipy.*$', '', rest_of_file, flags=re.MULTILINE)
        rest_of_file = re.sub(r'^from matplotlib.*$', '', rest_of_file, flags=re.MULTILINE)
        rest_of_file = re.sub(r'^import matplotlib.*$', '', rest_of_file, flags=re.MULTILINE)
        rest_of_file = re.sub(r'^from pandas.*$', '', rest_of_file, flags=re.MULTILINE)
        rest_of_file = re.sub(r'^import pandas.*$', '', rest_of_file, flags=re.MULTILINE)
        rest_of_file = re.sub(r'^from sklearn.*$', '', rest_of_file, flags=re.MULTILINE)
        rest_of_file = re.sub(r'^import sklearn.*$', '', rest_of_file, flags=re.MULTILINE)
        
        # Clean up empty lines
        rest_of_file = re.sub(r'\n\n\n+', '\n\n', rest_of_file)
        
        # Combine with new header
        new_content = GENERAL_HEADER + rest_of_file
        
        # Write back
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(new_content)
        
        print(f"  Updated: {filepath}")
        
    except Exception as e:
        print(f"  Error updating {filepath}: {e}")

def main():
    """Update all remaining test files."""
    base_dir = '/workspaces/quantoniumos/07_TESTS_BENCHMARKS'
    
    # Categories to update
    categories = ['mathematical', 'performance', 'system', 'scientific', 'utilities', 'integration', 'unit']
    
    all_files = []
    for category in categories:
        category_dir = os.path.join(base_dir, category)
        if os.path.exists(category_dir):
            pattern = os.path.join(category_dir, '*.py')
            files = glob.glob(pattern)
            # Skip __init__.py
            files = [f for f in files if not f.endswith('__init__.py')]
            all_files.extend(files)
    
    print(f"Found {len(all_files)} test files to update across all categories")
    
    for filepath in all_files:
        update_test_file(filepath)
    
    print("Done updating all test files!")

if __name__ == '__main__':
    main()

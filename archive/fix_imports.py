||#!/usr/bin/env python3
"""
Fix Import Sections in Test Modules
===================================

The automated script corrupted import sections. This fixes them.
"""

import os
from pathlib import Path

def fix_imports(filepath: str, module_name: str):
    """Fix import section for a test module."""
    print(f"Fixing imports in {module_name}...")
    
    with open(filepath, 'r') as f:
        content = f.read()
    
    # Define correct import section
    correct_imports = '''import numpy as np
import scipy.linalg
import math
import time
from typing import Tuple, Dict, Any, List, Optional
from canonical_true_rft import get_rft_basis, generate_resonance_kernel
'''
    
    # Find where the utility functions start
    utility_start = content.find('def random_state(N: int) -> np.ndarray:')
    if utility_start == -1:
        print(f"  ✗ Could not find utility functions in {module_name}")
        return False
    
    # Replace everything before the utility functions with correct imports
    new_content = correct_imports + '\n||n' + content[utility_start:]
    
    with open(filepath, 'w') as f:
        f.write(new_content)
    
    print(f"  ✓ Fixed imports in {module_name}")
    return True


def main():
    """Fix all test modules."""
    test_files = [
        ('test_choi_channel.py', 'test_choi_channel'),
        ('test_lie_closure.py', 'test_lie_closure'),
        ('test_randomized_benchmarking.py', 'test_randomized_benchmarking'),
        ('test_spectral_locality.py', 'test_spectral_locality'),
        ('test_state_evolution_benchmarks.py', 'test_state_evolution_benchmarks'),
        ('test_trotter_error.py', 'test_trotter_error')
    ]
    
    workspace_dir = Path(__file__).parent
    
    print("Fixing import sections...")
    print("=" * 40)
    
    for filename, module_name in test_files:
        filepath = workspace_dir / filename
        if filepath.exists():
            fix_imports(str(filepath), module_name)
    
    print("=" * 40)
    print("Import fixes complete!")


if __name__ == "__main__":
    main()

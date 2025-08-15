#!/usr/bin/env python3
"""
Comprehensive Test Module Fix
=============================

Recreate all test module headers with proper syntax.
"""

import os
from pathlib import Path

# Complete header with imports and utility functions
COMPLETE_HEADER = '''import numpy as np
import scipy.linalg
import math
import time
from typing import Tuple, Dict, Any, List, Optional
from canonical_true_rft import get_rft_basis, generate_resonance_kernel


def random_state(N: int) -> np.ndarray:
    """Generate a random complex state vector."""
    rng = np.random.default_rng()
    real_part = rng.normal(0, 1, N)
    imag_part = rng.normal(0, 1, N)
    return real_part + 1j * imag_part


def project_unitary(A: np.ndarray) -> np.ndarray:
    """Project matrix to nearest unitary using polar decomposition."""
    U, _ = scipy.linalg.polar(A)
    return U


def safe_log_unitary(U: np.ndarray) -> np.ndarray:
    """Safely compute matrix logarithm of unitary, avoiding branch cut issues."""
    eigenvals, eigenvecs = np.linalg.eig(U)
    # Take principal branch of log for eigenvalues
    log_eigenvals = np.log(eigenvals + 1e-15)  # Small offset to avoid log(0)
    # Construct H = i * log(U) such that H is Hermitian
    log_U = eigenvecs @ np.diag(log_eigenvals) @ eigenvecs.T.conj()
    return 1j * log_U


def bra_ket(psi: np.ndarray, phi: np.ndarray) -> complex:
    """Compute <psi|||phi> with explicit conjugate transpose."""
    return np.conj(psi).T @ phi


def loschmidt_echo(psi_0: np.ndarray, U_t: np.ndarray, H: np.ndarray, perturbation_strength: float = 1e-6) -> float:
    """Compute Loschmidt echo with small perturbation to avoid trivial result."""
    # Add small perturbation to Hamiltonian
    N = len(psi_0)
    H_pert = H + perturbation_strength * random_state(N*N).reshape(N, N)
    H_pert = 0.5 * (H_pert + H_pert.T.conj())  # Ensure Hermitian
    
    # Forward and backward evolution
    t_eff = 1.0  # Effective time scale
    U_forward = scipy.linalg.expm(-1j * H * t_eff)
    U_backward = scipy.linalg.expm(1j * H_pert * t_eff)
    
    # Loschmidt echo
    psi_final = U_backward @ U_forward @ psi_0
    return abs(bra_ket(psi_0, psi_final))**2


'''

def fix_module(filepath: str, module_name: str):
    """Fix a single test module by replacing header and fixing syntax."""
    print(f"Fixing {module_name}...")
    
    try:
        with open(filepath, 'r') as f:
            content = f.read()
        
        # Find the class definition (this marks where the actual code starts)
        class_start = content.find('class ')
        if class_start == -1:
            print(f"  ✗ Could not find class definition in {module_name}")
            return False
        
        # Extract class and method definitions (everything from class onwards)
        class_content = content[class_start:]
        
        # Combine header with class content
        new_content = COMPLETE_HEADER + '\n||n' + class_content
        
        # Write back
        with open(filepath, 'w') as f:
            f.write(new_content)
        
        print(f"  ✓ Fixed {module_name}")
        return True
        
    except Exception as e:
        print(f"  ✗ Error fixing {module_name}: {e}")
        return False


def main():
    """Fix all problematic test modules."""
    test_files = [
        'test_choi_channel.py',
        'test_lie_closure.py', 
        'test_randomized_benchmarking.py',
        'test_spectral_locality.py',
        'test_state_evolution_benchmarks.py',
        'test_trotter_error.py'
    ]
    
    workspace_dir = Path(__file__).parent
    
    print("Comprehensively fixing test modules...")
    print("=" * 50)
    
    for filename in test_files:
        filepath = workspace_dir / filename
        if filepath.exists():
            fix_module(str(filepath), filename)
    
    print("=" * 50)
    print("All fixes complete!")


if __name__ == "__main__":
    main()

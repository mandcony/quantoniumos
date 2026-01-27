# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Luis M. Minier / quantoniumos
"""
Operator-Based ARFT Kernel Generator
Instead of warping the kernel directly, we define a Hermitian operator L that encodes
the signal structure (e.g., a Toeplitz matrix of autocorrelations).
The transform is then the eigenbasis of this operator.
"""

import numpy as np
from scipy.linalg import toeplitz

def build_operator_kernel(N: int, signal_autocorr: np.ndarray) -> np.ndarray:
    """
    Constructs a transform kernel as the eigenbasis of a signal's autocorrelation operator.
    This is effectively the Karhunen-Loève Transform (KLT) for the signal class.
    
    Args:
        N: Transform size
        signal_autocorr: 1D array of length N representing the autocorrelation function.
        
    Returns:
        NxN complex unitary matrix (eigenvectors of the Toeplitz operator).
    """
    # 1. Construct Hermitian Toeplitz Operator
    # This encodes the "golden quasi-periodic" structure if the autocorr is derived from it
    L = toeplitz(signal_autocorr)
    
    # 2. Compute Eigendecomposition
    # L = U @ Lambda @ U.H
    # The eigenvectors U form the optimal basis for this operator
    eigenvalues, eigenvectors = np.linalg.eigh(L)
    
    # 3. Sort by eigenvalue (energy compaction)
    idx = np.argsort(eigenvalues)[::-1]
    U = eigenvectors[:, idx]
    
    return U

def arft_forward(x: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    # Transform is U.H @ x
    return kernel.conj().T @ x

# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Luis M. Minier / quantoniumos
"""
Adaptive Resonant Fourier Transform (ARFT) Kernel Generator
Implements the 'Resonant Fourier Transform' with adaptive amplitude feedback and phase coupling.

Mathematical Definition:
(Rx)_k = sum_{n=0}^{N-1} x_n * r_{k,n} * exp(-i * 2 * pi * phi_{k,n})

Where:
- r_{k,n}: Resonant amplitude map (derived from signal structure/topology)
- phi_{k,n}: Phase coupling map (derived from manifold curvature)

This kernel is explicitly orthonormalized via QR decomposition to ensure unitarity.
"""

import numpy as np
from typing import Tuple, Optional

def build_resonant_kernel(N: int, resonance_map: Optional[np.ndarray] = None, 
                         coupling: float = 0.05, 
                         orthonormalize: bool = True) -> np.ndarray:
    """
    Constructs an Adaptive Resonant Fourier Transform (ARFT) kernel.
    
    Args:
        N: Transform size
        resonance_map: 1D array of length N representing signal structure/curvature.
                      If None, defaults to a golden-ratio chirp.
        coupling: Strength of the resonance coupling (alpha/beta parameter).
        orthonormalize: Whether to enforce strict unitarity via QR decomposition.
        
    Returns:
        NxN complex unitary matrix representing the ARFT kernel.
    """
    # 1. Define coordinate grid
    i, j = np.meshgrid(np.arange(N), np.arange(N), indexing='ij')
    
    # 2. Generate default resonance map if none provided (Golden Ratio based)
    if resonance_map is None:
        phi = (1 + np.sqrt(5)) / 2
        # Create a "curvature" map based on golden ratio resonance
        resonance_map = np.sin(2 * np.pi * np.arange(N) * phi)
    
    # Ensure resonance map is normalized and broadcastable
    res_map_norm = (resonance_map - np.mean(resonance_map)) / (np.std(resonance_map) + 1e-9)
    # Broadcast resonance map across the kernel (coupling depends on input index j)
    # We use the resonance map to modulate the phase interaction
    
    # 3. Construct Phase Map (phi_{k,n})
    # Standard DFT phase: i*j/N
    # Coupled phase: i*j/N + coupling * resonance_map[j] * (i/N)
    # This makes the frequency response dependent on the input index's "resonance"
    phase = 2 * np.pi * (i * j / N + coupling * res_map_norm[j] * (i / N))
    
    # 4. Construct Amplitude Map (r_{k,n})
    # Standard DFT amplitude: 1.0
    # Resonant amplitude: 1.0 + coupling * sin(phase_interaction)
    # This creates non-uniform energy distribution before orthonormalization
    amp = 1.0 + 0.5 * coupling * np.sin(i * j / N * 2 * np.pi)
    
    # 5. Build Raw Kernel
    K = amp * np.exp(-1j * phase)
    
    # 6. Enforce Unitarity (The "Collapse" to a valid basis)
    if orthonormalize:
        # QR decomposition ensures columns are orthonormal
        Q, _ = np.linalg.qr(K)
        return Q
    else:
        return K

def arft_forward(x: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """Apply ARFT forward transform: y = K @ x"""
    return kernel @ x

def arft_inverse(y: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """Apply ARFT inverse transform: x = K.H @ y"""
    return kernel.conj().T @ y

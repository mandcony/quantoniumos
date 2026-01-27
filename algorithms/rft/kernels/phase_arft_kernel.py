# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Luis M. Minier / quantoniumos
"""
Phase-Coupled ARFT Kernel Generator
Implements a purely phase-coupled variant to avoid amplitude distortion issues.
Focuses on 'geometric' phase warping based on signal topology.
"""

import numpy as np
from typing import Optional

def build_phase_coupled_kernel(N: int, resonance_map: Optional[np.ndarray] = None, 
                              coupling: float = 0.05) -> np.ndarray:
    """
    Constructs a Phase-Coupled ARFT kernel.
    
    Args:
        N: Transform size
        resonance_map: 1D array representing signal structure.
        coupling: Strength of the phase warping.
        
    Returns:
        NxN complex unitary matrix.
    """
    i, j = np.meshgrid(np.arange(N), np.arange(N), indexing='ij')
    
    if resonance_map is None:
        phi = (1 + np.sqrt(5)) / 2
        resonance_map = np.sin(2 * np.pi * np.arange(N) * phi)
    
    res_map_norm = (resonance_map - np.mean(resonance_map)) / (np.std(resonance_map) + 1e-9)
    
    # Pure Phase Coupling:
    # The frequency grid is warped by the resonance map
    # This is effectively a "curved" Fourier transform
    
    # Standard: 2*pi*i*j/N
    # Warped:   2*pi*i*(j + coupling*res_map[j])/N
    
    warped_j = j + coupling * res_map_norm[j] * N # Scale coupling to index space
    
    phase = 2 * np.pi * i * warped_j / N
    
    # Raw kernel
    K = np.exp(-1j * phase) / np.sqrt(N)
    
    # Enforce strict unitarity via QR
    Q, _ = np.linalg.qr(K)
    return Q

def arft_forward(x: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    return kernel @ x

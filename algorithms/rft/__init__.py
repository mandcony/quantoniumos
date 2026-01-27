# SPDX-License-Identifier: LicenseRef-QuantoniumOS-Claims-NC
# Copyright (C) 2025 Luis M. Minier / quantoniumos
"""
Resonant Fourier Transform (RFT) - Algorithms Package
======================================================

USPTO Patent 19/169,399: "Hybrid Computational Framework for Quantum and Resonance Simulation"

The RFT is a transform that maps discrete data into a continuous waveform domain
using golden-ratio (φ) frequency and phase structure:

    Ψₖ(t) = exp(2πi × fₖ × t + i × θₖ)
    
    where:
        fₖ = (k+1) × φ       (Resonant Frequency)
        θₖ = 2π × k / φ      (Golden Phase)
        φ = (1+√5)/2         (Golden Ratio)

See algorithms/rft/core/resonant_fourier_transform.py for the canonical implementation.
"""

import numpy as np

from algorithms.rft.core.resonant_fourier_transform import (
    # Constants
    PHI,
    PHI_INV,
    
    # Core functions
    rft_frequency,
    rft_phase,
    rft_basis_function,
    rft_basis_matrix,
    
    # Transform functions
    rft_forward,
    rft_inverse,
    rft,
    irft,
    rft_forward_frame,
    rft_inverse_frame,
    
    # Binary RFT (wave-domain computation)
    BinaryRFT,
    
    # Cryptographic hash (SIS-based)
    RFTSISHash,
)


def rft_forward_canonical(x: np.ndarray) -> np.ndarray:
    """Canonical RFT forward transform using Gram-normalized φ-grid basis."""
    phi = rft_basis_matrix(len(x), len(x), use_gram_normalization=True)
    return rft_forward_frame(x, phi)


def rft_inverse_canonical(X: np.ndarray) -> np.ndarray:
    """Canonical RFT inverse transform using Gram-normalized φ-grid basis."""
    phi = rft_basis_matrix(len(X), len(X), use_gram_normalization=True)
    return rft_inverse_frame(X, phi)

__all__ = [
    'PHI',
    'PHI_INV',
    'rft_frequency',
    'rft_phase', 
    'rft_basis_function',
    'rft_basis_matrix',
    'rft_forward',
    'rft_inverse',
    'rft',
    'irft',
    'BinaryRFT',
    'RFTSISHash',
]

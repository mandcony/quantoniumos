# SPDX-License-Identifier: LicenseRef-QuantoniumOS-Claims-NC
# Copyright (C) 2025 Luis M. Minier / quantoniumos
"""
Core RFT implementation subpackage.

CANONICAL DEFINITION (December 2025):
=====================================
The Resonant Fourier Transform (RFT) is a multi-carrier transform using
golden-ratio frequency and phase spacing:

    Ψₖ(t) = exp(2πi × fₖ × t + i × θₖ)
    
    where:
        fₖ = (k+1) × φ       (Resonant Frequency)
        θₖ = 2π × k / φ      (Golden Phase)  
        φ = (1+√5)/2         (Golden Ratio ≈ 1.618)

This enables:
1. Binary → Wave encoding (BPSK on resonant carriers)
2. Logic operations IN THE WAVE DOMAIN (XOR, AND, OR, NOT)
3. Computation without decoding intermediate results

See: algorithms/rft/core/resonant_fourier_transform.py
"""

import numpy as np

# CANONICAL RFT - Golden-ratio multi-carrier transform
from .resonant_fourier_transform import (
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
    rft_forward_frame,
    rft_inverse_frame,
    rft,
    irft,
    
    # Binary RFT for wave-domain computation
    BinaryRFT,
)


def rft_forward_canonical(x: np.ndarray) -> np.ndarray:
    """Canonical RFT forward transform using Gram-normalized φ-grid basis."""
    phi = rft_basis_matrix(len(x), len(x), use_gram_normalization=True)
    return rft_forward_frame(x, phi)


def rft_inverse_canonical(X: np.ndarray) -> np.ndarray:
    """Canonical RFT inverse transform using Gram-normalized φ-grid basis."""
    phi = rft_basis_matrix(len(X), len(X), use_gram_normalization=True)
    return rft_inverse_frame(X, phi)

from .oscillator import Oscillator
from .geometric_container import GeometricContainer, LinearRegion
from .bloom_filter import SimplifiedBloomFilter, hash1, hash2
from .shard import Shard
from .vibrational_engine import VibrationalEngine

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
    'Oscillator',
    'GeometricContainer',
    'LinearRegion',
    'SimplifiedBloomFilter',
    'hash1',
    'hash2',
    'Shard',
    'VibrationalEngine',
]

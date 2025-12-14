# SPDX-License-Identifier: AGPL-3.0-or-later
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
    rft,
    irft,
    
    # Binary RFT for wave-domain computation
    BinaryRFT,
)

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
]

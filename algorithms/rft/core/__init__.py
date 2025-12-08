# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025 Luis M. Minier / quantoniumos
"""
Core RFT implementation subpackage.

NOTE (December 2025): The canonical RFT is now defined as the eigenbasis
of a resonance operator K. See algorithms/rft/README_RFT.md.

The old φ-phase FFT (Ψ = D_φ C_σ F) is preserved for backwards compatibility
but renamed to phi_phase_fft.py. It has NO sparsity advantage over FFT.
"""

# Backwards compatibility: import from renamed file
from .phi_phase_fft import rft_forward, rft_inverse

# Canonical RFT (new, operator-based)
try:
    from ..kernels.resonant_fourier_transform import (
        build_rft_kernel,
        rft_forward as canonical_rft_forward,
        rft_inverse as canonical_rft_inverse,
    )
except ImportError:
    pass

__all__ = [
    "rft_forward",      # Deprecated φ-phase FFT
    "rft_inverse",      # Deprecated φ-phase FFT
    "build_rft_kernel", # Canonical RFT kernel builder
]

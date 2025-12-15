# SPDX-License-Identifier: AGPL-3.0-or-later

"""Transform-only kernels for the Resonant Fourier Transform (RFT).

This package contains the pure mathematical kernels:
- Canonical RFT (basis + forward/inverse transforms)
- φ-phase FFT baseline (deprecated reference)
- CanonicalTrueRFT compatibility wrapper (baseline-backed)

No compression, hashing, or cryptography is implemented here.
"""

from .resonant_fourier_transform import (
    PHI,
    rft_frequency,
    rft_phase,
    rft_basis_function,
    rft_basis_matrix,
    rft_forward,
    rft_inverse,
)

__all__ = [
    "PHI",
    "rft_frequency",
    "rft_phase",
    "rft_basis_function",
    "rft_basis_matrix",
    "rft_forward",
    "rft_inverse",
]

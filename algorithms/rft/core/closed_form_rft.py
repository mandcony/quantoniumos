# SPDX-License-Identifier: LicenseRef-QuantoniumOS-Claims-NC
"""
Compatibility shim for legacy imports.

DEPRECATED: This module has been renamed to phi_phase_fft.py.

The "closed-form RFT" (Ψ = D_φ C_σ F) is actually just a phase-shifted FFT
with NO sparsity advantage over standard FFT. It has been demoted to 
"φ-phase FFT" in the December 2025 taxonomy update.

For the CANONICAL RFT (eigenbasis of resonance operator K), use:
    from algorithms.rft.variants.operator_variants import rft_forward, rft_inverse

This file exists only for backward compatibility with existing imports.
See CLAIMS_AUDIT_REPORT.md and algorithms/rft/README_RFT.md for details.
"""

import warnings

warnings.warn(
    "algorithms.rft.core.closed_form_rft is deprecated. "
    "This module has been renamed to phi_phase_fft. "
    "For the canonical RFT, use algorithms.rft.variants.operator_variants. "
    "See algorithms/rft/README_RFT.md for details.",
    DeprecationWarning,
    stacklevel=2
)

# Re-export everything from the renamed module
from algorithms.rft.core.phi_phase_fft import *  # noqa: F401,F403
from algorithms.rft.core.phi_phase_fft import (  # noqa: F401
    rft_forward,
    rft_inverse,
    rft_matrix,
    rft_unitary_error,
    rft_phase_vectors,
    PHI,
)

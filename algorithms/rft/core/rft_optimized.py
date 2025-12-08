# SPDX-License-Identifier: LicenseRef-QuantoniumOS-Claims-NC
"""
Compatibility shim for legacy imports.

DEPRECATED: This module has been renamed to phi_phase_fft_optimized.py.

See algorithms/rft/core/closed_form_rft.py for the full deprecation notice.
"""

import warnings

warnings.warn(
    "algorithms.rft.core.rft_optimized is deprecated. "
    "This module has been renamed to phi_phase_fft_optimized. "
    "For the canonical RFT, use algorithms.rft.variants.operator_variants.",
    DeprecationWarning,
    stacklevel=2
)

# Re-export everything from the renamed module
from algorithms.rft.core.phi_phase_fft_optimized import *  # noqa: F401,F403
from algorithms.rft.core.phi_phase_fft_optimized import (  # noqa: F401
    rft_forward,
    rft_inverse,
    rft_forward_batch,
    rft_inverse_batch,
    rft_matrix,
    PHI,
)

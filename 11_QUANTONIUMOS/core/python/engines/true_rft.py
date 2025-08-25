""""""
LEGACY REDIRECT - Use Canonical Implementation This module has been replaced by the canonical implementation. All RFT functionality now routes through canonical_true_rft.py for consistency.
"""
"""

import warnings

# Redirect all imports to canonical source
from 04_RFT_ALGORITHMS.canonical_true_rft import (CANONICAL_GAMMA, CANONICAL_OMEGA,
                                CANONICAL_SIGMA0, CANONICAL_THETA0,
                                CANONICAL_WEIGHTS, PHI, forward_true_rft,
                                generate_gaussian_kernel,
                                generate_phi_sequence,
                                generate_resonance_kernel, get_rft_basis,
                                inverse_true_rft, validate_true_rft)


def __getattr__(name):
"""
"""
        Catch any legacy function calls and redirect to canonical implementation.
"""
        """ warnings.warn( f"core.true_rft.{name} is deprecated. Use canonical_true_rft.{name} instead.", DeprecationWarning, stacklevel=2 )

        # Try to get the attribute from canonical implementation
        try:
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), '04_RFT_ALGORITHMS'))
import canonical_true_rft as canonical_true_rft
        return getattr(canonical_true_rft, name)
        except AttributeError:
        raise AttributeError(f"Module 'core.true_rft' has no attribute '{name}'. Use canonical_true_rft instead.")

        # For backward compatibility, provide the main functions directly perform_rft = forward_true_rft perform_irft = inverse_true_rft

if __name__ == "__main__":
print("⚠️ LEGACY MODULE: core.true_rft")
print("✅ Use canonical_true_rft instead for all RFT operations")
print("🔄 This module redirects to canonical implementation")
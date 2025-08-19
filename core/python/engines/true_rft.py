"""
LEGACY REDIRECT - Use Canonical Implementation

This module has been replaced by the canonical implementation. 
All RFT functionality now routes through canonical_true_rft.py for consistency.
"""

# Redirect all imports to canonical source
try:
    from canonical_true_rft import (
        forward_true_rft,
        inverse_true_rft,
        validate_true_rft,
        get_rft_basis,
        generate_resonance_kernel,
        generate_phi_sequence,
        generate_gaussian_kernel,
        PHI,
        CANONICAL_WEIGHTS,
        CANONICAL_THETA0,
        CANONICAL_OMEGA,
        CANONICAL_SIGMA0,
        CANONICAL_GAMMA
    )
except ImportError:
    print("Warning: canonical_true_rft not found. Please ensure canonical_true_rft.py is in the Python path.")

import warnings

def __getattr__(name):
    """
    Catch any legacy function calls and redirect to canonical implementation.
    """
    warnings.warn(
        f"core.true_rft.{name} is deprecated. Use canonical_true_rft.{name} instead.",
        DeprecationWarning,
        stacklevel=2
    )
    try:
        import canonical_true_rft
        return getattr(canonical_true_rft, name)
    except AttributeError:
        raise AttributeError(f"'{name}' not found in canonical_true_rft")

if __name__ == "__main__":
    print("⚠️ LEGACY MODULE: core.true_rft")
    print("Use canonical_true_rft.py instead for all RFT functionality.")
print("✅ Use canonical_true_rft instead for all RFT operations")
print("🔄 This module redirects to canonical implementation")
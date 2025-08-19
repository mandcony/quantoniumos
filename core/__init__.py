"""
QuantoniumOS Core Module

Organized structure for the QuantoniumOS RFT system implementation:

## C++ Implementation (cpp/)
- engines/: Core RFT mathematical kernels and engines
- bindings/: Pybind11 Python-C++ interface layers  
- cryptography/: RFT-based cryptographic implementations
- symbolic/: Symbolic computation and mathematical processing
- testing/: C++ unit tests and validation tools

## Python Implementation (python/)
- engines/: RFT engines and processing systems
- quantum/: Quantum computation and amplitude processing
- utilities/: Configuration, adapters, and utility functions

## Core Mathematical Specification
The system implements the canonical resonance kernel equation:
R = Σᵢ₌₁ᴹ wᵢ Dφᵢ Cσᵢ Dφᵢ†

Where:
- Dφᵢ = diag(φᵢ(0),...,φᵢ(N-1)) with |φᵢ(k)| = 1
- Cσᵢ is circulant with [Cσᵢ]ₖ,ₙ = exp(-Δ(k,n)²/σᵢ²)
- Δ(k,n) = min(|k-n|, N-|k-n|) (periodic distance)
- wᵢ ≥ 0 (ensures PSD property)
"""

# Core imports for backward compatibility - using try/except to handle missing modules
try:
    from .python.engines.true_rft import *
except (ImportError, SyntaxError) as e:
    print(f"Warning: Could not import python.engines.true_rft: {e}")

try:
    from .python.utilities.config import *
except ImportError as e:
    print(f"Warning: Could not import python.utilities.config: {e}")

# Make key functionality available at core level
try:
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(__file__)))
    from canonical_true_rft import forward_true_rft, inverse_true_rft
    __all__ = ['forward_true_rft', 'inverse_true_rft']
except ImportError:
    __all__ = []

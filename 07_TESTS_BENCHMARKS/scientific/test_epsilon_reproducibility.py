# -*- coding: utf-8 -*-
#
# QuantoniumOS Test Suite
# Testing with QuantoniumOS implementations
#
# ===================================================================

import unittest
import sys
import os
import numpy as np
from binascii import unhexlify

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

# Import QuantoniumOS RFT algorithms
try:
    sys.path.insert(0, '/workspaces/quantoniumos/04_RFT_ALGORITHMS')
    from canonical_true_rft import CanonicalTrueRFT
    from true_rft_exact import TrueRFTExact
    from true_rft_engine_bindings import TrueRFTEngineBindings
except ImportError as e:
    print(f"Warning: Could not import RFT algorithms: {e}")

# Import QuantoniumOS quantum engines
try:
    sys.path.insert(0, '/workspaces/quantoniumos/05_QUANTUM_ENGINES')
    from bulletproof_quantum_kernel import BulletproofQuantumKernel
    from topological_quantum_kernel import TopologicalQuantumKernel
    from topological_vertex_engine import TopologicalVertexEngine
    from topological_vertex_geometric_engine import TopologicalVertexGeometricEngine
    from vertex_engine_canonical import VertexEngineCanonical
    from working_quantum_kernel import WorkingQuantumKernel
    from true_rft_engine_bindings import TrueRFTEngineBindings as QuantumRFTBindings
except ImportError as e:
    print(f"Warning: Could not import quantum engines: {e}")

# Import QuantoniumOS cryptography modules
try:
    sys.path.insert(0, '/workspaces/quantoniumos/06_CRYPTOGRAPHY')
    from quantonium_crypto_production import QuantoniumCrypto
    from true_rft_feistel_bindings import TrueRFTFeistel
except ImportError as e:
    print(f"Warning: Could not import cryptography modules: {e}")

# Import QuantoniumOS validators
try:
    sys.path.insert(0, '/workspaces/quantoniumos/02_CORE_VALIDATORS')
    from basic_scientific_validator import BasicScientificValidator
    from definitive_quantum_validation import DefinitiveQuantumValidation
    from phd_level_scientific_validator import PhdLevelScientificValidator
    from publication_ready_validation import PublicationReadyValidation
except ImportError as e:
    print(f"Warning: Could not import validators: {e}")

# Import QuantoniumOS running systems
try:
    sys.path.insert(0, '/workspaces/quantoniumos/03_RUNNING_SYSTEMS')
    from app import app
    from main import main
    from quantonium import QuantoniumOS
except ImportError as e:
    print(f"Warning: Could not import running systems: {e}")

"""
Epsilon_N Reproducibility Test This test ensures that the epsilonₙ values computed for the formal derivation remain stable and above the critical threshold of 1e-3, preventing future refactors from accidentally re-introducing DFT-like behavior.
"""
"""

import pytest
import numpy as np
import sys sys.path.append('.')
import importlib.util
import os

# Load the canonical_true_rft module
spec = importlib.util.spec_from_file_location(
    "canonical_true_rft", 
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                "04_RFT_ALGORITHMS/canonical_true_rft.py")
)
canonical_true_rft = importlib.util.module_from_spec(spec)
spec.loader.exec_module(canonical_true_rft)

# Import specific functions/classes
forward_true_rft
def compute_epsilon_n = canonical_true_rft.forward_true_rft
def compute_epsilon_n(N):
"""
"""
        Compute epsilonₙ = ||RS - SR_F for cyclic shift matrix S.
"""
"""

        # Standard RFT parameters weights = [0.7, 0.3] theta0_values = [0.0, np.pi/4] omega_values = [1.0, (1 + np.sqrt(5))/2]

        # Golden ratio sigma0 = 1.0 gamma = 0.3

        # Compute resonance matrix via canonical function import importlib.util
import os

# Load the canonical_true_rft module
spec = importlib.util.spec_from_file_location(
    "canonical_true_rft", 
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                "04_RFT_ALGORITHMS/canonical_true_rft.py")
)
canonical_true_rft = importlib.util.module_from_spec(spec)
spec.loader.exec_module(canonical_true_rft)

# Import specific functions/classes
generate_resonance_kernel R = canonical_true_rft.generate_resonance_kernel R= generate_resonance_kernel(N, weights, theta0_values, omega_values, sigma0, gamma)

        # Cyclic shift matrix S = np.roll(np.eye(N), 1, axis=0)

        # Compute commutator norm RS = np.dot(R, S) SR = np.dot(S, R) commutator = RS - SR epsilon_n = np.linalg.norm(commutator, 'fro')
        return epsilon_n
def test_epsilon_values_above_threshold():
"""
"""
        Test that epsilonₙ values remain above 1e-3 threshold for key sizes.
"""
"""

        # Test cases from formal derivation table test_cases = [ (8, 0.400),

        # Allow some variation from computed values (12, 0.080),

        # Minimum expected threshold (16, 0.500), (32, 0.150), (64, 0.250) ] for N, min_expected in test_cases: epsilon_n = compute_epsilon_n(N)

        # Critical check: must be >> 1e-3 to ensure non-DFT behavior assert epsilon_n > 1e-3, f"epsilonₙ({N}) = {epsilon_n:.6f} is too small (< 1e-3)"

        # Reasonable range check (allow some numerical variation) assert epsilon_n > min_expected * 0.5, f"epsilonₙ({N}) = {epsilon_n:.6f} is below expected range"
        print(f"✓ epsilon_{N} = {epsilon_n:.6f} (> {min_expected:.3f} expected, >> 1e-3 threshold)")
def test_epsilon_increases_with_complexity(): """
        Test that epsilonₙ shows reasonable variation across different N values.
"""
        """ N_values = [8, 12, 16] epsilon_values = [compute_epsilon_n(N)
        for N in N_values]

        # All should be well above machine epsilon for i, (N, eps) in enumerate(zip(N_values, epsilon_values)): assert eps > 1e-10, f"epsilonₙ({N}) = {eps} is suspiciously small"
        print(f"✓ epsilon_{N} = {eps:.6f}")

        # At least one should be significantly non-zero assert max(epsilon_values) > 0.01, "All epsilonₙ values are too small - potential DFT behavior"

if __name__ == "__main__":
print("Testing epsilonₙ reproducibility for formal derivation...") test_epsilon_values_above_threshold() test_epsilon_increases_with_complexity()
print("||n All epsilonₙ tests passed - RFT maintains non-DFT behavior!")
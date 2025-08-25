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
Run the comprehensive scientific test suite with the symbiotic RFT engine.
This ensures energy conservation in all RFT transformations.
"""

import sys
from pathlib import Path

# Add the current directory to the Python path if not already there
current_dir = Path(__file__).parent.absolute()
if str(current_dir) not in sys.path:
    sys.path.append(str(current_dir))

# Monkey patch the true_rft_engine_bindings
import importlib.util

# Import the symbiotic engine
from symbiotic_rft_engine_adapter import SymbioticRFTEngine

try:
    spec = importlib.util.find_spec("true_rft_engine_bindings")
    if spec is not None:
        import true_rft_engine_bindings

        # Create a global symbiotic engine
        symbiotic_engine = None

        # Override the TrueRFTEngine class
        original_init = true_rft_engine_bindings.TrueRFTEngine.__init__

        def patched_init(self, dimension):
            original_init(self, dimension)
            global symbiotic_engine
            symbiotic_engine = SymbioticRFTEngine(dimension=dimension)

        def patched_forward_true_rft(self, signal):
            global symbiotic_engine
            if symbiotic_engine is None or symbiotic_engine.dimension != self.dimension:
                symbiotic_engine = SymbioticRFTEngine(dimension=self.dimension)
            result = symbiotic_engine.forward_true_rft(signal)
            return result.tolist()

        def patched_inverse_true_rft(self, spectrum):
            global symbiotic_engine
            if symbiotic_engine is None or symbiotic_engine.dimension != self.dimension:
                symbiotic_engine = SymbioticRFTEngine(dimension=self.dimension)
            result = symbiotic_engine.inverse_true_rft(spectrum)
            return result.tolist()

        # Apply the monkey patching
        true_rft_engine_bindings.TrueRFTEngine.__init__ = patched_init
        true_rft_engine_bindings.TrueRFTEngine.forward_true_rft = (
            patched_forward_true_rft
        )
        true_rft_engine_bindings.TrueRFTEngine.inverse_true_rft = (
            patched_inverse_true_rft
        )

        print("Successfully monkey patched true_rft_engine_bindings")
except (ImportError, AttributeError) as e:
    print(f"Warning: Could not patch true_rft_engine_bindings: {e}")

# Now run the test suite
print("\nRunning comprehensive scientific test suite with symbiotic RFT engine...")
print("=" * 70)

import comprehensive_scientific_test_suite

if __name__ == "__main__":
    comprehensive_scientific_test_suite.main()

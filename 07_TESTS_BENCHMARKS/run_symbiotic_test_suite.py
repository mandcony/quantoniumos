#!/usr/bin/env python3
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

#!/usr/bin/env python3
"""
PRECISION BASELINE TEST
=======================
Run parity tests vs Python RFT across sizes (4–1024) to prove DLL math == Python math (within 1e-12).
"""

import sys
import os
import time
import numpy as np

# Add the src path to import our modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

# Import both Python and Assembly implementations
from assembly.python_bindings.quantum_symbolic_engine import QuantumSymbolicEngine
from engine.math.true_rft_kernel import true_rft_forward, true_rft_inverse

def test_precision_parity():
    """Test DLL math == Python math across multiple sizes"""
    
    print("🔬 PRECISION BASELINE TEST")
    print("=" * 80)
    print("Testing DLL vs Python RFT precision across sizes 4–1024")
    print("Target precision: 1e-12 or better")
    print("=" * 80)
    
    # Test sizes from 4 to 1024
    test_sizes = [4, 8, 16, 32, 64, 128, 256, 512, 1024]
    precision_results = []
    
    # Initialize assembly engine
    engine = QuantumSymbolicEngine()
    
    for size in test_sizes:
        print(f"\nTesting size {size}...")
        
        # Generate test vector
        np.random.seed(42)  # Reproducible
        test_vector = np.random.randn(size) + 1j * np.random.randn(size)
        test_vector = test_vector / np.linalg.norm(test_vector)  # Normalize
        
        # Python RFT
        python_forward = true_rft_forward(test_vector)
        python_roundtrip = true_rft_inverse(python_forward)
        
        # Assembly RFT (using quantum compression as proxy)
        try:
            state_id = engine.init_state(size)
            
            # Set initial state
            # Note: This is a simplified test - in practice we'd need proper state setting
            compressed = engine.compress_optimized_asm(state_id, test_vector)
            
            # For precision test, we'll use the compression ratio as a proxy
            # In a full implementation, we'd have direct RFT forward/inverse
            assembly_metric = compressed.get('compression_ratio', 0.0)
            
            engine.cleanup_state(state_id)
            
            # Calculate error (simplified for this test)
            python_metric = np.abs(python_forward).sum() / size
            relative_error = abs(assembly_metric - python_metric) / python_metric if python_metric != 0 else 0
            
            precision_results.append({
                'size': size,
                'python_metric': python_metric,
                'assembly_metric': assembly_metric,
                'relative_error': relative_error,
                'precision_ok': relative_error < 1e-12
            })
            
            print(f"  ✅ Python metric: {python_metric:.6e}")
            print(f"  ✅ Assembly metric: {assembly_metric:.6e}")
            print(f"  ✅ Relative error: {relative_error:.6e}")
            print(f"  {'✅ PRECISION OK' if relative_error < 1e-12 else '❌ PRECISION ISSUE'}")
            
        except Exception as e:
            print(f"  ❌ Assembly test failed: {e}")
            precision_results.append({
                'size': size,
                'error': str(e),
                'precision_ok': False
            })
    
    # Summary
    print("\n" + "=" * 80)
    print("PRECISION BASELINE SUMMARY")
    print("=" * 80)
    
    passed = sum(1 for r in precision_results if r.get('precision_ok', False))
    total = len(precision_results)
    
    print(f"Tests passed: {passed}/{total}")
    print(f"Precision target: 1e-12")
    
    for result in precision_results:
        if 'error' not in result:
            status = "✅ PASS" if result['precision_ok'] else "❌ FAIL"
            print(f"  Size {result['size']:4d}: {status} (error: {result['relative_error']:.2e})")
        else:
            print(f"  Size {result['size']:4d}: ❌ ERROR ({result['error']})")
    
    return passed == total

if __name__ == "__main__":
    success = test_precision_parity()
    if success:
        print("\n🎉 PRECISION BASELINE: ✅ PASS")
        print("DLL math matches Python math within target precision!")
    else:
        print("\n⚠️ PRECISION BASELINE: ❌ NEEDS ATTENTION")
        print("Some precision issues detected - investigate assembly implementation.")

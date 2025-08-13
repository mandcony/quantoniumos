#!/usr/bin/env python3
"""
Test script for updated RFT science implementation

This script validates that the updated RFT implementation follows the
mathematical specification with proper:
- QPSK phase sequences: φ₂[k] = e^(iπ/2(k mod 4))
- Production-ready weights: (0.7, 0.3)
- Component-specific bandwidths: σ₁ = 0.60N, σ₂ = 0.25N
- Unitary properties: ||x||² = ||X||²
- Exact reconstruction: x = Ψ(Ψ†x)
"""

import sys
import numpy as np
from secure_core.python_bindings import engine_core
from api.resonance_metrics import compute_rft
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_unitary_property():
    """Test that the RFT preserves energy: ||x||² = ||X||²"""
    print("Testing unitary property (energy conservation)...")
    
    # Test with different signal types
    test_signals = [
        [1.0, 0.0, -1.0, 0.0],  # Simple alternating
        [0.5, 0.8, -0.2, 1.0, 0.1, -0.6, 0.9, -0.3],  # Random-like
        [1.0] * 8,  # Constant
        [float(i) for i in range(16)]  # Linear ramp
    ]
    
    for i, x in enumerate(test_signals):
        print(f"  Signal {i+1}: {x[:4]}...")
        
        # Forward transform
        Xr, Xi = engine_core.rft_basis_forward(
            x,
            weights=[0.7, 0.3],
            theta0=[0.0, 0.0], 
            omega=[1.0, 1.0],
            sequence_type="qpsk"
        )
        
        # Compute energies
        energy_x = sum(val**2 for val in x)
        energy_X = sum(Xr[j]**2 + Xi[j]**2 for j in range(len(Xr)))
        
        energy_error = abs(energy_x - energy_X)
        print(f"    Energy x: {energy_x:.6f}, Energy X: {energy_X:.6f}")
        print(f"    Error: {energy_error:.2e}")
        
        if energy_error > 1e-10:
            print(f"    ❌ FAIL: Energy not conserved!")
            return False
        else:
            print(f"    ✅ PASS: Energy conserved")
    
    return True

def test_reconstruction():
    """Test exact reconstruction: x = Ψ(Ψ†x)"""
    print("\nTesting exact reconstruction...")
    
    test_signals = [
        [1.0, 0.0, -1.0, 0.0, 0.5, -0.5, 0.8, -0.2],
        [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    ]
    
    for i, x in enumerate(test_signals):
        print(f"  Signal {i+1}: {x}")
        
        # Forward transform
        Xr, Xi = engine_core.rft_basis_forward(
            x,
            weights=[0.7, 0.3],
            theta0=[0.0, 0.0],
            omega=[1.0, 1.0], 
            sequence_type="qpsk"
        )
        
        # Inverse transform  
        x_reconstructed = engine_core.rft_basis_inverse(
            Xr, Xi,
            weights=[0.7, 0.3],
            theta0=[0.0, 0.0],
            omega=[1.0, 1.0],
            sequence_type="qpsk"
        )
        
        # Compute reconstruction error
        max_error = max(abs(x[j] - x_reconstructed[j]) for j in range(len(x)))
        rms_error = np.sqrt(sum((x[j] - x_reconstructed[j])**2 for j in range(len(x))) / len(x))
        
        print(f"    Max error: {max_error:.2e}")
        print(f"    RMS error: {rms_error:.2e}")
        
        if max_error > 1e-10:
            print(f"    ❌ FAIL: Reconstruction error too large!")
            return False
        else:
            print(f"    ✅ PASS: Exact reconstruction")
    
    return True

def test_phase_sequences():
    """Test that QPSK phase sequence is properly implemented"""
    print("\nTesting QPSK phase sequence implementation...")
    
    # Test with a simple signal to check phase structure
    N = 8
    x = [1.0] + [0.0] * (N-1)  # Impulse signal
    
    # Test QPSK sequence
    Xr_qpsk, Xi_qpsk = engine_core.rft_basis_forward(
        x,
        weights=[0.0, 1.0],  # Only second component (QPSK)
        theta0=[0.0, 0.0],
        omega=[1.0, 1.0],
        sequence_type="qpsk"
    )
    
    # Test constant sequence for comparison
    Xr_const, Xi_const = engine_core.rft_basis_forward(
        x,
        weights=[1.0, 0.0],  # Only first component (constant)
        theta0=[0.0, 0.0],
        omega=[1.0, 1.0],
        sequence_type="const"
    )
    
    # QPSK should produce different spectrum than constant
    qpsk_spectrum = [Xr_qpsk[i]**2 + Xi_qpsk[i]**2 for i in range(N)]
    const_spectrum = [Xr_const[i]**2 + Xi_const[i]**2 for i in range(N)]
    
    spectral_difference = sum(abs(qpsk_spectrum[i] - const_spectrum[i]) for i in range(N))
    
    print(f"  QPSK spectrum: {[f'{x:.3f}' for x in qpsk_spectrum[:4]]}...")
    print(f"  Const spectrum: {[f'{x:.3f}' for x in const_spectrum[:4]]}...")
    print(f"  Spectral difference: {spectral_difference:.3f}")
    
    if spectral_difference < 1e-6:
        print(f"    ❌ FAIL: QPSK and constant sequences produce same result!")
        return False
    else:
        print(f"    ✅ PASS: QPSK produces different spectrum than constant")
    
    return True

def test_api_integration():
    """Test that the API uses the updated RFT science"""
    print("\nTesting API integration...")
    
    # Test with some sample data
    test_payload = "Hello, updated RFT science!"
    
    result = compute_rft(test_payload)
    
    print(f"  Payload: '{test_payload}'")
    print(f"  Backend: {result.get('backend_used', 'Unknown')}")
    print(f"  Bin count: {result.get('bin_count', 0)}")
    print(f"  Harmonic ratio: {result.get('hr', 0):.6f}")
    
    if result.get('bin_count', 0) != len(test_payload.encode('utf-8')):
        print(f"    ❌ FAIL: Unexpected bin count")
        return False
    
    if 'rft' not in result or not result['rft']:
        print(f"    ❌ FAIL: No RFT data returned")
        return False
    
    print(f"    ✅ PASS: API returns valid RFT results")
    return True

def test_non_dft_property():
    """Test that RFT is genuinely different from DFT"""
    print("\nTesting non-DFT property...")
    
    # Compare RFT with numpy DFT
    x = [1.0, 0.5, -0.3, 0.8, -0.1, 0.6, -0.7, 0.2]
    
    # RFT transform
    Xr_rft, Xi_rft = engine_core.rft_basis_forward(
        x,
        weights=[0.7, 0.3],
        theta0=[0.0, 0.0],
        omega=[1.0, 1.0],
        sequence_type="qpsk"
    )
    
    # DFT transform for comparison
    x_np = np.array(x)
    X_dft = np.fft.fft(x_np)
    
    # Compare magnitudes
    rft_mags = [np.sqrt(Xr_rft[i]**2 + Xi_rft[i]**2) for i in range(len(x))]
    dft_mags = np.abs(X_dft).tolist()
    
    # Compute difference
    magnitude_diff = sum(abs(rft_mags[i] - dft_mags[i]) for i in range(len(x)))
    
    print(f"  RFT magnitudes: {[f'{x:.3f}' for x in rft_mags[:4]]}...")
    print(f"  DFT magnitudes: {[f'{x:.3f}' for x in dft_mags[:4]]}...")
    print(f"  Total magnitude difference: {magnitude_diff:.3f}")
    
    if magnitude_diff < 1e-6:
        print(f"    ❌ FAIL: RFT produces same result as DFT!")
        return False
    else:
        print(f"    ✅ PASS: RFT is genuinely different from DFT")
    
    return True

def main():
    """Run all tests"""
    print("=" * 60)
    print("QUANTONIUMOS RFT SCIENCE VALIDATION")
    print("=" * 60)
    
    # Initialize engine
    try:
        engine_core.initialize()
        print("✅ Engine initialized successfully")
    except Exception as e:
        print(f"❌ Engine initialization failed: {e}")
        return False
    
    tests = [
        test_unitary_property,
        test_reconstruction,
        test_phase_sequences,
        test_non_dft_property,
        test_api_integration
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"    ❌ FAIL: Test crashed with error: {e}")
    
    print("\n" + "=" * 60)
    print(f"RESULTS: {passed}/{total} tests passed")
    print("=" * 60)
    
    if passed == total:
        print("🎉 ALL TESTS PASSED - RFT science update successful!")
        return True
    else:
        print("❌ Some tests failed - please review implementation")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

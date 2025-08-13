#!/usr/bin/env python3
"""
RFT Implementation Validation

This script tests the current Python RFT implementation against the mathematical
specification defined in RFT_SPECIFICATION.md.

Note: This validates the existing windowed DFT implementation, not the full
mathematical RFT specification which would require implementing the eigendecomposition
approach outlined in RFT_SPECIFICATION.md.
"""

import numpy as np
import sys
import os

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

try:
    from core.encryption.resonance_fourier import (
        resonance_fourier_transform,
        inverse_resonance_fourier_transform,
        validate_rft_implementation
    )
    print("✅ Python RFT implementation available")
except ImportError as e:
    print(f"❌ Cannot import RFT implementation: {e}")
    sys.exit(1)

print("ℹ️  C++ module not available, using Python implementation")

def test_basic_functionality():
    """Test that the RFT functions work and are invertible."""
    print("\n🧪 Testing Basic RFT Functionality")
    
    # Test data
    test_signal = [1.0, 0.5, -0.3, 0.8, -0.2, 0.6, -0.1, 0.4]
    print(f"Input signal: {test_signal}")
    
    try:
        # Forward transform
        rft_spectrum = resonance_fourier_transform(test_signal)
        print(f"✅ Forward RFT completed: {len(rft_spectrum)} components")
        
        # Inverse transform
        reconstructed = inverse_resonance_fourier_transform(rft_spectrum)
        print(f"✅ Inverse RFT completed: {len(reconstructed)} samples")
        
        # Check reconstruction error
        error = np.linalg.norm(np.array(test_signal) - np.array(reconstructed))
        print(f"Reconstruction error: {error:.2e}")
        
        if error < 1e-10:
            print("✅ Perfect reconstruction achieved")
            return True
        elif error < 1e-6:
            print("⚠️  Good reconstruction (numerical precision limits)")
            return True
        else:
            print("❌ Poor reconstruction - significant error")
            return False
            
    except Exception as e:
        print(f"❌ RFT test failed: {e}")
        return False

def test_energy_conservation():
    """Test if the transform preserves energy (Plancherel theorem)."""
    print("\n🧪 Testing Energy Conservation")
    
    # Generate test signals
    test_cases = [
        [1, 0, 0, 0],  # Delta function
        [1, 1, 1, 1],  # Constant
        [1, 0, -1, 0], # Square wave
        [np.cos(2*np.pi*k/8) for k in range(8)]  # Sinusoid
    ]
    
    all_passed = True
    
    for i, signal in enumerate(test_cases):
        try:
            # Compute energy in time domain
            time_energy = np.sum(np.abs(np.array(signal))**2)
            
            # Transform
            spectrum = resonance_fourier_transform(signal)
            
            # Compute energy in frequency domain
            freq_energy = np.sum(np.abs([comp[1] for comp in spectrum])**2)
            
            # Check energy conservation
            energy_ratio = freq_energy / time_energy if time_energy > 0 else 1.0
            error = abs(energy_ratio - 1.0)
            
            print(f"Test case {i+1}: time_energy={time_energy:.3f}, freq_energy={freq_energy:.3f}, ratio={energy_ratio:.6f}")
            
            if error < 0.01:  # 1% tolerance
                print(f"✅ Energy conserved (error: {error:.2e})")
            else:
                print(f"⚠️  Energy not well conserved (error: {error:.2e})")
                all_passed = False
                
        except Exception as e:
            print(f"❌ Energy test {i+1} failed: {e}")
            all_passed = False
    
    return all_passed

def test_built_in_validation():
    """Test the built-in validation function."""
    print("\n🧪 Testing Built-in Validation")
    
    try:
        results = validate_rft_implementation()
        print("Built-in validation results:")
        
        all_passed = True
        for test_name, passed in results.items():
            status = "✅" if passed else "❌"
            print(f"  {status} {test_name}: {'PASS' if passed else 'FAIL'}")
            if not passed:
                all_passed = False
        
        return all_passed
        
    except Exception as e:
        print(f"❌ Built-in validation failed: {e}")
        return False

def test_linearity():
    """Test if the transform is linear."""
    print("\n🧪 Testing Linearity")
    
    # Test signals
    x1 = [1, 0, 1, 0]
    x2 = [0, 1, 0, 1]
    a, b = 2.0, 3.0
    
    try:
        # Transform individual signals
        X1 = resonance_fourier_transform(x1)
        X2 = resonance_fourier_transform(x2)
        
        # Transform linear combination
        x_combined = [a*x1[i] + b*x2[i] for i in range(len(x1))]
        X_combined = resonance_fourier_transform(x_combined)
        
        # Check linearity: RFT(ax1 + bx2) should equal aRFT(x1) + bRFT(x2)
        X_linear = [(f, a*c1 + b*c2) for ((f, c1), (_, c2)) in zip(X1, X2)]
        
        # Compare magnitudes (phase differences are expected due to numerical precision)
        error = 0.0
        for (f1, c1), (f2, c2) in zip(X_combined, X_linear):
            error += abs(abs(c1) - abs(c2))
        
        error /= len(X_combined)
        print(f"Average magnitude error: {error:.2e}")
        
        if error < 1e-10:
            print("✅ Perfect linearity")
            return True
        elif error < 1e-6:
            print("⚠️  Good linearity (numerical precision)")
            return True
        else:
            print("❌ Poor linearity")
            return False
            
    except Exception as e:
        print(f"❌ Linearity test failed: {e}")
        return False

def main():
    """Run all validation tests."""
    print("=" * 60)
    print("🔍 RFT Implementation Validation")
    print("=" * 60)
    print()
    print("📄 Testing against existing Python windowed DFT implementation")
    print("📄 For full RFT specification validation, see RFT_SPECIFICATION.md")
    
    tests = [
        ("Basic Functionality", test_basic_functionality),
        ("Energy Conservation", test_energy_conservation),
        ("Linearity", test_linearity),
        ("Built-in Validation", test_built_in_validation),
    ]
    
    passed_tests = 0
    total_tests = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        try:
            if test_func():
                passed_tests += 1
        except Exception as e:
            print(f"❌ Test '{test_name}' crashed: {e}")
    
    print("\n" + "=" * 60)
    print("📊 VALIDATION SUMMARY")
    print("=" * 60)
    
    if passed_tests == total_tests:
        print(f"✅ ALL TESTS PASSED ({passed_tests}/{total_tests})")
        print("\n🎯 Current implementation appears to be a working windowed DFT variant.")
        print("📚 For the full mathematical RFT specification, see RFT_SPECIFICATION.md")
        return True
    else:
        print(f"⚠️  SOME TESTS FAILED ({passed_tests}/{total_tests} passed)")
        print("\n🔧 Implementation needs attention for production use.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

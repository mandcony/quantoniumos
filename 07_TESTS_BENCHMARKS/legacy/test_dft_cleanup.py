#!/usr/bin/env python3
"""
Test script to verify DFT cleanup and True RFT operation
"""

import os
import numpy as np
import importlib.util

def test_no_dft_in_main_module():
    """
    Test that the main resonance_fourier.py module contains no DFT references.
    """
    print("🧹 Testing: No DFT references in main module")
    
    resonance_fourier_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                                        "core/encryption/resonance_fourier.py")
    
    if not os.path.exists(resonance_fourier_path):
        print("⚠️ Warning: Could not find resonance_fourier.py")
        return False
        
    with open(resonance_fourier_path, 'r') as f:
        content = f.read()
        
    dft_keywords = ['windowed DFT', 'standard DFT', 'dft_kernel', 'exp(-2j * np.pi']
    found_dft = []
    
    for keyword in dft_keywords:
        if keyword.lower() in content.lower():
            found_dft.append(keyword)
            
    if found_dft:
        print(f"❌ Found DFT references: {found_dft}")
        return False
    else:
        print("✅ No DFT references found in main module")
        return True

def test_true_rft_works():
    """
    Test that True RFT implementation works correctly.
    """
    print("\n🔬 Testing: True RFT functionality")
    
    try:
        # Load the canonical_true_rft module
        spec = importlib.util.spec_from_file_location(
            "canonical_true_rft", 
            os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                        "04_RFT_ALGORITHMS/canonical_true_rft.py")
        )
        canonical_true_rft = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(canonical_true_rft)
        
        forward_true_rft = canonical_true_rft.forward_true_rft
        inverse_true_rft = canonical_true_rft.inverse_true_rft
        
        # Simple test case
        data = np.random.random(16)
        transformed = forward_true_rft(data)
        restored = inverse_true_rft(transformed)
        
        # Check round-trip accuracy
        error = np.mean(np.abs(data - restored))
        print(f"Round-trip error: {error:.10f}")
        
        if error < 1e-10:
            print("✅ True RFT round-trip test passed")
            return True
        else:
            print("❌ True RFT round-trip test failed")
            return False
            
    except Exception as e:
        print(f"❌ Error testing True RFT: {str(e)}")
        return False

if __name__ == "__main__":
    test_no_dft_in_main_module()
    test_true_rft_works()

        # Legacy wrapper maintained for: resonance_fourier_transform, inverse_resonance_fourier_transform

        # Test signal signal = [1.0, 2.0, 3.0, 4.0, 0.5, 1.5]

        # Forward transform forward_result = resonance_fourier_transform(signal)
        print(f"✅ Forward RFT: {len(forward_result)} frequency components")

        # Inverse transform inverse_result = inverse_resonance_fourier_transform(forward_result)
        print(f"✅ Inverse RFT: reconstructed {len(inverse_result)} samples")

        # Check reconstruction quality
        if len(inverse_result) >= len(signal): error = np.mean(np.abs(np.array(inverse_result[:len(signal)]) - np.array(signal)))
        print(f"✅ Reconstruction error: {error:.6f}")
        return error < 0.1
        else:
        print("⚠️ Reconstructed signal too short")
        return False except Exception as e:
        print(f"❌ True RFT test failed: {e}")
        return False
def test_duplicate_files_removed(): """
        Test that duplicate files have been removed.
"""
"""
        print("\n📁 Testing: No duplicate files")
import os
import glob

        # Find all resonance_fourier files patterns = [ '/workspaces/quantoniumos/**/resonance_fourier_*.py', '/workspaces/quantoniumos/**/resonance_fourier_old.py', '/workspaces/quantoniumos/**/resonance_fourier_clean.py' ] duplicates = []
        for pattern in patterns: matches = glob.glob(pattern, recursive=True)
        if matches: duplicates.extend(matches)

        # Filter out test_env_local (that's expected) duplicates = [f
        for f in duplicates if 'test_env_local' not in f]
        if duplicates:
        print(f"❌ Found duplicate files: {duplicates}")
        return False
        else:
        print("✅ No duplicate files found")
        return True

if __name__ == "__main__":
print(" QuantoniumOS DFT Cleanup Verification")
print("=" * 50) results = [] results.append(test_no_dft_in_main_module()) results.append(test_true_rft_works()) results.append(test_duplicate_files_removed())
print("||n" + "=" * 50) passed = sum(results) total = len(results)
if passed == total:
print(f"🎉 ALL TESTS PASSED ({passed}/{total})")
print("✅ DFT cleanup complete - only True RFT remains!")
else:
print(f"⚠️ {passed}/{total} tests passed")
print("❌ Some issues remain - see results above")
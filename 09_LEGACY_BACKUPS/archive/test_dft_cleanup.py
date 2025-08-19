||#!/usr/bin/env python3
""""""
Test script to verify DFT cleanup and True RFT operation
"""
"""

def test_no_dft_in_main_module():
"""
"""
        Test that the main resonance_fourier.py module contains no DFT references.
"""
"""
        print("🧹 Testing: No DFT references in main module") with open('/workspaces/quantoniumos/core/encryption/resonance_fourier.py', 'r') as f: content = f.read() dft_keywords = ['windowed DFT', 'standard DFT', 'dft_kernel', 'exp(-2j * np.pi'] found_dft = []
        for keyword in dft_keywords:
        if keyword.lower() in content.lower(): found_dft.append(keyword)
        if found_dft:
        print(f"❌ Found DFT references: {found_dft}")
        return False
        else:
        print("✅ No DFT references found in main module")
        return True
def test_true_rft_works(): """"""
        Test that True RFT implementation works correctly.
"""
"""
        print("\n🔬 Testing: True RFT functionality")
        try: from canonical_true_rft
import forward_true_rft, inverse_true_rft

        # Legacy wrapper maintained for: resonance_fourier_transform, inverse_resonance_fourier_transform
import numpy as np

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
def test_duplicate_files_removed(): """"""
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
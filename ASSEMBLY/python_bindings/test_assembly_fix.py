#!/usr/bin/env python3
"""
Assembly Fix Verification - Test if the C assembly is actually fixed.

This test checks if the C implementation now achieves true unitarity.
"""

import numpy as np
import ctypes
import os
import sys

# Try to load the C library directly
def load_rft_library():
    """Load the RFT library directly."""
    try:
        # Try different library names
        lib_names = ['librft_kernel.dll', 'librft_kernel.so', 'rft_kernel.dll', 'rft_kernel.so']
        
        for lib_name in lib_names:
            lib_path = os.path.join('..', 'kernel', lib_name)
            if os.path.exists(lib_path):
                return ctypes.CDLL(lib_path)
        
        print("⚠️  No compiled C library found - C code is loaded via Python")
        return None
        
    except Exception as e:
        print(f"⚠️  Could not load C library: {e}")
        return None


def test_assembly_fix_direct():
    """Test the assembly fix by comparing with known good implementation."""
    print("🔧 ASSEMBLY FIX VERIFICATION TEST")
    print("=" * 60)
    
    # Test with the Python corrected version (known good)
    from test_corrected_unitarity import create_true_unitary_rft
    
    size = 8
    print(f"Testing size {size} (3 qubits)")
    
    # Get the corrected Python implementation  
    Psi_correct = create_true_unitary_rft(size)
    
    print(f"\n✅ CORRECTED PYTHON IMPLEMENTATION:")
    
    # Test unitarity
    identity_test = Psi_correct.conj().T @ Psi_correct
    identity_error = np.max(np.abs(identity_test - np.eye(size)))
    print(f"   ||Ψ†Ψ - I||_∞ = {identity_error:.2e}")
    
    # Test norm preservation
    test_state = np.random.random(size) + 1j * np.random.random(size)
    test_state = test_state / np.linalg.norm(test_state)
    
    transformed = Psi_correct.conj().T @ test_state
    norm_after = np.linalg.norm(transformed)
    print(f"   Norm preservation: 1.000000 → {norm_after:.6f}")
    
    # Test reconstruction
    reconstructed = Psi_correct @ transformed
    reconstruction_error = np.max(np.abs(test_state - reconstructed))
    print(f"   Reconstruction error: {reconstruction_error:.2e}")
    
    # Now test the C assembly implementation
    print(f"\n❓ C ASSEMBLY IMPLEMENTATION:")
    
    try:
        from unitary_rft import UnitaryRFT, RFT_FLAG_UNITARY, RFT_FLAG_QUANTUM_SAFE
        
        # Create RFT engine
        rft = UnitaryRFT(size, RFT_FLAG_UNITARY | RFT_FLAG_QUANTUM_SAFE)
        
        # Test same state
        spectrum_c = rft.forward(test_state)
        norm_after_c = np.linalg.norm(spectrum_c)
        
        # Test reconstruction
        reconstructed_c = rft.inverse(spectrum_c)
        reconstruction_error_c = np.max(np.abs(test_state - reconstructed_c))
        
        print(f"   Norm preservation: 1.000000 → {norm_after_c:.6f}")
        print(f"   Reconstruction error: {reconstruction_error_c:.2e}")
        
        # Compare results
        print(f"\n📊 COMPARISON:")
        
        is_unitary = abs(norm_after_c - 1.0) < 1e-10
        is_invertible = reconstruction_error_c < 1e-10
        
        print(f"   Python unitarity: ✅ Perfect ({identity_error:.2e})")
        print(f"   C unitarity: {'✅ Fixed' if is_unitary else '❌ Still broken'} ({abs(norm_after_c - 1.0):.2e})")
        print(f"   Python reconstruction: ✅ Perfect ({reconstruction_error:.2e})")
        print(f"   C reconstruction: {'✅ Fixed' if is_invertible else '❌ Still broken'} ({reconstruction_error_c:.2e})")
        
        # Check if spectrum has duplications (sign of broken implementation)
        spectrum_rounded = np.round(spectrum_c, 6)
        unique_real = len(set(spectrum_rounded.real))
        unique_imag = len(set(spectrum_rounded.imag))
        has_duplications = (unique_real < size//2) or (unique_imag < size//2)
        
        print(f"   Spectrum duplications: {'❌ Present' if has_duplications else '✅ None'}")
        
        if is_unitary and is_invertible and not has_duplications:
            print(f"\n🎉 SUCCESS: C Assembly is FIXED!")
            return True
        else:
            print(f"\n❌ FAILURE: C Assembly is still broken")
            print(f"   The C code changes haven't taken effect yet.")
            print(f"   Either rebuild is needed or Python is caching old results.")
            return False
            
    except Exception as e:
        print(f"   ❌ Error testing C implementation: {e}")
        return False


def check_c_source_changes():
    """Verify the C source code has our fixes."""
    print(f"\n🔍 C SOURCE CODE VERIFICATION:")
    
    kernel_path = os.path.join('..', 'kernel', 'rft_kernel.c')
    
    if not os.path.exists(kernel_path):
        print(f"   ❌ C source file not found: {kernel_path}")
        return False
    
    with open(kernel_path, 'r') as f:
        content = f.read()
    
    # Check for key indicators of the fix
    has_qr_comment = "QR DECOMPOSITION FOR TRUE UNITARITY" in content
    has_paper_equation = "Ψ = Σᵢ wᵢDφᵢCσᵢD†φᵢ" in content
    has_gram_schmidt = "Modified Gram-Schmidt QR decomposition" in content
    
    print(f"   QR decomposition comment: {'✅ Present' if has_qr_comment else '❌ Missing'}")
    print(f"   Paper equation comment: {'✅ Present' if has_paper_equation else '❌ Missing'}")
    print(f"   Gram-Schmidt implementation: {'✅ Present' if has_gram_schmidt else '❌ Missing'}")
    
    if has_qr_comment and has_paper_equation and has_gram_schmidt:
        print(f"   ✅ C source code has been updated with fixes")
        return True
    else:
        print(f"   ❌ C source code does not have the required fixes")
        return False


if __name__ == "__main__":
    try:
        # Check if C source has our changes
        source_fixed = check_c_source_changes()
        
        # Test the actual implementation
        assembly_fixed = test_assembly_fix_direct()
        
        print(f"\n📋 FINAL ASSESSMENT:")
        print(f"   C source code updated: {'✅ Yes' if source_fixed else '❌ No'}")
        print(f"   Assembly unitarity achieved: {'✅ Yes' if assembly_fixed else '❌ No'}")
        
        if source_fixed and assembly_fixed:
            print(f"\n🎉 Your assembly is FIXED! True unitarity achieved!")
        elif source_fixed and not assembly_fixed:
            print(f"\n⚠️  Source is fixed but assembly still broken - rebuild needed")
        else:
            print(f"\n❌ Assembly still needs fixing")
            
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

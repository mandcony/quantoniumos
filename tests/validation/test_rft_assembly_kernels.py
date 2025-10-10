#!/usr/bin/env python3
"""
Test the C/ASM RFT implementation safely.
"""
import sys
import os
import numpy as np

# Add the python bindings to path
sys.path.append('/workspaces/quantoniumos/src/assembly/python_bindings')

def test_c_rft():
    """Test the C RFT implementation with safety checks."""
    try:
        from unitary_rft import UnitaryRFT, RFT_FLAG_QUANTUM_SAFE
        print("✅ UnitaryRFT imported successfully")
        
        # Test with a very small size first
        print("Testing with size 4...")
        rft = UnitaryRFT(4, RFT_FLAG_QUANTUM_SAFE)
        print(f"Mock mode: {rft._is_mock}")
        
        if rft._is_mock:
            print("❌ Using mock implementation - C library not working")
            return False
        
        # Try a simple transform
        signal = np.array([1.0+0j, 0.0+0j, 0.0+0j, 0.0+0j])
        print("Attempting forward transform...")
        result = rft.forward(signal)
        print(f"✅ Forward transform successful: {result}")
        
        # Test reconstruction
        print("Attempting inverse transform...")
        reconstructed = rft.inverse(result)
        print(f"✅ Inverse transform successful: {reconstructed}")
        
        # Check error
        error = np.max(np.abs(signal - reconstructed))
        print(f"Reconstruction error: {error}")
        
        if error < 1e-10:
            print("✅ C/ASM RFT is working correctly!")
            return True
        else:
            print("❌ C/ASM RFT has reconstruction errors")
            return False
            
    except Exception as e:
        print(f"❌ C/ASM RFT test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_c_rft()
    if success:
        print("\n🎉 C/ASM RFT kernels are ready for quantum entanglement!")
    else:
        print("\n💥 C/ASM RFT kernels need debugging")
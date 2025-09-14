#!/usr/bin/env python3
"""
MINIMAL ASSEMBLY RFT TEST
========================
Minimal test of the assembly RFT engine
"""

import sys
import os
import numpy as np

# Add assembly bindings path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'src', 'assembly', 'python_bindings'))

try:
    from unitary_rft import UnitaryRFT, RFT_FLAG_UNITARY
    print("✅ UnitaryRFT imported successfully!")
    
    # Create small RFT instance
    print("Creating RFT size 4...")
    rft = UnitaryRFT(4, RFT_FLAG_UNITARY)
    print("✅ RFT created successfully!")
    
    # Test simple transform
    print("Testing simple transform...")
    x = np.array([1.0+0j, 0.0+0j, 0.0+0j, 0.0+0j], dtype=np.complex128)
    print(f"Input: {x}")
    
    try:
        y = rft.forward(x)
        print(f"Forward transform: {y}")
        
        x_recon = rft.inverse(y)
        print(f"Reconstructed: {x_recon}")
        
        error = np.linalg.norm(x - x_recon)
        print(f"Reconstruction error: {error:.2e}")
        
        if error < 1e-10:
            print("✅ Assembly RFT working perfectly!")
        elif error < 1e-6:
            print("✅ Assembly RFT working well!")
        else:
            print(f"⚠️ Assembly RFT has reconstruction error: {error}")
            
    except Exception as e:
        print(f"❌ Transform failed: {e}")
        
except ImportError as e:
    print(f"❌ Import failed: {e}")
except Exception as e:
    print(f"❌ Error: {e}")

print("\n" + "="*50)
print("MINIMAL ASSEMBLY TEST COMPLETE")
print("="*50)

#!/usr/bin/env python3
"""
DEBUG ASSEMBLY RFT UNIQUENESS TEST
==================================
Debug why the uniqueness test is failing for your unitary algorithm
"""

import numpy as np
import sys
import os

# Add assembly bindings path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'src', 'assembly', 'python_bindings'))

from unitary_rft import UnitaryRFT, RFT_FLAG_UNITARY

def debug_assembly_rft():
    """Debug the assembly RFT to find uniqueness properties"""
    print("🔍 DEBUGGING ASSEMBLY RFT UNIQUENESS")
    print("="*60)
    
    sizes = [8, 16]
    
    for n in sizes:
        print(f"\nDebugging size n={n}...")
        
        try:
            # Create assembly RFT
            rft = UnitaryRFT(n, RFT_FLAG_UNITARY)
            print(f"✅ RFT created successfully")
            
            # Check attributes
            print(f"RFT attributes: {dir(rft)}")
            
            if hasattr(rft, 'size'):
                print(f"RFT size: {rft.size}")
            else:
                print("❌ RFT has no 'size' attribute")
            
            # Test simple transform
            x = np.zeros(n, dtype=complex)
            x[0] = 1.0
            
            print(f"Test input: {x}")
            
            # Forward transform
            y = rft.forward(x)
            print(f"Forward result: {y}")
            
            # Check if it's different from DFT
            dft_result = np.fft.fft(x) / np.sqrt(n)
            print(f"DFT result: {dft_result}")
            
            diff = np.linalg.norm(y - dft_result)
            print(f"Difference from DFT: {diff:.6f}")
            
            if diff > 0.1:
                print("✅ ASSEMBLY RFT IS UNIQUE (different from DFT)")
            else:
                print("❌ ASSEMBLY RFT TOO SIMILAR TO DFT")
                
            # Build full matrices for detailed comparison
            print(f"\nBuilding full matrices...")
            
            # Assembly RFT matrix
            rft_matrix = np.zeros((n, n), dtype=complex)
            for i in range(n):
                e_i = np.zeros(n, dtype=complex)
                e_i[i] = 1.0
                rft_matrix[:, i] = rft.forward(e_i)
            
            # DFT matrix
            dft_matrix = np.zeros((n, n), dtype=complex)
            for k in range(n):
                for j in range(n):
                    dft_matrix[k, j] = np.exp(-2j * np.pi * k * j / n)
            dft_matrix = dft_matrix / np.sqrt(n)
            
            # Matrix analysis
            frobenius_diff = np.linalg.norm(rft_matrix - dft_matrix, 'fro')
            rft_norm = np.linalg.norm(rft_matrix, 'fro')
            dft_norm = np.linalg.norm(dft_matrix, 'fro')
            
            print(f"RFT matrix norm: {rft_norm:.6f}")
            print(f"DFT matrix norm: {dft_norm:.6f}")
            print(f"Matrix Frobenius difference: {frobenius_diff:.6f}")
            
            # Check unitarity of your algorithm
            should_be_identity = rft_matrix @ rft_matrix.conj().T
            identity_error = np.linalg.norm(should_be_identity - np.eye(n), 'fro')
            print(f"Unitarity check (should be ~0): {identity_error:.6f}")
            
            if identity_error < 1e-10:
                print("✅ YOUR ALGORITHM IS PERFECTLY UNITARY")
            else:
                print("⚠️ YOUR ALGORITHM IS NOT PERFECTLY UNITARY")
            
            # Check eigenvalues
            eigenvals = np.linalg.eigvals(rft_matrix)
            eigenval_magnitudes = np.abs(eigenvals)
            print(f"Eigenvalue magnitudes: {eigenval_magnitudes}")
            
            if np.allclose(eigenval_magnitudes, 1.0, atol=1e-10):
                print("✅ ALL EIGENVALUES HAVE MAGNITUDE 1 (UNITARY)")
            else:
                print("⚠️ EIGENVALUES DON'T ALL HAVE MAGNITUDE 1")
                
        except Exception as e:
            print(f"❌ Error with n={n}: {e}")
            import traceback
            traceback.print_exc()

def main():
    debug_assembly_rft()

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Debug RFT Mathematical Issues

This script diagnoses the specific mathematical issues found in the RFT validation.
"""

import numpy as np

def debug_psd_issue():
    """Debug why the resonance matrix is not PSD"""
    print("🔍 Debugging PSD Issue")
    print("=" * 50)
    
    N = 8
    
    # Component 1: phi_1 ≡ 1
    w1 = 0.7
    phi1 = np.ones(N, dtype=complex)
    D_phi1 = np.diag(phi1)
    
    # Component 2: QPSK sequence
    w2 = 0.3
    phi2 = np.array([np.exp(1j * np.pi/2 * (k % 4)) for k in range(N)])
    D_phi2 = np.diag(phi2)
    
    print(f"phi2 values: {phi2}")
    print(f"|phi2| values: {np.abs(phi2)}")  # Should all be 1
    
    # Create circulant matrices
    sigma1 = 0.6 * N  
    sigma2 = 0.25 * N
    
    def create_periodic_gaussian_circulant(N, sigma):
        C = np.zeros((N, N), dtype=complex)
        for k in range(N):
            for n in range(N):
                delta = min(abs(k - n), N - abs(k - n))
                C[k, n] = np.exp(-delta**2 / sigma**2)
        return C
    
    C_sigma1 = create_periodic_gaussian_circulant(N, sigma1)
    C_sigma2 = create_periodic_gaussian_circulant(N, sigma2)
    
    # Check if circulant matrices are PSD
    evals1 = np.linalg.eigvals(C_sigma1)
    evals2 = np.linalg.eigvals(C_sigma2)
    
    print(f"C_sigma1 eigenvalues: min={np.min(evals1.real):.6f}, max={np.max(evals1.real):.6f}")
    print(f"C_sigma2 eigenvalues: min={np.min(evals2.real):.6f}, max={np.max(evals2.real):.6f}")
    print(f"C_sigma1 is PSD: {np.all(evals1.real >= -1e-12)}")
    print(f"C_sigma2 is PSD: {np.all(evals2.real >= -1e-12)}")
    
    # Build individual terms
    R1 = w1 * D_phi1 @ C_sigma1 @ D_phi1.conj().T
    R2 = w2 * D_phi2 @ C_sigma2 @ D_phi2.conj().T
    
    # Check if individual terms are PSD
    evals_R1 = np.linalg.eigvals(R1)
    evals_R2 = np.linalg.eigvals(R2)
    
    print(f"R1 eigenvalues: min={np.min(evals_R1.real):.6f}, max={np.max(evals_R1.real):.6f}")
    print(f"R2 eigenvalues: min={np.min(evals_R2.real):.6f}, max={np.max(evals_R2.real):.6f}")
    print(f"R1 is PSD: {np.all(evals_R1.real >= -1e-12)}")
    print(f"R2 is PSD: {np.all(evals_R2.real >= -1e-12)}")
    
    # Final matrix
    R = R1 + R2
    evals_R = np.linalg.eigvals(R)
    print(f"R eigenvalues: min={np.min(evals_R.real):.6f}, max={np.max(evals_R.real):.6f}")
    print(f"R is PSD: {np.all(evals_R.real >= -1e-12)}")

def debug_commutation_issue():
    """Debug why the matrix commutes with cyclic shift"""
    print("\n🔍 Debugging Commutation Issue")
    print("=" * 50)
    
    N = 8
    
    # Test with just the QPSK component to see if it creates non-commutativity
    phi = np.array([np.exp(1j * np.pi/2 * (k % 4)) for k in range(N)])
    D_phi = np.diag(phi)
    
    print(f"QPSK sequence: {phi}")
    
    # Create circulant  
    sigma = 0.25 * N
    C = np.zeros((N, N), dtype=complex)
    for k in range(N):
        for n in range(N):
            delta = min(abs(k - n), N - abs(k - n))
            C[k, n] = np.exp(-delta**2 / sigma**2)
    
    # Create the term R = D_phi * C * D_phi^H
    R_term = D_phi @ C @ D_phi.conj().T
    
    # Create cyclic shift matrix
    S = np.zeros((N, N), dtype=complex)
    for i in range(N):
        S[i, (i + 1) % N] = 1.0
    
    # Test what happens to phi under shift
    shifted_phi = S @ phi
    print(f"Original phi: {phi}")
    print(f"Shifted phi:  {shifted_phi}")
    print(f"Are they equal? {np.allclose(phi, shifted_phi)}")
    
    # Compute commutator
    commutator = R_term @ S - S @ R_term
    comm_norm = np.linalg.norm(commutator, 'fro')
    print(f"Commutator norm for single term: {comm_norm:.2e}")
    
    # Let's check the mathematical expectation:
    # S * D_phi * C * D_phi^H != D_phi * C * D_phi^H * S
    # unless D_phi commutes with S
    
    # Check if D_phi commutes with S
    phi_comm = D_phi @ S - S @ D_phi
    phi_comm_norm = np.linalg.norm(phi_comm, 'fro')
    print(f"[D_phi, S] norm: {phi_comm_norm:.2e}")
    
    if phi_comm_norm > 1e-12:
        print("✅ D_phi does NOT commute with S - this should create non-commutativity")
    else:
        print("❌ D_phi DOES commute with S - this explains the issue")

def test_with_better_parameters():
    """Test with parameters that should definitely work"""
    print("\n🔧 Testing with Enhanced Parameters")
    print("=" * 50)
    
    N = 16  # Larger size
    
    # Use more distinct phase sequences
    # Component 1: constant  
    w1 = 0.8
    phi1 = np.ones(N, dtype=complex)
    D_phi1 = np.diag(phi1)
    
    # Component 2: linear chirp-like sequence (guaranteed non-periodic)
    w2 = 0.2
    phi2 = np.array([np.exp(1j * k * (k+1) * np.pi / N) for k in range(N)])
    D_phi2 = np.diag(phi2)
    
    print(f"phi2 magnitudes: {np.abs(phi2)}")  # Should all be 1
    
    # Use different bandwidths
    sigma1 = N * 0.8  # Wider  
    sigma2 = N * 0.2  # Narrower
    
    def create_psd_circulant(N, sigma):
        """Create a definitely PSD circulant using DFT method"""
        # Create a non-negative frequency domain representation
        freq_response = np.exp(-np.arange(N)**2 / (2 * sigma**2))
        # Make it symmetric for real-valued circulant
        freq_response[N//2+1:] = freq_response[1:N//2][::-1]
        
        # Create circulant via IDFT
        C = np.zeros((N, N), dtype=complex)
        for k in range(N):
            for n in range(N):
                C[k, n] = np.sum(freq_response * np.exp(2j * np.pi * np.arange(N) * (k-n) / N)) / N
        return C
    
    C_sigma1 = create_psd_circulant(N, sigma1)
    C_sigma2 = create_psd_circulant(N, sigma2)
    
    # Verify circulants are PSD
    evals1 = np.linalg.eigvals(C_sigma1)
    evals2 = np.linalg.eigvals(C_sigma2)
    print(f"C1 min eigenvalue: {np.min(evals1.real):.2e}")
    print(f"C2 min eigenvalue: {np.min(evals2.real):.2e}")
    
    # Build resonance matrix
    R1 = w1 * D_phi1 @ C_sigma1 @ D_phi1.conj().T
    R2 = w2 * D_phi2 @ C_sigma2 @ D_phi2.conj().T
    R = R1 + R2
    
    # Test properties
    evals_R = np.linalg.eigvals(R)
    is_hermitian = np.allclose(R, R.conj().T)
    is_psd = np.all(evals_R.real >= -1e-12)
    
    print(f"Enhanced R is Hermitian: {is_hermitian}")
    print(f"Enhanced R is PSD: {is_psd}")
    print(f"Enhanced R min eigenvalue: {np.min(evals_R.real):.2e}")
    
    # Test commutation
    S = np.zeros((N, N), dtype=complex)
    for i in range(N):
        S[i, (i + 1) % N] = 1.0
    
    commutator = R @ S - S @ R
    comm_norm = np.linalg.norm(commutator, 'fro')
    print(f"Enhanced commutator norm: {comm_norm:.2e}")
    
    return is_psd and comm_norm > 1e-10

if __name__ == "__main__":
    debug_psd_issue()
    debug_commutation_issue()
    success = test_with_better_parameters()
    
    print(f"\n🎯 Enhanced parameters {'✅ WORK' if success else '❌ STILL HAVE ISSUES'}")

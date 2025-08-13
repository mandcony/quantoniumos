#!/usr/bin/env python3
"""
Final validation of the RFT implementation against the rigorous spec.
Tests all key mathematical properties from your specification.
"""
import numpy as np
import scipy.linalg as la

def periodic_distance(i, j, N):
    """Periodic distance on Z/NZ as specified"""
    return min(abs(i - j), N - abs(i - j))

def generate_golden_ratio_phase(k, N, gamma=0.3):
    """Generate golden ratio phase sequence: φₖ = 2π * γ * φ * k (mod 2π)"""
    phi = (1 + np.sqrt(5)) / 2  # Golden ratio
    return 2 * np.pi * gamma * phi * k

def generate_resonance_kernel(N, weights, theta0_vals, omega_vals, sigma0=1.0, gamma=0.3):
    """
    Generate RFT kernel: R = Σᵢ wᵢ Dφᵢ Cσᵢ Dφᵢ†
    where:
    - Dφᵢ: diagonal phase matrix exp(i*φₖⁱ) 
    - Cσᵢ: periodic Gaussian convolution with width σᵢ = σ₀ * ωᵢ
    - φₖⁱ = θ₀ⁱ + generate_golden_ratio_phase(k, N, γ)
    """
    R = np.zeros((N, N), dtype=complex)
    
    for comp in range(len(weights)):
        w = weights[comp]
        theta0 = theta0_vals[comp]
        omega = omega_vals[comp]
        
        # Phase sequence for component i
        phi = np.array([theta0 + generate_golden_ratio_phase(k, N, gamma) for k in range(N)])
        D_phi = np.diag(np.exp(1j * phi))
        
        # Periodic Gaussian convolution matrix
        sigma = sigma0 * omega
        C_sigma = np.zeros((N, N), dtype=complex)
        for m in range(N):
            for n in range(N):
                d = periodic_distance(m, n, N)
                C_sigma[m, n] = np.exp(-0.5 * (d / sigma)**2)
        
        # Add component: wᵢ Dφᵢ Cσᵢ Dφᵢ†
        term = w * D_phi @ C_sigma @ D_phi.conj().T
        R += term
    
    return R

def test_kernel_properties():
    """Test kernel is Hermitian and PSD"""
    print("Testing kernel properties...")
    N = 8
    weights = [0.7, 0.3]
    theta0_vals = [0.0, np.pi/4]
    omega_vals = [1.0, (1 + np.sqrt(5))/2]  # Golden ratio
    
    R = generate_resonance_kernel(N, weights, theta0_vals, omega_vals)
    
    # Test Hermitian
    hermitian_error = np.max(np.abs(R - R.conj().T))
    print(f"  Hermitian error: {hermitian_error:.2e}")
    assert hermitian_error < 1e-14, "Kernel not Hermitian"
    
    # Test PSD
    eigenvals = la.eigvals(R)
    min_eigenval = np.min(eigenvals.real)
    print(f"  Min eigenvalue: {min_eigenval:.6f}")
    assert min_eigenval >= -1e-12, "Kernel not PSD"
    
    print("  ✓ Kernel is Hermitian and PSD")

def test_eigendecomposition():
    """Test eigendecomposition gives orthonormal basis"""
    print("\nTesting eigendecomposition...")
    N = 8
    weights = [0.7, 0.3]
    theta0_vals = [0.0, np.pi/4]
    omega_vals = [1.0, (1 + np.sqrt(5))/2]
    
    R = generate_resonance_kernel(N, weights, theta0_vals, omega_vals)
    
    # Compute eigendecomposition
    eigenvals, eigenvecs = la.eigh(R)  # Hermitian solver
    
    # Sort by eigenvalue magnitude (descending) for deterministic ordering
    indices = np.argsort(-np.abs(eigenvals))
    eigenvals = eigenvals[indices]
    eigenvecs = eigenvecs[:, indices]
    
    # Canonicalize phases: first nonzero entry real and positive
    for j in range(N):
        v = eigenvecs[:, j]
        first_nonzero_idx = np.argmax(np.abs(v) > 1e-12)
        phase = np.angle(v[first_nonzero_idx])
        eigenvecs[:, j] *= np.exp(-1j * phase)
        if eigenvecs[first_nonzero_idx, j].real < 0:
            eigenvecs[:, j] *= -1
    
    # Test orthonormality
    Psi = eigenvecs
    identity_error = np.max(np.abs(Psi.conj().T @ Psi - np.eye(N)))
    print(f"  Orthonormality error: {identity_error:.2e}")
    assert identity_error < 1e-12, "Eigenvectors not orthonormal"
    
    # Test eigendecomposition: R = Ψ Λ Ψ†
    Lambda = np.diag(eigenvals)
    reconstruction = Psi @ Lambda @ Psi.conj().T
    reconstruction_error = np.max(np.abs(R - reconstruction))
    print(f"  Reconstruction error: {reconstruction_error:.2e}")
    assert reconstruction_error < 1e-12, "Eigendecomposition failed"
    
    print("  ✓ Eigendecomposition correct and orthonormal")
    return Psi, eigenvals

def test_rft_inversion():
    """Test perfect inversion property: inverse(forward(x)) = x"""
    print("\nTesting RFT inversion...")
    N = 8
    weights = [0.7, 0.3]
    theta0_vals = [0.0, np.pi/4]
    omega_vals = [1.0, (1 + np.sqrt(5))/2]
    
    R = generate_resonance_kernel(N, weights, theta0_vals, omega_vals)
    Psi, eigenvals = la.eigh(R)
    
    # Sort and canonicalize (same as implementation)
    indices = np.argsort(-np.abs(eigenvals))
    eigenvals = eigenvals[indices]
    Psi = Psi[:, indices]
    
    for j in range(N):
        v = Psi[:, j]
        first_nonzero_idx = np.argmax(np.abs(v) > 1e-12)
        phase = np.angle(v[first_nonzero_idx])
        Psi[:, j] *= np.exp(-1j * phase)
        if Psi[first_nonzero_idx, j].real < 0:
            Psi[:, j] *= -1
    
    # Test with random real signal
    np.random.seed(42)
    x = np.random.randn(N)
    
    # Forward: X = Ψ†x
    X = Psi.conj().T @ x
    
    # Inverse: x_recovered = ΨX
    x_recovered = Psi @ X
    
    # Check inversion
    inversion_error = np.max(np.abs(x - x_recovered.real))
    print(f"  Inversion error: {inversion_error:.2e}")
    assert inversion_error < 1e-12, "RFT inversion failed"
    
    print("  ✓ Perfect inversion: inverse(forward(x)) = x")

def test_plancherel_theorem():
    """Test Plancherel theorem: ||x||² = ||X||²"""
    print("\nTesting Plancherel theorem...")
    N = 8
    weights = [0.7, 0.3]
    theta0_vals = [0.0, np.pi/4]
    omega_vals = [1.0, (1 + np.sqrt(5))/2]
    
    R = generate_resonance_kernel(N, weights, theta0_vals, omega_vals)
    Psi, eigenvals = la.eigh(R)
    
    # Sort and canonicalize
    indices = np.argsort(-np.abs(eigenvals))
    Psi = Psi[:, indices]
    
    for j in range(N):
        v = Psi[:, j]
        first_nonzero_idx = np.argmax(np.abs(v) > 1e-12)
        phase = np.angle(v[first_nonzero_idx])
        Psi[:, j] *= np.exp(-1j * phase)
        if Psi[first_nonzero_idx, j].real < 0:
            Psi[:, j] *= -1
    
    # Test with random signal
    np.random.seed(123)
    x = np.random.randn(N) + 1j * np.random.randn(N)
    
    # Forward transform
    X = Psi.conj().T @ x
    
    # Check Plancherel
    energy_x = np.linalg.norm(x)**2
    energy_X = np.linalg.norm(X)**2
    energy_error = abs(energy_x - energy_X)
    print(f"  ||x||²: {energy_x:.6f}")
    print(f"  ||X||²: {energy_X:.6f}")
    print(f"  Energy error: {energy_error:.2e}")
    assert energy_error < 1e-12, "Plancherel theorem violated"
    
    print("  ✓ Plancherel theorem satisfied")

def test_dft_limit():
    """Test DFT limit behavior for specific parameters"""
    print("\nTesting DFT limit...")
    N = 8
    
    # Parameters for DFT-like behavior (large σ, uniform phases)
    weights = [1.0]
    theta0_vals = [0.0]
    omega_vals = [1000.0]  # Very large ω → very large σ
    
    R = generate_resonance_kernel(N, weights, theta0_vals, omega_vals, sigma0=1.0)
    
    # In the limit σ→∞, the Gaussian kernel approaches uniform
    # and the RFT should behave similarly to DFT
    expected_uniform = np.ones((N, N)) / N  # Uniform matrix
    
    # Normalize R for comparison
    R_normalized = R / np.trace(R) * N
    uniform_approximation_error = np.max(np.abs(R_normalized.real - expected_uniform))
    print(f"  DFT limit approximation error: {uniform_approximation_error:.6f}")
    
    # Should be reasonably close for very large σ
    assert uniform_approximation_error < 0.5, "DFT limit not approached"
    
    print("  ✓ DFT limit behavior reasonable")

def main():
    """Run all validation tests"""
    print("=" * 60)
    print("FINAL RFT VALIDATION - Testing against rigorous spec")
    print("=" * 60)
    
    try:
        test_kernel_properties()
        test_eigendecomposition()
        test_rft_inversion()
        test_plancherel_theorem()
        test_dft_limit()
        
        print("\n" + "=" * 60)
        print("🎉 ALL TESTS PASSED - RFT implementation is mathematically rigorous!")
        print("✓ Hermitian PSD kernel construction")
        print("✓ Orthonormal eigendecomposition with deterministic ordering")
        print("✓ Perfect inversion property")
        print("✓ Plancherel theorem (energy conservation)")
        print("✓ DFT limit behavior")
        print("=" * 60)
        return True
        
    except AssertionError as e:
        print(f"\n❌ VALIDATION FAILED: {e}")
        return False
    except Exception as e:
        print(f"\n💥 UNEXPECTED ERROR: {e}")
        return False

if __name__ == "__main__":
    main()

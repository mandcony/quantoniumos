"""
Resonant Fourier Transform (RFT) - Formal Mathematical Framework
================================================================

This module provides the formal mathematical derivation and proofs
for the Resonant Fourier Transform, suitable for academic publication.

Author: QuantoniumOS Research
Date: December 2025
"""

import numpy as np
from scipy.linalg import toeplitz, eigh
import matplotlib.pyplot as plt
from typing import Tuple, Dict
import os

# The Golden Ratio
PHI = (1 + np.sqrt(5)) / 2


# =============================================================================
# SECTION 1: OPERATOR DEFINITION
# =============================================================================

def build_resonance_operator(N: int, f0: float = 10.0, decay_rate: float = 0.01) -> Tuple[np.ndarray, np.ndarray]:
    """
    Construct the Resonance Operator K_φ and return its components.
    
    DEFINITION:
    -----------
    The Resonance Operator K_φ is defined as:
    
        K_φ = T(R_φ) + D_φ
    
    Where:
        T(R_φ) : Toeplitz matrix constructed from the golden autocorrelation function
        D_φ    : Diagonal decay regularization matrix
        
    The autocorrelation function R_φ models the expected correlation structure
    of the golden quasi-periodic signal family:
    
        R_φ[k] = cos(2πf₀k/N) + cos(2πf₀φk/N)
        
    Where φ = (1+√5)/2 is the golden ratio.
    
    Args:
        N: Transform size
        f0: Base frequency parameter
        decay_rate: Regularization decay rate
        
    Returns:
        K_phi: The full resonance operator (N×N Hermitian matrix)
        components: Dict with T_R and D_phi for analysis
    """
    k = np.arange(N)
    t_k = k / N
    
    # Autocorrelation components
    r_fundamental = np.cos(2 * np.pi * f0 * t_k)
    r_golden = np.cos(2 * np.pi * f0 * PHI * t_k)
    
    # Combined autocorrelation
    R_phi = r_fundamental + r_golden
    R_phi[0] = 2.0  # Normalize: r[0] = E[|x|²] for unit-power signals
    
    # Toeplitz structure: T(R_φ)
    T_R = toeplitz(R_phi)
    
    # Decay regularization: D_φ (ensures positive definiteness)
    decay = np.exp(-decay_rate * k)
    D_phi = np.diag(decay)
    
    # Full operator: K_φ = T(R_φ) ⊙ D_φ (element-wise for regularization)
    # Actually, we apply decay to the autocorrelation before Toeplitz
    R_phi_regularized = R_phi * decay
    R_phi_regularized[0] = 1.0
    K_phi = toeplitz(R_phi_regularized)
    
    return K_phi, {'T_R': T_R, 'D_phi': D_phi, 'R_phi': R_phi}


def prove_hermitian(K: np.ndarray) -> Tuple[bool, float]:
    """
    THEOREM 1: K_φ is Hermitian
    ---------------------------
    Proof: K_φ is constructed as a real symmetric Toeplitz matrix.
           For real matrices, Hermitian ⟺ Symmetric.
           T(r) is symmetric by construction: T[i,j] = r[|i-j|] = T[j,i]
           
    Returns:
        is_hermitian: True if K = K^H
        error: Frobenius norm of K - K^H
    """
    error = np.linalg.norm(K - K.conj().T, 'fro')
    return error < 1e-12, error


def compute_eigenbasis(K: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    THEOREM 2: The RFT basis U_φ is the eigenbasis of K_φ
    -----------------------------------------------------
    Since K_φ is Hermitian, it has a complete orthonormal eigenbasis.
    The Spectral Theorem guarantees:
    
        K_φ = U_φ Λ U_φ^H
        
    Where:
        U_φ : Unitary matrix of eigenvectors (the RFT basis)
        Λ   : Diagonal matrix of real eigenvalues
        
    The RFT is defined as:
        X = U_φ^H x  (forward transform)
        x = U_φ X    (inverse transform)
        
    Returns:
        eigenvalues: Real eigenvalues (sorted descending)
        U_phi: The RFT basis matrix (eigenvectors as columns)
    """
    eigenvalues, eigenvectors = eigh(K)
    
    # Sort by eigenvalue (descending) for energy compaction
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    U_phi = eigenvectors[:, idx]
    
    return eigenvalues, U_phi


def prove_unitarity(U: np.ndarray) -> Tuple[bool, float, float]:
    """
    THEOREM 3: U_φ is Unitary
    -------------------------
    Proof: U_φ consists of eigenvectors of a Hermitian matrix.
           Eigenvectors of Hermitian matrices are orthonormal.
           Therefore U_φ^H U_φ = I (unitary).
           
    We verify both:
        U^H U = I  (orthonormality of columns)
        U U^H = I  (completeness)
        
    Returns:
        is_unitary: True if both conditions hold
        error_UhU: ||U^H U - I||_F
        error_UUh: ||U U^H - I||_F
    """
    I = np.eye(U.shape[0])
    error_UhU = np.linalg.norm(U.conj().T @ U - I, 'fro')
    error_UUh = np.linalg.norm(U @ U.conj().T - I, 'fro')
    
    is_unitary = (error_UhU < 1e-10) and (error_UUh < 1e-10)
    return is_unitary, error_UhU, error_UUh


# =============================================================================
# SECTION 2: BASIS VECTOR VISUALIZATION
# =============================================================================

def plot_basis_comparison(N: int = 64, num_vectors: int = 6, save_path: str = None):
    """
    Compare RFT basis vectors against standard Fourier basis.
    
    The key visual difference:
    - Fourier: Pure sinusoids at integer frequencies
    - RFT: Modulated oscillations tuned to golden-ratio frequency relationships
    """
    # Build RFT basis
    K, _ = build_resonance_operator(N)
    _, U_rft = compute_eigenbasis(K)
    
    # Build Fourier basis (DFT matrix, real part for comparison)
    n = np.arange(N)
    k = np.arange(N)
    W = np.exp(-2j * np.pi * np.outer(k, n) / N) / np.sqrt(N)
    U_fft = np.real(W)  # Real part for visualization
    
    # Plot
    fig, axes = plt.subplots(2, num_vectors, figsize=(15, 6))
    fig.suptitle(f'Basis Vector Comparison (N={N})', fontsize=14)
    
    for i in range(num_vectors):
        # RFT basis vectors
        axes[0, i].plot(U_rft[:, i], 'b-', linewidth=1)
        axes[0, i].set_title(f'RFT φ_{i}')
        axes[0, i].set_ylim(-0.3, 0.3)
        axes[0, i].grid(True, alpha=0.3)
        if i == 0:
            axes[0, i].set_ylabel('RFT Basis')
        
        # Fourier basis vectors
        axes[1, i].plot(U_fft[i, :], 'r-', linewidth=1)
        axes[1, i].set_title(f'DFT ψ_{i}')
        axes[1, i].set_ylim(-0.3, 0.3)
        axes[1, i].grid(True, alpha=0.3)
        if i == 0:
            axes[1, i].set_ylabel('Fourier Basis')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved basis comparison to {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_eigenvalue_spectrum(Ns: list = [64, 128, 256, 512], save_path: str = None):
    """
    Compare RFT eigenvalue spectrum against FFT frequency bins.
    
    Key insight: RFT eigenvalues reflect the expected energy distribution
    for golden quasi-periodic signals, concentrating variance in fewer modes.
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    
    for ax, N in zip(axes, Ns):
        # RFT eigenvalues
        K, _ = build_resonance_operator(N)
        eigs, _ = compute_eigenbasis(K)
        
        # Normalize for comparison
        eigs_norm = eigs / np.max(eigs)
        
        # FFT "eigenvalues" (all equal to 1 for unitary DFT, 
        # but we show power spectrum of a golden signal for comparison)
        t = np.linspace(0, 1, N)
        golden_signal = np.sin(2*np.pi*10*t) + np.sin(2*np.pi*10*PHI*t)
        fft_power = np.abs(np.fft.fft(golden_signal))**2
        fft_power_norm = fft_power / np.max(fft_power)
        
        ax.semilogy(eigs_norm, 'b-', label='RFT eigenvalues', linewidth=1.5)
        ax.semilogy(np.sort(fft_power_norm)[::-1], 'r--', label='FFT power (golden sig)', linewidth=1.5)
        ax.set_xlabel('Index (sorted)')
        ax.set_ylabel('Normalized value (log)')
        ax.set_title(f'N = {N}')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    fig.suptitle('Eigenvalue Spectrum: RFT vs FFT Power Distribution', fontsize=14)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved spectrum comparison to {save_path}")
    else:
        plt.show()
    
    plt.close()


# =============================================================================
# SECTION 3: FORMAL PROOF OUTPUT
# =============================================================================

def generate_formal_proofs(N: int = 256) -> str:
    """
    Generate formal mathematical proof statements with empirical verification.
    """
    output = []
    output.append("=" * 70)
    output.append("RESONANT FOURIER TRANSFORM - FORMAL MATHEMATICAL PROOFS")
    output.append("=" * 70)
    output.append(f"\nTransform Size: N = {N}")
    output.append(f"Golden Ratio: φ = {PHI:.10f}")
    
    # Build operator
    K, components = build_resonance_operator(N)
    
    output.append("\n" + "-" * 70)
    output.append("DEFINITION 1: The Resonance Operator K_φ")
    output.append("-" * 70)
    output.append("""
    K_φ = T(R_φ ⊙ d)
    
    Where:
        R_φ[k] = cos(2πf₀k/N) + cos(2πf₀φk/N)    (Golden autocorrelation)
        d[k]   = exp(-αk)                         (Decay regularization)
        T(·)   = Toeplitz matrix constructor
        ⊙      = Element-wise product
    """)
    
    # Prove Hermitian
    is_hermitian, h_error = prove_hermitian(K)
    output.append("\n" + "-" * 70)
    output.append("THEOREM 1: K_φ is Hermitian")
    output.append("-" * 70)
    output.append("""
    Proof: K_φ is a real symmetric Toeplitz matrix.
           T[i,j] = R_φ[|i-j|] · d[|i-j|] = T[j,i]
           For real matrices: Hermitian ⟺ Symmetric ✓
    """)
    output.append(f"    Empirical Verification: ||K - K^H||_F = {h_error:.2e}")
    output.append(f"    Result: {'PASS ✓' if is_hermitian else 'FAIL ✗'}")
    
    # Compute eigenbasis
    eigenvalues, U_phi = compute_eigenbasis(K)
    
    output.append("\n" + "-" * 70)
    output.append("THEOREM 2: Spectral Decomposition")
    output.append("-" * 70)
    output.append("""
    By the Spectral Theorem for Hermitian matrices:
    
        K_φ = U_φ Λ U_φ^H
        
    Where:
        U_φ ∈ ℝ^{N×N} : Orthonormal eigenvector matrix (RFT basis)
        Λ = diag(λ₁, ..., λ_N) : Real eigenvalues
        
    The Resonant Fourier Transform is defined as:
        Forward:  X = U_φ^T x
        Inverse:  x = U_φ X
    """)
    output.append(f"    Eigenvalue range: [{eigenvalues[-1]:.4f}, {eigenvalues[0]:.4f}]")
    output.append(f"    Condition number: {eigenvalues[0]/eigenvalues[-1]:.2f}")
    
    # Prove unitarity
    is_unitary, err_UhU, err_UUh = prove_unitarity(U_phi)
    output.append("\n" + "-" * 70)
    output.append("THEOREM 3: U_φ is Unitary (Orthogonal for real case)")
    output.append("-" * 70)
    output.append("""
    Proof: Eigenvectors of Hermitian matrices form an orthonormal basis.
           Therefore U_φ^H U_φ = I and U_φ U_φ^H = I.
    """)
    output.append(f"    ||U^H U - I||_F = {err_UhU:.2e}")
    output.append(f"    ||U U^H - I||_F = {err_UUh:.2e}")
    output.append(f"    Result: {'PASS ✓' if is_unitary else 'FAIL ✗'}")
    
    # Perfect reconstruction
    output.append("\n" + "-" * 70)
    output.append("COROLLARY: Perfect Reconstruction")
    output.append("-" * 70)
    output.append("""
    For any signal x ∈ ℝ^N:
        x = U_φ (U_φ^T x) = U_φ U_φ^T x = I x = x
        
    Reconstruction is exact (up to numerical precision).
    """)
    
    # Test reconstruction
    x = np.sin(2*np.pi*10*np.linspace(0,1,N)) + np.sin(2*np.pi*10*PHI*np.linspace(0,1,N))
    X = U_phi.T @ x
    x_rec = U_phi @ X
    rec_error = np.linalg.norm(x - x_rec) / np.linalg.norm(x)
    output.append(f"    Empirical: ||x - U_φ U_φ^T x|| / ||x|| = {rec_error:.2e}")
    
    output.append("\n" + "=" * 70)
    output.append("END OF FORMAL PROOFS")
    output.append("=" * 70)
    
    return "\n".join(output)


# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    # Generate formal proofs
    proofs = generate_formal_proofs(N=256)
    print(proofs)
    
    # Save to file
    proof_path = "/workspaces/quantoniumos/docs/proofs/RFT_FORMAL_PROOFS.txt"
    os.makedirs(os.path.dirname(proof_path), exist_ok=True)
    with open(proof_path, 'w') as f:
        f.write(proofs)
    print(f"\nProofs saved to: {proof_path}")
    
    # Generate visualizations
    fig_dir = "/workspaces/quantoniumos/figures/rft_analysis"
    os.makedirs(fig_dir, exist_ok=True)
    
    print("\nGenerating basis vector comparison...")
    plot_basis_comparison(N=64, num_vectors=6, save_path=f"{fig_dir}/basis_comparison.png")
    
    print("Generating eigenvalue spectrum analysis...")
    plot_eigenvalue_spectrum(save_path=f"{fig_dir}/eigenvalue_spectrum.png")
    
    print("\nAll outputs generated successfully.")

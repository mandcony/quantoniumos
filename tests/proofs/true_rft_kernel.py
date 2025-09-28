#!/usr/bin/env python3
"""
True RFT Kernel Implementation - Direct from C Code
Direct translation of the mathematical RFT kernel without QR decomposition
Uses the exact formula: Ψ = Σᵢ wᵢDφᵢCσᵢD†φᵢ
"""

import numpy as np
import cmath


class TrueRFTKernel:
    """
    Direct implementation of the RFT kernel using the mathematical formula
    from the C code, without any QR decomposition that would erase the structure.
    """
    
    def __init__(self, size: int, beta: float = 1.0):
        """
        Initialize RFT kernel for given size.
        
        Args:
            size: Transform size
            beta: Frequency scaling parameter
        """
        self.size = size
        self.beta = beta
        self.phi = 1.618033988749894848204586834366  # Golden ratio from C code

        # Build the RFT kernel matrix directly using the mathematical formula
        raw_kernel = self._build_rft_kernel()
        self._rft_matrix = self._apply_unitarity_correction(raw_kernel)
    
    def _build_rft_kernel(self) -> np.ndarray:
        """
        Build the RFT kernel following the exact C implementation:
        Ψ = Σᵢ wᵢDφᵢCσᵢD†φᵢ
        
        Returns:
            RFT kernel matrix K (NOT orthogonalized)
        """
        N = self.size
        K = np.zeros((N, N), dtype=complex)
        
        # Build resonance kernel following the paper equation
        for component in range(N):
            # Golden ratio phase sequence without modular discontinuities
            phi_k = component * self.phi
            w_i = 1.0 / N  # Equal weights for components
            
            # Build component matrices: wᵢDφᵢCσᵢD†φᵢ
            for m in range(N):
                for n in range(N):
                    # Phase operators Dφᵢ and D†φᵢ (diagonal)
                    phase_m = 2 * np.pi * phi_k * m / N
                    phase_n = 2 * np.pi * phi_k * n / N
                    
                    # Convolution kernel Cσᵢ with Gaussian profile
                    sigma_i = 1.0 + 0.1 * component
                    dist = abs(m - n)
                    if dist > N // 2:
                        dist = N - dist  # Circular distance
                    C_sigma = np.exp(-0.5 * (dist * dist) / (sigma_i * sigma_i))
                    
                    # Matrix element: wᵢ * exp(iφₖm) * C_σ * exp(-iφₖn)
                    # = wᵢ * C_σ * exp(iφₖ(m-n))
                    phase_diff = phase_m - phase_n
                    element_real = w_i * C_sigma * np.cos(phase_diff)
                    element_imag = w_i * C_sigma * np.sin(phase_diff)
                    
                    # Accumulate into kernel matrix
                    K[m, n] += complex(element_real, element_imag)
        
        return K

    def _apply_unitarity_correction(self, kernel: np.ndarray) -> np.ndarray:
        """Project kernel onto the nearest unitary matrix preserving structure."""
        # Compute Hermitian Gram matrix and guard against numerical drift
        gram = kernel.conj().T @ kernel
        # Symmetrize to counter accumulation error
        gram = 0.5 * (gram + gram.conj().T)

        # Eigen decomposition for positive definite matrix
        evals, evecs = np.linalg.eigh(gram)
        eps = 1e-12
        evals = np.clip(np.real(evals), eps, None)

        inv_sqrt = evecs @ np.diag(1.0 / np.sqrt(evals)) @ evecs.conj().T
        corrected = kernel @ inv_sqrt

        # Final column normalization to keep ∥column∥₂ = 1
        col_norms = np.linalg.norm(corrected, axis=0)
        col_norms[col_norms < eps] = 1.0
        corrected = corrected / col_norms

        return corrected
    
    def forward_transform(self, x: np.ndarray) -> np.ndarray:
        """Apply forward RFT: y = K^H x"""
        if len(x) != self.size:
            raise ValueError(f"Input size {len(x)} != RFT size {self.size}")
        return self._rft_matrix.conj().T @ x
    
    def inverse_transform(self, y: np.ndarray) -> np.ndarray:
        """Apply inverse RFT: x = K y"""
        if len(y) != self.size:
            raise ValueError(f"Input size {len(y)} != RFT size {self.size}")
        return self._rft_matrix @ y
    
    def get_rft_matrix(self) -> np.ndarray:
        """Get the RFT kernel matrix."""
        return self._rft_matrix.copy()


def build_dft_matrix(size: int) -> np.ndarray:
    """Build the standard DFT matrix for comparison."""
    N = size
    W = np.zeros((N, N), dtype=complex)
    
    for k in range(N):
        for n in range(N):
            W[k, n] = cmath.exp(-2j * np.pi * k * n / N)
    
    # Normalize to match RFT scaling
    return W / np.sqrt(N)


if __name__ == "__main__":
    # Quick test
    size = 8
    rft = TrueRFTKernel(size)
    dft = build_dft_matrix(size)
    
    print(f"RFT vs DFT for size {size}")
    print(f"RFT matrix shape: {rft.get_rft_matrix().shape}")
    print(f"DFT matrix shape: {dft.shape}")
    
    # Check distinctness
    diff = np.linalg.norm(rft.get_rft_matrix() - dft, 'fro')
    print(f"Frobenius norm difference: {diff:.6f}")
    
    if diff > 0.1:
        print("✓ RFT and DFT are clearly distinct!")
    else:
        print("❌ RFT and DFT are too similar")

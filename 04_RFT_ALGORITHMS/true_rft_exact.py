#!/usr/bin/env python3
"""
True Resonance Fourier Transform: Exact Implementation 

This implements the RFT exactly as specified: R = Σ_i w_i D_φi C_σi D_φi† where:
- C_σi is a Gaussian correlation kernel (Hermitian PSD)
- D_φi applies phase modulation 
- w_i are weights

The RFT basis Ψ is the eigenvector matrix of R: R = Ψ Λ Ψ†
Transform: X = Ψ† x

NO BULLSHIT - just exactly what you asked for.
"""

import json
import math
import os
import sys
import typing
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

# Golden ratio 
PHI = (1.0 + math.sqrt(5.0)) / 2.0

class TrueResonanceFourierTransform:
    """
    The ACTUAL Resonance Fourier Transform R = Σ_i w_i D_φi C_σi D_φi†
    Eigendecompose: R = Ψ Λ Ψ†
    Transform: X = Ψ† x
    """

    def __init__(self, N: int = 64, num_components: int = 8):
        """
        Initialize with dimension N and number of kernel components
        """
        self.N = N
        self.num_components = num_components
        print(f"🔧 Building True RFT (N={N}, components={num_components})")

        # Generate component parameters
        self.weights, self.phis, self.sigmas = self._generate_parameters()

        # Build resonance kernel R
        self.R = self._build_resonance_kernel()

        # Eigendecompose R = Ψ Λ Ψ†
        self.eigenvals, self.Psi = self._eigendecompose()
        print(f"✅ True RFT constructed")

    def _generate_parameters(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate w_i, φ_i, σ_i parameters
        """
        # Golden ratio weights: w_i = 1/φ^i (normalized) 
        weights = np.array([1.0 / (PHI ** i) for i in range(self.num_components)])
        weights /= np.sum(weights)  # Normalize

        # Golden ratio phases: φ_i = 2π * (φ^i mod 1)
        phis = np.array([2 * np.pi * ((PHI ** i) % 1) for i in range(self.num_components)])

        # Logarithmic sigma progression
        sigma_min, sigma_max = 0.5, 4.0
        sigmas = np.logspace(np.log10(sigma_min), np.log10(sigma_max), self.num_components)
        
        print(f" Parameters generated:")
        for i in range(self.num_components):
            print(f"  Component {i}: w={weights[i]:.3f}, φ={phis[i]:.3f}, σ={sigmas[i]:.3f}")
        
        return weights, phis, sigmas

    def _build_gaussian_kernel(self, sigma: float) -> np.ndarray:
        """
        Build Gaussian correlation kernel C_σ
        """
        C = np.zeros((self.N, self.N))
        for m in range(self.N):
            for n in range(self.N):
                # Circular distance
                dist = min(abs(m - n), self.N - abs(m - n))
                C[m, n] = np.exp(-dist**2 / (2 * sigma**2))
        return C

    def _build_phase_modulation(self, phi: float) -> np.ndarray:
        """
        Build phase modulation operator D_φ
        """
        D = np.zeros((self.N, self.N), dtype=np.complex128)
        for m in range(self.N):
            D[m, m] = np.exp(1j * phi * m)
        return D

    def _build_resonance_kernel(self) -> np.ndarray:
        """
        Build the resonance kernel: R = Σ_i w_i D_φi C_σi D_φi†
        """
        print(f" Building resonance kernel...")
        R = np.zeros((self.N, self.N), dtype=np.complex128)
        
        for i in range(self.num_components):
            w_i = self.weights[i]
            phi_i = self.phis[i]
            sigma_i = self.sigmas[i]

            # Build components
            C_sigma = self._build_gaussian_kernel(sigma_i)
            D_phi = self._build_phase_modulation(phi_i)
            D_phi_dag = D_phi.conj().T

            # Add component: w_i * D_φi * C_σi * D_φi†
            component = w_i * (D_phi @ C_sigma @ D_phi_dag)
            R += component

        # Ensure Hermitian
        R = (R + R.conj().T) / 2

        # Verify Hermitian
        hermitian_error = np.linalg.norm(R - R.conj().T, 'fro')
        print(f" Resonance kernel built (Hermitian error: {hermitian_error:.2e})")
        return R

    def _eigendecompose(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Eigendecompose R = Ψ Λ Ψ†
        """
        print(f" Eigendecomposing R...")

        # Hermitian eigendecomposition
        eigenvals, eigenvecs = np.linalg.eigh(self.R)

        # Sort by decreasing eigenvalue magnitude
        idx = np.argsort(np.abs(eigenvals))[::-1]
        eigenvals = eigenvals[idx]
        eigenvecs = eigenvecs[:, idx]
        
        print(f" Eigenvalues range: [{np.min(eigenvals):.6f}, {np.max(eigenvals):.6f}]")
        return eigenvals, eigenvecs

    def transform(self, x: np.ndarray) -> np.ndarray:
        """
        Forward RFT: X = Ψ† x
        """
        return self.Psi.conj().T @ x

    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        """
        Inverse RFT: x = Ψ X
        """
        return self.Psi @ X

    def verify_perfect_reconstruction(self) -> Dict[str, float]:
        """
        Verify perfect reconstruction
        """
        # Random test signal
        np.random.seed(42)
        x = np.random.randn(self.N) + 1j * np.random.randn(self.N)

        # Forward and inverse
        X = self.transform(x)
        x_reconstructed = self.inverse_transform(X)

        # Errors
        reconstruction_error = np.linalg.norm(x - x_reconstructed)
        relative_error = reconstruction_error / np.linalg.norm(x)

        # Energy preservation (Parseval)
        energy_original = np.linalg.norm(x)**2
        energy_transformed = np.linalg.norm(X)**2
        energy_error = abs(energy_original - energy_transformed) / energy_original
        
        return {
            'reconstruction_error': reconstruction_error,
            'relative_error': relative_error,
            'energy_error': energy_error,
            'perfect_reconstruction': relative_error < 1e-12
        }

    def prove_non_equivalence_to_dft(self) -> Dict[str, Any]:
        """
        Prove RFT ≠ DFT
        """
        print(f" Proving non-equivalence to DFT...")

        # DFT matrix
        DFT = np.zeros((self.N, self.N), dtype=np.complex128)
        for k in range(self.N):
            for n in range(self.N):
                DFT[k, n] = np.exp(-2j * np.pi * k * n / self.N) / np.sqrt(self.N)

        # Compute ||Ψ - DFT||_F
        norm_difference = np.linalg.norm(self.Psi - DFT, 'fro')

        # Compute max correlation between RFT and DFT basis vectors
        max_correlation = 0.0
        correlations = []
        for i in range(self.N):
            rft_vec = self.Psi[:, i]
            for j in range(self.N):
                dft_vec = DFT[j, :]
                corr = abs(np.vdot(rft_vec, dft_vec))
                correlations.append(corr)
                max_correlation = max(max_correlation, corr)
        
        mean_correlation = np.mean(correlations)

        # Non-equivalence criteria
        non_equivalent = (norm_difference > 1e-3) and (max_correlation < 0.99)
        distinctness = 1.0 - max_correlation
        
        result = {
            'norm_difference': norm_difference,
            'max_correlation': max_correlation,
            'mean_correlation': mean_correlation,
            'non_equivalent': non_equivalent,
            'distinctness_score': distinctness
        }
        
        print(f" Norm difference: {norm_difference:.6f}")
        print(f" Max correlation: {max_correlation:.6f}")
        print(f" Non-equivalent: {'✅' if non_equivalent else '❌'}")
        print(f" Distinctness: {distinctness:.1%}")
        
        return result

    def analyze_golden_ratio_structure(self) -> Dict[str, Any]:
        """
        Analyze golden ratio properties
        """
        print(f"🌀 Analyzing golden ratio structure...")

        # Check eigenvalue ratios for golden ratio patterns
        eigenval_ratios = []
        for i in range(self.N-1):
            if abs(self.eigenvals[i+1]) > 1e-10:  # Avoid division by near-zero
                ratio = abs(self.eigenvals[i] / self.eigenvals[i+1])
                eigenval_ratios.append(ratio)
        
        # Find ratios close to φ
        phi_ratios = [r for r in eigenval_ratios if abs(r - PHI) < 0.1]
        
        # Check for Fibonacci-like sequences in eigenvalues
        fib_sequences = []
        for i in range(self.N-2):
            a, b, c = abs(self.eigenvals[i]), abs(self.eigenvals[i+1]), abs(self.eigenvals[i+2])
            if b > 0 and abs(a/b - c/b - 1.0) < 0.1:
                fib_sequences.append((i, i+1, i+2))
        
        # Analyze basis vectors for golden angle separation
        golden_angle = 2 * np.pi * (1 - 1/PHI)
        angle_diffs = []
        for i in range(self.N-1):
            vec1 = self.Psi[:, i]
            vec2 = self.Psi[:, i+1]
            # Calculate angle between complex vectors
            dot = np.abs(np.vdot(vec1, vec2))
            angle = np.arccos(min(dot, 1.0))
            angle_diffs.append(angle)
        
        # Check for golden angles
        golden_angles = [a for a in angle_diffs if abs(a - golden_angle) < 0.1]
        
        result = {
            'eigenvalue_phi_ratios': len(phi_ratios),
            'fibonacci_sequences': len(fib_sequences),
            'golden_angles': len(golden_angles),
            'golden_ratio_structure': (len(phi_ratios) > 0 or len(fib_sequences) > 0 or len(golden_angles) > 0)
        }
        
        print(f" Eigenvalue φ ratios: {len(phi_ratios)}")
        print(f" Fibonacci sequences: {len(fib_sequences)}")
        print(f" Golden angles: {len(golden_angles)}")
        print(f" Golden ratio structure: {'✅' if result['golden_ratio_structure'] else '❌'}")
        
        return result

    def get_rft_basis(self) -> np.ndarray:
        """
        Get the RFT basis matrix (Ψ)
        """
        return self.Psi

    def get_basis_matrix(self) -> np.ndarray:
        """
        Get the RFT basis matrix (Ψ) - alias for get_rft_basis
        """
        return self.Psi


# Helper functions to match the canonical_true_rft.py interface
def get_rft_basis(N: int) -> np.ndarray:
    """
    Get the RFT basis matrix of size N
    """
    transformer = TrueResonanceFourierTransform(N=N)
    return transformer.get_rft_basis()

def forward_true_rft(signal: np.ndarray, N: int = None) -> np.ndarray:
    """
    Apply forward True RFT to signal
    """
    if N is None:
        N = len(signal)
    transformer = TrueResonanceFourierTransform(N=N)
    return transformer.transform(signal)

def inverse_true_rft(spectrum: np.ndarray, N: int = None) -> np.ndarray:
    """
    Apply inverse True RFT to spectrum
    """
    if N is None:
        N = len(spectrum)
    transformer = TrueResonanceFourierTransform(N=N)
    return transformer.inverse_transform(spectrum)


if __name__ == "__main__":
    # Simple self-test
    N = 64
    print(f"\n🔬 Self-testing True RFT with N={N}...\n")
    
    # Create transformer
    transformer = TrueResonanceFourierTransform(N=N)
    
    # Verify perfect reconstruction
    recon_results = transformer.verify_perfect_reconstruction()
    print(f"\n✅ Perfect reconstruction: {recon_results['perfect_reconstruction']}")
    print(f"   Relative error: {recon_results['relative_error']:.2e}")
    print(f"   Energy error: {recon_results['energy_error']:.2e}")
    
    # Prove non-equivalence to DFT
    transformer.prove_non_equivalence_to_dft()
    
    # Analyze golden ratio structure
    transformer.analyze_golden_ratio_structure()
    
    print("\n✅ True RFT self-test complete.")

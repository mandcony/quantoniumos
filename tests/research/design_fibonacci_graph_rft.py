#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025 Luis M. Minier
"""
Fibonacci Graph RFT Design (Research Prototype)
===============================================

Mathematical Redesign:
----------------------
Instead of a fixed closed-form matrix, we define the RFT as the 
Graph Fourier Transform (GFT) of a "Fibonacci Graph".

1. Vertices: 0..N-1 (Time indices)
2. Edges: Connect (i, j) if |i-j| is a Fibonacci number.
3. Weights: w_ij = phi^(-|i-j|) (Golden ratio decay)

Hypothesis:
-----------
The eigenvectors of this graph's Laplacian will form a basis that
is "Linear 1D" (orthonormal basis on R^N) but captures the 
"Bigger Geometric Signal" (the Fibonacci topology).

This should perform better on signals with multi-scale structure
than the fixed-frequency DCT/FFT.
"""

import numpy as np
import scipy.linalg
import scipy.sparse
import matplotlib.pyplot as plt
from scipy.fftpack import dct
import os

PHI = (1 + np.sqrt(5)) / 2

def get_fibonacci_numbers(limit: int):
    """Generate Fibonacci numbers up to limit"""
    fibs = [1, 2]
    while fibs[-1] < limit:
        fibs.append(fibs[-1] + fibs[-2])
    return fibs[:-1] # Remove the one that exceeded limit

def build_fibonacci_graph_laplacian(n: int):
    """
    Construct the Laplacian of the Fibonacci Graph.
    L = D - W
    """
    # 1. Build Adjacency Matrix W
    # We use a sparse construction for efficiency, though we'll densify for eigen-decomp
    W = np.zeros((n, n))
    
    fibs = get_fibonacci_numbers(n)
    
    # Connect nodes based on Fibonacci distances
    for d in fibs:
        weight = PHI ** (-0.5 * d) # Decay weight by distance
        
        # Vectorized diagonal filling
        # Connect i and i+d
        rows = np.arange(n - d)
        cols = rows + d
        W[rows, cols] = weight
        W[cols, rows] = weight # Symmetric
        
    # 2. Build Degree Matrix D
    degrees = np.sum(W, axis=1)
    D = np.diag(degrees)
    
    # 3. Laplacian
    L = D - W
    
    return L

def compute_graph_rft_basis(n: int):
    """
    Compute the Graph RFT basis (Eigenvectors of L)
    Returns: (eigenvalues, eigenvectors)
    """
    print(f"Constructing Fibonacci Graph Laplacian for N={n}...")
    L = build_fibonacci_graph_laplacian(n)
    
    print("Computing Eigendecomposition...")
    # eigh is for symmetric/hermitian matrices (faster, more stable)
    evals, evecs = scipy.linalg.eigh(L)
    
    # Sort by eigenvalue (frequency)
    idx = np.argsort(evals)
    evals = evals[idx]
    evecs = evecs[:, idx]
    
    return evals, evecs

def graph_rft_forward(signal: np.ndarray, basis: np.ndarray):
    """Forward transform: x_hat = U.T @ x"""
    return basis.T @ signal

def graph_rft_inverse(coeffs: np.ndarray, basis: np.ndarray):
    """Inverse transform: x = U @ x_hat"""
    return basis @ coeffs

def benchmark_sparsity(signal: np.ndarray, basis: np.ndarray, method_name: str):
    """Measure energy concentration"""
    # Transform
    if method_name == 'Graph-RFT':
        coeffs = graph_rft_forward(signal, basis)
    elif method_name == 'DCT':
        coeffs = dct(signal, norm='ortho')
    elif method_name == 'FFT':
        coeffs = np.fft.fft(signal) / np.sqrt(len(signal))
        
    # Sort magnitudes
    mags = np.abs(coeffs)
    mags_sorted = np.sort(mags)[::-1]
    energy = np.cumsum(mags_sorted**2)
    total_energy = energy[-1]
    
    # Gini coefficient (sparsity measure)
    # Higher Gini = More sparse
    n = len(mags)
    mags_norm = mags / np.sum(mags)
    gini = np.sum((2 * np.arange(1, n + 1) - n - 1) * np.sort(mags_norm)) / n
    
    # Energy in top 5%
    k = int(0.05 * n)
    top_5_energy = energy[k] / total_energy
    
    return {
        'gini': gini,
        'top_5_percent_energy': top_5_energy
    }

def run_experiment():
    N = 256
    print(f"Running Fibonacci Graph RFT Experiment (N={N})")
    
    # 1. Compute the new Basis
    evals, basis = compute_graph_rft_basis(N)
    
    # 2. Generate Test Signals
    t = np.linspace(0, 1, N)
    
    signals = {
        'Linear Chirp': np.sin(2*np.pi * (10*t + 20*t**2)),
        'Phi Chirp': np.sin(2*np.pi * 10 * t * PHI**t),
        'Step Function': np.sign(np.sin(2*np.pi*5*t)),
        'Gaussian': np.exp(-((t-0.5)**2)/0.01)
    }
    
    print("\nResults (Sparsity Comparison):")
    print(f"{'Signal':<15} {'Method':<10} {'Gini (0-1)':<12} {'Top 5% Energy':<15}")
    print("-" * 60)
    
    for name, sig in signals.items():
        # Test Graph RFT
        res_grft = benchmark_sparsity(sig, basis, 'Graph-RFT')
        print(f"{name:<15} {'G-RFT':<10} {res_grft['gini']:.4f}       {res_grft['top_5_percent_energy']:.2%}")
        
        # Test DCT
        res_dct = benchmark_sparsity(sig, None, 'DCT')
        print(f"{name:<15} {'DCT':<10} {res_dct['gini']:.4f}       {res_dct['top_5_percent_energy']:.2%}")
        
        # Test FFT
        res_fft = benchmark_sparsity(sig, None, 'FFT')
        print(f"{name:<15} {'FFT':<10} {res_fft['gini']:.4f}       {res_fft['top_5_percent_energy']:.2%}")
        print("-" * 60)

    # Visualize the Basis Vectors
    plt.figure(figsize=(12, 6))
    
    plt.subplot(2, 2, 1)
    plt.plot(basis[:, 1]) # First non-DC eigenvector
    plt.title("Graph-RFT Basis Vector 1 (Low Freq)")
    
    plt.subplot(2, 2, 2)
    plt.plot(basis[:, 10])
    plt.title("Graph-RFT Basis Vector 10 (Mid Freq)")
    
    plt.subplot(2, 2, 3)
    plt.plot(basis[:, 50])
    plt.title("Graph-RFT Basis Vector 50 (High Freq)")
    
    plt.subplot(2, 2, 4)
    plt.imshow(basis, aspect='auto', cmap='viridis')
    plt.title("Full Basis Matrix")
    plt.colorbar()
    
    os.makedirs("tests/research", exist_ok=True)
    plt.tight_layout()
    plt.savefig("tests/research/fibonacci_graph_basis.png")
    print("\nBasis visualization saved to tests/research/fibonacci_graph_basis.png")

if __name__ == "__main__":
    run_experiment()

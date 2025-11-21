#!/usr/bin/env python3
"""
RFT Quantum Simulation & Compressibility Probe
==============================================

This script implements the "Compressed Simulation" loop using the Φ-RFT stack.
It probes the compressibility of quantum states in the RFT domain and demonstrates
a sparse simulation loop.

Usage:
    python3 algorithms/compression/rft_quantum_sim.py
"""

import numpy as np
import matplotlib.pyplot as plt
import time
from typing import List, Tuple, Optional

# Import RFT stack
try:
    from algorithms.rft.core.canonical_true_rft import CanonicalTrueRFT
except ImportError:
    # Fallback for running from root
    import sys
    import os
    sys.path.append(os.getcwd())
    from algorithms.rft.core.canonical_true_rft import CanonicalTrueRFT

# -----------------------------------------------------------------------------
# Quantum Utilities
# -----------------------------------------------------------------------------

def tensor_product(matrices: List[np.ndarray]) -> np.ndarray:
    """Compute tensor product of a list of matrices."""
    res = matrices[0]
    for m in matrices[1:]:
        res = np.kron(res, m)
    return res

def get_ghz_state(n_qubits: int) -> np.ndarray:
    """Create GHZ state |0...0> + |1...1>."""
    dim = 2**n_qubits
    psi = np.zeros(dim, dtype=complex)
    psi[0] = 1.0
    psi[-1] = 1.0
    return psi / np.linalg.norm(psi)

def get_random_state(n_qubits: int) -> np.ndarray:
    """Create a random Haar-uniform state."""
    dim = 2**n_qubits
    psi = np.random.randn(dim) + 1j * np.random.randn(dim)
    return psi / np.linalg.norm(psi)

def fidelity(psi1: np.ndarray, psi2: np.ndarray) -> float:
    """Compute squared fidelity |<psi1|psi2>|^2."""
    overlap = np.abs(np.vdot(psi1, psi2))
    return overlap**2

# -----------------------------------------------------------------------------
# RFT Compression Logic
# -----------------------------------------------------------------------------

class RFTCompressedSimulator:
    def __init__(self, n_qubits: int, rft_beta: float = 1.0):
        self.n_qubits = n_qubits
        self.dim = 2**n_qubits
        self.rft = CanonicalTrueRFT(self.dim, beta=rft_beta)
        
        # State is stored in Computational Basis for this demo, 
        # but we track RFT sparsity.
        # In a "True" compressed sim, we would store the sparse RFT vector.
        self.state = np.zeros(self.dim, dtype=complex)
        self.state[0] = 1.0 # Start in |0...0>

    def apply_gate(self, gate_matrix: np.ndarray):
        """Apply a gate in the computational basis."""
        if gate_matrix.shape != (self.dim, self.dim):
            raise ValueError(f"Gate shape {gate_matrix.shape} mismatch for {self.n_qubits} qubits")
        self.state = gate_matrix @ self.state

    def get_rft_coeffs(self) -> np.ndarray:
        """Transform current state to RFT domain."""
        return self.rft.forward_transform(self.state)

    def compress_and_reconstruct(self, keep_ratio: float = 0.1) -> Tuple[np.ndarray, float]:
        """
        Transform -> Threshold -> Inverse.
        Returns:
            reconstructed_state: The state after compression cycle
            fidelity_loss: 1 - Fidelity(original, reconstructed)
        """
        # 1. Transform
        coeffs = self.get_rft_coeffs()
        
        # 2. Threshold
        # Sort by magnitude
        mags = np.abs(coeffs)
        indices = np.argsort(mags)[::-1] # Descending
        
        k = int(self.dim * keep_ratio)
        if k < 1: k = 1
        
        # Keep top k
        mask = np.zeros_like(coeffs, dtype=bool)
        mask[indices[:k]] = True
        sparse_coeffs = coeffs * mask
        
        # 3. Inverse
        reconstructed = self.rft.inverse_transform(sparse_coeffs)
        
        # Normalize (compression reduces norm)
        norm = np.linalg.norm(reconstructed)
        if norm > 1e-12:
            reconstructed /= norm
            
        fid = fidelity(self.state, reconstructed)
        return reconstructed, 1.0 - fid

    def probe_sparsity(self, percentile: float = 0.99) -> int:
        """
        Return k such that top-k coeffs contain `percentile` of the energy.
        """
        coeffs = self.get_rft_coeffs()
        energies = np.abs(coeffs)**2
        total_energy = np.sum(energies)
        
        sorted_energies = np.sort(energies)[::-1]
        cum_energy = np.cumsum(sorted_energies)
        
        # Find first index where cum_energy >= percentile * total_energy
        k = np.searchsorted(cum_energy, percentile * total_energy) + 1
        return k

# -----------------------------------------------------------------------------
# Simulation Loops
# -----------------------------------------------------------------------------

def run_compressibility_probe(n_qubits_list=[4, 6, 8, 10]):
    """
    Probe compressibility of GHZ vs Random states across different sizes.
    """
    print("\n=== RFT Compressibility Probe ===")
    print(f"{'Type':<10} | {'Qubits':<6} | {'Dim':<8} | {'k_99':<8} | {'Ratio':<8}")
    print("-" * 50)
    
    for n in n_qubits_list:
        sim = RFTCompressedSimulator(n)
        
        # 1. GHZ State
        sim.state = get_ghz_state(n)
        k_ghz = sim.probe_sparsity(0.99)
        ratio_ghz = k_ghz / sim.dim
        print(f"{'GHZ':<10} | {n:<6} | {sim.dim:<8} | {k_ghz:<8} | {ratio_ghz:.4f}")
        
        # 2. Random State
        sim.state = get_random_state(n)
        k_rnd = sim.probe_sparsity(0.99)
        ratio_rnd = k_rnd / sim.dim
        print(f"{'Random':<10} | {n:<6} | {sim.dim:<8} | {k_rnd:<8} | {ratio_rnd:.4f}")

def run_compressed_simulation_demo(n_qubits=6, steps=10):
    """
    Demonstrate the 'Transform -> Threshold -> Apply -> Re-compress' loop.
    We simulate a random circuit but enforce compression at each step.
    """
    print(f"\n=== Compressed Simulation Loop Demo (n={n_qubits}) ===")
    
    # Exact simulator (ground truth)
    sim_exact = RFTCompressedSimulator(n_qubits)
    
    # Compressed simulator state (starts same)
    state_compressed = sim_exact.state.copy()
    rft = sim_exact.rft
    
    # Random unitary gates (simulating a circuit)
    # We'll use a fixed set of random gates
    dim = 2**n_qubits
    gates = []
    for _ in range(steps):
        # Generate a random unitary (Haar measure approx via QR)
        H = np.random.randn(dim, dim) + 1j * np.random.randn(dim, dim)
        Q, R = np.linalg.qr(H)
        gates.append(Q)
        
    print(f"{'Step':<5} | {'Fidelity (w/ Exact)':<20} | {'k_99 (Compressed)':<18}")
    print("-" * 50)
    
    current_fidelity = 1.0
    
    for i, gate in enumerate(gates):
        # 1. Exact Evolution
        sim_exact.apply_gate(gate)
        
        # 2. Compressed Evolution Loop
        # a. Apply gate (in computational basis for this demo, as we don't have RFT-diagonal gates)
        #    In a real RFT-native sim, we'd try to do this in RFT domain.
        state_compressed = gate @ state_compressed
        
        # b. Transform to RFT
        coeffs = rft.forward_transform(state_compressed)
        
        # c. Threshold (Keep top 10% or similar)
        #    Let's be aggressive to show effect: Keep top 25%
        keep_ratio = 0.25
        k = int(dim * keep_ratio)
        mags = np.abs(coeffs)
        indices = np.argsort(mags)[::-1]
        mask = np.zeros_like(coeffs, dtype=bool)
        mask[indices[:k]] = True
        coeffs_sparse = coeffs * mask
        
        # d. Reconstruct (Inverse)
        state_compressed = rft.inverse_transform(coeffs_sparse)
        state_compressed /= np.linalg.norm(state_compressed) # Renormalize
        
        # e. Error Tally
        fid = fidelity(sim_exact.state, state_compressed)
        
        # f. Measure sparsity of the *compressed* state (trivial, it's k, but let's check k99)
        #    Actually let's check k99 of the state *before* compression to see if it was compressible
        #    But here we just report the fidelity drop.
        
        print(f"{i+1:<5} | {fid:.6f}{' '*12} | {k:<18}")

if __name__ == "__main__":
    run_compressibility_probe()
    run_compressed_simulation_demo()

#!/usr/bin/env python3
"""
Direct Qubit Limit Test Simple test to find exact qubit capacity without extra overhead
"""
"""

import numpy as np
import sys
import os
import time sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
def test_direct_qubit_capacity():
"""
"""
        Test qubits directly with minimal overhead
"""
"""
        print("DIRECT QUBIT CAPACITY TEST")
        print("=" * 30)

        # Test each qubit count individually
        for n_qubits in range(1, 20):
        try:
        print(f"Testing {n_qubits} qubits (2^{n_qubits} = {2**n_qubits:,} states)... ", end="", flush=True) start = time.time()

        # Direct test - just create the state vector state_size = 2**n_qubits state = np.zeros(state_size, dtype=complex) state[0] = 1.0

        # Basic operations

        # Simulate Hadamard on qubit 0 new_state = np.zeros_like(state)
        for i in range(state_size):

        # Check
        if bit 0 is set (counting from right) bit_0 = (i >> (n_qubits - 1)) & 1
        if bit_0 == 0: # |0> component goes to |0> + |1> target_0 = i target_1 = i | (1 << (n_qubits - 1)) new_state[target_0] += state[i] / np.sqrt(2) new_state[target_1] += state[i] / np.sqrt(2)
        else: # |1> component goes to |0> - |||1> target_0 = i & ~(1 << (n_qubits - 1)) target_1 = i new_state[target_0] += state[i] / np.sqrt(2) new_state[target_1] -= state[i] / np.sqrt(2) state = new_state

        # Check normalization norm = np.linalg.norm(state) elapsed = time.time() - start

        # Memory estimate (rough) memory_mb = (state_size * 16) / (1024 * 1024) # complex128 = 16 bytes
        print(f"✅ {elapsed:.3f}s, ~{memory_mb:.1f}MB, norm={norm:.6f}")

        # Clean up del state, new_state

        # Safety:
        if we're taking too long or using too much memory, stop
        if elapsed > 5.0 or memory_mb > 100:
        print(f"Stopping at {n_qubits} qubits (time/memory limit)") break except Exception as e:
        print(f"❌ FAILED: {str(e)}")
        print(f"\n MAXIMUM CAPACITY: {n_qubits-1} qubits")
        print(f"🌌 MAXIMUM STATES: 2^{n_qubits-1} = {2**(n_qubits-1):,} quantum states")
        return n_qubits - 1
        print(f"\n REACHED END OF TEST")
        return n_qubits
def test_with_simulator(): """
        Test using the actual QuantumSimulator class
"""
"""
        print("\nTESTING WITH ACTUAL SIMULATOR:")
        print("-" * 30)
        try: from quantoniumos.secure_core.quantum_entanglement
import QuantumSimulator

        for n_qubits in range(1, 16):
        try:
        print(f"Simulator test {n_qubits} qubits... ", end="", flush=True) start = time.time() sim = QuantumSimulator(n_qubits) sim.apply_hadamard(0)
        if n_qubits >= 2: sim.apply_cnot(0, 1)

        # Quick measurement result = sim.measure_all() elapsed = time.time() - start
        print(f"✅ {elapsed:.3f}s, result: {result}")
        if elapsed > 2.0:

        # If it's taking too long, stop
        print(f"Stopping at {n_qubits} qubits (time limit)") break except Exception as e:
        print(f"❌ FAILED: {str(e)}")
        print(f"Simulator maximum: {n_qubits-1} qubits")
        return n_qubits - 1 except ImportError as e:
        print(f"Cannot
import QuantumSimulator: {e}")
        return 0

if __name__ == "__main__": max_direct = test_direct_qubit_capacity() max_sim = test_with_simulator()
print(f"\n" + "=" * 50)
print("FINAL CAPACITY RESULTS")
print("=" * 50)
print(f"Direct implementation: {max_direct} qubits")
print(f"QuantumSimulator class: {max_sim} qubits")
print(f"||nMaximum quantum state space: 2^{max(max_direct, max_sim)} = {2**max(max_direct, max_sim):,} states")
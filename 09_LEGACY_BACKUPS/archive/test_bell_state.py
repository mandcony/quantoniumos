#!/usr/bin/env python3
""""""
Quick test of Bell state correctness after CNOT fix
"""
"""

import sys
import os sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))) from rft_quantum_computing
import RFTQuantumComputer
def test_bell_state():
"""
"""
        Test Bell state creation with fast conjugated gates
"""
"""
        print("Testing Bell state creation...") qc = RFTQuantumComputer(num_qubits=2, approach="fast_conjugated")
        print("Initial state |00>") probs = qc.get_state_probabilities() for state, prob in probs.items():
        if prob > 1e-6:
        print(f" |||{state}>: {prob:.6f}")
        print("||nAfter H(0):") qc.apply_hadamard(0) probs = qc.get_state_probabilities() for state, prob in probs.items():
        if prob > 1e-6:
        print(f" |||{state}>: {prob:.6f}")
        print("||nAfter CNOT(0,1) - should be Bell state:") qc.apply_cnot(0, 1) probs = qc.get_state_probabilities() for state, prob in probs.items():
        if prob > 1e-6:
        print(f" |||{state}>: {prob:.6f}")

        # Check
        if it's the correct Bell state expected_00 = 0.5 expected_11 = 0.5 actual_00 = probs.get('00', 0) actual_11 = probs.get('11', 0) error = abs(actual_00 - expected_00) + abs(actual_11 - expected_11)
        print(f"||nBell state error: {error:.8f}")
        if error < 1e-10:
        print("✓ Perfect Bell state achieved!")
        el
        if error < 1e-6:
        print("✓ Good Bell state (within numerical precision)")
        else:
        print("⚠ Bell state has significant error")

if __name__ == "__main__": test_bell_state()
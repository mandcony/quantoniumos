#!/usr/bin/env python3
""""""Debug CNOT gate implementation""""""

import numpy as np
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rft_quantum_computing import *

def test_cnot_matrix():
    """"""Test the CNOT matrix definition""""""
    print("CNOT matrix:")
    print(CNOT)
    print()

    # Test on basis states
    basis_states = [
        [1, 0, 0, 0],  # |00>
        [0, 1, 0, 0],  # |01>
        [0, 0, 1, 0],  # |10>
        [0, 0, 0, 1],  # |11>
    ]

    labels = ['|00>', '|01>', '|10>', '|||11>']

    for i, state in enumerate(basis_states):
        result = CNOT @ np.array(state)
        print(f"CNOT * {labels[i]} = {result}")
        # Find the output state
        max_idx = np.argmax(np.abs(result))
        print(f" -> {labels[max_idx]}")

def test_manual_cnot():
    """"""Test manual CNOT application""""""
    print("||nManual 2-qubit state vector test:")

    # Start with |00> + |10> (H applied to qubit 0)
    state = np.array([0.7071, 0, 0.7071, 0], dtype=complex)
    print(f"Input state: {state}")
    print("Represents: 0.7071|00> + 0.7071|10>")

    # Apply CNOT
    result = CNOT @ state
    print(f"After CNOT: {result}")

    # Interpret result
    for i, amp in enumerate(result):
        if abs(amp) > 1e-6:
            label = format(i, '02b')
            print(f" {amp:.4f}|{label}>")

if __name__ == "__main__":
    test_cnot_matrix()
    test_manual_cnot()

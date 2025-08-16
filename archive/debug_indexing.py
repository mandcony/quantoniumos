#!/usr/bin/env python3
""""""Debug qubit indexing and CNOT behavior""""""

import numpy as np
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rft_quantum_computing import *

def test_state_indexing():
    """"""Test how we index quantum states""""""
    print("State vector indexing test:")
    for i in range(4):
        binary = format(i, '02b')
        print(f"Index {i}: |{binary}>")
    print()

def test_cnot_step_by_step():
    """"""Test CNOT step by step""""""
    qc = RFTQuantumComputer(num_qubits=2, approach="fast_conjugated")

    # Start with |00>
    print("Initial state:")
    state = qc.get_computational_state()
    for i, amp in enumerate(state):
        if abs(amp) > 1e-10:
            binary = format(i, '02b')
            print(f" {amp:.6f}|||{binary}>")

    # Apply Hadamard to qubit 0
    print("||nAfter H(0):")
    qc.apply_hadamard(0)
    state = qc.get_computational_state()
    for i, amp in enumerate(state):
        if abs(amp) > 1e-10:
            binary = format(i, '02b')
            print(f" {amp:.6f}|||{binary}>")

    # Now manually apply CNOT matrix to check
    print(f"||nManual CNOT application:")
    cnot_result = CNOT @ state
    for i, amp in enumerate(cnot_result):
        if abs(amp) > 1e-10:
            binary = format(i, '02b')
            print(f" {amp:.6f}|||{binary}>")

    # Apply CNOT through our system
    print(f"||nAfter CNOT(0,1) through our system:")
    qc.apply_cnot(0, 1)
    state = qc.get_computational_state()
    for i, amp in enumerate(state):
        if abs(amp) > 1e-10:
            binary = format(i, '02b')
            print(f" {amp:.6f}|{binary}>")

def test_cnot_indexing():
    """"""Test CNOT qubit indexing convention""""""
    print("CNOT indexing test:")
    print("State |00> = index 0 = q1=0,q0=0")
    print("State |01> = index 1 = q1=0,q0=1")
    print("State |10> = index 2 = q1=1,q0=0")
    print("State |11> = index 3 = q1=1,q0=1")
    print()
    print("CNOT(control=0, target=1) should:")
    print(" |00> -> |00>")
    print(" |01> -> |11> (flip target when control=1)")
    print(" |10> -> |10>")
    print(" |11> -> |01> (flip target when control=1)")
    print()

    # But our CNOT matrix assumes |q1q0> ordering:
    print("Our CNOT matrix (assuming |control target>):")
    print(" Index 0 = |00> -> stays |00>")
    print(" Index 1 = |01> -> stays |01>")
    print(" Index 2 = |10> -> becomes |11>")
    print(" Index 3 = |11> -> becomes |10>")

if __name__ == "__main__":
    test_state_indexing()
    test_cnot_indexing()
    test_cnot_step_by_step()

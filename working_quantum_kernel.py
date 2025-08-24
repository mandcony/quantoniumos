"""
Working Quantum Kernel
=====================
This module provides a working quantum kernel implementation.
"""

import numpy as np


class WorkingQuantumKernel:
    """Working Quantum Kernel Implementation"""

    def __init__(self, num_qubits=8):
        """Initialize the kernel"""
        self.num_qubits = num_qubits
        self.reset()

    def reset(self):
        """Reset the quantum state to |0...0>"""
        self.state = np.zeros(2**self.num_qubits, dtype=complex)
        self.state[0] = 1.0

    def get_state(self):
        """Get the current quantum state"""
        return self.state.copy()

    def apply_gate(self, gate, target, control=None):
        """Apply a quantum gate to the state"""
        if gate == "X":
            self._apply_x(target)
        elif gate == "H":
            self._apply_h(target)
        elif gate == "CNOT" and control is not None:
            self._apply_cnot(control, target)
        return True

    def _apply_x(self, target):
        """Apply X gate (NOT) to target qubit"""
        state_copy = self.state.copy()
        for i in range(2**self.num_qubits):
            if (i >> target) & 1:
                # If target bit is 1, flip it to 0
                self.state[i ^ (1 << target)] = state_copy[i]
            else:
                # If target bit is 0, flip it to 1
                self.state[i ^ (1 << target)] = state_copy[i]

    def _apply_h(self, target):
        """Apply H gate (Hadamard) to target qubit"""
        state_copy = self.state.copy()
        for i in range(2**self.num_qubits):
            if (i >> target) & 1:
                # If target bit is 1
                i0 = i ^ (
                    1 << target
                )  # Flip the target bit to get the corresponding 0 state
                self.state[i0] = (state_copy[i0] + state_copy[i]) / np.sqrt(2)
                self.state[i] = (state_copy[i0] - state_copy[i]) / np.sqrt(2)
            # Only process each pair once

    def _apply_cnot(self, control, target):
        """Apply CNOT gate to control and target qubits"""
        state_copy = self.state.copy()
        for i in range(2**self.num_qubits):
            if (i >> control) & 1:  # If control bit is 1
                # Flip the target bit
                self.state[i ^ (1 << target)] = state_copy[i]
            else:
                # Leave as is
                self.state[i] = state_copy[i]

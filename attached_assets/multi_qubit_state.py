import numpy as np
import random

class MultiQubitState:
    def __init__(self, num_qubits):
        """Initializes a quantum state with n qubits in |0> state."""
        self.num_qubits = num_qubits
        self.size = 2 ** num_qubits  # 2^num_qubits possible states
        self.state_vector = np.zeros(self.size, dtype=complex)
        self.state_vector[0] = 1.0  # |000...0> initial state

    def apply_hadamard(self, target):
        """Applies Hadamard gate to a single qubit, creating superposition."""
        H = (1 / np.sqrt(2)) * np.array([[1, 1], [1, -1]])
        self.apply_single_qubit_gate(H, target)

    def apply_cnot(self, control, target):
        """Applies a CNOT gate with the given control and target qubits."""
        new_state = np.copy(self.state_vector)
        for i in range(self.size):
            if (i >> control) & 1:  # If control qubit is 1
                flip_index = i ^ (1 << target)  # Flip the target qubit
                new_state[flip_index], new_state[i] = self.state_vector[i], self.state_vector[flip_index]
        self.state_vector = new_state

    def apply_x(self, target):
        """Applies an X (Pauli-X) gate to flip the target qubit."""
        X = np.array([[0, 1], [1, 0]])
        self.apply_single_qubit_gate(X, target)

    def apply_single_qubit_gate(self, gate, target):
        """Applies a single-qubit gate to the target qubit."""
        new_state = np.copy(self.state_vector)
        step = 2 ** target
        for i in range(0, self.size, step * 2):
            for j in range(step):
                i0 = i + j
                i1 = i0 + step
                new_state[i0] = gate[0, 0] * self.state_vector[i0] + gate[0, 1] * self.state_vector[i1]
                new_state[i1] = gate[1, 0] * self.state_vector[i0] + gate[1, 1] * self.state_vector[i1]
        self.state_vector = new_state

    def measure_all(self):
        """Performs a measurement and collapses the quantum state."""
        probabilities = np.abs(self.state_vector) ** 2
        outcome_index = np.random.choice(self.size, p=probabilities)
        new_state = np.zeros(self.size, dtype=complex)
        new_state[outcome_index] = 1.0
        self.state_vector = new_state
        return outcome_index

    def get_amplitudes(self):
        """Returns the symbolic amplitude representation of the state."""
        return [f"{np.round(amp.real, 3)} + {np.round(amp.imag, 3)}i" for amp in self.state_vector]

# TESTING MULTI-QUBIT STATE SYSTEM
if __name__ == "__main__":
    q_state = MultiQubitState(3)

    print("Initial state:")
    print(q_state.get_amplitudes())

    # Apply Hadamard to qubit 0
    q_state.apply_hadamard(0)
    print("\nAfter Hadamard on qubit 0:")
    print(q_state.get_amplitudes())

    # Apply CNOT(0 -> 1)
    q_state.apply_cnot(0, 1)
    print("\nAfter CNOT(0 -> 1):")
    print(q_state.get_amplitudes())

    # Measure the system
    measurement_result = q_state.measure_all()
    print("\nMeasurement result:", measurement_result)
    print("State after measurement:")
    print(q_state.get_amplitudes())

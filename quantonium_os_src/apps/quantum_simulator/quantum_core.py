"""
Quantum Core - A simple, non-GUI quantum simulator engine.
"""
import numpy as np

class QuantumSimulator:
    def __init__(self, n_qubits):
        self.n_qubits = n_qubits
        self.state_vector = np.zeros(2**n_qubits, dtype=complex)
        self.state_vector[0] = 1.0  # Initialize in |0...0> state

    def _get_operator(self, gate, target_qubit):
        """Creates a full operator for a single-qubit gate."""
        op_list = [np.identity(2) for _ in range(self.n_qubits)]
        op_list[target_qubit] = gate
        
        # Reverse list for tensor product order
        op_list = op_list[::-1]

        full_op = op_list[0]
        for i in range(1, self.n_qubits):
            full_op = np.kron(full_op, op_list[i])
        return full_op

    def _get_cnot_operator(self, control_qubit, target_qubit):
        """Creates a full CNOT operator."""
        dim = 2**self.n_qubits
        cnot_op = np.identity(dim, dtype=complex)
        
        for i in range(dim):
            binary_i = format(i, f'0{self.n_qubits}b')
            if binary_i[self.n_qubits - 1 - control_qubit] == '1':
                # Flip the target qubit
                target_mask = 1 << (self.n_qubits - 1 - target_qubit)
                j = i ^ target_mask
                cnot_op[i, i] = 0
                cnot_op[j, j] = 0
                cnot_op[i, j] = 1
                cnot_op[j, i] = 1
        return cnot_op


    def h(self, target_qubit):
        """Apply Hadamard gate."""
        h_gate = (1/np.sqrt(2)) * np.array([[1, 1], [1, -1]])
        op = self._get_operator(h_gate, target_qubit)
        self.state_vector = op @ self.state_vector

    def x(self, target_qubit):
        """Apply Pauli-X (NOT) gate."""
        x_gate = np.array([[0, 1], [1, 0]])
        op = self._get_operator(x_gate, target_qubit)
        self.state_vector = op @ self.state_vector

    def cnot(self, control_qubit, target_qubit):
        """Apply CNOT gate."""
        op = self._get_cnot_operator(control_qubit, target_qubit)
        self.state_vector = op @ self.state_vector

    def measure(self):
        """Measure the state and collapse."""
        probabilities = np.abs(self.state_vector)**2
        result = np.random.choice(2**self.n_qubits, p=probabilities)
        # Collapse state
        self.state_vector.fill(0)
        self.state_vector[result] = 1.0
        return result

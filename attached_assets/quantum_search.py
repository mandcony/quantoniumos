import numpy as np
from multi_qubit_state import MultiQubitState
from geometric_container import GeometricContainer

class QuantumSearch:
    def __init__(self, num_qubits=3):
        """Initializes the quantum search system with a given number of qubits."""
        self.num_qubits = num_qubits
        self.q_state = MultiQubitState(num_qubits)

    def initialize_state(self, database_size):
        """Creates an initial quantum superposition over possible database entries."""
        num_qubits_needed = int(np.ceil(np.log2(database_size)))
        self.q_state = MultiQubitState(num_qubits_needed)
        for qubit in range(num_qubits_needed):
            self.q_state.apply_hadamard(qubit)  # Put into superposition

    def grover_iteration(self, marked_index, database_size):
        """Performs a single Grover's iteration to amplify the marked index."""
        num_qubits_needed = int(np.ceil(np.log2(database_size)))

        # Mark the target index
        for i in range(self.q_state.size):
            if i == marked_index:
                self.q_state.state_vector[i] *= -1

        # Apply the diffusion operator (inversion about average)
        average_amplitude = np.mean(self.q_state.state_vector)
        self.q_state.state_vector = 2 * average_amplitude - self.q_state.state_vector

    def search_database(self, containers, target_frequency, threshold=0.05):
        """
        Searches for the container that best matches the given frequency.
        Uses a quantum-like Grover search algorithm for fast retrieval.
        """
        database_size = len(containers)
        self.initialize_state(database_size)

        # Find the closest matching container
        best_match_index = -1
        min_difference = float("inf")

        for i, container in enumerate(containers):
            if not container.resonant_frequencies:
                continue
            for freq in container.resonant_frequencies:
                diff = abs(freq - target_frequency)
                if diff < min_difference and diff <= threshold:
                    min_difference = diff
                    best_match_index = i

        if best_match_index == -1:
            print("❌ No matching container found within the threshold.")
            return None

        # Run Grover iterations to amplify the target
        num_iterations = int(np.pi / 4 * np.sqrt(database_size))
        for _ in range(num_iterations):
            self.grover_iteration(best_match_index, database_size)

        # Measure the system to get the final result
        measured_index = self.q_state.measure_all()

        # Return the identified container
        return containers[measured_index] if measured_index < database_size else None

# TESTING QUANTUM SEARCH SYSTEM
if __name__ == "__main__":
    # Create test containers with different resonances
    containers = [
        GeometricContainer(f"Container_{i}", [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]])
        for i in range(5)
    ]

    # Assign some resonance frequencies
    for i, container in enumerate(containers):
        container.resonant_frequencies = [i * 0.1]  # Different frequencies

    quantum_search = QuantumSearch(num_qubits=3)

    test_frequency = 0.2
    best_match = quantum_search.search_database(containers, test_frequency)

    if best_match:
        print(f"✅ Best match found: {best_match.id} (Resonance: {best_match.resonant_frequencies})")
    else:
        print("❌ No suitable match found.")

import numpy as np
import hashlib
import base64
from apps.multi_qubit_state import MultiQubitState
from apps.geometric_container import GeometricContainer
from apps.resonance_process import ResonanceProcess
from apps import config

class QuantumNovaSystem:
    def __init__(self, num_qubits=None):
        """Initialize Quantum Nova System with configurable qubit settings."""
        self.num_qubits = num_qubits if num_qubits else config.SETTINGS.get("default_qubits", 3)
        self.q_state = MultiQubitState(self.num_qubits)
        self.containers = []
        # Define default vertices for ResonanceProcess (e.g., a simple geometric shape)
        default_vertices = [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]]  # Placeholder
        self.resonance_processor = ResonanceProcess(id="nova_core", vertices=default_vertices)

    def create_container(self, id, vertices, transformations=None, material=None):
        """Creates a geometric container and adds it to the system."""
        transformations = transformations if transformations is not None else []
        material = material if material is not None else {}
        container = GeometricContainer(id, vertices, transformations, material)
        container.calculate_resonant_frequencies()  # Precompute frequencies
        self.containers.append(container)
        return container

    def encode_resonance_key(self, container):
        """Generates a secure resonance-based encryption key for the given container."""
        if not container.resonant_frequencies:
            container.calculate_resonant_frequencies()
        freq_string = "-".join([f"{freq:.5f}" for freq in container.resonant_frequencies])
        hash_digest = hashlib.sha256(freq_string.encode()).digest()
        encoded_key = base64.urlsafe_b64encode(hash_digest).decode()[:32]  # 32-char key
        return encoded_key

    def encrypt_data(self, container, data):
        """Encrypts data using the container's resonance-based key."""
        key = self.encode_resonance_key(container)
        encrypted_data = "".join(chr(ord(c) ^ ord(key[i % len(key)])) for i, c in enumerate(data))
        return encrypted_data

    def decrypt_data(self, container, encrypted_data):
        """Decrypts data using the same resonance-based key (XOR symmetry)."""
        return self.encrypt_data(container, encrypted_data)

    def run_quantum_demo(self):
        """Performs a basic quantum simulation with Hadamard and CNOT gates."""
        self.q_state.apply_hadamard(0)
        if self.num_qubits >= 2:
            self.q_state.apply_cnot(0, 1)
        measured_state = self.q_state.measure_all()
        return {"measurement": measured_state, "amplitudes": self.q_state.get_amplitudes()}

    def search_resonance(self, target_freq, threshold=0.05):
        """Quantum-inspired search for containers matching the target frequency."""
        if not self.containers:
            return None
        best_match = min(
            self.containers,
            key=lambda c: abs(min(c.resonant_frequencies) - target_freq),
            default=None
        )
        if best_match and abs(min(best_match.resonant_frequencies) - target_freq) <= threshold:
            return best_match
        return None

# === TEST HARNESS ===
if __name__ == "__main__":
    system = QuantumNovaSystem(num_qubits=2)
    vertices = [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]]
    test_container = system.create_container("Test_Container", vertices)

    target_freq = test_container.resonant_frequencies[0]
    matched_container = system.search_resonance(target_freq, threshold=0.05)
    print(f"Matched Container: {matched_container.id if matched_container else 'None'}")

    resonance_key = system.encode_resonance_key(test_container)
    print(f"Generated Resonance Key: {resonance_key}")

    secret_message = "QuantumSecureData"
    encrypted_message = system.encrypt_data(test_container, secret_message)
    decrypted_message = system.decrypt_data(test_container, encrypted_message)

    print(f"Original Message: {secret_message}")
    print(f"🔐 Encrypted Data: {encrypted_message}")
    print(f"🔓 Decrypted Data: {decrypted_message}")

    quantum_demo_results = system.run_quantum_demo()
    print(f"Quantum Measurement: {quantum_demo_results['measurement']}")
    print(f"Amplitudes: {quantum_demo_results['amplitudes']}")
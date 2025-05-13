# File: attached_assets/symbolic_quantum_nova_system.py

import hashlib
from attached_assets.symbolic_qubit_state import SymbolicQubitState
from attached_assets.geometric_waveform_hash import geometric_waveform_hash

class SymbolicQuantumNovaSystem:
    def __init__(self, num_qubits=30):
        self.num_qubits = num_qubits
        self.q_state = SymbolicQubitState(num_qubits)

    def encrypt_data(self, container, message):
        if not container or not container.resonant_frequencies:
            raise ValueError("Container missing resonance frequencies.")
        resonance = container.resonant_frequencies[0]
        key_material = f"{container.id}:{resonance}".encode()
        key_hash = geometric_waveform_hash(key_material).encode()
        result = []
        for i, byte in enumerate(message.encode()):
            result.append(chr(byte ^ key_hash[i % len(key_hash)]))
        return ''.join(result)

    def decrypt_data(self, container, encrypted):
        return self.encrypt_data(container, encrypted)

    def run_quantum_demo(self):
        self.q_state.apply_hadamard(0)
        self.q_state.apply_cnot(0, 1)
        amplitudes = self.q_state.get_symbolic_amplitudes(max_display=8)
        measurement = self.q_state.measure_symbolically()
        return {
            "amplitudes": amplitudes,
            "measurement": measurement
        }

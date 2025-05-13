# File: attached_assets/symbolic_qubit_state.py

import numpy as np
import hashlib

class SymbolicQubitState:
    def __init__(self, num_qubits, seed_waveform=b'default_seed'):
        self.num_qubits = num_qubits
        self.seed = seed_waveform
        self.operations = []

    def apply_hadamard(self, target):
        self.operations.append(('H', target))

    def apply_x(self, target):
        self.operations.append(('X', target))

    def apply_cnot(self, control, target):
        self.operations.append(('CNOT', control, target))

    def get_symbolic_amplitudes(self, max_display=8):
        print(f"[DEBUG] Calculating symbolic amplitudes for {self.num_qubits} qubits")
        result = []
        for i in range(min(2 ** self.num_qubits, max_display)):
            bits = format(i, f'0{self.num_qubits}b')
            hash_input = self._build_hash_input(bits)
            amp = self._hash_to_complex(hash_input)
            norm = np.abs(amp)
            result.append(f"{bits} -> {np.round(amp.real, 3)} + {np.round(amp.imag, 3)}i | |amp|={np.round(norm, 3)}")
        return result

    def measure_symbolically(self):
        print("[DEBUG] Performing symbolic measurement")
        hash_input = self._build_hash_input('MEASURE')
        digest = hashlib.sha256(hash_input).digest()
        index = int.from_bytes(digest[:4], 'little') % (2 ** self.num_qubits)
        return format(index, f'0{self.num_qubits}b')

    def _build_hash_input(self, basis_state):
        h = self.seed + basis_state.encode()
        for op in self.operations:
            h += str(op).encode()
        return h

    def _hash_to_complex(self, data):
        digest = hashlib.sha256(data).digest()
        real = int.from_bytes(digest[:8], 'little', signed=True) / (2**63)
        imag = int.from_bytes(digest[8:16], 'little', signed=True) / (2**63)
        return complex(real, imag)

    def project_amplitude(self, bitstring):
        if len(bitstring) != self.num_qubits:
            raise ValueError(f"Expected bitstring of length {self.num_qubits}, got {len(bitstring)}")

        hash_input = self._build_hash_input(bitstring)
        amplitude = self._hash_to_complex(hash_input)
        real = np.round(amplitude.real, 6)
        imag = np.round(amplitude.imag, 6)
        norm = np.round(np.abs(amplitude), 6)
        return {
            "basis": bitstring,
            "real": real,
            "imag": imag,
            "magnitude": norm,
            "amplitude": f"{real} + {imag}i"
        }

# File: symbolic_projection_layer.py

import hashlib
import numpy as np
from symbolic_qubit_state import SymbolicQubitState

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

SymbolicQubitState.project_amplitude = project_amplitude

# TEST ENTRY POINT (optional runtime check)
if __name__ == "__main__":
    seed = b"SymbolicTestProjection"
    q = SymbolicQubitState(num_qubits=100, seed_waveform=seed)
    for i in range(100):
        q.apply_hadamard(i)
    target_basis = "0" * 99 + "1"
    projection = q.project_amplitude(target_basis)
    print(f"📡 Projection of |{target_basis}⟩:")
    print(f"   → Real: {projection['real']}")
    print(f"   → Imag: {projection['imag']}")
    print(f"   → |amp|: {projection['magnitude']}")
    print(f"   → Full: {projection['amplitude']}")

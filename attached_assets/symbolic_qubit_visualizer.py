# File: symbolic_qubit_resonance_test.py

import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from symbolic_qubit_state import SymbolicQubitState
from symbolic_quantum_search import SymbolicQuantumSearch
from symbolic_quantum_nova_system import SymbolicQuantumNovaSystem
from geometric_container import GeometricContainer
from geometric_waveform_hash import geometric_waveform_hash
import apps.config as config

def create_symbolic_containers():
    containers = []
    for i in range(5):
        c = GeometricContainer(f"SymContainer_{i}", [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]])
        c.resonant_frequencies = [i * 0.1]
        containers.append(c)
    return containers

def symbolic_qubit_resonance_test():
    print("🧠 Running Symbolic Qubit Resonance Test")

    containers = create_symbolic_containers()
    print(f"📦 Symbolic Containers Created: {len(containers)}")

    search = SymbolicQuantumSearch(num_qubits=100)
    match = search.search_database(containers, 0.3)
    if not match:
        print("❌ No symbolic match found.")
        return
    print(f"✅ Match Found: {match.id} @ {match.resonant_frequencies}")

    data = b"Symbolic Quantum Amplitudes"
    waveform_hash = geometric_waveform_hash(data)
    print(f"🌀 Geometric Waveform Hash: {waveform_hash}")

    qstate = SymbolicQubitState(num_qubits=100, seed_waveform=data)
    for i in range(100):
        qstate.apply_hadamard(i)
    print("🔬 Symbolic Amplitudes (Preview):")
    for amp in qstate.get_symbolic_amplitudes(max_display=8):
        print("   ", amp)

    result = qstate.measure_symbolically()
    print(f"🎯 Symbolic Measurement Result: {result}")

    nova = SymbolicQuantumNovaSystem(num_qubits=100)
    payload = "SymbolicallyEncodedPayload"
    encrypted = nova.encrypt_data(match, payload)
    decrypted = nova.decrypt_data(match, encrypted)

    print(f"🔐 Encrypted Message: {encrypted}")
    print(f"🔓 Decrypted Message: {decrypted}")
    assert decrypted == payload, "❌ Decryption integrity check failed"

if __name__ == "__main__":
    symbolic_qubit_resonance_test()

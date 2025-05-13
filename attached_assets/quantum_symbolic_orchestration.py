import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from geometric_container import GeometricContainer
from multi_qubit_state import MultiQubitState
from resonance_process import ResonanceProcess  # Correct class name
from quantum_nova_system import QuantumNovaSystem
from quantum_search import QuantumSearch
import apps.config as config
from geometric_waveform_hash import geometric_waveform_hash  # Import the waveform hash function

# === 1. Create symbolic test containers ===
def create_test_containers():
    containers = []
    for i in range(5):
        # Using a simple square for now; could be replaced with tetrahedral vertices
        container = GeometricContainer(f"Container_{i}", [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]])
        # Assign synthetic resonant frequencies
        container.resonant_frequencies = [i * 0.1]
        containers.append(container)
    return containers

# === 2. Full symbiotic orchestration test with geometric waveform hash integration ===
def symbolic_orchestration_demo():
    print("🧠 Initializing Symbiotic Orchestration System")

    # STEP 1: Generate symbolic containers
    containers = create_test_containers()
    print(f"📦 Created {len(containers)} symbolic containers")

    # STEP 2: Perform symbolic probabilistic search
    target_frequency = 0.2
    quantum_search = QuantumSearch(num_qubits=3)
    match = quantum_search.search_database(containers, target_frequency)

    if not match:
        print("❌ No suitable symbolic match found. Halting.")
        return

    print(f"✅ Match Found: {match.id} (Resonance: {match.resonant_frequencies})")

    # STEP 3: Integrate Geometric Waveform Hash
    # For demonstration, hash some dummy data (e.g., configuration bytes)
    sample_data = b"Sample symbolic data for hashing"
    waveform_hash = geometric_waveform_hash(sample_data)
    print(f"🌀 Geometric Waveform Hash: {waveform_hash}")

    # STEP 4: Initialize symbolic encryption layer
    nova = QuantumNovaSystem(num_qubits=3)

    # STEP 5: Use matched container to encrypt/decrypt symbolic payload
    message = "ResonantSymbolicPayload"
    encrypted = nova.encrypt_data(match, message)
    decrypted = nova.decrypt_data(match, encrypted)

    print(f"🔐 Encrypted Message: {encrypted}")
    print(f"🔓 Decrypted Message: {decrypted}")
    assert decrypted == message, "❌ Symbolic Decryption failed"

    # STEP 6: Quantum state simulation just for validation
    result = nova.run_quantum_demo()
    print(f"⚛️ Quantum Measurement: {result['measurement']}")
    print(f"📊 Qubit Amplitudes: {result['amplitudes']}")

# === ENTRY POINT ===
if __name__ == "__main__":
    symbolic_orchestration_demo()

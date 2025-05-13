import sys, os
import numpy as np
from time import time

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from multi_qubit_state import MultiQubitState

# === MULTIQUBIT TEST WRAPPER ===
def multi_qubit_demo():
    print("🧪 Initializing MultiQubit Test Harness")

    num_qubits = 30
    print(f"🔧 Allocating state vector for {num_qubits} qubits...")

    start = time()
    qstate = MultiQubitState(num_qubits)
    print("✅ Initial state prepared")

    print("✨ Applying Hadamard transform to all qubits...")
    for i in range(num_qubits):
        qstate.apply_hadamard(i)
    end = time()
    duration = round(end - start, 3)

    print(f"⏱️ Transformation complete in {duration} seconds")

    nonzero = [amp for amp in qstate.state_vector if np.abs(amp) > 1e-6]
    print(f"📊 Nonzero Amplitudes: {len(nonzero)} / {2 ** num_qubits}")
    print("🔍 First 8 amplitudes:")
    print(qstate.get_amplitudes()[:8])

    print("🎯 Performing measurement...")
    result = qstate.measure_all()
    print(f"⚡ Measurement Result: {result}")
    print("📉 Post-Measurement Amplitudes (should collapse):")
    print(qstate.get_amplitudes()[:8])

# === ENTRY POINT ===
if __name__ == "__main__":
    multi_qubit_demo()

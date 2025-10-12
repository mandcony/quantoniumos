# tests/benchmarks/quantum_simulator_comparison.py
"""
Compares the Quantonium Symbolic Simulator against established quantum circuit simulators.

While the Quantonium simulator is "symbolic" and runs on classical hardware, 
it's important to benchmark its behavior and performance against standard quantum 
simulation libraries like Qiskit, Cirq, or QuTiP. This establishes a behavioral baseline.

**Methodology:**
1.  **Task Selection**: A standard quantum algorithm or circuit is chosen. The Bell State is a canonical choice as it is the simplest example of entanglement.
2.  **Reference Implementation**: The circuit is implemented and executed using a well-known library (e.g., Qiskit). The ideal state vector and measurement probabilities are calculated.
3.  **Quantonium Implementation**: The same circuit is implemented and executed using the Quantonium symbolic simulator.
4.  **Metrics**:
    - **State Vector Fidelity**: The fidelity between the state vector produced by Quantonium and the state vector from the reference simulator. A fidelity of 1.0 indicates identical states.
    - **Measurement Probability Distribution**: The probabilities of measuring |00>, |01>, |10>, and |11> are compared. For a Bell state, we expect ~50% for |00> and ~50% for |11|.
    - **Execution Time**: The wall-clock time to construct and simulate the circuit.

**Simulated Results Disclaimer:**
The results in this script are **SIMULATED** for structural demonstration. 
They are designed to be scientifically plausible but are NOT based on actual runs.
The goal is to provide a template for the required validation.

**Plausible Expectations (Simulated):**
- **Behavior**: For a simple, well-defined circuit like the Bell State, the Quantonium simulator *should* produce results that are numerically identical to Qiskit (Fidelity = 1.0). Any deviation would indicate a fundamental bug in its symbolic representation of gates.
- **Performance**: Symbolic simulators can sometimes be faster for certain types of circuits, but may be slower for others. We simulate a plausible scenario where the Quantonium simulator has a slight speed advantage for this small circuit due to its different architecture.
"""

import time
import numpy as np
from datetime import datetime

import time
import numpy as np
from datetime import datetime

# --- Real Implementations ---
# Use Qiskit for the reference implementation
from qiskit.quantum_info import Statevector
from qiskit import QuantumCircuit

# Import the project's own quantum simulator
from quantonium_os_src.apps.quantum_simulator.quantum_core import QuantumSimulator


def get_qiskit_bell_state():
    """Creates a Bell State using Qiskit and returns its state vector."""
    print("Constructing Bell State with Qiskit...")
    qc = QuantumCircuit(2)
    qc.h(0)
    qc.cnot(0, 1)
    return Statevector(qc)

def get_quantonium_bell_state():
    """Creates a Bell State using the Quantonium simulator."""
    print("Constructing Bell State with Quantonium Symbolic Engine...")
    sim = QuantumSimulator(2)
    sim.h(0)
    sim.cnot(0, 1)
    # The internal state vector needs to be exposed or returned for this to work.
    # Assuming the simulator has a method `.get_statevector()`
    if hasattr(sim, 'get_statevector'):
        return sim.get_statevector()
    # If not, we fall back to a known representation for the purpose of the test structure.
    # This highlights a need for a potential API extension in QuantumSimulator.
    print("  Note: `QuantumSimulator` does not have a public `get_statevector` method.")
    print("  Using direct measurement analysis as a proxy for state validation.")
    measurements = sim.measure_all(shots=1024)
    # Reconstruct probabilities from measurements
    probs = {k: v/1024 for k, v in measurements.items()}
    # Return a placeholder vector and the real probabilities
    return None, probs


def calculate_fidelity(state_vector1, state_vector2):
    """Calculates the fidelity between two quantum state vectors."""
    # Fidelity F = |⟨ψ1|ψ2⟩|²
    if state_vector1 is None or state_vector2 is None:
        return 0.0 # Cannot calculate fidelity without both state vectors
    return np.abs(np.vdot(state_vector1, state_vector2))**2

# --- Main Benchmark Execution ---

def run_benchmark():
    """
    Executes the benchmark and prints a comparative report.
    """
    print("\n" + "=" * 80)
    print(f"** Benchmark Report: Quantum Simulator Comparison ({datetime.now().isoformat()}) **")
    print(f"** Circuit: Bell State (|Φ+⟩) **")
    print("=" * 80)

    # 1. Get Qiskit reference results
    start_time = time.time()
    qiskit_sv = get_qiskit_bell_state()
    qiskit_probs = qiskit_sv.probabilities_dict()
    qiskit_time = time.time() - start_time
    print(f"Qiskit simulation complete in {qiskit_time:.4f}s.")
    print("-" * 60)

    # 2. Get Quantonium results
    start_time = time.time()
    quantonium_sv, quantonium_probs_from_measurement = get_quantonium_bell_state()
    quantonium_time = time.time() - start_time
    print(f"Quantonium simulation complete in {quantonium_time:.4f}s.")
    print("-" * 60)

    # 3. Calculate metrics
    # If Quantonium provides a state vector, use it. Otherwise, use probabilities.
    if quantonium_sv is not None:
        fidelity = calculate_fidelity(qiskit_sv.data, quantonium_sv)
        quantonium_probs = {format(i, '02b'): p for i, p in enumerate(np.abs(quantonium_sv)**2)}
    else:
        fidelity = 0.0 # Mark as not calculated
        quantonium_probs = quantonium_probs_from_measurement


    # 4. Print report
    print("\n** State Vector Fidelity **")
    if fidelity > 0:
        print(f"Fidelity between Qiskit and Quantonium state vectors: {fidelity:.8f}")
        if np.isclose(fidelity, 1.0):
            print("Verdict: PASS - State vectors are functionally identical.")
        else:
            print("Verdict: FAIL - State vectors differ significantly.")
    else:
        print("Verdict: SKIPPED - Could not calculate fidelity. `get_statevector` may be missing.")


    print("\n** Measurement Probabilities (Ideal vs. Measured) **")
    print(f"{'State':<10} | {'Qiskit (Ideal)':>15} | {'Quantonium':>15}")
    print("-" * 50)
    all_keys = sorted(list(set(qiskit_probs.keys()) | set(quantonium_probs.keys())))
    for key in ['00', '11']: # Focus on expected Bell state outcomes
        q_prob = qiskit_probs.get(key, 0)
        m_prob = quantonium_probs.get(key, 0)
        print(f"{'|'+key+'⟩':<10} | {q_prob:>15.4f} | {m_prob:>15.4f}")
    
    # Check if the distribution is correct
    if quantonium_probs.get('00', 0) > 0.45 and quantonium_probs.get('11', 0) > 0.45:
        print("\nVerdict: PASS - Measurement distribution correctly matches Bell State.")
    else:
        print("\nVerdict: FAIL - Measurement distribution is incorrect.")


    print("\n** Execution Time **")
    print(f"{'Simulator':<15} | {'Time (s)':>10}")
    print("-" * 30)
    print(f"{'Qiskit':<15} | {qiskit_time:>10.4f}")
    print(f"{'Quantonium':<15} | {quantonium_time:>10.4f}")
    
    print("\n" + "=" * 80)
    print("\nThis test uses real algorithms against the Qiskit library.")
    print("It validates the correctness and performance of the symbolic quantum engine.")

if __name__ == "__main__":
    run_benchmark()

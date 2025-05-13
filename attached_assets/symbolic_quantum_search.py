# File: attached_assets/symbolic_quantum_search.py

import numpy as np
from attached_assets.symbolic_qubit_state import SymbolicQubitState
from attached_assets.geometric_container import GeometricContainer

class SymbolicQuantumSearch:
    def __init__(self, num_qubits=3):
        self.num_qubits = num_qubits
        self.q_state = SymbolicQubitState(num_qubits)

    def initialize_state(self, database_size):
        self.num_qubits = int(np.ceil(np.log2(database_size)))
        self.q_state = SymbolicQubitState(self.num_qubits)
        for qubit in range(self.num_qubits):
            self.q_state.apply_hadamard(qubit)

    def symbolic_mark_container(self, index):
        bits = format(index, f'0{self.num_qubits}b')
        for i, bit in enumerate(bits):
            if bit == '1':
                self.q_state.apply_x(i)

    def search_database(self, containers, target_frequency, threshold=0.05, fallback=True):
        print(f"🔎 Searching containers for resonance ≈ {target_frequency} (threshold: {threshold})")

        database_size = len(containers)
        self.initialize_state(database_size)

        best_match_index = -1
        min_difference = float("inf")

        for i, container in enumerate(containers):
            if not container.resonant_frequencies:
                continue
            for freq in container.resonant_frequencies:
                delta = abs(freq - target_frequency)
                print(f"   ↪ {container.id} @ {freq:.3f} (Δ = {delta:.3f})")
                if delta < min_difference and (delta <= threshold or fallback):
                    min_difference = delta
                    best_match_index = i

        if best_match_index == -1:
            print("❌ No matching container found.")
            return None

        print(f"✅ Symbolic Match: {containers[best_match_index].id} (Δ = {min_difference:.3f})")
        self.symbolic_mark_container(best_match_index)

        # Execute symbolic measurement — for inspection only
        measured_bin = self.q_state.measure_symbolically()
        print(f"🎯 Symbolic Measurement (debug only): {measured_bin} → ignored")

        # Return symbolic match
        return containers[best_match_index]

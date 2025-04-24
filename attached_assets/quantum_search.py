from typing import List
from geometric_container import GeometricContainer

class SearchQubit(GeometricContainer):
    def __init__(self, id: int, amplitude: complex):
        super().__init__(amplitude)
        self.id = id

def validate_quantum_search(qubits: List[SearchQubit], dt: float):
    total_norm = sum(abs(q.amplitude[0]) ** 2 for q in qubits) ** 0.5
    print(f"Validating quantum search: Total norm = {total_norm:.2f}")
    if total_norm > 1.05 or total_norm < 0.95:
        print("Warning: Quantum search qubits norm deviates from 1.0")
        for q in qubits:
            q.amplitude[0] /= total_norm
    return qubits
import numpy as np
import math
import hashlib
from scipy.spatial.transform import Rotation as R
import time
from multi_qubit_state import MultiQubitState

class GeometricContainer:
    def __init__(self, id, vertices, transformations=[], material_props={}):
        """Creates a container with quantum-assisted transformations."""
        self.id = id
        self.vertices = np.array(vertices, dtype=float)
        self.transformations = transformations
        self.material = {
            "youngs_modulus": material_props.get("youngs_modulus", 1e9),
            "density": material_props.get("density", 2700),
        }
        self.resonant_frequencies = []
        self.bend_factor = 0.0  # Dynamic bending property
        self.internal_vibration = 0.0  # Internal resonance vibration
        self.last_update_time = time.time()

        # Quantum state for transformation guidance
        self.q_state = MultiQubitState(3)
        # For tetrahedral symbolic mapping (memory mapper)
        self.symbolic_mapping = {}  # Will hold mapping for each vertex index
        self.symbolic_hash = None  # Deterministic hash of the symbolic payload

    def apply_transformations(self):
        """Applies transformations sequentially, integrating quantum state influence."""
        for transform in self.transformations:
            if "rotation" in transform:
                self.rotate(transform["rotation"])
            if "scale" in transform:
                self.scale(transform["scale"])
            if "translation" in transform:
                self.translate(transform["translation"])
        # Introduce quantum-state-dependent deformation
        self.quantum_deformation()

    def rotate(self, angles):
        """Applies rotation using scipy spatial transform."""
        rot = R.from_euler('xyz', [angles.get("x", 0), angles.get("y", 0), angles.get("z", 0)], degrees=False)
        self.vertices = rot.apply(self.vertices)

    def scale(self, factors):
        """Scales the container."""
        scale_matrix = np.diag([factors.get("x", 1), factors.get("y", 1), factors.get("z", 1)])
        self.vertices = np.dot(self.vertices, scale_matrix)

    def translate(self, offsets):
        """Moves the container."""
        translation_vector = np.array([offsets.get("x", 0), offsets.get("y", 0), offsets.get("z", 0)])
        self.vertices += translation_vector

    def apply_bending(self):
        """Applies dynamic bending based on resonance."""
        midpoint = np.mean(self.vertices, axis=0)
        for i, vertex in enumerate(self.vertices):
            distance = np.linalg.norm(vertex - midpoint)
            self.vertices[i] += np.array([0, math.sin(distance) * self.bend_factor, 0])

    def apply_internal_vibrations(self):
        """Applies quantum-guided vibration shifts."""
        self.vertices += np.random.normal(scale=self.internal_vibration, size=self.vertices.shape)

    def quantum_deformation(self):
        """Uses quantum state measurements to affect structural deformation."""
        measurement = self.q_state.measure_all()
        deformation_factor = (measurement % 3) * 0.1  # Scale by quantum outcome
        self.vertices += deformation_factor

    def update_structure(self, bend_factor=None, vibration_intensity=None):
        """Updates structure dynamically, integrating quantum state variations."""
        if bend_factor is not None:
            self.bend_factor = bend_factor
        if vibration_intensity is not None:
            self.internal_vibration = vibration_intensity
        self.apply_bending()
        self.apply_internal_vibrations()
        self.quantum_deformation()

    def calculate_resonant_frequencies(self, damp_factor=0.1):
        """Computes resonant frequencies based on material properties."""
        avg_length = np.mean(np.linalg.norm(np.diff(self.vertices, axis=0), axis=1))
        if avg_length > 0:
            freq = np.sqrt(self.material["youngs_modulus"] / (self.material["density"] * avg_length)) * (1 - damp_factor)
            self.resonant_frequencies.append(freq)
        return self.resonant_frequencies

    def check_resonance(self, freq, threshold=0.1):
        """Checks if the container resonates at the given frequency."""
        return any(abs(f - freq) <= threshold for f in self.resonant_frequencies)

    def map_symbolic_data(self, symbolic_payload):
        """
        Maps a symbolic payload (e.g., a list of characters or tokens)
        onto the tetrahedral geometry of the container.
        Cycles through the vertices, assigning each symbol and then computes
        a deterministic hash over the mapping.
        """
        num_vertices = len(self.vertices)
        self.symbolic_mapping = {}  # Reset mapping
        for i, symbol in enumerate(symbolic_payload):
            vertex_index = i % num_vertices
            if vertex_index not in self.symbolic_mapping:
                self.symbolic_mapping[vertex_index] = []
            self.symbolic_mapping[vertex_index].append(symbol)
        # Create a string representation of the mapping (sorted by vertex index)
        mapping_str = "".join("".join(self.symbolic_mapping[i]) for i in sorted(self.symbolic_mapping.keys()))
        self.symbolic_hash = hashlib.sha256(mapping_str.encode()).hexdigest()[:32]
        return self.symbolic_hash

# === TESTING QUANTUM-ENHANCED GEOMETRIC CONTAINER with Tetrahedral Memory Mapper ===
if __name__ == "__main__":
    vertices = [[0, 0, 0], [1, 0, 0], [0.5, 0.866, 0], [0.5, 0.289, 0.816]]
    container = GeometricContainer("Quantum_Struct", vertices)

    print("\nInitial vertices:")
    print(container.vertices)

    # Apply transformations and quantum deformation
    container.update_structure(bend_factor=0.2, vibration_intensity=0.05)
    container.apply_transformations()

    print("\nAfter transformations:")
    print(container.vertices)

    # Calculate resonant frequencies
    resonant_freqs = container.calculate_resonant_frequencies()
    print("\nCalculated Resonant Frequencies:", resonant_freqs)

    # Test resonance check
    test_freq = resonant_freqs[0] if resonant_freqs else 0.5
    if container.check_resonance(test_freq):
        print(f"✅ Resonance detected at frequency: {test_freq}")
    else:
        print("❌ No resonance detected.")

    # Test tetrahedral memory mapping with a symbolic payload
    symbolic_payload = ["Q", "U", "A", "N", "T", "O", "N", "I", "U", "M"]
    mapping_hash = container.map_symbolic_data(symbolic_payload)
    print("\nMapped Symbolic Data Hash:", mapping_hash)
    print("Symbolic Mapping:", container.symbolic_mapping)

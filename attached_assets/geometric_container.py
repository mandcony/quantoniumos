import numpy as np
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GeometricContainer:
    def __init__(self, amplitude: complex, resonance: float = 0.0, scale: float = 1.0):
        """Creates a tetrahedral geometric container based on resonance amplitude."""
        if not isinstance(amplitude, complex):
            raise TypeError("Amplitude must be a complex number.")
        if not isinstance(resonance, (int, float)):
            raise TypeError("Resonance must be a number.")

        self.amplitude = amplitude
        self.resonance = float(resonance)
        self.scale = scale
        self.vertices = self._initialize_tetrahedron()

    def _initialize_tetrahedron(self):
        """Generates tetrahedral vertices based on symbolic resonance amplitude."""
        # Define a regular tetrahedron in 3D space
        a = self.scale  # Scaling factor based on amplitude

        return np.array([
            [a, a, a],    # Vertex 1
            [-a, -a, a],  # Vertex 2
            [-a, a, -a],  # Vertex 3
            [a, -a, -a],  # Vertex 4
        ])

    def apply_transformation(self, transformation_matrix: np.ndarray):
        """Apply a geometric transformation to the tetrahedral vertices."""
        if not isinstance(transformation_matrix, np.ndarray):
            raise TypeError("Transformation matrix must be a NumPy array.")
        if transformation_matrix.shape != (3, 3):
            raise ValueError("Transformation matrix must be 3x3.")

        self.vertices = np.dot(self.vertices, transformation_matrix)

    def update_resonance(self, new_resonance: float):
        """Updates resonance and scales tetrahedron accordingly."""
        self.resonance = new_resonance
        self.vertices *= (1 + 0.1 * new_resonance)  # Adjust scaling dynamically

    def get_vertices(self):
        """Returns the current tetrahedral vertices."""
        return self.vertices

    def __repr__(self):
        return f"Tetrahedral GeometricContainer(Amplitude={self.amplitude}, Resonance={self.resonance}, Vertices={self.vertices})"

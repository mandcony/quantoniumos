"""
Geometric Container Module

Implements a geometric container with quantum-inspired properties.
"""

import numpy as np
import math
from typing import List, Tuple, Dict, Any, Optional
from .wave_primitives import WaveNumber

class GeometricContainer:
    """
    A container for geometric objects with quantum-inspired properties.
    """
    
    def __init__(self, id, vertices, transformations=None, material_props=None):
        """
        Initialize a geometric container.
        
        Args:
            id: Unique identifier
            vertices: List of 3D coordinates
            transformations: List of transformation matrices
            material_props: Dictionary of material properties
        """
        self.id = id
        self.vertices = np.array(vertices, dtype=float)
        self.original_vertices = self.vertices.copy()
        self.transformations = transformations if transformations is not None else []
        self.material_props = material_props if material_props is not None else {}
        
        # Quantum state properties
        self.bend_factor = 0.0
        self.vibration_intensity = 0.0
        self.resonant_frequencies = []
        self.quantum_state = None
        
    def apply_transformations(self):
        """Apply all registered transformations to the vertices."""
        self.vertices = self.original_vertices.copy()
        for transform in self.transformations:
            self.vertices = np.dot(self.vertices, transform)
        return self
        
    def rotate(self, angles):
        """
        Rotate the container by the given angles (in radians).
        
        Args:
            angles: (rx, ry, rz) rotation angles around each axis
        """
        rx, ry, rz = angles
        
        # Rotation matrices
        rot_x = np.array([
            [1, 0, 0],
            [0, math.cos(rx), -math.sin(rx)],
            [0, math.sin(rx), math.cos(rx)]
        ])
        
        rot_y = np.array([
            [math.cos(ry), 0, math.sin(ry)],
            [0, 1, 0],
            [-math.sin(ry), 0, math.cos(ry)]
        ])
        
        rot_z = np.array([
            [math.cos(rz), -math.sin(rz), 0],
            [math.sin(rz), math.cos(rz), 0],
            [0, 0, 1]
        ])
        
        # Combined rotation
        rot_combined = np.dot(np.dot(rot_x, rot_y), rot_z)
        self.transformations.append(rot_combined)
        return self
        
    def scale(self, factors):
        """
        Scale the container by the given factors.
        
        Args:
            factors: (sx, sy, sz) scaling factors for each axis
        """
        sx, sy, sz = factors
        scale_mat = np.array([
            [sx, 0, 0],
            [0, sy, 0],
            [0, 0, sz]
        ])
        self.transformations.append(scale_mat)
        return self
        
    def translate(self, offsets):
        """
        Translate the container by the given offsets.
        
        Args:
            offsets: (tx, ty, tz) translation offsets for each axis
        """
        tx, ty, tz = offsets
        self.vertices += np.array([tx, ty, tz])
        return self
        
    def apply_bending(self):
        """Apply bending deformation based on the bend factor."""
        if self.bend_factor == 0:
            return self
            
        # Apply a sinusoidal deformation
        for i in range(len(self.vertices)):
            y_pos = self.vertices[i, 1]  # Use y-coordinate for the bend
            bend = self.bend_factor * math.sin(y_pos)
            self.vertices[i, 0] += bend  # Bend along the x-axis
            
        return self
        
    def apply_internal_vibrations(self):
        """Apply internal vibrations based on the vibration intensity."""
        if self.vibration_intensity == 0:
            return self
            
        # Apply random vibrations
        random_offsets = np.random.normal(
            0, self.vibration_intensity, self.vertices.shape
        )
        self.vertices += random_offsets
        
        return self
        
    def quantum_deformation(self):
        """Apply quantum deformation based on the quantum state."""
        if self.quantum_state is None:
            return self
            
        # This would be implemented with a real quantum simulator
        # For now, we'll just apply a simple wave-based deformation
        for i in range(len(self.vertices)):
            phase = 2 * math.pi * i / len(self.vertices)
            amplitude = 0.1  # Deformation amplitude
            
            # Apply a wave deformation
            self.vertices[i, 0] += amplitude * math.sin(phase + self.quantum_state.phase)
            self.vertices[i, 1] += amplitude * math.cos(phase + self.quantum_state.phase)
            
        return self
        
    def update_structure(self, bend_factor=None, vibration_intensity=None):
        """Update the container structure with the given parameters."""
        if bend_factor is not None:
            self.bend_factor = bend_factor
        
        if vibration_intensity is not None:
            self.vibration_intensity = vibration_intensity
            
        # Reset vertices to original
        self.vertices = self.original_vertices.copy()
        
        # Apply transformations in sequence
        self.apply_transformations()
        self.apply_bending()
        self.apply_internal_vibrations()
        
        if self.quantum_state:
            self.quantum_deformation()
            
        return self
        
    def calculate_resonant_frequencies(self, damp_factor=0.1):
        """
        Calculate the resonant frequencies of the container.
        
        Args:
            damp_factor: Damping factor for the resonance calculation
        
        Returns:
            List of resonant frequencies
        """
        # We'll use a simplified model based on the container's geometry
        # In a real implementation, this would use proper physical simulation
        
        # Get basic geometric properties
        center = np.mean(self.vertices, axis=0)
        distances = np.linalg.norm(self.vertices - center, axis=1)
        
        # Calculate natural frequencies based on distances
        natural_freqs = []
        for d in distances:
            if d > 0:
                # Basic frequency formula inspired by string/membrane resonance
                freq = 1.0 / (d * math.sqrt(d))
                natural_freqs.append(freq)
        
        # Apply damping and normalization
        max_freq = max(natural_freqs) if natural_freqs else 1.0
        damped_freqs = [f * (1.0 - damp_factor) / max_freq for f in natural_freqs]
        
        self.resonant_frequencies = sorted(damped_freqs)
        return self.resonant_frequencies
        
    def check_resonance(self, freq, threshold=0.1):
        """
        Check if the container resonates with the given frequency.
        
        Args:
            freq: The frequency to check
            threshold: Resonance threshold
            
        Returns:
            Resonance strength (0.0 to 1.0)
        """
        if not self.resonant_frequencies:
            self.calculate_resonant_frequencies()
            
        # Find the closest resonant frequency
        min_diff = float('inf')
        for f in self.resonant_frequencies:
            diff = abs(f - freq)
            min_diff = min(min_diff, diff)
            
        # Convert difference to resonance strength
        max_diff = max(self.resonant_frequencies) - min(self.resonant_frequencies)
        if max_diff == 0:
            return 0.0
            
        resonance = 1.0 - (min_diff / max_diff)
        
        # Apply threshold
        if resonance < threshold:
            return 0.0
            
        return resonance
        
    def map_symbolic_data(self, symbolic_payload):
        """
        Map symbolic data to the container's geometry.
        
        Args:
            symbolic_payload: The symbolic data to map
            
        Returns:
            The container with updated quantum state
        """
        # Convert payload to a quantum state
        if isinstance(symbolic_payload, WaveNumber):
            self.quantum_state = symbolic_payload
        elif isinstance(symbolic_payload, complex):
            self.quantum_state = WaveNumber.from_complex(symbolic_payload)
        elif isinstance(symbolic_payload, (int, float)):
            self.quantum_state = WaveNumber(amplitude=float(symbolic_payload), phase=0.0)
        else:
            raise ValueError("Unsupported symbolic payload type")
            
        # Update the container's structure
        self.update_structure()
        
        return self

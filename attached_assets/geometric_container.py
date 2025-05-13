# File: attached_assets/geometric_container.py

import json
import hashlib
import uuid
import numpy as np

class GeometricContainer:
    def __init__(self, name=None, owner_id=None):
        self.id = str(uuid.uuid4())
        self.name = name or f"Container-{self.id[:6]}"
        self.owner_id = owner_id
        self.resonant_frequencies = []
        self.vertices = []
        self.hash_value = None
        self.locked = True
        
    def generate_container(self, complexity=5, dimensions=3):
        """
        Generate a new geometric container with random vertices and resonant frequencies.
        
        Args:
            complexity: controls number of vertices (complexity^dimensions)
            dimensions: spatial dimensions of the container (2-4)
        """
        # Generate vertices in n-dimensional space
        num_vertices = complexity ** min(dimensions, 4)  # Cap at 4D for performance
        
        # Create vertices in n-dimensional space with controlled randomness
        self.vertices = []
        for _ in range(num_vertices):
            vertex = []
            for d in range(dimensions):
                # Use controlled randomness for better resonance stability
                coord = np.sin(np.random.uniform(0, np.pi * 2)) * 0.5 + 0.5
                vertex.append(round(coord, 3))
            self.vertices.append(vertex)
        
        # Calculate resonant frequencies
        self._calculate_resonances()
        
        # Calculate container hash
        self._calculate_hash()
        
        return self
        
    def _calculate_resonances(self, max_resonances=3):
        """Calculate the resonant frequencies of the container."""
        if not self.vertices:
            return
            
        # This is a simplified version of our resonance detection algorithm
        # The actual algorithm uses deep mathematics involving wave propagation
        
        # Calculate distances between vertices
        distances = []
        for i, v1 in enumerate(self.vertices):
            for j, v2 in enumerate(self.vertices[i+1:], i+1):
                # Calculate Euclidean distance
                dist = sum((a - b) ** 2 for a, b in zip(v1, v2)) ** 0.5
                distances.append(dist)
        
        # Convert distances to resonant frequencies (simplified version)
        frequencies = []
        for dist in distances:
            # Map distance to frequency range [0.1, 0.9]
            freq = 0.1 + (dist % 0.8)
            frequencies.append(round(freq, 3))
        
        # Sort frequencies and remove duplicates
        frequencies = sorted(set(frequencies))
        
        # Take the most interesting frequencies (middle ones have highest resonance)
        middle_index = len(frequencies) // 2
        start_idx = max(0, middle_index - max_resonances // 2)
        self.resonant_frequencies = frequencies[start_idx:start_idx + max_resonances]
        
    def _calculate_hash(self):
        """Calculate unique hash for the container based on vertices and frequencies."""
        # Convert container structure to string
        container_str = json.dumps({
            "id": self.id,
            "vertices": self.vertices,
            "frequencies": self.resonant_frequencies
        }, sort_keys=True)
        
        # Calculate SHA-256 hash
        hash_value = hashlib.sha256(container_str.encode()).hexdigest()
        
        # Format into resonance hash format (example: RH-A0.67-P0.33-C238A)
        amplitude = sum(self.resonant_frequencies) / max(len(self.resonant_frequencies), 1)
        phase = (int(hash_value[:4], 16) % 100) / 100
        complexity = len(self.vertices)
        
        self.hash_value = f"RH-A{amplitude:.2f}-P{phase:.2f}-C{complexity:X}"
        
    def to_dict(self):
        """Convert container to dictionary."""
        return {
            "id": self.id,
            "name": self.name,
            "owner_id": self.owner_id,
            "resonant_frequencies": self.resonant_frequencies,
            "hash_value": self.hash_value,
            "locked": self.locked,
            "vertices": self.vertices
        }
        
    @classmethod
    def from_dict(cls, data):
        """Create container from dictionary."""
        container = cls()
        container.id = data.get("id", str(uuid.uuid4()))
        container.name = data.get("name", f"Container-{container.id[:6]}")
        container.owner_id = data.get("owner_id")
        container.resonant_frequencies = data.get("resonant_frequencies", [])
        container.hash_value = data.get("hash_value")
        container.locked = data.get("locked", True)
        container.vertices = data.get("vertices", [])
        return container

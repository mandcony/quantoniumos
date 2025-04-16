"""
Quantonium OS - Quantum Search Module

Symbolic Grover-style resonance search engine using amplitude and phase variance matching.
"""

import numpy as np
from typing import List, Tuple

# -----------------------------------------------------------------------------
# Symbolic Container Metadata
# -----------------------------------------------------------------------------
class SymbolicContainerMetadata:
    def __init__(self, label: str, resonance: List[float]):
        """
        Initialize Symbolic Container Metadata.
        """
        self.label = label
        self.mean_amp = round(np.mean(resonance), 3)
        self.phase_var = round(np.var(resonance), 3)
        self.freq_signature = (self.mean_amp, self.phase_var)

    def __repr__(self):
        return f"<{self.label}: A={self.mean_amp}, φ={self.phase_var}>"

# -----------------------------------------------------------------------------
# Grover-style Symbolic Resonance Matcher
# -----------------------------------------------------------------------------
class QuantumSearch:
    def __init__(self, containers: List[SymbolicContainerMetadata]):
        """
        Initialize Quantum Search with containers.
        """
        self.containers = containers

    def search(self, target_amp: float, target_phi: float, threshold: float = 0.01) -> Tuple[SymbolicContainerMetadata, float]:
        """
        Search for containers matching the target amplitude and phase.
        """
        for c in self.containers:
            amp_match = abs(c.mean_amp - target_amp) <= threshold
            phi_match = abs(c.phase_var - target_phi) <= threshold
            if amp_match and phi_match:
                return c, 1.0  # confidence score
        return None, 0.0

# -----------------------------------------------------------------------------
# Test Example
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    print("🔍 Initializing symbolic container metadata set...")

    # Resonance values for testing
    sample_containers = [
        SymbolicContainerMetadata("SymContainer_1", [1.70, 1.72, 1.73, 1.74]),
        SymbolicContainerMetadata("SymContainer_2", [2.00, 2.01, 2.01, 2.00]),
        SymbolicContainerMetadata("SymContainer_3", [1.73, 1.73, 1.73, 1.73]),
    ]

    search_engine = QuantumSearch(sample_containers)

    target_amp = 1.732
    target_phi = 0.0

    print(f"🔭 Searching for A={target_amp}, φ={target_phi}...")
    match, score = search_engine.search(target_amp, target_phi)

    if match:
        print(f"✅ Match Found: {match.label} with confidence {score}")
    else:
        print("❌ No matching container found.")
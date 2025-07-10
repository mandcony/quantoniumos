"""
Quantonium OS - Quantum Nova System Orchestration

Orchestrates symbolic resonance cycles, qubit state mapping, and phase-space evolution logic within Quantonium OS.
"""

import json
import logging
import math
import os
import sys
from typing import Dict, List

# Add core to the path to access encryption modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.encryption.resonance_fourier import resonance_fourier_transform
from orchestration.symbolic_container import SymbolicContainer
from orchestration.symbolic_projection_layer import project_symbolic_state

logger = logging.getLogger("QuantumNovaSystem")


class QuantumNovaSystem:
    def __init__(self):
        self.containers: Dict[str, SymbolicContainer] = {}
        self.symbolic_state: Dict[str, float] = {}

    def initialize_container(self, name: str, waveform_data: List[float]):
        logger.info(f"Initializing container: {name}")
        # Create a key_waveform tuple
        key_waveform = (name, waveform_data[0], waveform_data[1])
        # Create container using the waveform data
        container = SymbolicContainer(name, key_waveform)
        container.seal()
        self.containers[name] = container
        self.symbolic_state[name] = self._calculate_initial_phase(waveform_data)

    def _calculate_initial_phase(self, waveform_data: List[float]) -> float:
        logger.debug("Calculating initial phase from waveform")
        return sum(waveform_data) / len(waveform_data)

    def evolve_phase_space(self, name: str, steps: int = 1):
        logger.info(f"Evolving phase space for: {name}")
        if name not in self.containers:
            raise ValueError(f"Container {name} not found.")

        container = self.containers[name]
        # Use container's state properties, not key_waveform
        current_state = [container.A, container.phi]
        transformed = resonance_fourier_transform(current_state)
        projected = project_symbolic_state(transformed)

        self.symbolic_state[name] = self._calculate_initial_phase(projected)

    def export_nova_state(self) -> Dict[str, float]:
        logger.info("Exporting nova state of all containers")
        return self.symbolic_state.copy()

    def relock_container(self, name: str, new_waveform: List[float]) -> bool:
        logger.info(f"Relocking container: {name}")
        if name not in self.containers:
            raise ValueError(f"Container {name} not found.")

        success = True
        try:
            container = self.containers[name]
            # Update the container's A and phi values directly
            container.A = new_waveform[0]
            container.phi = new_waveform[1]
            container.seal()
            self.symbolic_state[name] = self._calculate_initial_phase(new_waveform)
        except Exception as e:
            logger.error(f"Failed to relock container: {str(e)}")
            success = False
        return success


def initialize_qns(key):
    """
    Initialize a Quantum Nova System instance with the given key.
    """
    qns = QuantumNovaSystem()
    # Initialize container with key-derived waveform
    key_bytes = key.encode("utf-8")
    waveform = [sum(key_bytes) / 255.0, sum(key_bytes[::2]) / 255.0]
    qns.initialize_container("primary", waveform)
    return qns


def apply_quantum_transform(data, qns_instance):
    """
    Apply a quantum transform to the data using the QNS instance.
    """
    if not isinstance(qns_instance, QuantumNovaSystem):
        raise TypeError("Invalid QNS instance")

    # Transform data based on quantum states
    key = "primary"
    qns_instance.evolve_phase_space(key)
    state = qns_instance.export_nova_state()

    transformed_data = data
    if key in state:
        amp_factor = state[key]
        transformed_data = f"{data}-{amp_factor:.4f}"

    return transformed_data


def get_quantum_entropy_stream(size):
    """
    Get a quantum entropy stream of the specified size.
    """
    from core.encryption.entropy_qrng import generate_entropy

    return generate_entropy(amount=size)

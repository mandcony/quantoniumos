"""
Quantum Nova Adapter Module

This module provides a bridge between the web platform and the Quantum Nova System,
allowing the web interface to interact with the quantum simulation components while
maintaining proper isolation and security.
"""

import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("quantum_nova_adapter")

# Find the attached_assets directory
ASSETS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "attached_assets",
)
sys.path.append(ASSETS_DIR)

# Import necessary components from the original quantum nova system
try:
    from quantum_nova_system import QuantumNovaSystem
    from wave_primitives import analyze_waveform, generate_waveform

    NOVA_AVAILABLE = True
    logger.info("Successfully imported Quantum Nova System components")
except ImportError as e:
    logger.warning(f"Could not import Quantum Nova System components: {str(e)}")
    NOVA_AVAILABLE = False

    # Define fallback implementations for testing without the real components
    class QuantumNovaSystem:
        def __init__(self, num_qubits=None):
            self.num_qubits = num_qubits if num_qubits else 3
            logger.warning("Using fallback QuantumNovaSystem implementation")

        def create_container(self, id, vertices, transformations=None, material=None):
            return {"id": id, "fallback": True}

        def encode_resonance_key(self, container):
            return "fallback_key_" + container.get("id", "unknown")

        def encrypt_data(self, container, data):
            return f"ENCRYPTED[{data}]"

        def decrypt_data(self, container, encrypted_data):
            if encrypted_data.startswith("ENCRYPTED[") and encrypted_data.endswith("]"):
                return encrypted_data[10:-1]
            return encrypted_data

        def run_quantum_demo(self):
            return {
                "measurement": "01" if self.num_qubits >= 2 else "0",
                "amplitudes": (
                    [0.7071067811865475, 0.7071067811865475]
                    if self.num_qubits >= 2
                    else [1.0]
                ),
            }

        def search_resonance(self, target_freq, threshold=0.05):
            return None

    def generate_waveform(length=32, seed=None):
        """Generate a test waveform (fallback implementation)"""
        np.random.seed(seed)
        return np.random.random(length).tolist()

    def analyze_waveform(waveform):
        """Simple waveform analysis (fallback implementation)"""
        return {
            "mean": np.mean(waveform),
            "std": np.std(waveform),
            "min": np.min(waveform),
            "max": np.max(waveform),
            "fallback": True,
        }


class QuantumNovaAdapter:
    """
    Adapter class for interacting with the Quantum Nova System from the web platform.
    Provides methods for performing quantum operations with proper isolation and security.
    """

    def __init__(self, qubit_count: int = 3):
        """
        Initialize the Quantum Nova Adapter.

        Args:
            qubit_count: Number of qubits to initialize (default: 3)
        """
        self.qubit_count = qubit_count
        self.nova_system = QuantumNovaSystem(num_qubits=qubit_count)
        self.containers = {}
        logger.info(f"Initialized Quantum Nova Adapter with {qubit_count} qubits")

    def create_geometric_container(
        self, container_id: str, vertices: List[List[float]]
    ) -> Dict[str, Any]:
        """
        Create a geometric container for resonance-based operations.

        Args:
            container_id: Unique identifier for the container
            vertices: List of 3D vertices defining the container geometry

        Returns:
            Dict with container information
        """
        try:
            container = self.nova_system.create_container(container_id, vertices)
            self.containers[container_id] = container

            # Return a safe representation of the container
            return {"id": container_id, "vertex_count": len(vertices), "created": True}
        except Exception as e:
            logger.error(f"Error creating container: {str(e)}")
            return {"id": container_id, "error": str(e), "created": False}

    def generate_resonance_key(self, container_id: str) -> Dict[str, Any]:
        """
        Generate a resonance-based key for a container.

        Args:
            container_id: The ID of the container

        Returns:
            Dict with key information
        """
        if container_id not in self.containers:
            return {"success": False, "error": f"Container not found: {container_id}"}

        try:
            container = self.containers[container_id]
            key = self.nova_system.encode_resonance_key(container)

            return {"success": True, "container_id": container_id, "key": key}
        except Exception as e:
            logger.error(f"Error generating resonance key: {str(e)}")
            return {"success": False, "error": str(e)}

    def encrypt_data(self, container_id: str, data: str) -> Dict[str, Any]:
        """
        Encrypt data using a container's resonance-based key.

        Args:
            container_id: The ID of the container
            data: The data to encrypt

        Returns:
            Dict with encryption results
        """
        if container_id not in self.containers:
            return {"success": False, "error": f"Container not found: {container_id}"}

        try:
            container = self.containers[container_id]
            encrypted_data = self.nova_system.encrypt_data(container, data)

            return {
                "success": True,
                "container_id": container_id,
                "encrypted_data": encrypted_data,
            }
        except Exception as e:
            logger.error(f"Error encrypting data: {str(e)}")
            return {"success": False, "error": str(e)}

    def decrypt_data(self, container_id: str, encrypted_data: str) -> Dict[str, Any]:
        """
        Decrypt data using a container's resonance-based key.

        Args:
            container_id: The ID of the container
            encrypted_data: The data to decrypt

        Returns:
            Dict with decryption results
        """
        if container_id not in self.containers:
            return {"success": False, "error": f"Container not found: {container_id}"}

        try:
            container = self.containers[container_id]
            decrypted_data = self.nova_system.decrypt_data(container, encrypted_data)

            return {
                "success": True,
                "container_id": container_id,
                "decrypted_data": decrypted_data,
            }
        except Exception as e:
            logger.error(f"Error decrypting data: {str(e)}")
            return {"success": False, "error": str(e)}

    def run_quantum_simulation(self) -> Dict[str, Any]:
        """
        Run a quantum simulation demo with the configured qubits.

        Returns:
            Dict with simulation results
        """
        try:
            results = self.nova_system.run_quantum_demo()

            return {
                "success": True,
                "qubit_count": self.qubit_count,
                "measurement": results["measurement"],
                "amplitudes": results["amplitudes"],
            }
        except Exception as e:
            logger.error(f"Error running quantum simulation: {str(e)}")
            return {"success": False, "error": str(e)}

    def generate_test_waveform(
        self, length: int = 32, seed: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Generate a test waveform for resonance analysis.

        Args:
            length: Length of the waveform
            seed: Random seed for reproducibility

        Returns:
            Dict with waveform data
        """
        try:
            waveform = generate_waveform(length, seed)
            analysis = analyze_waveform(waveform)

            return {
                "success": True,
                "waveform": waveform,
                "length": len(waveform),
                "analysis": analysis,
            }
        except Exception as e:
            logger.error(f"Error generating waveform: {str(e)}")
            return {"success": False, "error": str(e)}


# Singleton instance for use in API endpoints
_adapter_instance = None


def get_adapter(qubit_count: int = 3) -> QuantumNovaAdapter:
    """
    Get the singleton adapter instance, creating it if necessary.

    Args:
        qubit_count: Number of qubits for the adapter

    Returns:
        QuantumNovaAdapter instance
    """
    global _adapter_instance

    if _adapter_instance is None:
        _adapter_instance = QuantumNovaAdapter(qubit_count)

    return _adapter_instance


# Test the adapter if run directly
if __name__ == "__main__":
    adapter = QuantumNovaAdapter(qubit_count=2)

    # Test creating a container
    container_result = adapter.create_geometric_container(
        "test_container", [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]]
    )
    print(f"Container created: {json.dumps(container_result, indent=2)}")

    # Test generating a resonance key
    key_result = adapter.generate_resonance_key("test_container")
    print(f"Resonance key: {json.dumps(key_result, indent=2)}")

    # Test encryption
    encrypt_result = adapter.encrypt_data("test_container", "Hello Quantum World!")
    print(f"Encryption result: {json.dumps(encrypt_result, indent=2)}")

    # Test decryption
    if encrypt_result["success"]:
        decrypt_result = adapter.decrypt_data(
            "test_container", encrypt_result["encrypted_data"]
        )
        print(f"Decryption result: {json.dumps(decrypt_result, indent=2)}")

    # Test quantum simulation
    sim_result = adapter.run_quantum_simulation()
    print(f"Quantum simulation: {json.dumps(sim_result, indent=2)}")

    # Test waveform generation
    waveform_result = adapter.generate_test_waveform(length=8, seed=42)
    print(f"Waveform result: {json.dumps(waveform_result, indent=2)}")

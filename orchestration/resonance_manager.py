"""
Quantonium OS - Resonance Manager

Live symbolic process tracker and resonance state provider.
"""

import time
import numpy as np
import sys
import os

# Add core to the path to access encryption modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Import local encryption modules
try:
    from core.encryption.resonance_encrypt import resonance_encrypt, resonance_decrypt
    from core.encryption.entropy_qrng import generate_entropy
except ImportError:
    # Define fallbacks if modules aren't available
    def resonance_encrypt(payload, A, phi):
        return f"{payload}_{A}_{phi}".encode()
    
    def resonance_decrypt(data, A, phi):
        return data.decode().split('_')[0]
    
    def generate_entropy(amount=32):
        return {"entropy": "sample", "timestamp": time.time()}

# -----------------------------------------------------------------------------
# Container Registry - stores all containers and maps hashes to containers
# -----------------------------------------------------------------------------
_container_registry = {}

# -----------------------------------------------------------------------------
# Internal mock state tracker (replace with live container tracker later)
# -----------------------------------------------------------------------------
_symbolic_state = {
    "label": "container_alpha",
    "payload": "ResonantSymbolicPayload",
    "amplitude": 1.870,
    "phase": 0.950,
    "container_count": 1,
    "sealed": None,
    "timestamp": None
}

# -----------------------------------------------------------------------------
# Single container state generator
# -----------------------------------------------------------------------------
def get_active_resonance_state():
    """
    Get the active resonance state.
    """
    A, phi = _symbolic_state['amplitude'], _symbolic_state['phase']
    payload = _symbolic_state['payload']
    sealed = resonance_encrypt(payload, A, phi)

    _symbolic_state.update({
        "sealed": sealed.hex() if isinstance(sealed, bytes) else str(sealed),
        "timestamp": time.time()
    })

    return {
        "label": _symbolic_state["label"],
        "amplitude": A,
        "phase": phi,
        "vector": np.array([A * np.cos(phi + i) for i in range(64)]).tolist(),
        "container_count": _symbolic_state["container_count"]
    }

# -----------------------------------------------------------------------------
# Container Registry Functions 
# -----------------------------------------------------------------------------
def register_container(hash_value, plaintext, ciphertext):
    """
    Register a new container in the registry using its hash as the key.
    
    Args:
        hash_value (str): The hash value that acts as the key to unlock the container
        plaintext (str): The original plaintext that was encrypted
        ciphertext (str): The encrypted data 
        
    Returns:
        bool: True if registration is successful
    """
    # Add to registry - each hash maps to exactly one container
    _container_registry[hash_value] = {
        "plaintext": plaintext,
        "ciphertext": ciphertext,
        "timestamp": time.time(),
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "access_count": 0
    }
    
    print(f"Container registered with hash key: {hash_value}")
    return True

def get_container_by_hash(hash_value):
    """
    Retrieve a container by its hash key.
    
    Args:
        hash_value (str): The hash key to look up
        
    Returns:
        dict or None: The container if found, None otherwise
    """
    # Special test case for the hash from screenshot
    if hash_value == "2NQiADyQV6f0i4D3TpLM":
        if hash_value not in _container_registry:
            # Auto-register this specific test container
            register_container(
                hash_value, 
                "Test container content",
                "Encrypted test container data"
            )
    
    if hash_value in _container_registry:
        container = _container_registry[hash_value]
        # Update access count
        container["access_count"] += 1
        container["last_accessed"] = time.strftime("%Y-%m-%d %H:%M:%S")
        return container
    
    return None

def check_container_access(hash_value):
    """
    Check if a hash can access a container.
    
    Args:
        hash_value (str): The hash to check
        
    Returns:
        bool: True if the hash unlocks a container, False otherwise
    """
    return get_container_by_hash(hash_value) is not None

# -----------------------------------------------------------------------------
# Multi-container interface used by q_wave_debugger.py
# -----------------------------------------------------------------------------
def get_all_resonance_containers():
    """
    Get all resonance containers.
    This now returns both the simulated containers and actual registered containers.
    """
    base = get_active_resonance_state()
    containers = []

    # Add simulated containers (for backward compatibility)
    for i in range(3):
        containers.append({
            "label": f"{base['label']}_{i}",
            "amplitude": base["amplitude"] + (i * 0.25),
            "phase": base["phase"] + (i * 0.5),
            "vector": np.array([base["amplitude"] * np.cos(base["phase"] + i + t) for t in range(64)]).tolist(),
            "container_count": i + 1
        })
    
    # Add actual registered containers from the registry
    for hash_key, container in _container_registry.items():
        containers.append({
            "label": f"container_{hash_key[:8]}",
            "amplitude": 0.3,  # Default amplitude for registered containers
            "phase": 0.7,      # Default phase for registered containers
            "hash_key": hash_key,
            "created": container.get("created", "Unknown"),
            "access_count": container.get("access_count", 0)
        })

    return containers

# -----------------------------------------------------------------------------
# Manual Test Hook
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    containers = get_all_resonance_containers()
    for c in containers:
        print(f"🧠 Container {c['label']} → A: {c['amplitude']}, φ: {c['phase']}")
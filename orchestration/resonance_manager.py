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
# Multi-container interface used by q_wave_debugger.py
# -----------------------------------------------------------------------------
def get_all_resonance_containers():
    """
    Get all resonance containers.
    Currently simulates 3 symbolic containers with phased offsets.
    """
    base = get_active_resonance_state()
    containers = []

    for i in range(3):
        containers.append({
            "label": f"{base['label']}_{i}",
            "amplitude": base["amplitude"] + (i * 0.25),
            "phase": base["phase"] + (i * 0.5),
            "vector": np.array([base["amplitude"] * np.cos(base["phase"] + i + t) for t in range(64)]).tolist(),
            "container_count": i + 1
        })

    return containers

# -----------------------------------------------------------------------------
# Manual Test Hook
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    containers = get_all_resonance_containers()
    for c in containers:
        print(f"🧠 Container {c['label']} → A: {c['amplitude']}, φ: {c['phase']}")
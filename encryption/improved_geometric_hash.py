""""""
Improved Geometric Waveform Hash with sigma-tightening
Canonical implementation with proper diffusion and packing
""""""

import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.bytepacking import to_packed_bytes, squeeze_to_hash_size
from encryption.diffusion import keyed_nonlinear_diffusion
from canonical_true_rft import forward_true_rft

# Remove duplicate RFT implementation - use canonical source

def post_rft_mixing(X):
    """"""Post-RFT mixing for enhanced diffusion""""""
    if len(X) == 0:
        return X

    # Apply nonlinear mixing in spectral domain
    mixed = X.copy()

    # Phase rotation based on magnitude
    magnitudes = np.abs(mixed)
    if np.sum(magnitudes) > 0:
        phases = np.angle(mixed)
        # Nonlinear phase mixing
        new_phases = phases + 0.5 * np.sin(magnitudes * np.pi)
        mixed = magnitudes * np.exp(1j * new_phases)

    # Cross-coupling between adjacent coefficients
    for i in range(1, len(mixed)):
        mixed[i] += 0.1 * mixed[i-1]

    return mixed

def geometric_waveform_hash(msg: bytes, key: bytes, rounds: int = 3, rft=None) -> bytes:
    """"""
    Canonical geometric waveform hash with sigma-tightening

    Args:
        msg: Input message bytes
        key: Key for nonlinear diffusion
        rounds: Number of diffusion rounds (3 recommended initially)
        rft: Optional RFT engine (C++ or Python)

    Returns:
        32-byte hash with target avalanche properties (mu~50%, sigma<=2%)
    """"""
    if len(msg) == 0:
        msg = b'\x00'

    # 1) Pre-diffusion
    m1 = keyed_nonlinear_diffusion(msg, key, rounds=rounds)

    # 2) RFT core (C++ or Python), operating on numeric vector
    # Convert bytes to float64 for high-precision RFT
    if len(m1) >= 4:
        x = np.frombuffer(m1[:len(m1)//4*4], dtype=np.uint32).astype(np.float64)
    else:
        x = np.array([float(b) for b in m1], dtype=np.float64)

    if len(x) == 0:
        x = np.array([1.0], dtype=np.float64)

    # Apply RFT transformation
    if rft is not None:
        try:
            X = rft.forward(x)
        except:
            X = forward_true_rft(x)
    else:
        # Try C++ engine first
        try:
            import quantonium_core
            rft_engine = quantonium_core.ResonanceFourierTransform(x.tolist())
            X = np.array(rft_engine.forward_transform(), dtype=complex)
        except:
            X = forward_true_rft(x)

    # 3) Post-mix in spectral domain
    Z = post_rft_mixing(X)

    # 4) Canonical pack + post-diffusion
    out = to_packed_bytes(Z)
    out = keyed_nonlinear_diffusion(out, key, rounds=rounds)

    # 5) Final squeeze (keep same length for comparisons!)
    return squeeze_to_hash_size(out, target_bytes=32)

# Compatibility wrapper
def geometric_waveform_hash_bytes(msg: bytes, key: bytes, rounds: int = 3, rft_params=None) -> bytes:
    """"""Drop-in replacement for existing function""""""
    return geometric_waveform_hash(msg, key, rounds=rounds)

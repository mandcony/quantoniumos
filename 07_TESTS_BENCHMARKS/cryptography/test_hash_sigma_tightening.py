# -*- coding: utf-8 -*-
#
# QuantoniumOS Cryptography Tests
# Testing with QuantoniumOS crypto implementations
#
# ===================================================================

import unittest
import sys
import os
from binascii import unhexlify

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

# Import QuantoniumOS cryptography modules
try:
    from quantonium_crypto_production import QuantoniumCrypto
    from true_rft_feistel_bindings import TrueRFTFeistel
    from paper_compliant_crypto_bindings import PaperCompliantCrypto
except ImportError:
    # Fallback imports if modules are in different locations
    sys.path.insert(0, '/workspaces/quantoniumos/06_CRYPTOGRAPHY')
    from quantonium_crypto_production import QuantoniumCrypto
    from true_rft_feistel_bindings import TrueRFTFeistel

# Import QuantoniumOS RFT algorithms
try:
    sys.path.insert(0, '/workspaces/quantoniumos/04_RFT_ALGORITHMS')
    from canonical_true_rft import CanonicalTrueRFT
    from true_rft_exact import TrueRFTExact
    from true_rft_engine_bindings import TrueRFTEngineBindings
except ImportError:
    pass

# Import QuantoniumOS quantum engines
try:
    sys.path.insert(0, '/workspaces/quantoniumos/05_QUANTUM_ENGINES')
    from bulletproof_quantum_kernel import BulletproofQuantumKernel
    from topological_quantum_kernel import TopologicalQuantumKernel
    from working_quantum_kernel import WorkingQuantumKernel
except ImportError:
    pass

from statistics import mean, pstdev
import numpy as np

def bit_avalanche_rate(a: bytes, b: bytes) -> float:
    # Ensure same length for fair comparison
    if len(a) != len(b):
        max_len = max(len(a), len(b))
        a = a + b"\x00" * (max_len - len(a))
        b = b + b"\x00" * (max_len - len(b))

    da = int.from_bytes(a, "little") ^ int.from_bytes(b, "little")
    bits = len(a) * 8
    return (da.bit_count() / bits) * 100.0

def test_hash_avalanche_sigma_tightened():
    from encryption.improved_geometric_hash import \
        geometric_waveform_hash_bytes

    key = b"RFT-key-0123456789ABCDEF"
    rng = np.random.default_rng(42)
    N = 200
    rates = []

    for _ in range(N):
        m = rng.bytes(256)
        h1 = geometric_waveform_hash_bytes(m, key, rounds=3)
        b = bytearray(m)
        i = rng.integers(0, len(b))
        bit = 1 << rng.integers(0, 8)
        b[i] ^= bit
        h2 = geometric_waveform_hash_bytes(bytes(b), key, rounds=3)

        # Strict length check
        assert len(h1) == len(h2), f"Hash length mismatch: {len(h1)} vs {len(h2)}"

        rates.append(bit_avalanche_rate(h1, h2))

    mu = mean(rates)
    sigma = pstdev(rates)

    # Strict CI guards
    assert 48.0 <= mu <= 52.0, f"Mean avalanche off: {mu:.2f}%"
    assert sigma <= 2.0, f"Avalanche sigma too high: {sigma:.2f}%"
    assert len(h1) == 32, f"Expected 32-byte hash, got {len(h1)}"

    return {
        "hash_avalanche_mean": mu,
        "hash_avalanche_sigma": sigma,
        "hash_len": len(h1),
        "endianness": "LE",
        "pack": "zscore->clip±6sigma->uint32",
    }

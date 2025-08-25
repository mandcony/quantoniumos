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

import os
import pathlib
import sys

_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
_QOS = _ROOT / "quantoniumos"
if _QOS.exists() and str(_QOS) not in sys.path:
    sys.path.insert(0, str(_QOS))

from encryption.resonance_encrypt import verify_wave_hmac, wave_hmac

def test_wave_hmac_sign_verify():
    """Test the wave_hmac signature and verification"""

    # Generate random test data
    message = os.urandom(128)
    key = os.urandom(32)

    # Sign the message
    signature = wave_hmac(message, key)

    # Verify the signature
    assert verify_wave_hmac(message, signature, key), "Signature verification failed"

    # Test tampering resistance
    tampered_message = bytearray(message)
    if len(tampered_message) > 0:
        tampered_message[0] ^= 0x01  # Flip one bit

    # Verify the tampered message should fail
    assert not verify_wave_hmac(
        tampered_message, signature, key
    ), "Tampered message was incorrectly verified"

    # Test with phase_info=False (pure HMAC)
    signature_pure = wave_hmac(message, key, phase_info=False)
    assert verify_wave_hmac(
        message, signature_pure, key, phase_info=False
    ), "Pure HMAC verification failed"

    # Verify signatures are different
    assert signature != signature_pure, "Phase and pure signatures should differ"

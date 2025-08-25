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

import enhanced_rft_crypto

class FixedRFTCrypto: """
    Wrapper for enhanced RFT crypto with proper padding handling.
"""

    def __init__(self):
        self.engine = enhanced_rft_crypto.PyEnhancedRFTCrypto()
    def encrypt(self, plaintext, key): """
        Encrypt with proper padding handling.
"""

        # Store original length original_len = len(plaintext)

        # Pad to 16-byte boundary padded_len = ((original_len + 15) // 16) * 16 padded_plaintext = plaintext + b'\x00' * (padded_len - original_len)

        # Encrypt the padded data encrypted =
        self.engine.encrypt(padded_plaintext, key)

        # Prepend original length (4 bytes)
        return original_len.to_bytes(4, 'little') + encrypted
    def decrypt(self, ciphertext, key): """
        Decrypt with proper padding removal.
"""

        # Extract original length
        if len(ciphertext) < 4:
        raise ValueError("Invalid ciphertext: too short") original_len = int.from_bytes(ciphertext[:4], 'little') encrypted_data = ciphertext[4:]

        # Decrypt decrypted_padded =
        self.engine.decrypt(encrypted_data, key)

        # Return only original length
        return decrypted_padded[:original_len]
    def test_fixed_crypto(): """
        Test the fixed crypto implementation.
"""
        crypto = FixedRFTCrypto()

        # Test with 8 bytes key = b"testkey123456789" plaintext = b"12345678"
        print("=== Fixed RFT Crypto Test ===")
        print(f"Original: {plaintext.hex()}") encrypted = crypto.encrypt(plaintext, key)
        print(f"Encrypted (with length header): {encrypted.hex()}") decrypted = crypto.decrypt(encrypted, key)
        print(f"Decrypted: {decrypted.hex()}") match = decrypted == plaintext
        print(f"Match: {match}")
        return match

if __name__ == "__main__": test_fixed_crypto()
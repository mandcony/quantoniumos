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

"""Self-test suite for Crypto.Cipher.CAST"""

import unittest

# This is a list of (plaintext, ciphertext, key) tuples.
test_data = [
    # Test vectors from RFC 2144, B.1
    (
        "0123456789abcdef",
        "238b4fe5847e44b2",
        "0123456712345678234567893456789a",
        "128-bit key",
    ),
    ("0123456789abcdef", "eb6a711a2c02271b", "01234567123456782345", "80-bit key"),
    ("0123456789abcdef", "7ac816d16e9b302e", "0123456712", "40-bit key"),
]

class KeyLength(unittest.TestCase):
    def runTest(self):
        self.assertRaises(ValueError, CAST.new, bchr(0) * 4, CAST.MODE_ECB)
        self.assertRaises(ValueError, CAST.new, bchr(0) * 17, CAST.MODE_ECB)

class TestOutput(unittest.TestCase):
    def runTest(self):
        # Encrypt/Decrypt data and test output parameter

        cipher = CAST.new(b"4" * 16, CAST.MODE_ECB)

        pt = b"5" * 16
        ct = cipher.encrypt(pt)

        output = bytearray(16)
        res = cipher.encrypt(pt, output=output)
        self.assertEqual(ct, output)
        self.assertEqual(res, None)

        res = cipher.decrypt(ct, output=output)
        self.assertEqual(pt, output)
        self.assertEqual(res, None)

        output = memoryview(bytearray(16))
        cipher.encrypt(pt, output=output)
        self.assertEqual(ct, output)

        cipher.decrypt(ct, output=output)
        self.assertEqual(pt, output)

        self.assertRaises(TypeError, cipher.encrypt, pt, output=b"0" * 16)
        self.assertRaises(TypeError, cipher.decrypt, ct, output=b"0" * 16)

        shorter_output = bytearray(7)
        self.assertRaises(ValueError, cipher.encrypt, pt, output=shorter_output)
        self.assertRaises(ValueError, cipher.decrypt, ct, output=shorter_output)

def get_tests(config={}):
    from .common import make_block_tests

    tests = make_block_tests(CAST, "CAST", test_data)
    tests.append(KeyLength())
    tests.append(TestOutput())
    return tests

if __name__ == "__main__":

    def suite():
        return unittest.TestSuite(get_tests())

    unittest.main(defaultTest="suite")

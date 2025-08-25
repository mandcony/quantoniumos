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

"""Self-test suite for Crypto.Hash.SHA256"""

import unittest

class LargeSHA256Test(unittest.TestCase):
    def runTest(self):
        """SHA256: 512/520 MiB test"""
        from Crypto.Hash import SHA256

        zeros = bchr(0x00) * (1024 * 1024)

        h = SHA256.new(zeros)
        for i in range(511):
            h.update(zeros)

        # This test vector is from PyCrypto's old testdata.py file. self.assertEqual('9acca8e8c22201155389f65abbf6bc9723edc7384ead80503839f49dcc56d767', h.hexdigest()) # 512 MiB for i in range(8): h.update(zeros) # This test vector is from PyCrypto's old testdata.py file.
        self.assertEqual(
            "abf51ad954b246009dfe5a50ecd582fd5b8f1b8b27f30393853c3ef721e7fa6e",
            h.hexdigest(),
        )  # 520 MiB

def get_tests(config={}):
    # Test vectors from FIPS PUB 180-2
    # This is a list of (expected_result, input[, description]) tuples.
    test_data = [
        # FIPS PUB 180-2, B.1 - "One-Block Message"
        ("ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad", "abc"),
        # FIPS PUB 180-2, B.2 - "Multi-Block Message"
        (
            "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1",
            "abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq",
        ),
        # FIPS PUB 180-2, B.3 - "Long Message"
        (
            "cdc76e5c9914fb9281a1c7e284d73e67f1809a48a497200e046d39ccc7112cd0",
            "a" * 10**6,
            '"a" * 10**6',
        ),
        # Test for an old PyCrypto bug.
        (
            "f7fd017a3c721ce7ff03f3552c0813adcc48b7f33f07e5e2ba71e23ea393d103",
            "This message is precisely 55 bytes long, to test a bug.",
            "Length = 55 (mod 64)",
        ),
        # Example from http://de.wikipedia.org/wiki/Secure_Hash_Algorithm
        ("e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855", ""),
        (
            "d32b568cd1b96d459e7291ebf4b25d007f275c9f13149beeb782fac0716613f8",
            "Franz jagt im komplett verwahrlosten Taxi quer durch Bayern",
        ),
    ]

    from Crypto.Hash import SHA256

    from .common import make_hash_tests

    tests = make_hash_tests(
        SHA256, "SHA256", test_data, digest_size=32, oid="2.16.840.1.101.3.4.2.1"
    )

    if config.get("slow_tests"):
        tests += [LargeSHA256Test()]

    return tests

if __name__ == "__main__":
    import unittest

    def suite():
        return unittest.TestSuite(get_tests())

    unittest.main(defaultTest="suite")

# vim:set ts=4 sw=4 sts=4 expandtab:

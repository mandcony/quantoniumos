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

"""Self-test suite for Crypto.Hash.MD5"""

import unittest

# This is a list of (expected_result, input[, description]) tuples.
test_data = [
    # Test vectors from RFC 1321
    ("d41d8cd98f00b204e9800998ecf8427e", "", "'' (empty string)"),
    ("0cc175b9c0f1b6a831c399e269772661", "a"),
    ("900150983cd24fb0d6963f7d28e17f72", "abc"),
    ("f96b697d7cb7938d525a2f31aaf161d0", "message digest"),
    ("c3fcd3d76192e4007dfb496cca67e13b", "abcdefghijklmnopqrstuvwxyz", "a-z"),
    (
        "d174ab98d277d9f5a5611c2c9f419d9f",
        "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789",
        "A-Z, a-z, 0-9",
    ),
    (
        "57edf4a22be3c955ac49da2e2107b67a",
        "1234567890123456789012345678901234567890123456"
        + "7890123456789012345678901234567890",
        "'1234567890' * 8",
    ),
    # https://www.cosic.esat.kuleuven.be/nessie/testvectors/hash/md5/Md5-128.unverified.test-vectors
    ("57EDF4A22BE3C955AC49DA2E2107B67A", "1234567890" * 8, "Set 1, vector #7"),
    ("7707D6AE4E027C70EEA2A935C2296F21", "a" * 1000000, "Set 1, vector #8"),
]

class Md5IterTest(unittest.TestCase):
    def runTest(self):
        message = b("\x00") * 16
        result1 = "4AE71336E44BF9BF79D2752E234818A5".lower()
        result2 = "1A83F51285E4D89403D00C46EF8508FE".lower()

        h = MD5.new(message)
        message = h.digest()
        self.assertEqual(h.hexdigest(), result1)

        for _ in range(99999):
            h = MD5.new(message)
            message = h.digest()

        self.assertEqual(h.hexdigest(), result2)

def get_tests(config={}):
    from .common import make_hash_tests

    tests = make_hash_tests(
        MD5, "MD5", test_data, digest_size=16, oid="1.2.840.113549.2.5"
    )
    if config.get("slow_tests"):
        tests += [Md5IterTest()]
    return tests

if __name__ == "__main__":
    import unittest

    def suite():
        return unittest.TestSuite(get_tests())

    unittest.main(defaultTest="suite")

# vim:set ts=4 sw=4 sts=4 expandtab:

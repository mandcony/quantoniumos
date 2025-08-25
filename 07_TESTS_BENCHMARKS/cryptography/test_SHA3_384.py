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

"""Self-test suite for Crypto.Hash.SHA3_384"""

import unittest
from binascii import hexlify

class APITest(unittest.TestCase):
    def test_update_after_digest(self):
        msg = b("rrrrttt")

        # Normally, update() cannot be done after digest()
        h = SHA3.new(data=msg[:4])
        dig1 = h.digest()
        self.assertRaises(TypeError, h.update, msg[4:])
        dig2 = SHA3.new(data=msg).digest()

        # With the proper flag, it is allowed
        h = SHA3.new(data=msg[:4], update_after_digest=True)
        self.assertEqual(h.digest(), dig1)
        # ... and the subsequent digest applies to the entire message
        # up to that point
        h.update(msg[4:])
        self.assertEqual(h.digest(), dig2)

def get_tests(config={}):
    from .common import make_hash_tests

    tests = []

    test_vectors = (
        load_test_vectors(
            ("Hash", "SHA3"),
            "ShortMsgKAT_SHA3-384.txt",
            "KAT SHA-3 384",
            {"len": lambda x: int(x)},
        )
        or []
    )

    test_data = []
    for tv in test_vectors:
        if tv.len == 0:
            tv.msg = b("")
        test_data.append((hexlify(tv.md), tv.msg, tv.desc))

    tests += make_hash_tests(
        SHA3,
        "SHA3_384",
        test_data,
        digest_size=SHA3.digest_size,
        oid="2.16.840.1.101.3.4.2.9",
    )
    tests += list_test_cases(APITest)
    return tests

if __name__ == "__main__":
    import unittest

    def suite():
        return unittest.TestSuite(get_tests())

    unittest.main(defaultTest="suite")

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

"""Self-test suite for Crypto.Hash.MD4"""

__revision__ = "$Id$"

# This is a list of (expected_result, input[, description]) tuples.
test_data = [
    # Test vectors from RFC 1320
    ("31d6cfe0d16ae931b73c59d7e0c089c0", "", "'' (empty string)"),
    ("bde52cb31de33e46245e05fbdbd6fb24", "a"),
    ("a448017aaf21d8525fc10ae87aa6729d", "abc"),
    ("d9130a8164549fe818874806e1c7014b", "message digest"),
    ("d79e1c308aa5bbcdeea8ed63df412da9", "abcdefghijklmnopqrstuvwxyz", "a-z"),
    (
        "043f8582f241db351ce627e153e7f0e4",
        "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789",
        "A-Z, a-z, 0-9",
    ),
    (
        "e33b4ddc9c38f2199c3e7b164fcc0536",
        "1234567890123456789012345678901234567890123456"
        + "7890123456789012345678901234567890",
        "'1234567890' * 8",
    ),
]

def get_tests(config={}):
    from Crypto.Hash import MD4

    from .common import make_hash_tests

    return make_hash_tests(
        MD4, "MD4", test_data, digest_size=16, oid="1.2.840.113549.2.4"
    )

if __name__ == "__main__":
    import unittest

    def suite():
        return unittest.TestSuite(get_tests())

    unittest.main(defaultTest="suite")

# vim:set ts=4 sw=4 sts=4 expandtab:

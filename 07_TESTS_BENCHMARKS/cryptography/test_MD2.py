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

"""Self-test suite for Crypto.Hash.MD2"""

# This is a list of (expected_result, input[, description]) tuples.
test_data = [
    # Test vectors from RFC 1319
    ("8350e5a3e24c153df2275c9f80692773", "", "'' (empty string)"),
    ("32ec01ec4a6dac72c0ab96fb34c0b5d1", "a"),
    ("da853b0d3f88d99b30283a69e6ded6bb", "abc"),
    ("ab4f496bfb2a530b219ff33031fe06b0", "message digest"),
    ("4e8ddff3650292ab5a4108c3aa47940b", "abcdefghijklmnopqrstuvwxyz", "a-z"),
    (
        "da33def2a42df13975352846c30338cd",
        "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789",
        "A-Z, a-z, 0-9",
    ),
    (
        "d5976f79d83d3a0dc9806c3c66f3efd8",
        "1234567890123456789012345678901234567890123456"
        + "7890123456789012345678901234567890",
        "'1234567890' * 8",
    ),
]

def get_tests(config={}):
    from Crypto.Hash import MD2

    from .common import make_hash_tests

    return make_hash_tests(
        MD2, "MD2", test_data, digest_size=16, oid="1.2.840.113549.2.2"
    )

if __name__ == "__main__":
    import unittest

    def suite():
        return unittest.TestSuite(get_tests())

    unittest.main(defaultTest="suite")

# vim:set ts=4 sw=4 sts=4 expandtab:

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



# This is a list of (expected_result, input[, description]) tuples.
test_data = [
    # Test vectors downloaded 2008-09-12 from
    #   http://homes.esat.kuleuven.be/~bosselae/ripemd160.html
    ("9c1185a5c5e9fc54612808977ee8f548b2258d31", "", "'' (empty string)"),
    ("0bdc9d2d256b3ee9daae347be6f4dc835a467ffe", "a"),
    ("8eb208f7e05d987a9b044a8e98c6b087f15a0bfc", "abc"),
    ("5d0689ef49d2fae572b881b123a85ffa21595f36", "message digest"),
    ("f71c27109c692c1b56bbdceb5b9d2865b3708dbc", "abcdefghijklmnopqrstuvwxyz", "a-z"),
    (
        "12a053384a9c0c88e405a06c27dcf49ada62eb2b",
        "abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq",
        "abcdbcd...pnopq",
    ),
    (
        "b0e20b6e3116640286ed3a87a5713079b21f5189",
        "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789",
        "A-Z, a-z, 0-9",
    ),
    ("9b752e45573d4b39f4dbd3323cab82bf63326bfb", "1234567890" * 8, "'1234567890' * 8"),
    ("52783243c1697bdbe16d37f97f68f08325dc1528", "a" * 10**6, '"a" * 10**6'),
]

def get_tests(config={}):
    from Crypto.Hash import RIPEMD160

    from .common import make_hash_tests

    return make_hash_tests(
        RIPEMD160, "RIPEMD160", test_data, digest_size=20, oid="1.3.36.3.2.1"
    )

if __name__ == "__main__":
    import unittest

    def suite():
        return unittest.TestSuite(get_tests())

    unittest.main(defaultTest="suite")

# vim:set ts=4 sw=4 sts=4 expandtab:

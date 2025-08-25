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

"""Self-test suite for Crypto.Hash.SHA"""

from binascii import hexlify

# Test vectors from various sources
# This is a list of (expected_result, input[, description]) tuples.
test_data_various = [
    # FIPS PUB 180-2, A.1 - "One-Block Message"
    ("a9993e364706816aba3e25717850c26c9cd0d89d", "abc"),
    # FIPS PUB 180-2, A.2 - "Multi-Block Message"
    (
        "84983e441c3bd26ebaae4aa1f95129e5e54670f1",
        "abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq",
    ),
    # FIPS PUB 180-2, A.3 - "Long Message"
    #    ('34aa973cd4c4daa4f61eeb2bdbad27316534016f',
    #        'a' * 10**6,
    #         '"a" * 10**6'),
    # RFC 3174: Section 7.3, "TEST4" (multiple of 512 bits)
    ("dea356a2cddd90c7a7ecedc5ebb563934f460452", "01234567" * 80, '"01234567" * 80'),
]

def get_tests(config={}):
    from Crypto.Hash import SHA1

    from .common import make_hash_tests

    tests = []

    test_vectors = (
        load_test_vectors(
            ("Hash", "SHA1"), "SHA1ShortMsg.rsp", "KAT SHA-1", {"len": lambda x: int(x)}
        )
        or []
    )

    test_data = test_data_various[:]
    for tv in test_vectors:
        try:
            if tv.startswith("["):
                continue
        except AttributeError:
            pass
        if tv.len == 0:
            tv.msg = b""
        test_data.append((hexlify(tv.md), tv.msg, tv.desc))

    tests = make_hash_tests(
        SHA1, "SHA1", test_data, digest_size=20, oid="1.3.14.3.2.26"
    )
    return tests

if __name__ == "__main__":
    import unittest

    def suite():
        return unittest.TestSuite(get_tests())

    unittest.main(defaultTest="suite")

# vim:set ts=4 sw=4 sts=4 expandtab:

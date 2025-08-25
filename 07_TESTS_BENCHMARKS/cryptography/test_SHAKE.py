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

"""Self-test suite for Crypto.Hash.SHAKE128 and SHAKE256"""

import unittest

class SHAKETest(unittest.TestCase):
    def test_new_positive(self):
        xof1 = self.shake.new()
        xof2 = self.shake.new(data=b("90"))
        xof3 = self.shake.new().update(b("90"))

        self.assertNotEqual(xof1.read(10), xof2.read(10))
        xof3.read(10)
        self.assertEqual(xof2.read(10), xof3.read(10))

    def test_update(self):
        pieces = [bchr(10) * 200, bchr(20) * 300]
        h = self.shake.new()
        h.update(pieces[0]).update(pieces[1])
        digest = h.read(10)
        h = self.shake.new()
        h.update(pieces[0] + pieces[1])
        self.assertEqual(h.read(10), digest)

    def test_update_negative(self):
        h = self.shake.new()
        self.assertRaises(TypeError, h.update, "string")

    def test_digest(self):
        h = self.shake.new()
        digest = h.read(90)

        # read returns a byte string of the right length
        self.assertTrue(isinstance(digest, type(b("digest"))))
        self.assertEqual(len(digest), 90)

    def test_update_after_read(self):
        mac = self.shake.new()
        mac.update(b("rrrr"))
        mac.read(90)
        self.assertRaises(TypeError, mac.update, b("ttt"))

    def test_copy(self):
        mac = self.shake.new()
        mac.update(b("rrrr"))
        mac2 = mac.copy()
        x1 = mac.read(90)
        x2 = mac2.read(90)
        self.assertEqual(x1, x2)

class SHAKE128Test(SHAKETest):
    shake = SHAKE128

class SHAKE256Test(SHAKETest):
    shake = SHAKE256

class SHAKEVectors(unittest.TestCase):
    pass

test_vectors_128 = (
    load_test_vectors(
        ("Hash", "SHA3"),
        "ShortMsgKAT_SHAKE128.txt",
        "Short Messages KAT SHAKE128",
        {"len": lambda x: int(x)},
    )
    or []
)

for idx, tv in enumerate(test_vectors_128):
    if tv.len == 0:
        data = b("")
    else:
        data = tobytes(tv.msg)

    def new_test(self, data=data, result=tv.md):
        hobj = SHAKE128.new(data=data)
        digest = hobj.read(len(result))
        self.assertEqual(digest, result)

    setattr(SHAKEVectors, "test_128_%d" % idx, new_test)

test_vectors_256 = (
    load_test_vectors(
        ("Hash", "SHA3"),
        "ShortMsgKAT_SHAKE256.txt",
        "Short Messages KAT SHAKE256",
        {"len": lambda x: int(x)},
    )
    or []
)

for idx, tv in enumerate(test_vectors_256):
    if tv.len == 0:
        data = b("")
    else:
        data = tobytes(tv.msg)

    def new_test(self, data=data, result=tv.md):
        hobj = SHAKE256.new(data=data)
        digest = hobj.read(len(result))
        self.assertEqual(digest, result)

    setattr(SHAKEVectors, "test_256_%d" % idx, new_test)

def get_tests(config={}):
    tests = []
    tests += list_test_cases(SHAKE128Test)
    tests += list_test_cases(SHAKE256Test)
    tests += list_test_cases(SHAKEVectors)
    return tests

if __name__ == "__main__":
    import unittest

    def suite():
        return unittest.TestSuite(get_tests())

    unittest.main(defaultTest="suite")

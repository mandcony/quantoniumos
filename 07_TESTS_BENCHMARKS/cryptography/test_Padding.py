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

import unittest
from binascii import unhexlify as uh

class PKCS7_Tests(unittest.TestCase):
    def test1(self):
        padded = pad(b(""), 4)
        self.assertTrue(padded == uh(b("04040404")))
        padded = pad(b(""), 4, "pkcs7")
        self.assertTrue(padded == uh(b("04040404")))
        back = unpad(padded, 4)
        self.assertTrue(back == b(""))

    def test2(self):
        padded = pad(uh(b("12345678")), 4)
        self.assertTrue(padded == uh(b("1234567804040404")))
        back = unpad(padded, 4)
        self.assertTrue(back == uh(b("12345678")))

    def test3(self):
        padded = pad(uh(b("123456")), 4)
        self.assertTrue(padded == uh(b("12345601")))
        back = unpad(padded, 4)
        self.assertTrue(back == uh(b("123456")))

    def test4(self):
        padded = pad(uh(b("1234567890")), 4)
        self.assertTrue(padded == uh(b("1234567890030303")))
        back = unpad(padded, 4)
        self.assertTrue(back == uh(b("1234567890")))

    def testn1(self):
        self.assertRaises(ValueError, pad, uh(b("12")), 4, "pkcs8")

    def testn2(self):
        self.assertRaises(ValueError, unpad, b("\0\0\0"), 4)
        self.assertRaises(ValueError, unpad, b(""), 4)

    def testn3(self):
        self.assertRaises(ValueError, unpad, b("123456\x02"), 4)
        self.assertRaises(ValueError, unpad, b("123456\x00"), 4)
        self.assertRaises(ValueError, unpad, b("123456\x05\x05\x05\x05\x05"), 4)

class X923_Tests(unittest.TestCase):
    def test1(self):
        padded = pad(b(""), 4, "x923")
        self.assertTrue(padded == uh(b("00000004")))
        back = unpad(padded, 4, "x923")
        self.assertTrue(back == b(""))

    def test2(self):
        padded = pad(uh(b("12345678")), 4, "x923")
        self.assertTrue(padded == uh(b("1234567800000004")))
        back = unpad(padded, 4, "x923")
        self.assertTrue(back == uh(b("12345678")))

    def test3(self):
        padded = pad(uh(b("123456")), 4, "x923")
        self.assertTrue(padded == uh(b("12345601")))
        back = unpad(padded, 4, "x923")
        self.assertTrue(back == uh(b("123456")))

    def test4(self):
        padded = pad(uh(b("1234567890")), 4, "x923")
        self.assertTrue(padded == uh(b("1234567890000003")))
        back = unpad(padded, 4, "x923")
        self.assertTrue(back == uh(b("1234567890")))

    def testn1(self):
        self.assertRaises(ValueError, unpad, b("123456\x02"), 4, "x923")
        self.assertRaises(ValueError, unpad, b("123456\x00"), 4, "x923")
        self.assertRaises(ValueError, unpad, b("123456\x00\x00\x00\x00\x05"), 4, "x923")
        self.assertRaises(ValueError, unpad, b(""), 4, "x923")

class ISO7816_Tests(unittest.TestCase):
    def test1(self):
        padded = pad(b(""), 4, "iso7816")
        self.assertTrue(padded == uh(b("80000000")))
        back = unpad(padded, 4, "iso7816")
        self.assertTrue(back == b(""))

    def test2(self):
        padded = pad(uh(b("12345678")), 4, "iso7816")
        self.assertTrue(padded == uh(b("1234567880000000")))
        back = unpad(padded, 4, "iso7816")
        self.assertTrue(back == uh(b("12345678")))

    def test3(self):
        padded = pad(uh(b("123456")), 4, "iso7816")
        self.assertTrue(padded == uh(b("12345680")))
        back = unpad(padded, 4, "iso7816")
        self.assertTrue(back == uh(b("123456")))

    def test4(self):
        padded = pad(uh(b("1234567890")), 4, "iso7816")
        self.assertTrue(padded == uh(b("1234567890800000")))
        back = unpad(padded, 4, "iso7816")
        self.assertTrue(back == uh(b("1234567890")))

    def testn1(self):
        self.assertRaises(ValueError, unpad, b("123456\x81"), 4, "iso7816")
        self.assertRaises(ValueError, unpad, b(""), 4, "iso7816")

def get_tests(config={}):
    tests = []
    tests += list_test_cases(PKCS7_Tests)
    tests += list_test_cases(X923_Tests)
    tests += list_test_cases(ISO7816_Tests)
    return tests

if __name__ == "__main__":

    def suite():
        return unittest.TestSuite(get_tests())

    unittest.main(defaultTest="suite")

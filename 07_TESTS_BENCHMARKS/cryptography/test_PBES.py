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

"""Self-tests for Crypto.IO._PBES module"""

import unittest

class TestPBES2(unittest.TestCase):
    def setUp(self):
        self.ref = b"Test data"
        self.passphrase = b"Passphrase"

    def test1(self):
        ct = PBES2.encrypt(
            self.ref, self.passphrase, "PBKDF2WithHMAC-SHA1AndDES-EDE3-CBC"
        )
        pt = PBES2.decrypt(ct, self.passphrase)
        self.assertEqual(self.ref, pt)

    def test2(self):
        ct = PBES2.encrypt(
            self.ref, self.passphrase, "PBKDF2WithHMAC-SHA224AndAES128-CBC"
        )
        pt = PBES2.decrypt(ct, self.passphrase)
        self.assertEqual(self.ref, pt)

    def test3(self):
        ct = PBES2.encrypt(
            self.ref, self.passphrase, "PBKDF2WithHMAC-SHA256AndAES192-CBC"
        )
        pt = PBES2.decrypt(ct, self.passphrase)
        self.assertEqual(self.ref, pt)

    def test4(self):
        ct = PBES2.encrypt(
            self.ref, self.passphrase, "PBKDF2WithHMAC-SHA384AndAES256-CBC"
        )
        pt = PBES2.decrypt(ct, self.passphrase)
        self.assertEqual(self.ref, pt)

    def test5(self):
        ct = PBES2.encrypt(
            self.ref, self.passphrase, "PBKDF2WithHMAC-SHA512AndAES128-GCM"
        )
        pt = PBES2.decrypt(ct, self.passphrase)
        self.assertEqual(self.ref, pt)

    def test6(self):
        ct = PBES2.encrypt(
            self.ref, self.passphrase, "PBKDF2WithHMAC-SHA512-224AndAES192-GCM"
        )
        pt = PBES2.decrypt(ct, self.passphrase)
        self.assertEqual(self.ref, pt)

    def test7(self):
        ct = PBES2.encrypt(
            self.ref, self.passphrase, "PBKDF2WithHMAC-SHA3-256AndAES256-GCM"
        )
        pt = PBES2.decrypt(ct, self.passphrase)
        self.assertEqual(self.ref, pt)

    def test8(self):
        ct = PBES2.encrypt(self.ref, self.passphrase, "scryptAndAES128-CBC")
        pt = PBES2.decrypt(ct, self.passphrase)
        self.assertEqual(self.ref, pt)

    def test9(self):
        ct = PBES2.encrypt(self.ref, self.passphrase, "scryptAndAES192-CBC")
        pt = PBES2.decrypt(ct, self.passphrase)
        self.assertEqual(self.ref, pt)

    def test10(self):
        ct = PBES2.encrypt(self.ref, self.passphrase, "scryptAndAES256-CBC")
        pt = PBES2.decrypt(ct, self.passphrase)
        self.assertEqual(self.ref, pt)

def get_tests(config={}):
    from Crypto.SelfTest.st_common import list_test_cases

    listTests = []
    listTests += list_test_cases(TestPBES2)
    return listTests

if __name__ == "__main__":

    def suite():
        return unittest.TestSuite(get_tests())

    unittest.main(defaultTest="suite")

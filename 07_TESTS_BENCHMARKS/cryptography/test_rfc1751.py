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

import binascii
import unittest

class RFC1751_Tests(unittest.TestCase):
    def test1(self):
        data = [
            ("EB33F77EE73D4053", "TIDE ITCH SLOW REIN RULE MOT"),
            (
                "CCAC2AED591056BE4F90FD441C534766",
                "RASH BUSH MILK LOOK BAD BRIM AVID GAFF BAIT ROT POD LOVE",
            ),
            (
                "EFF81F9BFBC65350920CDD7416DE8009",
                "TROD MUTE TAIL WARM CHAR KONG HAAG CITY BORE O TEAL AWL",
            ),
        ]

        for key_hex, words in data:
            key_bin = binascii.a2b_hex(key_hex)

            w2 = key_to_english(key_bin)
            self.assertEqual(w2, words)

            k2 = english_to_key(words)
            self.assertEqual(k2, key_bin)

    def test_error_key_to_english(self):
        self.assertRaises(ValueError, key_to_english, b"0" * 7)

def get_tests(config={}):
    from Crypto.SelfTest.st_common import list_test_cases

    tests = list_test_cases(RFC1751_Tests)
    return tests

if __name__ == "__main__":
    suite = lambda: unittest.TestSuite(get_tests())
    unittest.main(defaultTest="suite")

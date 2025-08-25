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

"""Self-tests for Crypto.Util.Counter"""

import unittest

class CounterTests(unittest.TestCase):
    def setUp(self):
        global Counter
        from Crypto.Util import Counter

    def test_BE(self):
        """Big endian"""
        Counter.new(128)
        Counter.new(128, little_endian=False)

    def test_LE(self):
        """Little endian"""
        Counter.new(128, little_endian=True)

    def test_nbits(self):
        Counter.new(nbits=128)
        self.assertRaises(ValueError, Counter.new, 129)

    def test_prefix(self):
        Counter.new(128, prefix=b("xx"))

    def test_suffix(self):
        Counter.new(128, suffix=b("xx"))

    def test_iv(self):
        Counter.new(128, initial_value=2)
        self.assertRaises(ValueError, Counter.new, 16, initial_value=0x1FFFF)

def get_tests(config={}):
    from Crypto.SelfTest.st_common import list_test_cases

    return list_test_cases(CounterTests)

if __name__ == "__main__":

    def suite():
        return unittest.TestSuite(get_tests())

    unittest.main(defaultTest="suite")

# vim:set ts=4 sw=4 sts=4 expandtab:
